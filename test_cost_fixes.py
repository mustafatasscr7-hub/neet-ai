"""Batch 1 of the 2026-10-17 cost/quality work in server.py.

Run with: python test_cost_fixes.py
Plain pass/fail harness like test_answer_quality.py. Fully offline: Supabase is a recording fake,
every model call is a fake, and the peak-hour clock is patched.

1. answer_cache: only a chat's first message (no history) is looked up or stored; key covers
   prompt version + language + answer style; 30-day expiry; never stores clarifying, corrected,
   self-correcting, cut-off, personalised or name-bearing answers; every hit/miss is logged.
2. DeepSeek model name + current rates.
3. provider_usage_log: cache-hit tokens + output-cap columns, new endpoints logged.
4. Budget fairness: cached input never counts, on Qwen or DeepSeek.
5. Hinglish on Qwen: call-time guard, and a bare "DOUBT_TYPE: conversational" retries on Qwen.
6. Hindi answers get a 1,400-token cap; any cap hit is logged and kept out of the caches.
7. (batch 2) Earlier AI answers in chat history are cut to ~600 characters.
8. (batch 2) End-of-chapter exercise lists are kept out of NCERT context.
"""
import asyncio
import io
import os
import sys
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server
from starlette.requests import Request

_passed = []
_failed = []


def check(label, condition, detail=""):
    if condition:
        _passed.append(label)
        print(f"  PASS: {label}")
    else:
        _failed.append(label)
        print(f"  FAIL: {label}" + (f" -- {detail}" if detail else ""))


# ---------- fakes ----------

class FakeResp:
    def __init__(self, status_code=200, data=None, text=""):
        self.status_code = status_code
        self._data = data if data is not None else []
        self.text = text

    def json(self):
        return self._data


def _table(url):
    return str(url).split("/rest/v1/")[-1].split("?")[0]


class FakeClient:
    """Records every call. GET returns get_rows[table]; POST to a table returns the next status in
    post_statuses[table] (default 201)."""
    def __init__(self, get_rows=None, post_statuses=None):
        self.posts, self.gets, self.patches = [], [], []
        self.get_rows = get_rows or {}
        self.post_statuses = post_statuses or {}

    async def post(self, url, headers=None, json=None, params=None, **kw):
        self.posts.append(SimpleNamespace(table=_table(url), headers=headers or {}, json=json, params=params or {}))
        queue = self.post_statuses.get(_table(url))
        status, text = queue.pop(0) if queue else (201, "")
        return FakeResp(status, [], text)

    async def get(self, url, headers=None, params=None, **kw):
        self.gets.append(SimpleNamespace(table=_table(url), params=params or {}))
        return FakeResp(200, list(self.get_rows.get(_table(url), [])))

    async def patch(self, url, headers=None, params=None, json=None, **kw):
        self.patches.append(SimpleNamespace(table=_table(url), params=params or {}, json=json))
        return FakeResp(204)

    def posts_to(self, table):
        return [p for p in self.posts if p.table == table]

    def gets_of(self, table):
        return [g for g in self.gets if g.table == table]


GOOD = "VISUAL_INTENT: no\n\nNEET Importance: 4/5\n\nAnswer:\n- Osmosis is the movement of water across a semi-permeable membrane.\n\nKey Points:\n- Water moves to higher solute concentration."
CONVERSATIONAL = "DOUBT_TYPE: conversational\n\nHappy to help -- ask me any NEET doubt!"


class Harness:
    """Patches server's HTTP client, lookups and the two text-model helpers (_stream_qwen /
    _stream_deepseek), so stream_response's real routing, retry and cache logic all run."""
    def __init__(self, qwen_outputs=(), deepseek_outputs=(), peak=False, get_rows=None, ncert=None, mistakes=None,
                 qwen_flags=None, deepseek_flags=None):
        self.qwen_outputs = list(qwen_outputs)
        self.deepseek_outputs = list(deepseek_outputs)
        self.qwen_flags = qwen_flags or {}
        self.deepseek_flags = deepseek_flags or {}
        self.peak = peak
        self.ncert = ncert or []
        self.mistakes = mistakes or []
        self.calls = []
        self.spawned = []
        self.client = FakeClient(get_rows=get_rows)

    def __enter__(self):
        h = self
        names = ("async_client", "_stream_qwen", "_stream_deepseek", "check_max_usage_tier", "_spawn_background",
                 "search_ncert", "_diagram_exists_for", "fetch_student_mistake_rows", "_is_deepseek_peak_hour")
        self._saved = {n: getattr(server, n) for n in names}

        async def fake_qwen(system, messages, user_id, ip, endpoint, billing_context=None, max_tokens=1024):
            h.calls.append(SimpleNamespace(model="qwen", system=system, messages=messages, max_tokens=max_tokens, endpoint=endpoint))
            if billing_context is not None:
                billing_context.update(h.qwen_flags)
            yield h.qwen_outputs.pop(0)

        async def fake_deepseek(system, messages, user_id, ip, is_peak, endpoint, billing_context=None, max_tokens=1024):
            h.calls.append(SimpleNamespace(model="deepseek", system=system, messages=messages, max_tokens=max_tokens, endpoint=endpoint))
            if billing_context is not None:
                billing_context.update(h.deepseek_flags)
            yield h.deepseek_outputs.pop(0)

        async def fake_none(*a, **k):
            return None

        async def fake_search(text, limit=3):
            return h.ncert

        async def fake_false(text):
            return False

        async def fake_mistakes(user_id):
            return h.mistakes

        server.async_client = self.client
        server._stream_qwen = fake_qwen
        server._stream_deepseek = fake_deepseek
        server.check_max_usage_tier = fake_none
        server._spawn_background = lambda coro: h.spawned.append(coro)
        server.search_ncert = fake_search
        server._diagram_exists_for = fake_false
        server.fetch_student_mistake_rows = fake_mistakes
        server._is_deepseek_peak_hour = lambda: h.peak
        return self

    def __exit__(self, *exc):
        for name, value in self._saved.items():
            setattr(server, name, value)
        for coro in self.spawned:
            coro.close()

    def models(self):
        return [c.model for c in self.calls]

    def cache_events(self):
        return [(p.json["event"], p.json["reason"]) for p in self.client.posts_to("answer_cache_log")]


async def _drain(gen):
    out = "".join([c async for c in gen])
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    return out


def chat(text, **kw):
    kw.setdefault("ip", "1.2.3.4")
    return asyncio.run(_drain(server.stream_response(text, **kw)))


def fake_request():
    return Request({"type": "http", "method": "POST", "path": "/", "headers": [], "client": ("1.2.3.4", 1234)})


# ---------- 1. answer_cache ----------

def test_cache_key_and_eligibility():
    print("\n[1a] answer_cache key + who may use it")
    k = server._answer_cache_key
    check("key ignores case and spacing", k("What is  Osmosis?", "en", "detailed") == k("what is osmosis?", "en", "detailed"))
    check("key differs by language", k("osmosis", "en", "detailed") != k("osmosis", "hi", "detailed"))
    check("key differs by answer style", k("osmosis", "en", "detailed") != k("osmosis", "en", "concise"))
    check("unknown style counts as detailed", k("osmosis", "en", "detailed") == k("osmosis", "en", ""))
    import hashlib
    legacy = hashlib.sha256("en:osmosis".encode()).hexdigest()
    check("old-format key (language:text) never matches", legacy != k("osmosis", "en", "detailed"))
    check("version = format tag + prompt hash", server.ANSWER_CACHE_PROMPT_VERSION.startswith(server.ANSWER_CACHE_FORMAT_TAG + "-"))
    expected = hashlib.sha256("\x00".join([
        server.SYSTEM_PROMPT, server.SYSTEM_PROMPT_NO_CLARIFY, server.CONCISE_STYLE_CONTEXT, server.HINDI_LANG_CONTEXT,
        server.HINGLISH_GUARD, server.MNEMONIC_QUALITY_GUARD, server.INERT_PAIR_GUARD, server.MCQ_AMBIGUOUS_GUARD,
    ]).encode()).hexdigest()[:10]
    check("version changes with the prompt text", server.ANSWER_CACHE_PROMPT_VERSION.endswith(expected))
    skip = server._answer_cache_skip_reason
    check("first message, text only -> eligible", skip([], None, [], "", False) is None)
    check("history -> skip", skip([], None, [{"role": "user", "text": "x"}], "", False) == "history")
    check("running summary alone -> skip", skip([], None, [], "earlier: osmosis", False) == "history")
    check("retry -> skip", skip([], None, [], "", True) == "retry")
    check("image -> skip", skip(["img"], None, [], "", False) == "media")
    check("pdf -> skip", skip([], "pdfdata", [], "", False) == "media")


def test_cache_store_rules():
    print("\n[1b] what may be stored")
    block = server._answer_cache_store_block
    check("clean answer -> storable", block(GOOD, "en", "", False, {}) is None)
    check("conversational reply -> storable", block(CONVERSATIONAL, "en", "", False, {}) is None)
    check("bare marker -> never", block("DOUBT_TYPE: conversational", "en", "", False, {}) == "empty")
    check("AMBIGUOUS clarify -> never", block("AMBIGUOUS: yes\nCLARIFY_TYPE: format\nQUESTION: x", "en", "", False, {}) == "clarifying")
    check("cut off at cap -> never", block(GOOD, "en", "", False, {"output_capped": True}) == "capped")
    check("verifier-corrected -> never", block(GOOD, "en", "", False, {"corrected": True}) == "corrected")
    check("weak-chapter context used -> never", block(GOOD, "en", "", True, {}) == "personalised")
    check("contains the student's full name -> never", block("Great question, Riya Sharma! " + GOOD, "en", "Riya Sharma", False, {}) == "name")
    check("contains just the first name -> never", block("Riya, here it is.\n" + GOOD, "en", "Riya Sharma", False, {}) == "name")
    check("name given but not used -> storable", block(GOOD, "en", "Riya Sharma", False, {}) is None)
    check("name inside a longer word doesn't count", block(GOOD.replace("Osmosis", "Riyadh osmosis"), "en", "Riya", False, {}) is None)
    check("Hindi answer written with a name in the prompt -> never (can't check Devanagari spelling)",
          block("उत्तर: परासरण", "hi", "Riya", False, {}) == "name_unverifiable")
    check("Hindi answer with no name -> storable", block("उत्तर: परासरण", "hi", "", False, {}) is None)
    check("phone-number 'name' is ignored", block(GOOD, "en", "+91 98765 43210", False, {}) is None)
    check("self-correction -> never (and checked last)", block("Wait -- let me re-check. " + GOOD, "en", "", False, {}) == "self_correction")
    check("self-correction + personalised -> personalised (no regeneration)", block("Wait -- let me re-check.", "en", "", True, {}) == "personalised")


def test_cache_flow():
    print("\n[1c] lookup / store / logging through stream_response")
    with Harness(deepseek_outputs=[GOOD]) as h:
        out = chat("What is osmosis?", user_id="u1", student_name="Riya", personalize=True)
        gets = h.client.gets_of("answer_cache")
        check("first message: looked up", len(gets) == 1)
        p = gets[0].params if gets else {}
        check("lookup pins the current prompt version", p.get("prompt_version") == f"eq.{server.ANSWER_CACHE_PROMPT_VERSION}")
        cutoff = datetime.fromisoformat(p.get("created_at", "gte.2000-01-01T00:00:00+00:00")[4:])
        check("lookup ignores rows older than 30 days", abs((datetime.now(timezone.utc) - cutoff) - timedelta(days=30)) < timedelta(minutes=1))
        check("miss -> model answers", out == GOOD and h.models() == ["deepseek"])
        posts = h.client.posts_to("answer_cache")
        row = posts[0].json if posts else {}
        check("logged-in student's first-message answer is stored (no name used, no weak chapters)", len(posts) == 1)
        check("stored with version, language, style, fresh created_at, cleared alternate",
              row.get("prompt_version") == server.ANSWER_CACHE_PROMPT_VERSION and row.get("language") == "en"
              and row.get("answer_style") == "detailed" and row.get("created_at") and row.get("alternate_method", "x") is None, str(row))
        check("stored by upsert on question_hash", posts and posts[0].headers.get("Prefer") == "resolution=merge-duplicates"
              and posts[0].params.get("on_conflict") == "question_hash")
        check("key = _answer_cache_key(text, language, style)", row.get("question_hash") == server._answer_cache_key("What is osmosis?", "en", "detailed"))
        check("logged: miss then stored", h.cache_events() == [("miss", None), ("stored", None)], str(h.cache_events()))
        log = h.client.posts_to("answer_cache_log")[0].json
        check("log row has no user id or text", "user_id" not in log and "text" not in log and log["is_guest"] is False and log["prompt_version"])

    hit_row = {"answer": "cached osmosis answer", "alternate_method": None}
    with Harness(get_rows={"answer_cache": [hit_row]}) as h:
        out = chat("What is osmosis?", personalize=False)
        check("hit: cached answer served, no model call, no store", out == "cached osmosis answer" and not h.calls and not h.client.posts_to("answer_cache"))
        check("hit logged, as a guest", h.cache_events() == [("hit", None)] and h.client.posts_to("answer_cache_log")[0].json["is_guest"] is True)

    history = [{"role": "user", "text": "what is krebs cycle"}, {"role": "ai", "text": "VISUAL_INTENT: no\n...\nWant the steps?"}]
    with Harness(deepseek_outputs=[GOOD], get_rows={"answer_cache": [hit_row]}) as h:
        out = chat("yes", history=history)
        check("'yes' after history: never looked up (even though a row exists)", not h.client.gets_of("answer_cache") and out == GOOD)
        check("...never stored", not h.client.posts_to("answer_cache"))
        check("...logged as skip/history", h.cache_events() == [("skip", "history")], str(h.cache_events()))

    with Harness(deepseek_outputs=[CONVERSATIONAL]) as h:
        chat("yes")
        check("'yes' as a chat's first message: generic reply is stored", len(h.client.posts_to("answer_cache")) == 1)

    with Harness(deepseek_outputs=[GOOD]) as h:
        chat("What is osmosis?", running_summary="earlier: diffusion")
        check("running summary without history: skipped", not h.client.gets_of("answer_cache") and not h.client.posts_to("answer_cache"))

    with Harness(deepseek_outputs=[GOOD]) as h:
        chat("What is osmosis?", skip_cache=True)
        check("Retry: skipped, logged skip/retry", not h.client.gets_of("answer_cache") and h.cache_events() == [("skip", "retry")])

    with Harness(deepseek_outputs=[GOOD]) as h:
        chat("What is osmosis?", answer_style="concise", language="hi")
        row = h.client.posts_to("answer_cache")[0].json
        check("Concise + Hindi stored under their own key", row["question_hash"] == server._answer_cache_key("What is osmosis?", "hi", "concise")
              and row["answer_style"] == "concise" and row["language"] == "hi")

    clarify = "AMBIGUOUS: yes\nCLARIFY_TYPE: format\nQUESTION: Explain or show?\nOPTION: Explain it\nOPTION: Show me the diagram"
    cases = [
        ("clarifying reply", dict(deepseek_outputs=[clarify]), {}, "clarifying"),
        ("answer naming the student", dict(deepseek_outputs=["Great question, Riya!\n" + GOOD]), dict(user_id="u1", student_name="Riya"), "name"),
        ("answer cut off at its cap", dict(deepseek_outputs=[GOOD], deepseek_flags={"output_capped": True}), {}, "capped"),
        ("verifier-corrected answer", dict(deepseek_outputs=[GOOD], deepseek_flags={"corrected": True}), {}, "corrected"),
        ("weak-chapter context used", dict(deepseek_outputs=[GOOD], ncert=[{"subject": "Biology", "class": 11, "chapter_name": "Transport in Plants", "content": "osmosis"}],
                                           mistakes=[{"subject": "Biology", "chapter": "Transport in Plants"}]), dict(user_id="u1"), "personalised"),
    ]
    for label, harness_kw, chat_kw, reason in cases:
        with Harness(**harness_kw) as h:
            chat("What is osmosis?", **chat_kw)
            check(f"not stored: {label} (logged '{reason}')", not h.client.posts_to("answer_cache")
                  and ("not_stored", reason) in h.cache_events(), str(h.cache_events()))

    dirty = "VISUAL_INTENT: no\n\nAnswer:\n- Wait -- let me re-check the direction of water flow."
    with Harness(deepseek_outputs=[dirty, GOOD]) as h:
        chat("What is osmosis?", language="hi", answer_style="concise")
        check("self-correcting answer: not stored, regeneration scheduled", not h.client.posts_to("answer_cache") and len(h.spawned) == 1)
        regen = h.spawned.pop()
        asyncio.run(_drain(_as_gen(regen)))
        posts = h.client.posts_to("answer_cache")
        check("regeneration stores under the same key/language/style", len(posts) == 1
              and posts[0].json["question_hash"] == server._answer_cache_key("What is osmosis?", "hi", "concise")
              and posts[0].json["answer_style"] == "concise", str([p.json for p in posts]))
        check("regeneration gets the Hindi cap", h.calls[-1].max_tokens == server.ANSWER_MAX_TOKENS_HI)
        check("regeneration logged as stored", ("stored", "regenerated") in h.cache_events(), str(h.cache_events()))


async def _as_gen(coro):
    await coro
    if False:
        yield ""


def test_cache_expiry_and_migration_safety():
    print("\n[1d] lookup failure modes")
    class ErrorClient(FakeClient):
        async def get(self, url, headers=None, params=None, **kw):
            return FakeResp(400, {"message": "column answer_cache.prompt_version does not exist"})
    saved = server.async_client
    try:
        server.async_client = ErrorClient()
        check("migration not run yet (400): treated as a miss, no crash", asyncio.run(server._answer_cache_lookup("k")) is None)
        server.async_client = FakeClient(get_rows={"answer_cache": [{"answer": "", "alternate_method": None}]})
        check("row with an empty answer is not a hit", asyncio.run(server._answer_cache_lookup("k")) is None)
    finally:
        server.async_client = saved


def test_chat_alternate_method_cache():
    print("\n[1e] /chat-alternate-method uses the same first-message rows")
    saved = {n: getattr(server, n) for n in ("async_client", "enforce_daily_budget", "log_token_usage", "generate_alternate_method", "log_provider_usage")}
    logged = []

    async def no_budget(*a, **k):
        return None

    async def fake_log(*a, **k):
        logged.append((a, k))

    def fake_generate(question, primary_answer, options_context="", language="en", usage_out=None):
        if usage_out is not None:
            usage_out["response"] = SimpleNamespace(usage=SimpleNamespace(input_tokens=300, output_tokens=200, cache_read_input_tokens=0, cache_creation_input_tokens=0), stop_reason="end_turn")
        return "Use energy conservation.", 500

    try:
        server.enforce_daily_budget = no_budget
        server.log_token_usage = no_budget
        server.log_provider_usage = fake_log
        server.generate_alternate_method = fake_generate
        row = {"answer": "cached answer", "alternate_method": None}
        for first_turn in (False, True):
            server.async_client = FakeClient(get_rows={"answer_cache": [row]})
            req = server.ChatAlternateMethodRequest(question="block on a ramp", primary_answer="kinematics", answer_style="concise", first_turn=first_turn)
            out = asyncio.run(server.chat_alternate_method_endpoint(req, fake_request(), None, "uid-1"))
            gets, patches = server.async_client.gets_of("answer_cache"), server.async_client.patches
            if first_turn:
                key = server._answer_cache_key("block on a ramp", "en", "concise")
                check("first message: looks up and saves on the shared row (same key as /chat)",
                      len(gets) == 1 and gets[0].params["question_hash"] == f"eq.{key}" and len(patches) == 1
                      and patches[0].params["question_hash"] == f"eq.{key}" and out["alternate_method"].startswith("Use energy"))
            else:
                check("not a first message: shared row never read or written", not gets and not patches and out["alternate_method"].startswith("Use energy"))
        check("alternate-method generations are cost-logged under their own endpoint",
              [a[5] for a, k in logged] == ["/chat-alternate-method", "/chat-alternate-method"], str([a[5] for a, k in logged]))
        server.async_client = FakeClient(get_rows={"answer_cache": [{"answer": "a", "alternate_method": "cached method"}]})
        req = server.ChatAlternateMethodRequest(question="q", primary_answer="p", first_turn=True)
        check("cached alternate method served for a first message", asyncio.run(server.chat_alternate_method_endpoint(req, fake_request(), None, "uid-1")) == {"alternate_method": "cached method"})
    finally:
        for n, v in saved.items():
            setattr(server, n, v)


# ---------- 2. DeepSeek model + rates ----------

def test_model_and_rates():
    print("\n[2] DeepSeek model name + rates")
    check("model is deepseek-flash", server.DEEPSEEK_MODEL == "deepseek-flash")
    src = open(server.__file__, encoding="utf-8").read()
    check("no call still names deepseek-v4-flash", 'model="deepseek-v4-flash"' not in src and 'log_provider_usage("deepseek-v4-flash"' not in src)
    r = server.DEEPSEEK_RATES
    check("off-peak 0.003 / 0.15 / 0.60", r["off_peak"] == {"cache_hit": 0.003, "cache_miss": 0.15, "output": 0.60})
    check("peak 0.006 / 0.30 / 1.20", r["peak"] == {"cache_hit": 0.006, "cache_miss": 0.30, "output": 1.20})
    cost = server._deepseek_cost(2000, 9728, 500, False)
    check("cost of a typical off-peak /chat call", abs(cost - (2000 * 0.15 + 9728 * 0.003 + 500 * 0.6) / 1e6) < 1e-12, cost)


# ---------- 3 + 4. provider_usage_log columns, new endpoints, budget fairness ----------

class FakeQwenStream:
    def __init__(self, chunks):
        self.chunks = chunks

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for c in self.chunks:
            yield c


class FakeQwenClient:
    def __init__(self, chunks):
        self.chunks = chunks
        self.kwargs = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        self.kwargs = kw
        return FakeQwenStream(self.chunks)


def qwen_chunks(text, finish, prompt, cached, completion):
    delta = SimpleNamespace(content=text)
    return [
        SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=None)], usage=None),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason=finish)], usage=None),
        SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                                                          prompt_tokens_details=SimpleNamespace(cached_tokens=cached))),
    ]


class FakeDeepSeekStream:
    def __init__(self, text, final):
        self.text = text
        self.final = final

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @property
    def text_stream(self):
        async def gen():
            yield self.text
        return gen()

    async def get_final_message(self):
        return self.final


class FakeDeepSeekClient:
    def __init__(self, text, final):
        self.kwargs = None
        self._stream = FakeDeepSeekStream(text, final)
        self.messages = SimpleNamespace(stream=self._make)

    def _make(self, **kw):
        self.kwargs = kw
        return self._stream


class Recorder:
    """Patches log_provider_usage and log_token_usage_with_bonus to record their arguments."""
    def __enter__(self):
        self.provider, self.budget = [], []
        self._saved = (server.log_provider_usage, server.log_token_usage_with_bonus, server.qwen_async_client, server.deepseek_async_client)

        async def rec_provider(*a, **k):
            self.provider.append(SimpleNamespace(args=a, **k))

        async def rec_budget(user_id, tokens, ip=""):
            self.budget.append(tokens)

        server.log_provider_usage = rec_provider
        server.log_token_usage_with_bonus = rec_budget
        return self

    def __exit__(self, *exc):
        server.log_provider_usage, server.log_token_usage_with_bonus, server.qwen_async_client, server.deepseek_async_client = self._saved


def test_provider_helpers_and_fairness():
    print("\n[3/4] provider helpers: cache hits, caps, budget")
    # Same request shape on both providers: 11,728 prompt tokens of which 9,728 cached, 500 out.
    with Recorder() as rec:
        server.qwen_async_client = FakeQwenClient(qwen_chunks("answer", "stop", 11728, 9728, 500))
        meta = {"bill": True}
        out = asyncio.run(_drain(server._stream_qwen("SYS", [], "u1", "1.2.3.4", "/chat", meta, max_tokens=1400)))
        row = rec.provider[0]
        check("Qwen: streams the text", out == "answer")
        check("Qwen: max_tokens passed through", server.qwen_async_client.kwargs["max_tokens"] == 1400)
        check("Qwen: cache-hit tokens logged", row.cache_hit_tokens == 9728 and row.args[2] == 11728 and row.output_capped is False)
        check("Qwen: budget = cache-miss input + output (cached input never counts)", rec.budget == [2000 + 500], str(rec.budget))
        qwen_budget = rec.budget[0]

    with Recorder() as rec:
        final = SimpleNamespace(usage=SimpleNamespace(input_tokens=2000, cache_creation_input_tokens=0, cache_read_input_tokens=9728, output_tokens=500), stop_reason="end_turn")
        server.deepseek_async_client = FakeDeepSeekClient("answer", final)
        asyncio.run(_drain(server._stream_deepseek("SYS", [], "u1", "1.2.3.4", False, "/chat", {"bill": True})))
        row = rec.provider[0]
        check("DeepSeek: logged as deepseek-flash with cache hits", row.args[0] == "deepseek-flash" and row.cache_hit_tokens == 9728 and row.args[2] == 11728)
        check("DeepSeek: default cap is 1024", server.deepseek_async_client.kwargs["max_tokens"] == 1024 and server.deepseek_async_client.kwargs["model"] == "deepseek-flash")
        check("budget fairness: same doubt costs the same on Qwen and DeepSeek", rec.budget == [qwen_budget], f"{rec.budget} vs {qwen_budget}")

    with Recorder() as rec:
        server.qwen_async_client = FakeQwenClient(qwen_chunks("cut off mid-wo", "length", 12000, 9728, 1400))
        meta = {"bill": True}
        asyncio.run(_drain(server._stream_qwen("SYS", [], "u1", "1.2.3.4", "/chat", meta, max_tokens=1400)))
        check("Qwen cap hit: logged as output_capped", rec.provider[0].output_capped is True)
        check("Qwen cap hit: flags the response (kept out of caches)", meta.get("output_capped") is True)
        meta = {"bill": True}
        server.qwen_async_client = FakeQwenClient(qwen_chunks("A: TRUE", "length", 900, 0, 220))
        asyncio.run(_drain(server._stream_qwen("SYS", [], "u1", "1.2.3.4", "/chat-verify", meta, max_tokens=220)))
        check("verification call hitting ITS cap doesn't flag the answer", "output_capped" not in meta and rec.provider[-1].output_capped is True)

    with Recorder() as rec:
        final = SimpleNamespace(usage=SimpleNamespace(input_tokens=1500, cache_creation_input_tokens=0, cache_read_input_tokens=9728, output_tokens=1400), stop_reason="max_tokens")
        server.deepseek_async_client = FakeDeepSeekClient("cut off", final)
        meta = {"bill": True}
        asyncio.run(_drain(server._stream_deepseek("SYS", [], "u1", "1.2.3.4", False, "/solve", meta, max_tokens=1400)))
        check("DeepSeek cap hit: logged + flagged", rec.provider[0].output_capped is True and meta.get("output_capped") is True)

    FR = server.genai_types.FinishReason
    check("Gemini MAX_TOKENS detected", server._gemini_output_capped(SimpleNamespace(candidates=[SimpleNamespace(finish_reason=FR.MAX_TOKENS)])))
    check("Gemini STOP is not a cap hit", not server._gemini_output_capped(SimpleNamespace(candidates=[SimpleNamespace(finish_reason=FR.STOP)])))
    check("Gemini chunk without candidates is not a cap hit", not server._gemini_output_capped(SimpleNamespace(candidates=None)))


def test_log_provider_usage_columns():
    print("\n[3] provider_usage_log row + pre-migration fallback")
    saved = server.async_client
    try:
        server.async_client = FakeClient()
        asyncio.run(server.log_provider_usage("deepseek-flash", False, 11728, 500, 0.0006, "/chat", "u1", cache_hit_tokens=9728, output_capped=True))
        row = server.async_client.posts_to("provider_usage_log")[0].json
        check("row carries cache_hit_tokens + output_capped", row["cache_hit_tokens"] == 9728 and row["output_capped"] is True and row["input_tokens"] == 11728)
        server.async_client = FakeClient(post_statuses={"provider_usage_log": [(400, '{"message":"Could not find the \'cache_hit_tokens\' column"}')]})
        asyncio.run(server.log_provider_usage("qwen-flash", True, 100, 10, 0.0, "/chat-verify", "", cache_hit_tokens=5))
        posts = server.async_client.posts_to("provider_usage_log")
        check("before the migration: retried without the new columns", len(posts) == 2 and "cache_hit_tokens" not in posts[1].json
              and "output_capped" not in posts[1].json and posts[1].json["endpoint"] == "/chat-verify")
    finally:
        server.async_client = saved


def test_new_endpoints_logged():
    print("\n[3] /title, /summarize-answer, alternate methods, usage summary")
    saved = {n: getattr(server, n) for n in ("deepseek_client", "log_provider_usage", "log_token_usage", "enforce_daily_budget",
                                             "async_client", "generate_alternate_method", "get_user_plan", "_fetch_usage_rows")}
    logged = []

    async def rec(*a, **k):
        logged.append(SimpleNamespace(args=a, **k))

    async def noop(*a, **k):
        return None

    def response(text, stop="end_turn"):
        return SimpleNamespace(content=[SimpleNamespace(text=text)], stop_reason=stop,
                               usage=SimpleNamespace(input_tokens=40, output_tokens=15, cache_read_input_tokens=320, cache_creation_input_tokens=0))

    try:
        server.log_provider_usage = rec
        server.log_token_usage = noop
        server.enforce_daily_budget = noop
        server.deepseek_client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: response("Osmosis Explained", stop="max_tokens")))
        out = asyncio.run(server.generate_title(server.Message(text="explain osmosis"), fake_request(), None, "u1"))
        check("/title still returns its title", out == {"title": "Osmosis Explained"})
        check("/title cost-logged, with cache hits and its cap hit", logged and logged[-1].args[5] == "/title" and logged[-1].cache_hit_tokens == 320
              and logged[-1].output_capped is True and logged[-1].args[0] == "deepseek-flash")
        server.deepseek_client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: response("- point")))
        asyncio.run(server.summarize_answer(server.SummarizeAnswerRequest(text="long answer"), fake_request(), None, "u1"))
        check("/summarize-answer cost-logged", logged[-1].args[5] == "/summarize-answer" and logged[-1].output_capped is False)

        def fake_generate(question, primary_answer, options_context="", language="en", usage_out=None):
            usage_out["response"] = response("ALTERNATE EXISTS")
            return "method", 55
        server.generate_alternate_method = fake_generate
        server.async_client = FakeClient()
        asyncio.run(server.solve_alternate_method_endpoint(server.SolveAlternateMethodRequest(pyq_id="p1", question="q", primary_solution="s"), fake_request(), None, ""))
        check("/solve-alternate-method cost-logged", logged[-1].args[5] == "/solve-alternate-method")

        async def plan(user_id):
            return "free"

        async def rows(table, column, value):
            return server._ist_today(), "2000-01-01", {}
        server.get_user_plan = plan
        server._fetch_usage_rows = rows
        server.async_client = FakeClient()
        asyncio.run(server.usage_summary("u1"))
        pul = server.async_client.gets_of("provider_usage_log")
        check("Usage tab counts only /chat and /solve rows as doubts", pul and pul[0].params.get("endpoint") == "in.(/chat,/solve)")
    finally:
        for n, v in saved.items():
            setattr(server, n, v)


# ---------- 5. Hinglish on Qwen ----------

def test_hinglish():
    print("\n[5] Hinglish on Qwen")
    for q in ["ohm ka niyam kya hai", "photosynthesis kaise hota hai", "plants mein osmosis kya hota hai samjhao", "tum kaun ho",
              "newton ka teesra niyam samjhao", "Mitosis KYA hai?"]:
        check(f"Hinglish: {q!r}", server._looks_hinglish(q))
    for q in ["what is osmosis", "Explain the SN1 reaction mechanism", "yes", "khaitan", "पौधों में परासरण क्या है", "kayak", ""]:
        check(f"not Hinglish: {q!r}", not server._looks_hinglish(q))

    with Harness(deepseek_outputs=[GOOD]) as h:
        chat("ohm ka niyam kya hai")
        check("Hinglish doubt gets the guard", server.HINGLISH_GUARD in h.calls[0].system)
    with Harness(deepseek_outputs=[GOOD]) as h:
        chat("what is ohm's law")
        check("English doubt doesn't", server.HINGLISH_GUARD not in h.calls[0].system)

    bare = "DOUBT_TYPE: conversational"
    with Harness(peak=True, qwen_outputs=[bare, CONVERSATIONAL]) as h:
        out = chat("tum kaun ho")
        check("bare marker on Qwen -> retried on Qwen, no DeepSeek call", h.models() == ["qwen", "qwen"], str(h.models()))
        check("...retry carries the extra note", server.QWEN_EMPTY_CONVERSATIONAL_RETRY_NOTE in h.calls[1].system)
        check("...student gets the real reply", out == CONVERSATIONAL)
    with Harness(peak=True, qwen_outputs=[bare, GOOD]) as h:
        out = chat("ohm ka niyam kya hai")
        check("Qwen retry that answers academically is streamed as-is", out == GOOD and h.models() == ["qwen", "qwen"])
    with Harness(peak=True, qwen_outputs=[bare, bare], deepseek_outputs=[CONVERSATIONAL]) as h:
        out = chat("tum kaun ho")
        check("Qwen bare again -> one DeepSeek fallback", h.models() == ["qwen", "qwen", "deepseek"] and out == CONVERSATIONAL, str(h.models()))
        check("...and the bare marker never reaches the student", "DOUBT_TYPE: conversational\n\nHappy" in out and not out.startswith(bare + bare))
    with Harness(peak=False, deepseek_outputs=[bare, CONVERSATIONAL]) as h:
        out = chat("yes")
        check("DeepSeek-served bare marker still retries on DeepSeek", h.models() == ["deepseek", "deepseek"] and out == CONVERSATIONAL, str(h.models()))
    with Harness(peak=True, qwen_outputs=[bare, CONVERSATIONAL]) as h:
        chat("tum kaun ho", language="hi")
        check("Qwen retry keeps the Hindi cap", h.calls[1].max_tokens == server.ANSWER_MAX_TOKENS_HI)


# ---------- 6. Hindi cap ----------

def test_hindi_cap():
    print("\n[6] Hindi output cap")
    check("caps: en 1024, hi 1400", server._answer_max_tokens("en") == 1024 and server._answer_max_tokens("hi") == 1400 and server._answer_max_tokens("") == 1024)
    with Harness(deepseek_outputs=[GOOD]) as h:
        chat("परासरण क्या है?", language="hi")
        check("Hindi /chat on DeepSeek -> 1400", h.calls[0].max_tokens == 1400)
    with Harness(peak=True, qwen_outputs=[GOOD]) as h:
        chat("परासरण क्या है?", language="hi")
        check("Hindi /chat on Qwen -> 1400", h.calls[0].model == "qwen" and h.calls[0].max_tokens == 1400)
    with Harness(deepseek_outputs=[GOOD]) as h:
        chat("what is osmosis", language="en")
        check("English /chat -> 1024", h.calls[0].max_tokens == 1024)

    solution = "✅ उत्तर: B — 20 m\n\n📝 हल:\n- $h = u^2/2g$"
    req = dict(pyq_id="p1", question="Q?", option_a="1", option_b="2", option_c="3", option_d="4", correct_answer="b")
    with Harness(deepseek_outputs=[solution]) as h:
        asyncio.run(_drain(server.stream_solve_response(server.SolveRequest(**req, language="hi"), None, "1.2.3.4")))
        check("Hindi /solve -> 1400, solution cached", h.calls[0].max_tokens == 1400 and len(h.client.posts_to("pyq_solution_cache")) == 1)
    with Harness(deepseek_outputs=[solution], deepseek_flags={"output_capped": True}) as h:
        out = asyncio.run(_drain(server.stream_solve_response(server.SolveRequest(**req, language="hi"), None, "1.2.3.4")))
        check("cut-off /solve solution streams but is never cached", out == solution and not h.client.posts_to("pyq_solution_cache") and not h.spawned)
    with Harness(deepseek_outputs=[solution]) as h:
        asyncio.run(_drain(server.stream_solve_response(server.SolveRequest(**req, language="en"), None, "1.2.3.4")))
        check("English /solve -> 1024", h.calls[0].max_tokens == 1024)


# ---------- 7. history: earlier AI answers capped at ~600 characters (batch 2) ----------

LONG_ANSWER = ("NEET Importance: 4/5\n\n📚 Chapter: NCERT Class 11, Chapter 11 — Transport in Plants\n\nAnswer:\n"
               + "\n".join(f"- Point {i}: osmosis moves water from higher to lower water potential across a membrane." for i in range(1, 15))
               + "\n\nKey Points:\n- The last key point about turgor pressure.")


def test_history_cap():
    print("\n[7] earlier AI answers capped in history")
    cap = server._cap_history_answer
    check("short answer unchanged", cap("Osmosis is diffusion of water.") == "Osmosis is diffusion of water.")
    capped = cap(LONG_ANSWER)
    body = capped[:-len(server._HISTORY_ANSWER_CUT_NOTE)]
    check("long answer cut to <= 600 chars + a note", capped.endswith(server._HISTORY_ANSWER_CUT_NOTE) and len(body) <= 600, len(body))
    check("cut on a line boundary (no half bullet)", LONG_ANSWER.startswith(body) and LONG_ANSWER[len(body)] == "\n")
    check("keeps the start of the answer", body.startswith("NEET Importance: 4/5"))
    no_break = "word " * 200
    check("no line break -> cut on a space", cap(no_break)[:-len(server._HISTORY_ANSWER_CUT_NOTE)].endswith("word"))
    long_question = "Explain " + "osmosis " * 120
    history = [{"role": "user", "text": long_question}, {"role": "ai", "text": LONG_ANSWER}]
    with Harness(deepseek_outputs=[GOOD]) as h:
        chat("explain point 2 again", history=history)
        msgs = h.calls[0].messages
        check("student's own earlier message is never cut", msgs[0]["content"] == long_question)
        check("earlier AI answer is capped in what the model sees", msgs[1]["content"] == capped)
        check("current question untouched", msgs[-1]["content"].endswith("Student Question: explain point 2 again"))


# ---------- 8. end-of-chapter exercise lists kept out of NCERT context (batch 2) ----------

EN_EXERCISE_LIST = ("EXERCISES 1.What do you think is the significance of reproductive health in a society? "
                    "2.Suggest the aspects of reproductive health which need to be given special attention in the present scenario. "
                    "3.Is sex education necessary in schools? Why? 4.Do you think that reproductive health in our country has improved "
                    "in the past 50 years? If yes, mention some such areas of improvement. 5.What are the suggested reasons for population "
                    "explosion? 6.Is the use of contraceptives justified? Give reasons.")
EN_CONTINUATION = ("of male accessory ducts and glands? 12. What is oogenesis? Give a brief account of oogenesis. 13. Draw a labelled "
                   "diagram of a section through ovary. 14. Draw a labelled diagram of a Graafian follicle? 15. Name the functions of the "
                   "following: (a) Corpus luteum (b) Endometrium (c) Acrosome (d) Sperm tail 16. Identify True/False statements.")
EN_CONTENT = ("Osmosis is the diffusion of water across a semi-permeable membrane. The net direction and rate of osmosis depends on "
              "both the pressure gradient and concentration gradient. Water will move from its region of higher chemical potential "
              "to its region of lower chemical potential until equilibrium is reached. At equilibrium the two chambers should have "
              "the same water potential. Plant cells placed in a hypertonic solution lose water and the protoplast shrinks away "
              "from the cell wall, which is called plasmolysis.")
EN_MIXED = EN_CONTENT + " " + EN_EXERCISE_LIST
EN_TOC = ("CHAPTER TWO MOTION IN A STRAIGHT LINE 2.1 Introduction 2.2 Instantaneous velocity and speed 2.3 Acceleration "
          "2.4 Kinematic equations for uniformly accelerated motion Summary Points to ponder Exercises2.1 INTRODUCTION Motion is "
          "common to everything in the universe. We walk, run and ride a bicycle.")
EN_WORKED_EXAMPLE = ("Example 2.3 A ball is thrown vertically upwards with a velocity of 20 m s–1 from the top of a building. "
                     "The height of the point from where the ball is thrown is 25.0 m from the ground. (a) How high will the ball rise? "
                     "Answer (a) Let us take the y-axis in the vertically upward direction with zero at the ground. Now v = 0 at the "
                     "maximum height, so using v2 = v02 – 2 a (y – y0) we get 0 = 400 - 20 (y - y0), which gives y – y0 = 20 m.")
EN_INTEXT_BOX = ("Intext Questions 3.3 For a reaction, A + B -> Product; the rate law is given by r = k [A]1/2 [B]2. What is the "
                 "order of the reaction? 3.4 The conversion of molecules X to Y follows second order kinetics. If concentration of X "
                 "is increased to three times how will it affect the rate of formation of Y? " + EN_CONTENT)
HI_EXERCISE_LIST = ("**अभ्यास** 1. वर्गीकरण की पद्धतियों में समय के साथ आए परिवर्तनों की व्याख्या कीजिए। 2. निम्नलिखित के बारे में आर्थिक "
                    "दृष्टि से दो महत्वपूर्ण उपयोगों को लिखिए। 3. डाइऐटम की कोशिका भित्ति के क्या लक्षण हैं? 4. शैवाल पुष्पन तथा "
                    "लाल तरंगें क्या दर्शाती हैं? 5. वाइरस से विरोज़्ड कैसे भिन्न होते हैं?")
HI_CONTENT = ("परासरण वह प्रक्रिया है जिसमें जल के अणु अर्धपारगम्य झिल्ली के पार उच्च जल विभव वाले क्षेत्र से निम्न जल विभव वाले क्षेत्र "
              "की ओर गति करते हैं। जब किसी पादप कोशिका को अतिपरासारी विलयन में रखा जाता है, तो जल कोशिका से बाहर निकल जाता है और "
              "जीवद्रव्य कोशिका भित्ति से सिकुड़कर अलग हो जाता है। इस प्रक्रिया को जीवद्रव्यकुंचन कहते हैं। ")
HI_WORKED_EXAMPLE = ("इस प्रकार, 1 हर्ट्ज़ = 1 दोलन प्रति सेकंड (13.2) ध्यान दीजिए, आवृत्ति का सदैव ही पूर्णांक होना आवश्यक नहीं है। "
                     "**उदाहरण 13.1** कोई मानव हृदय एक मिनट में औसतन 75 बार धड़कता है। इसकी आवृत्ति तथा आवर्तकाल परिकलित कीजिए। "
                     "**हल** हृदय की धड़कन की आवृत्ति = 75/(1 min) = 1.25 s⁻¹ = 1.25 Hz, आवर्तकाल T = 1/1.25 = 0.8 s.")


def test_exercise_filter():
    print("\n[8] exercise-list chunks")
    t = server._exercise_trim
    check("EN exercise list -> dropped", t(EN_EXERCISE_LIST, "en") is None)
    check("EN mid-list continuation (no header) -> dropped", t(EN_CONTINUATION, "en") is None)
    check("EN content -> unchanged", t(EN_CONTENT, "en") == EN_CONTENT)
    check("EN content + EXERCISES -> only the exercises cut off", t(EN_MIXED, "en") == EN_CONTENT)
    check("EN contents list ('Exercises2.1 INTRODUCTION') -> unchanged", t(EN_TOC, "en") == EN_TOC)
    check("EN worked example with its answer -> unchanged", t(EN_WORKED_EXAMPLE, "en") == EN_WORKED_EXAMPLE)
    check("EN in-text question box followed by the chapter -> unchanged", t(EN_INTEXT_BOX, "en") == EN_INTEXT_BOX)
    check("HI exercise list -> dropped", t(HI_EXERCISE_LIST, "hi") is None)
    check("HI content -> unchanged", t(HI_CONTENT, "hi") == HI_CONTENT)
    check("HI content + अभ्यास -> only the exercises cut off", t(HI_CONTENT + " " + HI_EXERCISE_LIST, "hi") == HI_CONTENT.rstrip())
    check("HI worked example ('उदाहरण 13.1', 'ध्यान दीजिए,') -> unchanged", t(HI_WORKED_EXAMPLE, "hi") == HI_WORKED_EXAMPLE)
    check("too little content before the exercises -> dropped", t("Summary. " + EN_EXERCISE_LIST, "en") is None)

    rows = [
        {"subject": "Biology", "class": 12, "chapter_name": "Reproductive Health", "language": "en", "content": EN_EXERCISE_LIST, "similarity": 0.80},
        {"subject": "Biology", "class": 11, "chapter_name": "Transport in Plants", "language": "en", "content": EN_MIXED, "similarity": 0.70},
        {"subject": "Biology", "class": 11, "chapter_name": "Transport in Plants", "language": "en", "content": EN_CONTENT, "similarity": 0.60},
        {"subject": "Physics", "class": 11, "chapter_name": "Motion in a Straight Line", "language": "en", "content": EN_WORKED_EXAMPLE, "similarity": 0.55},
        {"subject": "Physics", "class": 11, "chapter_name": "Motion in a Straight Line", "language": "en", "content": EN_TOC, "similarity": 0.40},
    ]

    class RpcClient(FakeClient):
        async def post(self, url, headers=None, json=None, params=None, **kw):
            self.posts.append(SimpleNamespace(table=_table(url), headers=headers or {}, json=json, params=params or {}))
            return FakeResp(200, [dict(r) for r in rows])

    async def fake_embedding(text):
        return [0.0] * 8

    saved = (server.async_client, server.get_embedding, server.NCERT_EXCLUDE_EXERCISES)
    try:
        server.async_client, server.get_embedding = RpcClient(), fake_embedding
        res = asyncio.run(server.search_ncert("what is osmosis in plants"))
        check("search: exercise list dropped, next chunks fill the 3 slots", [r["similarity"] for r in res] == [0.70, 0.60, 0.55], str([r["similarity"] for r in res]))
        check("search: mixed chunk's exercise tail cut off", res and res[0]["content"] == EN_CONTENT)
        check("search: threshold still applies (0.40 TOC chunk not used)", all(r["similarity"] >= server.NCERT_MATCH_THRESHOLD_EN for r in res))
        server.NCERT_EXCLUDE_EXERCISES = False
        res = asyncio.run(server.search_ncert("what is osmosis in plants"))
        check("switch off -> previous behaviour (exercise list first)", [r["similarity"] for r in res] == [0.80, 0.70, 0.60] and res[1]["content"] == EN_MIXED)
    finally:
        server.async_client, server.get_embedding, server.NCERT_EXCLUDE_EXERCISES = saved


if __name__ == "__main__":
    test_cache_key_and_eligibility()
    test_cache_store_rules()
    test_cache_flow()
    test_cache_expiry_and_migration_safety()
    test_chat_alternate_method_cache()
    test_model_and_rates()
    test_provider_helpers_and_fairness()
    test_log_provider_usage_columns()
    test_new_endpoints_logged()
    test_hinglish()
    test_hindi_cap()
    test_history_cap()
    test_exercise_filter()
    print(f"\nTOTAL: {len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
