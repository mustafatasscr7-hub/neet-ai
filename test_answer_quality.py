"""Regression tests for the 2026-10-13 pre-launch answer-quality fixes in server.py.

Run with: python test_answer_quality.py
No pytest dependency -- plain asserts and the same small pass/fail harness as
test_student_name_sanitization.py. Fully offline: every network/model call is monkeypatched.

1. Self-correction text ("Wait -- let me re-verify...") is never written to the shared caches
   (pyq_solution_cache, answer_cache); a background regeneration replaces it only if clean.
2. The Group 14 inert-pair worked example is no longer in every chat prompt (it leaked
   "Pb(II) is a reducing agent" into an Ohm's Law answer) -- only appended for doubts about it.
3. Weak-chapter personalisation only uses chapters from the retrieved subject, and the
   mock-test average line is gone.
"""
import asyncio
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server

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
    def __init__(self, data=None):
        self._data = data if data is not None else []
        self.status_code = 200

    def json(self):
        return self._data


class FakeClient:
    """Records every POST; GETs return an empty list (cache miss)."""
    def __init__(self):
        self.posts = []

    async def post(self, url, headers=None, json=None, **kw):
        self.posts.append((url, headers or {}, json))
        return FakeResp()

    async def get(self, url, headers=None, params=None, **kw):
        return FakeResp([])

    def posts_to(self, table):
        # Exact table: "/rest/v1/answer_cache" must not also match ".../answer_cache_log".
        return [p for p in self.posts if p[0].split("?")[0].endswith(f"/rest/v1/{table}")]


class Harness:
    """Monkeypatches server's model stream, HTTP client and lookups; restores on exit."""
    def __init__(self, model_outputs, ncert_results=None, mistake_rows=None):
        self.model_outputs = list(model_outputs)
        self.ncert_results = ncert_results or []
        self.mistake_rows = mistake_rows or []
        self.systems = []
        self.spawned = []
        self.client = FakeClient()

    def __enter__(self):
        h = self
        self._saved = {n: getattr(server, n) for n in (
            "async_client", "_stream_with_peak_fallback", "check_max_usage_tier", "_spawn_background",
            "search_ncert", "_diagram_exists_for", "fetch_student_mistake_rows")}

        async def fake_stream(system, messages, user_id, ip, endpoint, billing_context=None, force_qwen=False, max_tokens=None):
            h.systems.append(system)
            yield h.model_outputs.pop(0)

        async def fake_tier(user_id):
            return None

        def fake_spawn(coro):
            h.spawned.append(coro)

        async def fake_search(text, limit=3):
            return h.ncert_results

        async def fake_diagram(text):
            return False

        async def fake_mistakes(user_id):
            return h.mistake_rows

        server.async_client = self.client
        server._stream_with_peak_fallback = fake_stream
        server.check_max_usage_tier = fake_tier
        server._spawn_background = fake_spawn
        server.search_ncert = fake_search
        server._diagram_exists_for = fake_diagram
        server.fetch_student_mistake_rows = fake_mistakes
        return self

    def __exit__(self, *exc):
        for name, value in self._saved.items():
            setattr(server, name, value)
        for coro in self.spawned:
            coro.close()


async def drain(gen):
    return "".join([c async for c in gen])


# Excerpts of the 4 real flagged pyq_solution_cache rows (2026-10-13 scan).
REAL_DIRTY = [
    "$$I = \\frac{-5}{5} = -1\\text{ A}$$ (Magnitude is $1\\text{ A}$? Wait, let us re-verify node potentials).\n- Alternative simpler method:",
    "- **D** has a-i ❌ (should be a-ii) → Wait, let me recheck:\n  - Option D: a-i, b-iii, c-iv, d-ii",
    "- From the figure, the line for $P_1$ is steeper upward? Wait — actually, steeper slope = higher $V$",
    "- But the question says \"non-transition element\" — wait, group 11 elements are transition metals.\n\n⚠️ Correction: Actually, the given options are all false",
]
CLEAN = [
    "✅ Answer: C — It is a transition element.\n\n📝 Solution:\n- Unununium = element 111 (Rg).\n- Group 11, period 7 → d-block.\n\n🔑 Key Concept:\nIUPAC roots: un=1.",
    "- End correction: $e = 0.3d$ is added to the resonance column length.",
    "Let me verify each statement for vertical projection upward:",
    "However, the resistance of a semiconductor actually decreases with temperature.",
    "The cell must wait for the next signal before depolarising again.",
    "- Let us check the options one by one: A is wrong because...",
    "✅ उत्तर: C — यह एक संक्रमण तत्व है।\n\n📝 हल:\n- अनअनअनियम = तत्व 111",
]
SOLVE_REQ = dict(pyq_id="test-pyq-1", question="Q?", option_a="1", option_b="2", option_c="3", option_d="4", correct_answer="b")


# ---------- 1. self-correction detector + cache gating ----------

def test_detector():
    print("\n[1a] self-correction detector")
    for i, text in enumerate(REAL_DIRTY):
        check(f"flags real cached row #{i + 1}", server._has_self_correction(text))
    check("flags Hindi self-correction", server._has_self_correction("रुकिए, दोबारा जाँच करते हैं।"))
    check("flags answer-key disagreement line",
          server._has_self_correction("⚠️ The given answer appears to be incorrect: Mega is 10^6, not 10^9"))
    for i, text in enumerate(CLEAN):
        check(f"passes clean text #{i + 1}", not server._has_self_correction(text), text[:50])


def test_solve_cache_gating():
    print("\n[1b] /solve cache gating")
    req = server.SolveRequest(**SOLVE_REQ)
    with Harness([REAL_DIRTY[1]]) as h:
        out = asyncio.run(drain(server.stream_solve_response(req, None, "1.2.3.4")))
        check("dirty solution still streams to the requester", "Wait" in out)
        check("dirty solution NOT written to pyq_solution_cache", not h.client.posts_to("pyq_solution_cache"))
        check("background regeneration scheduled", len(h.spawned) == 1)

    with Harness([CLEAN[0]]) as h:
        asyncio.run(server._regenerate_and_cache_solution(req, "1.2.3.4"))
        posts = h.client.posts_to("pyq_solution_cache")
        check("clean regeneration is cached", len(posts) == 1 and posts[0][2]["solution"] == CLEAN[0])
        check("regeneration upserts", posts and posts[0][1].get("Prefer") == "resolution=merge-duplicates")
        check("regeneration prompt carries the retry guard", server.SELF_CORRECTION_RETRY_GUARD in h.systems[0])

    with Harness([REAL_DIRTY[3]]) as h:
        asyncio.run(server._regenerate_and_cache_solution(req, "1.2.3.4"))
        check("still-dirty regeneration is NOT cached", not h.client.posts_to("pyq_solution_cache"))

    with Harness([CLEAN[0]]) as h:
        asyncio.run(drain(server.stream_solve_response(req, None, "1.2.3.4")))
        check("clean live solution cached as before", len(h.client.posts_to("pyq_solution_cache")) == 1)
        check("no regeneration for a clean solution", not h.spawned)

    with Harness([]) as h:
        out = asyncio.run(drain(server.stream_solve_response(req, "cached text", "1.2.3.4")))
        check("cache hit still served verbatim, no writes", out == "cached text" and not h.client.posts)

    prompt = server._solve_system_prompt("en", False)
    check("solve prompt forbids visible self-correction", "Do all checking and re-checking silently" in prompt)
    check("Hindi rule renumbered to 6", "\n6. " in server._solve_system_prompt("hi", False))
    blank = server.SolveRequest(**{**SOLVE_REQ, "correct_answer": ""})
    check("blank answer key spelled out as 'not provided'",
          "Correct Answer: not provided -- determine it yourself" in server._solve_text_messages(blank)[0]["content"])
    check("given answer key passed through unchanged", server._solve_text_messages(req)[0]["content"].endswith("Correct Answer: b"))
    check("retry guard keeps the answer-key warning line allowed",
          "The given answer appears to be incorrect" in server.SELF_CORRECTION_RETRY_GUARD)


GOOD_CHAT = "VISUAL_INTENT: no\n\nNEET Importance: 4/5\n\nAnswer:\n- Ohm's law: $V = IR$.\n\nKey Points:\n- Holds only for ohmic conductors."
DIRTY_CHAT = "VISUAL_INTENT: no\n\nNEET Importance: 4/5\n\nAnswer:\n- $V = IR$. Wait — let me re-check: actually R depends on temperature."


def test_chat_cache_gating():
    print("\n[1c] /chat answer_cache gating")
    with Harness([DIRTY_CHAT]) as h:
        asyncio.run(drain(server.stream_response("What is Ohm's law?", personalize=False)))
        check("dirty chat answer NOT written to answer_cache", not h.client.posts_to("answer_cache"))
        check("background regeneration scheduled", len(h.spawned) == 1)

    with Harness([GOOD_CHAT]) as h:
        asyncio.run(drain(server.stream_response("What is Ohm's law?", personalize=False)))
        check("clean chat answer cached as before", len(h.client.posts_to("answer_cache")) == 1)

    for label, output, expect_cached in [("clean", GOOD_CHAT, True), ("still dirty", DIRTY_CHAT, False),
                                         ("clarifying question", "AMBIGUOUS: yes\nCLARIFY_TYPE: topic", False)]:
        with Harness([output]) as h:
            asyncio.run(server._regenerate_and_cache_chat_answer("SYS", [], "hash123", False, False, "", ""))
            cached = len(h.client.posts_to("answer_cache")) == 1
            check(f"chat regeneration: {label} -> {'cached' if expect_cached else 'not cached'}", cached == expect_cached)


# ---------- 2. inert-pair example scoped to relevant doubts ----------

def test_inert_pair_scoping():
    print("\n[2] inert-pair worked example scoping")
    facts = server._INERT_PAIR_EFFECT_SN_PB_FACTS
    check("SYSTEM_PROMPT no longer contains the Sn/Pb facts", facts not in server.SYSTEM_PROMPT)
    check("SYSTEM_PROMPT_NO_CLARIFY has no Sn/Pb facts either", facts not in server.SYSTEM_PROMPT_NO_CLARIFY)
    check("SYSTEM_PROMPT_NO_CLARIFY still replaces rule 11", "already been confirmed, by a separate server-side check"
          in server.SYSTEM_PROMPT_NO_CLARIFY and "CLARIFY_TYPE: [topic or format]" not in server.SYSTEM_PROMPT_NO_CLARIFY)
    check("MCQ hedge verifier keeps the facts", facts in server._VERIFY_MCQ_HEDGE_SYSTEM)

    for q in ["Ohms law", "What is Ohm's law?", "SN1 vs SN2 reaction", "Sn1 reaction mechanism", "what leads to anaemia",
              "explain snapping of bonds", "Plumbing in kidneys?"]:
        check(f"not inert-pair: {q!r}", not server._is_inert_pair_query(q))
    for q in ["inert pair effect", "Why is PbCl4 less stable than PbCl2?", "Is Sn2+ a reducing agent?",
              "stannous chloride uses", "stability of lead(II)", "group 14 oxidation states", "अक्रिय युग्म प्रभाव"]:
        check(f"inert-pair: {q!r}", server._is_inert_pair_query(q))

    with Harness([GOOD_CHAT]) as h:
        asyncio.run(drain(server.stream_response("Ohms law", personalize=False, skip_cache=True)))
        check("Ohm's law prompt has no Sn/Pb facts", facts not in h.systems[0])
    with Harness([GOOD_CHAT]) as h:
        asyncio.run(drain(server.stream_response("Why is PbCl4 less stable than PbCl2?", personalize=False, skip_cache=True)))
        check("inert-pair prompt gets the Sn/Pb facts", facts in h.systems[0])


# ---------- 3. subject-scoped personalisation ----------

BIO_MISTAKES = [{"subject": "Biology", "chapter": "Locomotion and Movement"},
                {"subject": "Biology", "chapter": "Excretory Products and their Elimination"}]
PHY_MISTAKE = [{"subject": "Physics", "chapter": "Laws of Motion"}]
PHYSICS_NCERT = [{"subject": "Physics", "class": 12, "chapter_name": "Current Electricity", "content": "V = IR", "similarity": 0.6}]


def test_personalisation():
    print("\n[3] subject-scoped personalisation")
    check("Biology chapters dropped for a Physics doubt", server.build_student_context(BIO_MISTAKES, {"Physics"}) == "")
    ctx = server.build_student_context(BIO_MISTAKES + PHY_MISTAKE, {"Physics"})
    check("same-subject chapter kept", "Laws of Motion" in ctx and "Locomotion" not in ctx, ctx)
    check("no retrieval -> no weak-chapter context", server.build_student_context(BIO_MISTAKES + PHY_MISTAKE, set()) == "")
    check("context forbids naming chapters off-topic", "Never mention these chapters" in ctx)

    with Harness([GOOD_CHAT], ncert_results=PHYSICS_NCERT, mistake_rows=BIO_MISTAKES) as h:
        asyncio.run(drain(server.stream_response("Ohms law", user_id="u1", personalize=True)))
        system = h.systems[0]
        check("Physics doubt prompt has no Biology chapters", "Locomotion" not in system and "Excretory" not in system)
        check("no mock-test average in prompt", "mock test" not in system.lower())
        check("no STUDENT CONTEXT block at all", "STUDENT CONTEXT" not in system)
    with Harness([GOOD_CHAT], ncert_results=PHYSICS_NCERT, mistake_rows=BIO_MISTAKES + PHY_MISTAKE) as h:
        asyncio.run(drain(server.stream_response("Ohms law", user_id="u1", personalize=True)))
        check("Physics weak chapter still reaches a Physics doubt", "Laws of Motion" in h.systems[0])


if __name__ == "__main__":
    test_detector()
    test_solve_cache_gating()
    test_chat_cache_gating()
    test_inert_pair_scoping()
    test_personalisation()
    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
