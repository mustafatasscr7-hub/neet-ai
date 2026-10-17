"""NCERT retrieval relevance check + retrieval log in /chat (stream_response).

Run with: python test_ncert_relevance.py
Plain pass/fail harness like test_answer_quality.py. Fully offline: the model stream, NCERT search
and HTTP client are fakes.

1. _question_subject: clear single-subject questions get a subject; overlap/mixed ones get none.
2. Wrong-subject chunks: set aside, no "Retrieved from:", model told to answer, Chapter line stripped.
3. Normal case: the model prompt is byte-for-byte what it was before this change.
4. Mixed chunks: off-subject ones dropped; citations highest-scoring chapter first, near-ties only.
5. search_ncert fills stats; one ncert_retrieval_log row per answer, ids/scores only, no text.
6. Short English questions (<=5 words) fall back to a 0.42 cutoff when nothing reaches 0.5;
   longer ones, and short ones that already match, are unchanged; small talk still never retries.
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


class FakeResp:
    def __init__(self, data=None):
        self._data = data if data is not None else []
        self.status_code = 200

    def json(self):
        return self._data


class FakeClient:
    def __init__(self, rpc_rows=None):
        self.posts = []
        self.rpc_rows = rpc_rows or []

    async def post(self, url, headers=None, json=None, **kw):
        self.posts.append((url, json))
        return FakeResp(self.rpc_rows if "/rpc/" in url else [])

    async def get(self, url, headers=None, params=None, **kw):
        return FakeResp([])

    def posts_to(self, table):
        return [p for p in self.posts if p[0].split("?")[0].endswith(f"/rest/v1/{table}")]


class Harness:
    def __init__(self, model_output, ncert_results, top_raw_score=0.6):
        self.model_output = model_output
        self.ncert_results = ncert_results
        self.top_raw_score = top_raw_score
        self.messages = []
        self.client = FakeClient()

    def __enter__(self):
        h = self
        self._saved = {n: getattr(server, n) for n in (
            "async_client", "_stream_with_peak_fallback", "check_max_usage_tier", "_spawn_background",
            "search_ncert", "_diagram_exists_for")}

        async def fake_stream(system, messages, user_id, ip, endpoint, billing_context=None, force_qwen=False, max_tokens=None):
            h.messages.append(messages)
            yield h.model_output

        async def fake_tier(user_id):
            return None

        def fake_spawn(coro):
            coro.close()

        async def fake_search(text, limit=3, stats=None):
            if stats is not None:
                stats.update(language="en", top_raw_score=h.top_raw_score)
            return h.ncert_results

        async def fake_diagram(text):
            return False

        server.async_client = self.client
        server._stream_with_peak_fallback = fake_stream
        server.check_max_usage_tier = fake_tier
        server._spawn_background = fake_spawn
        server.search_ncert = fake_search
        server._diagram_exists_for = fake_diagram
        return self

    def __exit__(self, *exc):
        for name, value in self._saved.items():
            setattr(server, name, value)

    def user_message(self):
        return self.messages[0][-1]["content"]


async def drain(gen):
    out = "".join([c async for c in gen])
    await asyncio.sleep(0)  # let the fire-and-forget log task post
    return out


def row(id_, subject, cls, chapter, sim, content="..."):
    return {"id": id_, "subject": subject, "class": cls, "chapter_name": chapter, "chapter_name_en": None,
            "language": "en", "content": content, "similarity": sim}


CHEM_UNITS = row("c1", "Chemistry", 11, "Some Basic Concepts of Chemistry", 0.58, "SI base units: metre, kilogram, second ...")
PHY_UNITS = row("p1", "Physics", 11, "Units and Measurement", 0.55, "The SI unit of force is the newton ...")
PHY_OHM = row("p2", "Physics", 12, "Current Electricity", 0.71, "Ohm's law: V = IR ...")
PHY_LAWS = row("p3", "Physics", 11, "Laws of Motion", 0.57, "Force is the rate of change of momentum ...")

ANSWER_WITH_CITATION = ("VISUAL_INTENT: no\n\nNEET Importance: 4/5\n\n📚 Chapter: NCERT Class 11, Chapter 1 — Some Basic Concepts of Chemistry\n\n"
                        "Answer:\n- The SI unit of force is the newton (N) = kg m s⁻².\n\nKey Points:\n- 1 N = 10⁵ dyne.")


def test_question_subject():
    print("\n1. question subject")
    for q, want in [("What is the SI unit of force?", "Physics"), ("Explain SN1 mechanism", "Chemistry"),
                    ("stages of meiosis", "Biology"), ("intermolecular forces in liquids", None),
                    ("structure of DNA", None), ("what is osmosis", None), ("बल का SI मात्रक क्या है?", None),
                    ("de Broglie wavelength of an electron", None)]:
        got = server._question_subject(q)
        check(f"{q!r} -> {want}", got == want, got)


def test_wrong_subject_no_citation():
    print("\n2. wrong-subject chunks give no citation")
    with Harness(ANSWER_WITH_CITATION, [CHEM_UNITS]) as h:
        out = asyncio.run(drain(server.stream_response("What is the SI unit of force?", personalize=False, skip_cache=True)))
        msg = h.user_message()
        check("chunk content not sent", "SI base units" not in msg and "NCERT Content" not in msg)
        check("no 'Retrieved from:' list", "Retrieved from" not in msg)
        check("model told to set aside and answer", "set aside" in msg and "do not refuse" in msg)
        check("Chapter line stripped from the answer", "📚 Chapter" not in out and "Some Basic Concepts" not in out, out)
        check("answer itself still streams", "newton (N)" in out)
        log = h.client.posts_to("ncert_retrieval_log")
        check("logged as set_aside", len(log) == 1 and log[0][1]["outcome"] == "set_aside", log)
        check("set-aside chunk logged with used=false", log and log[0][1]["chunks"][0]["used"] is False)


def test_normal_case_unchanged():
    print("\n3. normal case unchanged")
    with Harness(ANSWER_WITH_CITATION, [PHY_OHM]) as h:
        asyncio.run(drain(server.stream_response("What is Ohm's law?", personalize=False, skip_cache=True)))
        expected = ("NCERT Content:\n[Physics - Class 12 - Current Electricity]\nOhm's law: V = IR ...\n\n"
                    "Retrieved from (copy EXACTLY ONE of these verbatim into your Chapter: line -- never a different name or number):\n"
                    "- NCERT Class 12, Chapter 3 — Current Electricity\n\nStudent Question: What is Ohm's law?")
        check("prompt identical to before", h.user_message() == expected, h.user_message())
        log = h.client.posts_to("ncert_retrieval_log")
        check("logged as grounded", len(log) == 1 and log[0][1]["outcome"] == "grounded")
    # No clear subject -> nothing set aside even across subjects.
    with Harness(ANSWER_WITH_CITATION, [CHEM_UNITS, PHY_UNITS]) as h:
        asyncio.run(drain(server.stream_response("what are SI units", personalize=False, skip_cache=True)))
        msg = h.user_message()
        check("unclear subject keeps every chunk", "SI base units" in msg and "SI unit of force" in msg)
    # Search found nothing -> same plain message as before.
    with Harness(ANSWER_WITH_CITATION, []) as h:
        asyncio.run(drain(server.stream_response("What is Ohm's law?", personalize=False, skip_cache=True)))
        check("no retrieval: plain question", h.user_message() == "Student Question: What is Ohm's law?")
        log = h.client.posts_to("ncert_retrieval_log")
        check("logged as none", len(log) == 1 and log[0][1]["outcome"] == "none")


def test_mixed_and_citation_order():
    print("\n4. mixed subjects + highest-scoring chapter first")
    with Harness(ANSWER_WITH_CITATION, [CHEM_UNITS, PHY_LAWS, PHY_UNITS]) as h:
        asyncio.run(drain(server.stream_response("What is the SI unit of force?", personalize=False, skip_cache=True)))
        msg = h.user_message()
        check("off-subject chunk dropped", "Some Basic Concepts" not in msg and "SI base units" not in msg)
        laws, units = msg.find("- NCERT Class 11, Chapter 4 — Laws of Motion"), msg.find("- NCERT Class 11, Chapter 1 — Units and Measurement")
        check("both near-tie physics citations, best first", 0 < laws < units, msg)
        check("prefer-first instruction present", "use the first unless" in msg)
        used = {c["id"]: c["used"] for c in h.client.posts_to("ncert_retrieval_log")[0][1]["chunks"]}
        check("log marks used/unused per chunk", used == {"c1": False, "p3": True, "p1": True}, used)
    far = row("p4", "Physics", 11, "Units and Measurement", 0.60)
    best = row("p5", "Physics", 11, "Laws of Motion", 0.72)
    check("chapter far below the best is not offered", server._ncert_citations([far, best]) == ["NCERT Class 11, Chapter 4 — Laws of Motion"])


def test_search_stats_and_log_privacy():
    print("\n5. search stats + log privacy")
    rows = [row("x1", "Physics", 11, "Laws of Motion", 0.66), row("x2", "Physics", 11, "Laws of Motion", 0.31)]

    async def fake_embed(text):
        return [0.0] * 1536

    saved = server.async_client, server.get_embedding
    server.async_client, server.get_embedding = FakeClient(rpc_rows=rows), fake_embed
    try:
        stats = {}
        got = asyncio.run(server.search_ncert("newton's second law", stats=stats))
    finally:
        server.async_client, server.get_embedding = saved
    check("stats has language + raw top score", stats == {"language": "en", "top_raw_score": 0.66}, stats)
    check("threshold still applied to results", [r["id"] for r in got] == ["x1"])

    question = "What is the SI unit of force?"
    with Harness(ANSWER_WITH_CITATION, [PHY_UNITS], top_raw_score=0.61234) as h:
        out = asyncio.run(drain(server.stream_response(question, personalize=False, skip_cache=True)))
        payload = h.client.posts_to("ncert_retrieval_log")[0][1]
        check("log fields", set(payload) == {"language", "question_subject", "outcome", "chunks", "top_raw_score"}, payload)
        check("chunk entry shape", payload["chunks"] == [{"id": "p1", "score": 0.55, "subject": "Physics", "class": 11,
                                                          "chapter": "Units and Measurement", "used": True}], payload["chunks"])
        check("top raw score + subject logged", payload["top_raw_score"] == 0.6123 and payload["question_subject"] == "Physics")
        text = repr(payload)
        check("no question or answer text in the log", "SI unit" not in text and "newton" not in text and out[:20] not in text)


def _search_with(rows, query):
    """Real search_ncert over fixed RPC rows; returns (result ids, number of embed calls)."""
    embeds = []

    async def fake_embed(text):
        embeds.append(text)
        return [0.0] * 1536

    saved = server.async_client, server.get_embedding
    server.async_client, server.get_embedding = FakeClient(rpc_rows=rows), fake_embed
    try:
        got = asyncio.run(server.search_ncert(query))
    finally:
        server.async_client, server.get_embedding = saved
    return [r["id"] for r in got], len(embeds)


def test_short_query_threshold():
    print("\n6. short-question cutoff")
    check("<=5 words -> 0.42", server._ncert_threshold_en("function of ribosomes") == 0.42)
    check("5 words -> 0.42", server._ncert_threshold_en("who is the prime minister") == 0.42)
    check("6+ words -> 0.5", server._ncert_threshold_en("explain the function of ribosomes please") == 0.5)

    near = [row("r1", "Biology", 11, "Cell: The Unit of Life", 0.46), row("r2", "Biology", 12, "Molecular Basis of Inheritance", 0.44),
            row("r3", "Biology", 11, "Biomolecules", 0.40)]
    ids, _ = _search_with(near, "function of ribosomes")
    check("short question at 0.46 now retrieves (>= 0.42 only)", ids == ["r1", "r2"], ids)
    ids, _ = _search_with(near, "what is the function of ribosomes in a cell")
    check("long question at 0.46 still retrieves nothing", ids == [], ids)

    strong = [row("s1", "Physics", 11, "Gravitation", 0.55), row("s2", "Physics", 11, "Motion in a Plane", 0.46)]
    ids, _ = _search_with(strong, "escape velocity")
    check("short question already over 0.5 keeps only its 0.5 chunks", ids == ["s1"], ids)

    small_talk = [row("t1", "Chemistry", 11, "Equilibrium", 0.36)]
    ids, embeds = _search_with(small_talk, "how are you")
    check("small talk at 0.36 retrieves nothing", ids == [], ids)
    check("small talk skips the merge-variant retries (one embed)", embeds == 1, embeds)


if __name__ == "__main__":
    test_question_subject()
    test_wrong_subject_no_citation()
    test_normal_case_unchanged()
    test_mixed_and_citation_order()
    test_search_stats_and_log_privacy()
    test_short_query_threshold()
    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
