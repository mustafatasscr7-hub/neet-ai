"""Unit tests for wiring search_ncert_v2 into the /chat NCERT retrieval path (server.py).

Run with: python test_ncert_v2_chat_wiring.py
Plain pass/fail harness like test_ncert_rechunk_v2.py. Fully offline -- mocked HTTP client and
get_embedding, no network/DB.

Covers _search_ncert_for_chat, the single switch point stream_response calls instead of
search_ncert directly:
1. NCERT_V2_ENABLED true + English -> calls search_ncert_v2, never search_ncert directly.
2. v2 empty, or the v2 RPC raising -> falls back to search_ncert (real search_ncert_v2 fallback,
   not mocked away), and the fallback is reflected in stats["version"] == "v1".
3. NCERT_V2_ENABLED false -> search_ncert_v2 is never even called; behaviour matches today exactly.
4. Hindi query -> always search_ncert directly, regardless of NCERT_V2_ENABLED.
5. End-to-end through stream_response: ncert_retrieval_log's chunks carry "version" when known
   (v2 success, and v1-via-fallback), and carry no such key at all when the flag is off -- so an
   existing log row written before this change round-trips through the same assertions unchanged.
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
    def __init__(self, status, body):
        self.status_code = status
        self._body = body

    def json(self):
        return self._body


async def fake_embed(text):
    return [0.0] * 1536


def test_flag_on_english_routes_to_v2():
    print("\n1. NCERT_V2_ENABLED true + English -> search_ncert_v2, not search_ncert directly")

    class FakeClient:
        async def post(self, url, headers=None, json=None):
            assert url.endswith("/rpc/match_ncert_v2_hybrid"), url
            return FakeResp(200, [{"id": "v2-1", "subject": "Biology", "class": 11,
                                   "chapter_name": "Cell: The Unit of Life", "chapter_name_en": None,
                                   "language": "en", "content": "...", "similarity": 0.7}])

    def fail_if_called(*a, **kw):
        raise AssertionError("search_ncert (v1) must not be called when v2 has rows")

    saved = (server.async_client, server.get_embedding, server.search_ncert, server.NCERT_V2_ENABLED)
    server.async_client, server.get_embedding, server.search_ncert = FakeClient(), fake_embed, fail_if_called
    try:
        server.NCERT_V2_ENABLED = True
        stats = {}
        results = asyncio.run(server._search_ncert_for_chat("function of golgi apparatus", "en", stats=stats))
        check("v2's row returned", results and results[0]["id"] == "v2-1", results)
        check("stats marks version v2", stats.get("version") == "v2", stats)
        check("stats still carries language + top_raw_score", stats.get("language") == "en" and stats.get("top_raw_score") == 0.7, stats)
    finally:
        server.async_client, server.get_embedding, server.search_ncert, server.NCERT_V2_ENABLED = saved


def test_v2_empty_or_erroring_falls_back_to_v1():
    print("\n2. v2 empty or raising -> real fallback to search_ncert, stats marks version v1")

    class EmptyClient:
        async def post(self, url, headers=None, json=None):
            return FakeResp(200, [])

    v1_calls = []

    async def fake_v1(query, limit=3, stats=None):
        v1_calls.append((query, limit))
        if stats is not None:
            stats["language"] = "en"
            stats["top_raw_score"] = 0.33
        return [{"id": "v1-1", "subject": "Biology", "class": 11, "chapter_name": "Biomolecules",
                 "chapter_name_en": None, "language": "en", "content": "...", "similarity": 0.33}]

    saved = (server.async_client, server.get_embedding, server.search_ncert, server.NCERT_V2_ENABLED)
    server.async_client, server.get_embedding, server.search_ncert = EmptyClient(), fake_embed, fake_v1
    try:
        server.NCERT_V2_ENABLED = True
        stats = {}
        results = asyncio.run(server._search_ncert_for_chat("a chapter not loaded into v2 yet", "en", stats=stats))
        check("v1's fallback row returned", results and results[0]["id"] == "v1-1", results)
        check("search_ncert called once with the same query/limit", v1_calls == [("a chapter not loaded into v2 yet", 3)], v1_calls)
        check("stats marks version v1 (fallback), not v2", stats.get("version") == "v1", stats)
        check("fallback's own stats (language/top_raw_score) preserved", stats.get("top_raw_score") == 0.33, stats)

        # RPC raising outright (not just empty) -- same fallback, no crash.
        v1_calls.clear()

        class FailingClient:
            async def post(self, url, headers=None, json=None):
                raise ConnectionError("boom")

        server.async_client = FailingClient()
        stats2 = {}
        results2 = asyncio.run(server._search_ncert_for_chat("a chapter not loaded into v2 yet", "en", stats=stats2))
        check("RPC failure also falls back, doesn't raise", v1_calls == [("a chapter not loaded into v2 yet", 3)], v1_calls)
        check("fallback row returned despite the RPC exception", results2 and results2[0]["id"] == "v1-1", results2)
        check("stats marks version v1 on the exception path too", stats2.get("version") == "v1", stats2)
    finally:
        server.async_client, server.get_embedding, server.search_ncert, server.NCERT_V2_ENABLED = saved


def test_flag_off_never_touches_v2():
    print("\n3. NCERT_V2_ENABLED false -> search_ncert_v2 never called, behaviour unchanged")

    def fail_if_called(*a, **kw):
        raise AssertionError("search_ncert_v2 must not be called while the flag is off")

    v1_calls = []

    async def fake_v1(query, limit=3, stats=None):
        v1_calls.append((query, limit))
        if stats is not None:
            stats["language"] = "en"
            stats["top_raw_score"] = 0.6
        return [{"id": "v1-only", "content": "..."}]

    saved = (server.search_ncert_v2, server.search_ncert, server.NCERT_V2_ENABLED)
    server.search_ncert_v2, server.search_ncert = fail_if_called, fake_v1
    try:
        server.NCERT_V2_ENABLED = False
        stats = {}
        results = asyncio.run(server._search_ncert_for_chat("what is osmosis", "en", stats=stats))
        check("search_ncert's row returned", results == [{"id": "v1-only", "content": "..."}], results)
        check("search_ncert called with the same query/limit as before", v1_calls == [("what is osmosis", 3)], v1_calls)
        check("no version key added when v2 is never in play", "version" not in stats, stats)
    finally:
        server.search_ncert_v2, server.search_ncert, server.NCERT_V2_ENABLED = saved


def test_hindi_never_routes_to_v2():
    print("\n4. Hindi query -> always search_ncert directly, even with the flag on")

    def fail_if_called(*a, **kw):
        raise AssertionError("search_ncert_v2 is English-only; must never be called for language='hi'")

    v1_calls = []

    async def fake_v1(query, limit=3, stats=None):
        v1_calls.append((query, limit))
        return [{"id": "hi-1", "content": "..."}]

    saved = (server.search_ncert_v2, server.search_ncert, server.NCERT_V2_ENABLED)
    server.search_ncert_v2, server.search_ncert = fail_if_called, fake_v1
    try:
        server.NCERT_V2_ENABLED = True
        results = asyncio.run(server._search_ncert_for_chat("परासरण क्या है?", "hi"))
        check("search_ncert's row returned for Hindi", results == [{"id": "hi-1", "content": "..."}], results)
        check("search_ncert called once for the Hindi query", v1_calls == [("परासरण क्या है?", 3)], v1_calls)
    finally:
        server.search_ncert_v2, server.search_ncert, server.NCERT_V2_ENABLED = saved


class StreamFakeClient:
    """Records every POST (used to inspect the ncert_retrieval_log write), and answers the v2
    hybrid RPC itself so search_ncert_v2's real fallback logic runs unmocked."""
    def __init__(self, v2_rows):
        self.posts = []
        self.v2_rows = v2_rows

    async def post(self, url, headers=None, json=None, **kw):
        self.posts.append((url, json))
        if "/rpc/match_ncert_v2_hybrid" in url:
            return FakeResp(200, self.v2_rows)
        return FakeResp(200, [])

    async def get(self, url, headers=None, params=None, **kw):
        return FakeResp(200, [])

    def posts_to(self, table):
        return [p for p in self.posts if p[0].split("?")[0].endswith(f"/rest/v1/{table}")]


def _run_stream_response_with_v2(v2_rows, question="What is Ohm's law?"):
    saved = {n: getattr(server, n) for n in (
        "async_client", "_stream_with_peak_fallback", "check_max_usage_tier", "_spawn_background",
        "get_embedding", "_diagram_exists_for", "NCERT_V2_ENABLED")}
    client = StreamFakeClient(v2_rows)

    async def fake_stream(system, messages, user_id, ip, endpoint, billing_context=None, force_qwen=False, max_tokens=None):
        yield "VISUAL_INTENT: no\n\nAnswer:\n- Ohm's law: V = IR."

    async def fake_tier(user_id):
        return None

    def fake_spawn(coro):
        coro.close()

    async def fake_diagram(text):
        return False

    server.async_client = client
    server._stream_with_peak_fallback = fake_stream
    server.check_max_usage_tier = fake_tier
    server._spawn_background = fake_spawn
    server.get_embedding = fake_embed
    server._diagram_exists_for = fake_diagram
    server.NCERT_V2_ENABLED = True
    try:
        async def drain():
            out = "".join([c async for c in server.stream_response(question, personalize=False, skip_cache=True)])
            await asyncio.sleep(0)
            return out
        asyncio.run(drain())
    finally:
        for name, value in saved.items():
            setattr(server, name, value)
    return client


def test_end_to_end_log_carries_version():
    print("\n5. end-to-end through stream_response: ncert_retrieval_log carries version")

    v2_row = {"id": "p2", "subject": "Physics", "class": 12, "chapter_name": "Current Electricity",
              "chapter_name_en": None, "language": "en", "content": "Ohm's law: V = IR ...", "similarity": 0.71}
    client = _run_stream_response_with_v2([v2_row])
    log = client.posts_to("ncert_retrieval_log")
    check("one log row written", len(log) == 1, log)
    chunks = log[0][1]["chunks"] if log else None
    check("v2 success is tagged version v2 in the log", chunks and chunks[0].get("version") == "v2", chunks)

    client2 = _run_stream_response_with_v2([])  # v2 has nothing -> real fallback to search_ncert
    log2 = client2.posts_to("ncert_retrieval_log")
    check("fallback-to-v1 case still logs (as 'none', nothing in ncert_content either)",
          len(log2) == 1 and log2[0][1]["outcome"] == "none", log2)
    check("no chunks to tag a version onto when nothing was retrieved at all", log2[0][1]["chunks"] == [], log2)


if __name__ == "__main__":
    test_flag_on_english_routes_to_v2()
    test_v2_empty_or_erroring_falls_back_to_v1()
    test_flag_off_never_touches_v2()
    test_hindi_never_routes_to_v2()
    test_end_to_end_log_carries_version()
    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
