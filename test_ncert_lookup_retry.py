"""A failed NCERT/diagram lookup is retried once with the raw query, never fanned out.

Run with: python test_ncert_lookup_retry.py
Plain pass/fail harness like test_cost_fixes.py. Fully offline: the Supabase RPC and the Gemini
embedding are fakes.

1. _match_with_merge_fallback: a failed lookup (results None) retries the same embedding once;
   a second failure returns [] with no merge variants tried.
2. A genuine empty result (not a failure) still tries the merge variants, as before.
3. search_ncert's Hindi path end to end: one HTTP 500 costs one embed + two RPC calls.
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


QUERY = "SN1 और SN2 अभिक्रिया में क्या अंतर है?"  # 8 words -> 7 merge variants
ROW = {"id": "a", "subject": "Chemistry", "class": 12, "chapter_name": "हैलोऐल्केन तथा हैलोऐरीन",
       "chapter_name_en": "Haloalkanes and Haloarenes", "language": "hi",
       "content": "SN1 अभिक्रिया दो पदों में होती है।", "similarity": 0.74}


def run_fallback(outcomes):
    """outcomes: what each successive search_fn call returns."""
    embedded, searched = [], []
    outcomes = list(outcomes)

    def embed_fn(text):
        embedded.append(text)
        return [0.0]

    async def search_fn(embedding):
        searched.append(embedding)
        return outcomes.pop(0) if outcomes else ([], 0.6)

    result = asyncio.run(server._match_with_merge_fallback(QUERY, embed_fn, search_fn, 0.57))
    return result, embedded, searched


def test_failed_lookup_retries_once():
    print("\n1. failed lookup -> one retry, no fan-out")
    result, embedded, searched = run_fallback([(None, None), ([ROW], 0.74)])
    check("retry's results returned", result == [ROW], result)
    check("raw query embedded once, no variants", embedded == [QUERY], embedded)
    check("two lookups (first + retry)", len(searched) == 2, len(searched))

    result, embedded, searched = run_fallback([(None, None), (None, None)])
    check("second failure returns []", result == [], result)
    check("still no variants after two failures", embedded == [QUERY], embedded)
    check("still exactly two lookups", len(searched) == 2, len(searched))

    result, embedded, searched = run_fallback([(None, None), ([], 0.4)])
    check("retry below off-topic gate returns [] with no variants", result == [] and embedded == [QUERY], embedded)


def test_genuine_empty_still_fans_out():
    print("\n2. genuine empty result above the gate -> merge variants, unchanged")
    result, embedded, searched = run_fallback([([], 0.6)])
    variants = server._generate_space_merge_variants(QUERY)
    check("every merge variant tried", embedded == [QUERY] + variants, len(embedded))
    check("empty result stays []", result == [], result)


class FakeResp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body

    def json(self):
        return self._body


def test_search_ncert_hindi_500():
    print("\n3. search_ncert (Hindi): one HTTP 500")
    posts, embeds = [], []
    statuses = [500, 200]

    class FakeClient:
        async def post(self, url, headers=None, json=None):
            posts.append(url)
            status = statuses.pop(0) if statuses else 200
            return FakeResp(status, [ROW] if status == 200 else {"message": "canceling statement due to statement timeout"})

    def fake_embed(text):
        embeds.append(text)
        return [0.0] * 3072

    saved = server.async_client, server._get_gemini_query_embedding
    server.async_client, server._get_gemini_query_embedding = FakeClient(), fake_embed
    try:
        rows = asyncio.run(server.search_ncert(QUERY))
    finally:
        server.async_client, server._get_gemini_query_embedding = saved
    check("Hindi RPC used", posts and all(u.endswith("/rpc/match_ncert_hi_gemini") for u in posts), posts)
    check("one embed + two RPC calls", len(embeds) == 1 and len(posts) == 2, (len(embeds), len(posts)))
    check("retry's row returned", [r["id"] for r in rows] == ["a"], rows)


if __name__ == "__main__":
    test_failed_lookup_retries_once()
    test_genuine_empty_still_fans_out()
    test_search_ncert_hindi_500()
    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
