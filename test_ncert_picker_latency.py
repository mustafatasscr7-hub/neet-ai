"""Unit tests for the NCERT picker latency work (2026-11-16) -- the plan/cache concurrency fold-in,
the picker skip rule (NCERT_PICKER_SKIP_ENABLED/NCERT_PICKER_SKIP_MIN_SCORE, see
_shortlist_top_score/_safe_shortlist_top_score), the small in-memory picker-result cache (see
_ncert_picker_cache_key/_ncert_picker_result_cache_get/_ncert_picker_result_cache_set), and the
tightened _pick_ncert_sections max_tokens/timeout, in server.py.

Run with: python test_ncert_picker_latency.py
Plain pass/fail harness, same style as test_ncert_block_retry.py. Fully offline: the model client
is a fake, no network, no DB.

1. Fail-open on timeout/error: _pick_ncert_sections returns [] (never raises) when the DeepSeek
   call times out or raises, same as the existing NONE/empty-index paths.
2. Skip rule: _shortlist_top_score ranks a real lexical match above an unrelated query; the
   NCERT_PICKER_SKIP_MIN_SCORE comparison a caller does with it separates the two; a scoring
   exception fails OPEN (+inf, never skips) via _safe_shortlist_top_score.
3. Cache: a fresh key misses; set-then-get round-trips the exact ids; a different normalized
   question (whitespace/case only) still hits the same key; a different section-set version
   misses even for the same question text; the FIFO cap evicts the oldest entry once full.
"""
import asyncio
import io
import os
import sys
from types import SimpleNamespace

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


def _cache(rows):
    return {"loaded_at": 0.0, "by_id": {r["id"]: r for r in rows}, "index_lines": [server._ncert_section_index_line(r) for r in rows]}


def _row(id_, chapter, heading, keywords=""):
    return {"id": id_, "class": "11", "subject": "Biology", "chapter": chapter,
            "heading": heading, "ordinal": id_, "text": "irrelevant body text", "keywords": keywords}


# ---------- 1. fail-open on timeout/error ----------

class FakePickerClient:
    def __init__(self, text=None, raise_exc=None, input_tokens=1500, output_tokens=3):
        self.kwargs = None
        self.text, self.raise_exc = text, raise_exc
        self.input_tokens, self.output_tokens = input_tokens, output_tokens
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kw):
        self.kwargs = kw
        if self.raise_exc:
            raise self.raise_exc
        usage = SimpleNamespace(input_tokens=self.input_tokens, cache_creation_input_tokens=0,
                                 cache_read_input_tokens=0, output_tokens=self.output_tokens)
        return SimpleNamespace(content=[SimpleNamespace(text=self.text)], usage=usage)


class PickerClientSwap:
    def __enter__(self):
        self._saved = (server.deepseek_async_client, server.log_provider_usage)

        async def rec(*a, **k):
            pass

        server.log_provider_usage = rec
        return self

    def __exit__(self, *exc):
        server.deepseek_async_client, server.log_provider_usage = self._saved


PICKER_CACHE = _cache([_row(1, "Chapter 1", "The Living World", "life growth reproduction")])


def test_fail_open_on_timeout():
    print("\n1a. a timeout from the DeepSeek call returns [] rather than raising")
    with PickerClientSwap():
        server.deepseek_async_client = FakePickerClient(raise_exc=asyncio.TimeoutError("slow"))
        picked = asyncio.run(server._pick_ncert_sections("what is life", PICKER_CACHE, "u1"))
        check("returns empty list", picked == [], picked)


def test_fail_open_on_generic_exception():
    print("\n1b. a generic exception from the DeepSeek call also returns [] rather than raising")
    with PickerClientSwap():
        server.deepseek_async_client = FakePickerClient(raise_exc=RuntimeError("boom"))
        picked = asyncio.run(server._pick_ncert_sections("what is life", PICKER_CACHE, "u1"))
        check("returns empty list", picked == [], picked)


def test_still_picks_normally_when_the_call_succeeds():
    print("\n1c. sanity: a normal successful reply still parses correctly after the tightening")
    with PickerClientSwap():
        server.deepseek_async_client = FakePickerClient(text="1")
        picked = asyncio.run(server._pick_ncert_sections("what is life", PICKER_CACHE, "u1"))
        check("picked id 1", picked == [1], picked)
        check("max_tokens tightened to 12", server.deepseek_async_client.kwargs["max_tokens"] == 12,
              server.deepseek_async_client.kwargs.get("max_tokens"))


# ---------- 2. skip rule ----------

def test_top_score_ranks_real_match_above_unrelated_query():
    print("\n2a. a question sharing heading/keyword vocabulary scores far above an unrelated one")
    rows = [
        _row(1, "Biological Classification", "Kingdom Monera", "bacteria archaebacteria cyanobacteria"),
        _row(2, "Biological Classification", "Kingdom Fungi", "mushroom mould yeast"),
    ]
    cache = _cache(rows)
    on_topic_score = server._shortlist_top_score("explain kingdom monera bacteria", cache)
    off_topic_score = server._shortlist_top_score("thanks a lot for the help see you later", cache)
    check("on-topic score is clearly higher", on_topic_score > off_topic_score, (on_topic_score, off_topic_score))
    check("off-topic score is at or near zero", off_topic_score < 1.0, off_topic_score)


def test_skip_threshold_separates_the_two():
    print("\n2b. a threshold between the two scores, applied the same way the chat handler does, skips only the off-topic case")
    # A tiny fixture's absolute BM25 scores aren't comparable to the real ~109-row cache
    # NCERT_PICKER_SKIP_MIN_SCORE's own default was calibrated against (BM25's idf term scales
    # with corpus size) -- this test checks the comparison MECHANISM the chat handler uses
    # (skip iff top_score < threshold) with a threshold picked for this fixture's own scores,
    # not the production constant itself.
    rows = [_row(1, "Biological Classification", "Kingdom Monera", "bacteria archaebacteria cyanobacteria")]
    cache = _cache(rows)
    on_score = server._shortlist_top_score("explain kingdom monera bacteria", cache)
    off_score = server._shortlist_top_score("what is the weather like today", cache)
    local_threshold = (on_score + off_score) / 2
    check("on-topic score is above the off-topic score", on_score > off_score, (on_score, off_score))
    check("on-topic stays above a threshold set between them", not (on_score < local_threshold), (on_score, local_threshold))
    check("off-topic falls below that same threshold", off_score < local_threshold, (off_score, local_threshold))


def test_empty_cache_scores_zero():
    print("\n2c. an empty cache scores 0.0 rather than raising")
    check("zero for empty cache", server._shortlist_top_score("anything", _cache([])) == 0.0)


def test_safe_wrapper_fails_open_on_scoring_error():
    print("\n2d. a scoring exception fails OPEN (+inf, never skips) via the safe wrapper")
    rows = [_row(1, "Chapter", "Heading", "keywords")]
    cache = _cache(rows)
    saved = server._bm25_score_docs
    try:
        server._bm25_score_docs = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        check("raw function raises", _raises(lambda: server._shortlist_top_score("q", cache)))
        score = server._safe_shortlist_top_score("q", cache)
        check("safe wrapper returns +inf instead of raising", score == float("inf"), score)
        check("+inf never falls below any real skip floor", not (score < server.NCERT_PICKER_SKIP_MIN_SCORE))
    finally:
        server._bm25_score_docs = saved


def _raises(fn):
    try:
        fn()
        return False
    except Exception:
        return True


# ---------- 3. picker-result cache ----------

def test_cache_miss_then_hit():
    print("\n3a. a fresh key misses; set-then-get round-trips the exact ids")
    server._ncert_picker_result_cache.clear()
    key = server._ncert_picker_cache_key("What causes photosynthesis?", 123.0)
    check("fresh key misses", server._ncert_picker_result_cache_get(key) is None)
    server._ncert_picker_result_cache_set(key, [4, 7])
    check("hit returns the exact ids", server._ncert_picker_result_cache_get(key) == [4, 7])


def test_cache_key_normalizes_whitespace_and_case():
    print("\n3b. the same question, differently cased/spaced, hits the same cache entry")
    server._ncert_picker_result_cache.clear()
    key1 = server._ncert_picker_cache_key("What Causes Photosynthesis?", 123.0)
    server._ncert_picker_result_cache_set(key1, [9])
    key2 = server._ncert_picker_cache_key("  what causes   photosynthesis?  ", 123.0)
    check("normalized variant hits the same entry", server._ncert_picker_result_cache_get(key2) == [9])


def test_cache_invalidates_on_version_change():
    print("\n3c. the same question text under a different section-set version misses")
    server._ncert_picker_result_cache.clear()
    key_v1 = server._ncert_picker_cache_key("What causes photosynthesis?", 123.0)
    server._ncert_picker_result_cache_set(key_v1, [4])
    key_v2 = server._ncert_picker_cache_key("What causes photosynthesis?", 456.0)
    check("new version misses despite identical text", server._ncert_picker_result_cache_get(key_v2) is None)
    check("old version is still intact", server._ncert_picker_result_cache_get(key_v1) == [4])


def test_cache_fifo_eviction_bounds_size():
    print("\n3d. once full, the FIFO cap evicts the oldest entry so the cache never grows unbounded")
    server._ncert_picker_result_cache.clear()
    saved_max = server._NCERT_PICKER_RESULT_CACHE_MAX
    try:
        server._NCERT_PICKER_RESULT_CACHE_MAX = 3
        keys = [server._ncert_picker_cache_key(f"question {i}", 1.0) for i in range(4)]
        for i, k in enumerate(keys):
            server._ncert_picker_result_cache_set(k, [i])
        check("cache stays at the cap", len(server._ncert_picker_result_cache) == 3, len(server._ncert_picker_result_cache))
        check("oldest entry was evicted", server._ncert_picker_result_cache_get(keys[0]) is None)
        check("newest entry survives", server._ncert_picker_result_cache_get(keys[3]) == [3])
    finally:
        server._NCERT_PICKER_RESULT_CACHE_MAX = saved_max
        server._ncert_picker_result_cache.clear()


def test_cache_get_set_never_raise_on_bad_input():
    print("\n3e. get/set degrade silently (never raise) even if the cache dict is replaced with something broken")
    saved = server._ncert_picker_result_cache
    try:
        server._ncert_picker_result_cache = None
        check("get on a broken cache returns None, not an exception", server._ncert_picker_result_cache_get(("x", 1.0)) is None)
        check("set on a broken cache does not raise", not _raises(lambda: server._ncert_picker_result_cache_set(("x", 1.0), [1])))
    finally:
        server._ncert_picker_result_cache = saved


if __name__ == "__main__":
    test_fail_open_on_timeout()
    test_fail_open_on_generic_exception()
    test_still_picks_normally_when_the_call_succeeds()
    test_top_score_ranks_real_match_above_unrelated_query()
    test_skip_threshold_separates_the_two()
    test_empty_cache_scores_zero()
    test_safe_wrapper_fails_open_on_scoring_error()
    test_cache_miss_then_hit()
    test_cache_key_normalizes_whitespace_and_case()
    test_cache_invalidates_on_version_change()
    test_cache_fifo_eviction_bounds_size()
    test_cache_get_set_never_raise_on_bad_input()
    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
