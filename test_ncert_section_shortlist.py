"""Unit tests for the NCERT section shortlist step -- NCERT_SHORTLIST_ENABLED, see
_shortlist_ncert_sections / _bm25_score_docs / _should_shortlist_sections /
_safe_shortlist_ncert_sections / _ncert_sections_cache_for_ids in server.py.

Run with: python test_ncert_section_shortlist.py
Plain pass/fail harness, same style as test_ncert_block_retry.py. Fully offline: no model calls,
no network, no DB.

1. BM25 ranking order on a tiny fake set -- a section whose heading/keywords actually mention the
   query terms outranks one that only shares unrelated chapter text.
2. Heading/keywords are weighted above the chapter name.
3. Threshold behavior: _should_shortlist_sections only flips on past NCERT_SHORTLIST_MIN_SECTIONS.
4. Disabled switch: NCERT_SHORTLIST_ENABLED=False keeps _should_shortlist_sections False
   regardless of size.
5. Fallback on empty cache / scoring error: _shortlist_ncert_sections/_safe_shortlist_ncert_sections
   degrade to [] rather than raising.
6. Chapter hint boost: a hinted chapter's sections rank above an equally-unmatched other chapter,
   but a genuinely better lexical match elsewhere can still outrank the hint (boost, not filter).
7. _ncert_sections_cache_for_ids builds a correctly-restricted {"by_id", "index_lines"} subset.
"""
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


def _row(id_, chapter, heading, keywords=""):
    return {"id": id_, "class": "11", "subject": "Biology", "chapter": chapter,
            "heading": heading, "ordinal": id_, "text": "irrelevant body text", "keywords": keywords}


def _cache(rows):
    return {"loaded_at": 0.0, "by_id": {r["id"]: r for r in rows}, "index_lines": []}


# ---------- 1-2. BM25 ranking order + field weighting ----------

def test_ranking_prefers_heading_and_keyword_matches():
    print("\n1. a section whose heading/keywords mention the query terms outranks one that doesn't")
    rows = [
        _row(1, "Biological Classification", "Kingdom Monera", "bacteria archaebacteria cyanobacteria"),
        _row(2, "Biological Classification", "Kingdom Fungi", "mushroom mould yeast"),
        _row(3, "Biological Classification", "Kingdom Plantae", "algae bryophyte pteridophyte"),
    ]
    ids = server._shortlist_ncert_sections("what organisms are found in kingdom monera bacteria", _cache(rows))
    check("Kingdom Monera section ranked first", ids[0] == 1, ids)


def test_heading_and_keywords_weighted_above_chapter():
    print("\n2. heading/keywords tokens outweigh the same word appearing only in the chapter name")
    rows = [
        # "photosynthesis" only in the chapter name, not in this section's own heading/keywords.
        _row(1, "Photosynthesis in Higher Plants", "Site of Photosynthesis", "chloroplast structure"),
        # "photosynthesis" in the heading itself -- should outrank the chapter-only match.
        _row(2, "Photosynthesis in Higher Plants", "Photosynthesis Pigments", "chlorophyll carotenoid"),
    ]
    ids = server._shortlist_ncert_sections("explain photosynthesis pigments", _cache(rows))
    check("heading match ranked above chapter-only match", ids[0] == 2, ids)


# ---------- 3-4. threshold + kill switch ----------

def test_threshold_behavior():
    print("\n3. _should_shortlist_sections only turns on past NCERT_SHORTLIST_MIN_SECTIONS")
    saved_enabled, saved_min = server.NCERT_SHORTLIST_ENABLED, server.NCERT_SHORTLIST_MIN_SECTIONS
    try:
        server.NCERT_SHORTLIST_ENABLED = True
        server.NCERT_SHORTLIST_MIN_SECTIONS = 80
        check("at threshold (80) stays off", server._should_shortlist_sections(80) is False)
        check("below threshold (79) stays off", server._should_shortlist_sections(79) is False)
        check("above threshold (81) turns on", server._should_shortlist_sections(81) is True)
    finally:
        server.NCERT_SHORTLIST_ENABLED, server.NCERT_SHORTLIST_MIN_SECTIONS = saved_enabled, saved_min


def test_disabled_switch():
    print("\n4. NCERT_SHORTLIST_ENABLED=False keeps the step off regardless of size")
    saved_enabled, saved_min = server.NCERT_SHORTLIST_ENABLED, server.NCERT_SHORTLIST_MIN_SECTIONS
    try:
        server.NCERT_SHORTLIST_ENABLED = False
        server.NCERT_SHORTLIST_MIN_SECTIONS = 80
        check("off even with 500 sections", server._should_shortlist_sections(500) is False)
    finally:
        server.NCERT_SHORTLIST_ENABLED, server.NCERT_SHORTLIST_MIN_SECTIONS = saved_enabled, saved_min


# ---------- 5. fallback on empty cache / scoring error ----------

def test_empty_cache_returns_empty():
    print("\n5a. an empty cache returns [] rather than raising")
    ids = server._shortlist_ncert_sections("any question", _cache([]))
    check("empty list returned", ids == [])


def test_safe_wrapper_falls_back_on_scoring_error():
    print("\n5b. a scoring exception is caught by the safe wrapper and returns []")
    rows = [_row(1, "Chapter", "Heading", "keywords")]
    saved = server._bm25_score_docs
    try:
        server._bm25_score_docs = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        check("raw function raises", _raises(lambda: server._shortlist_ncert_sections("q", _cache(rows))))
        ids = server._safe_shortlist_ncert_sections("q", _cache(rows))
        check("safe wrapper returns [] instead of raising", ids == [])
    finally:
        server._bm25_score_docs = saved


def _raises(fn):
    try:
        fn()
        return False
    except Exception:
        return True


def test_shortlist_size_cap():
    print("\n5c. the shortlist never returns more than NCERT_SHORTLIST_SIZE ids")
    saved_size = server.NCERT_SHORTLIST_SIZE
    try:
        server.NCERT_SHORTLIST_SIZE = 3
        rows = [_row(i, "Chapter", f"Heading {i}", f"term{i}") for i in range(1, 11)]
        ids = server._shortlist_ncert_sections("term5 term6 term7", _cache(rows))
        check("exactly 3 ids returned", len(ids) == 3, ids)
    finally:
        server.NCERT_SHORTLIST_SIZE = saved_size


# ---------- 6. chapter hint boost (boost, not restrict) ----------

def test_chapter_hint_boosts_without_restricting():
    print("\n6a. a chapter hint lifts same-chapter sections above an equally-unmatched other chapter")
    rows = [
        _row(1, "Animal Kingdom", "Phylum Porifera", "sponges spicules"),
        _row(2, "Plant Kingdom", "Algae", "chlorophyceae rhodophyceae"),
    ]
    # Query shares no real words with either section -- without the hint this is a pure tie.
    ids = server._shortlist_ncert_sections("describe this topic", _cache(rows), chapter_hint="Animal Kingdom")
    check("hinted chapter's section ranked first", ids[0] == 1, ids)


def test_chapter_hint_does_not_override_a_real_match_elsewhere():
    print("\n6b. a genuinely strong match in a different chapter still outranks the hint")
    rows = [
        _row(1, "Animal Kingdom", "Phylum Porifera", "sponges spicules"),
        _row(2, "Plant Kingdom", "Chlorophyceae Algae", "chlorophyceae rhodophyceae green algae"),
    ]
    ids = server._shortlist_ncert_sections(
        "what is chlorophyceae algae", _cache(rows), chapter_hint="Animal Kingdom"
    )
    check("the real match wins despite the other chapter's hint", ids[0] == 2, ids)


# ---------- 7. cache subsetting for the picker prompt ----------

def test_cache_for_ids_subset():
    print("\n7. _ncert_sections_cache_for_ids restricts by_id/index_lines to the given ids, in order")
    rows = [_row(1, "Ch", "Head A", "kw"), _row(2, "Ch", "Head B"), _row(3, "Ch", "Head C", "kw")]
    full = _cache(rows)
    full["index_lines"] = [server._ncert_section_index_line(r) for r in rows]
    subset = server._ncert_sections_cache_for_ids(full, [3, 1])
    check("by_id has only the requested ids", set(subset["by_id"].keys()) == {1, 3}, subset["by_id"])
    check("index_lines follows the requested order", subset["index_lines"][0].startswith("3:"), subset["index_lines"])
    check("index_lines second entry is id 1", subset["index_lines"][1].startswith("1:"), subset["index_lines"])
    missing = server._ncert_sections_cache_for_ids(full, [1, 999])
    check("a nonexistent id is silently dropped", list(missing["by_id"].keys()) == [1], missing)


if __name__ == "__main__":
    test_ranking_prefers_heading_and_keyword_matches()
    test_heading_and_keywords_weighted_above_chapter()
    test_threshold_behavior()
    test_disabled_switch()
    test_empty_cache_returns_empty()
    test_safe_wrapper_falls_back_on_scoring_error()
    test_shortlist_size_cap()
    test_chapter_hint_boosts_without_restricting()
    test_chapter_hint_does_not_override_a_real_match_elsewhere()
    test_cache_for_ids_subset()
    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
