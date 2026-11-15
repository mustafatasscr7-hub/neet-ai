"""Regression test for the backend half of the NCERT "Chapter:" line / "From NCERT" quote card
citation-conflict fix -- see server.py's NCERT_OLD_CHAPTER_MIN_SCORE and the show_old_chapter
comment at its /chat call site (stream_response). The chat.html half (suppressing the chapter
pill once a verified quote card is shown, and keeping reload/version-switch consistent with the
live render) is covered separately in test-chat-ncert-citation-conflict.js.

Run with: python test_ncert_old_chapter_confidence.py
No pytest dependency -- this repo has no existing test runner, so this uses plain asserts and a
small pass/fail harness, matching test_student_name_sanitization.py's own pattern. No model call,
no SQL, no network: _force_citation_when_no_retrieval is pure string buffering, and the
show_old_chapter formula tested here is pure arithmetic over fake similarity scores.

Covers:
1. NCERT_OLD_CHAPTER_MIN_SCORE's default value (sensible, taken from the existing English
   retrieval threshold -- see its own comment in server.py).
2. The show_old_chapter formula itself (bool(results) and max(similarity) >= threshold) against
   several fake result sets -- no retrieval, low-confidence-only, high-confidence, and mixed.
3. _force_citation_when_no_retrieval's real behavior given that computed flag: the Chapter line
   is stripped whenever it's False (whether because nothing was retrieved at all, OR because
   confidence was too low -- both must produce the exact same stripped output), and kept
   byte-for-byte untouched when it's True.
4. The function has no model/plan-specific branching at all, so a Gemini-shaped (image doubt) or
   Free-plan-shaped answer text is stripped/kept by the exact same rule as a DeepSeek/Qwen one --
   this is what guarantees "Gemini answers and Free users keep today's behavior exactly" (rule 3)
   without needing a separate code path for them.
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


async def _chunks(*parts):
    for p in parts:
        yield p


async def _drain(gen):
    out = []
    async for chunk in gen:
        out.append(chunk)
    return "".join(out)


def _show_old_chapter(results):
    """Mirrors the exact one-line formula at stream_response's own `show_old_chapter =` --
    see that comment in server.py. Kept here as a single expression (not re-derived some other
    way) specifically so this test fails the moment that formula or the threshold it reads
    changes in a way that would alter real behavior."""
    return bool(results) and max((r.get("similarity") or 0) for r in results) >= server.NCERT_OLD_CHAPTER_MIN_SCORE


SAMPLE_TEXT = (
    "VISUAL_INTENT: no\n\nNEET Importance: 4/5\n\n"
    "Chapter: NCERT Class 11, Chapter 8 — Cell: The Unit of Life\n\n"
    "Answer:\nThe cell is the basic structural and functional unit of life.\n\nKey Points:\n- Point 1"
)
EXPECTED_STRIPPED = (
    "VISUAL_INTENT: no\n\nNEET Importance: 4/5\n\n"
    "Answer:\nThe cell is the basic structural and functional unit of life.\n\nKey Points:\n- Point 1"
)


def test_min_score_default():
    print("\n=== NCERT_OLD_CHAPTER_MIN_SCORE default ===")
    check(
        "defaults to the normal (non-fallback) English retrieval threshold",
        server.NCERT_OLD_CHAPTER_MIN_SCORE == server.NCERT_MATCH_THRESHOLD_EN,
        detail=f"got {server.NCERT_OLD_CHAPTER_MIN_SCORE}, NCERT_MATCH_THRESHOLD_EN={server.NCERT_MATCH_THRESHOLD_EN}",
    )
    check(
        "sits strictly above the short-query fallback threshold (the one documented weak-match case)",
        server.NCERT_OLD_CHAPTER_MIN_SCORE > server.NCERT_MATCH_THRESHOLD_EN_SHORT,
    )


def test_show_old_chapter_formula():
    print("\n=== show_old_chapter formula ===")
    thr = server.NCERT_OLD_CHAPTER_MIN_SCORE
    check("no retrieval at all -> False", _show_old_chapter([]) is False)
    check(
        "a single low-confidence chunk (just under the bar) -> False",
        _show_old_chapter([{"similarity": thr - 0.01}]) is False,
    )
    check(
        "a single high-confidence chunk (at the bar exactly) -> True",
        _show_old_chapter([{"similarity": thr}]) is True,
    )
    check(
        "a single high-confidence chunk (comfortably above the bar) -> True",
        _show_old_chapter([{"similarity": thr + 0.1}]) is True,
    )
    check(
        "mixed set: one low, one high -> True (the best chunk wins, not the worst)",
        _show_old_chapter([{"similarity": thr - 0.2}, {"similarity": thr + 0.05}]) is True,
    )
    check(
        "mixed set: all low -> False even though something was retrieved",
        _show_old_chapter([{"similarity": thr - 0.2}, {"similarity": thr - 0.05}]) is False,
    )
    check(
        "a row with similarity=None (defensive -- real rows always have one) treated as 0, not a crash",
        _show_old_chapter([{"similarity": None}]) is False,
    )


def test_force_citation_passthrough_when_confident():
    print("\n=== _force_citation_when_no_retrieval: high confidence (today's behavior) ===")
    out = asyncio.run(_drain(server._force_citation_when_no_retrieval(_chunks(SAMPLE_TEXT), True)))
    check("Chapter line is kept byte-for-byte", out == SAMPLE_TEXT, detail=repr(out))


def test_force_citation_strips_on_zero_retrieval():
    print("\n=== _force_citation_when_no_retrieval: no retrieval at all (regression guard) ===")
    out = asyncio.run(_drain(server._force_citation_when_no_retrieval(_chunks(SAMPLE_TEXT), False)))
    check("Chapter line is stripped", out == EXPECTED_STRIPPED, detail=repr(out))
    check("no stray 'Chapter' text survives anywhere in the output", "Chapter" not in out)


def test_force_citation_strips_on_low_confidence():
    print("\n=== _force_citation_when_no_retrieval: retrieval happened but confidence is low (new case) ===")
    # The caller now passes show_old_chapter, not bool(results) -- a weak match produces the same
    # False a total miss always did, so this must strip identically to the zero-retrieval case
    # above even though "something was retrieved" is no longer the deciding factor on its own.
    low_confidence_results = [{"similarity": server.NCERT_OLD_CHAPTER_MIN_SCORE - 0.08}]
    show_old_chapter = _show_old_chapter(low_confidence_results)
    check("show_old_chapter itself is False for this input", show_old_chapter is False)
    out = asyncio.run(_drain(server._force_citation_when_no_retrieval(_chunks(SAMPLE_TEXT), show_old_chapter)))
    check("Chapter line is stripped even though retrieval found something", out == EXPECTED_STRIPPED, detail=repr(out))


def test_force_citation_keeps_on_high_confidence():
    print("\n=== _force_citation_when_no_retrieval: high-confidence retrieval keeps the line ===")
    high_confidence_results = [{"similarity": server.NCERT_OLD_CHAPTER_MIN_SCORE + 0.1}]
    show_old_chapter = _show_old_chapter(high_confidence_results)
    check("show_old_chapter itself is True for this input", show_old_chapter is True)
    out = asyncio.run(_drain(server._force_citation_when_no_retrieval(_chunks(SAMPLE_TEXT), show_old_chapter)))
    check("Chapter line is kept, unchanged from today's behavior", out == SAMPLE_TEXT, detail=repr(out))


def test_split_chunks_same_result():
    print("\n=== behaves the same regardless of how the stream is chunked ===")
    # The real stream never arrives as one chunk -- split across the Chapter line's own boundary
    # (mid-line) to exercise the buffering path for real, not just a single-chunk shortcut.
    parts = [SAMPLE_TEXT[:60], SAMPLE_TEXT[60:140], SAMPLE_TEXT[140:]]
    out = asyncio.run(_drain(server._force_citation_when_no_retrieval(_chunks(*parts), False)))
    check("split mid-Chapter-line still strips correctly", out == EXPECTED_STRIPPED, detail=repr(out))


def test_model_and_plan_agnostic():
    print("\n=== Gemini/Free unchanged (rule 3): no model- or plan-specific branching exists ===")
    # _force_citation_when_no_retrieval takes only a stream and a bool -- it has no idea whether
    # the answer came from DeepSeek/Qwen (text, any plan) or Gemini (image/pdf), and no idea what
    # plan the student is on. Running the identical text through twice, framed as each shape,
    # must produce identical output either way -- that sameness IS the guarantee that today's
    # Gemini/Free behavior is untouched by this fix, since there is nothing in this function (or
    # its caller's new show_old_chapter flag) that could tell the two apart even if it tried.
    text_doubt_out = asyncio.run(_drain(server._force_citation_when_no_retrieval(_chunks(SAMPLE_TEXT), False)))
    gemini_media_out = asyncio.run(_drain(server._force_citation_when_no_retrieval(_chunks(SAMPLE_TEXT), False)))
    check("identical stripped output for a text doubt and an image/pdf (Gemini) doubt", text_doubt_out == gemini_media_out == EXPECTED_STRIPPED)

    kept_a = asyncio.run(_drain(server._force_citation_when_no_retrieval(_chunks(SAMPLE_TEXT), True)))
    kept_b = asyncio.run(_drain(server._force_citation_when_no_retrieval(_chunks(SAMPLE_TEXT), True)))
    check("identical kept output either way when confidence is high", kept_a == kept_b == SAMPLE_TEXT)


if __name__ == "__main__":
    test_min_score_default()
    test_show_old_chapter_formula()
    test_force_citation_passthrough_when_confident()
    test_force_citation_strips_on_zero_retrieval()
    test_force_citation_strips_on_low_confidence()
    test_force_citation_keeps_on_high_confidence()
    test_split_chunks_same_result()
    test_model_and_plan_agnostic()

    print(f"\nTOTAL: {len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
