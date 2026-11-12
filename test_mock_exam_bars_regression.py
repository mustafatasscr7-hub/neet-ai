"""Regression guard for the modulus-bar fix (2026-11-11): runs the real deterministic Mock exam
scan (process_mock_exam_text.scan_mock_exam_pdf) against the real sample PDF (MockTest1.pdf) when
a copy can be found locally, and fails if the flagged count regresses above its own known-good
baseline, or if the reconstruction path's own engagement rate collapses - the two numbers a past
regression (modulus bars, commit 451d495) was believed to have moved; see
process_mock_exam_math.py's own "Regression" comment (in its modulus-bars section) for the full
story of what actually changed and what did not. Skips (exit 0) when no copy of the sample is
found locally - this is a real exam paper, never committed to this repo, so it only ever runs
opportunistically, on whatever machine happens to already have a local copy.

Run with: python test_mock_exam_bars_regression.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import process_mock_exam_math as pmm  # noqa: E402
import process_mock_exam_text as pmt  # noqa: E402

# Known-good baseline on the real sample, measured directly against this repo's own git history:
# 31 flagged / 159 engaged at commit a3f92fd (the math-reconstruction commit immediately before
# the modulus-bar work), reproduced exactly by reverting 451d495 and rerunning this same scan -
# and reproduced again after re-adding modulus-bar handling the way this fix does it. Updated
# again after the reconstruct-before-splitting + font-aware-table + order-mismatch work (see
# process_mock_exam_math.py's own module docstring and process_mock_exam_text.py's own
# _check_order_mismatch): 11 flagged / 178 engaged, reproduced live. Updated AGAIN after fixing
# three confirmed-live silent errors (Q6/Q10 fraction-denominator digits, Q19's subscript-instead-
# of-superscript, Q29's cross-question leak -- all three previously WRONG and unflagged), two
# further confirmed-live silent errors found during the SAME session's own verification pass
# (Q18's multi-line-stem baseline misclassification, Q39's fraction over-claiming an adjacent
# "tan^-1"-style exponent character) -- all five fixed at the root -- and adding new conservative
# geometry/text-confidence gates (requirement 2: an ambiguous fraction/hat-vector/baseline-
# sensitive-shift/base-text-size attachment, a drawn root glyph -- this sample's own root sign is
# never a real character, only ever a drawn shape, so a dropped one previously left NO trace at
# all, see process_mock_exam_math.detect_drawn_root_glyphs -- and two decimal numbers merged with
# no operator between them, see process_mock_exam_text._has_merged_decimal_numbers): 21 flagged /
# 178 engaged, reproduced live -- every one of those NEW flags (Physics 3, 4, 11, 14, 22, 27, 39,
# 40, Chemistry 1, 27, 28, Biology 35) verified against the real page image as a genuine,
# previously-silent issue, not a false positive. A real regression must never push flagged ABOVE
# this bound, or engagement BELOW it. The bounds below are a few points looser than the exact
# reproduced numbers (not 21/178 themselves) for the same reason the original bound was already
# looser than its own exact number -- this layout's own bold-marker detection picking up a
# slightly different split on some other real sample should not by itself trip this guard without
# masking an actual collapse.
_MAX_FLAGGED = 25
_MIN_ENGAGED = 175
_EXPECTED_TOTAL_QUESTIONS = 180  # Physics 45 + Chemistry 45 + Biology 90, this layout's own format

# Search order: this machine's own Claude-session cache first (where a previous session's own
# testing left a copy specifically for this kind of regression check to find again, and the one
# confirmed live to reproduce this script's own known-good baseline below), then the project
# folder itself, then the ordinary places a human would keep a sample exam PDF. More than one
# same-named "MockTest1.pdf" can genuinely exist on one machine - confirmed live on the machine
# this fix was developed on: the file actually on Desktop/OneDrive Desktop there is a DIFFERENT,
# unrelated mock paper that only happens to share the filename, and scans to its own different
# (also 180-question - not a usable disambiguator on its own) result. Every candidate found is
# still scanned and checked for a 180-question total as a cheap sanity floor, but the ORDER here
# is what actually decides which one wins when more than one real candidate exists.
_SEARCH_PATHS = [
    os.path.expandvars(r"%LOCALAPPDATA%\Temp\claude\mockexam_sample\MockTest1.pdf"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "MockTest1.pdf"),
    os.path.expanduser(r"~\Desktop\MockTest1.pdf"),
    os.path.expanduser(r"~\OneDrive\Desktop\MockTest1.pdf"),
    os.path.expanduser(r"~\Downloads\MockTest1.pdf"),
]


def _candidates():
    seen = set()
    for path in _SEARCH_PATHS:
        norm = os.path.normcase(os.path.abspath(path))
        if norm in seen:
            continue
        seen.add(norm)
        if os.path.isfile(path):
            yield path


def _scan_with_engagement_count(pdf_bytes):
    """Runs the real scan once, counting how many of its own build_option_zones calls engaged
    (returned non-None) alongside the scan's own normal result - same shape as the ad-hoc
    measurement harness this fix was developed against, kept here so a future regression shows up
    the same way it was originally found."""
    engaged = 0
    orig_build = pmm.build_option_zones

    def _counting_build(flat_spans, question_bbox):
        nonlocal engaged
        r = orig_build(flat_spans, question_bbox)
        if r is not None:
            engaged += 1
        return r

    pmm.build_option_zones = _counting_build
    pmt.build_option_zones = _counting_build
    try:
        result = pmt.scan_mock_exam_pdf(pdf_bytes, "RegressionCheck")
    finally:
        pmm.build_option_zones = orig_build
        pmt.build_option_zones = orig_build
    return result, engaged


def main():
    candidates = list(_candidates())
    if not candidates:
        print("SKIP: MockTest1.pdf not found in any known location - nothing to check here.")
        return 0

    chosen = None
    for path in candidates:
        with open(path, "rb") as f:
            pdf_bytes = f.read()
        result, engaged = _scan_with_engagement_count(pdf_bytes)
        if len(result["questions"]) == _EXPECTED_TOTAL_QUESTIONS:
            chosen = (path, result, engaged)
            break
        print(f"  (skipping {path}: scanned to {len(result['questions'])} questions, "
              f"not this layout's own {_EXPECTED_TOTAL_QUESTIONS})")

    if chosen is None:
        print(f"SKIP: found {len(candidates)} file(s) named MockTest1.pdf, but none scanned to "
              f"{_EXPECTED_TOTAL_QUESTIONS} questions - likely a different PDF entirely, not this "
              f"layout's own sample.")
        return 0

    path, result, engaged = chosen
    flagged = [q for q in result["questions"] if q["needs_review"]]

    print(f"Sample: {path}")
    print(f"flagged: {len(flagged)} (max allowed {_MAX_FLAGGED})")
    print(f"reconstruction path engaged: {engaged}/{_EXPECTED_TOTAL_QUESTIONS} "
          f"(min required {_MIN_ENGAGED})")

    ok = True
    if len(flagged) > _MAX_FLAGGED:
        print(f"FAIL: flagged count {len(flagged)} exceeds the known-good baseline of {_MAX_FLAGGED}")
        ok = False
    if engaged < _MIN_ENGAGED:
        print(f"FAIL: reconstruction path engaged on only {engaged} questions, below the "
              f"known-good baseline of {_MIN_ENGAGED}")
        ok = False
    if ok:
        print("PASS: both counts within their known-good bounds")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
