"""Unit tests for ncert_verify.py: the PDF-vs-sections text cross-check used by the admin NCERT
Content page's upload preview, its "Verify against PDF" action, and ncert_ingest.py's CLI. Fully
offline -- every PDF is a small synthetic one built in-process with PyMuPDF, same style as
test_admin_ncert_content.py's make_chapter_pdf. No network, no model calls.

Run with: python test_ncert_verify.py
Plain pass/fail harness like test_ncert_ingest.py.
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import fitz

import ncert_ingest_core as core
import ncert_verify as v

_passed = []
_failed = []


def check(label, condition, detail=""):
    if condition:
        _passed.append(label)
        print(f"  PASS: {label}")
    else:
        _failed.append(label)
        print(f"  FAIL: {label}" + (f" -- {detail}" if detail else ""))


CLEAN_PARA = (
    "How wonderful is the living world and the wide range of living types that are found in "
    "the forests and the oceans and the ponds and the rivers and the mountains of the earth."
)


def make_pdf(lines, page_width=595):
    doc = fitz.open()
    page = doc.new_page(width=page_width)
    y = 72
    for line in lines:
        if y > 780:
            page = doc.new_page(width=page_width)
            y = 72
        page.insert_text((72, y), line, fontsize=11)
        y += 18
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def good_chapter_pdf():
    return make_pdf([
        "CHAPTER 1",
        "THE LIVING WORLD",
        "1.1 Diversity In The Living World",
        CLEAN_PARA,
        "1.2 Taxonomic Categories",
        CLEAN_PARA,
        "1.3 Taxonomical Aids",
        CLEAN_PARA,
    ])


def test_pass_on_identical_text():
    pdf_bytes = good_chapter_pdf()
    result = core.process_chapter_pdf_bytes(pdf_bytes)
    check("the synthetic chapter has no validate_chapter flags", result["flags"] == [], result["flags"])

    verify = v.verify_text_coverage(pdf_bytes, result["sections"])
    check("identical text passes", verify["status"] == "PASS", verify)
    check("no sentence mismatches", verify["sentence_mismatches"] == [], verify)
    check("no missing paragraphs", verify["missing_paragraphs"] == [], verify)
    check("coverage_pct is a positive number", isinstance(verify["coverage_pct"], (int, float)) and verify["coverage_pct"] > 0, verify)
    check("kept_words equals the sum of the sections' own word counts",
          verify["kept_words"] == sum(s["word_count"] for s in result["sections"]), verify)
    check("total_words is at least kept_words (some front matter is always dropped)",
          verify["total_words"] >= verify["kept_words"], verify)


def test_coverage_calculation():
    pdf_bytes = good_chapter_pdf()
    result = core.process_chapter_pdf_bytes(pdf_bytes)
    verify = v.verify_text_coverage(pdf_bytes, result["sections"])
    expected_pct = round(verify["kept_words"] / verify["total_words"] * 100, 2)
    check("coverage_pct matches kept/total*100", verify["coverage_pct"] == expected_pct, (verify["coverage_pct"], expected_pct))
    check("word_diff is total minus kept", verify["word_diff"] == verify["total_words"] - verify["kept_words"], verify)


def test_fail_on_changed_sentence():
    pdf_bytes = good_chapter_pdf()
    result = core.process_chapter_pdf_bytes(pdf_bytes)
    sections = [dict(s) for s in result["sections"]]
    sections[0]["text"] = "This sentence was never anywhere in the original source PDF at all."
    verify = v.verify_text_coverage(pdf_bytes, sections)
    check("a sentence not in the PDF fails verification", verify["status"] == "FAIL", verify)
    check("the mismatch is reported", len(verify["sentence_mismatches"]) > 0, verify)
    check("no missing paragraphs are reported for this case", verify["missing_paragraphs"] == [], verify)


def test_fail_on_missing_paragraph():
    pdf_bytes = make_pdf([
        "CHAPTER 1",
        "THE LIVING WORLD",
        "1.1 Diversity In The Living World",
        CLEAN_PARA,
        "1.2 Taxonomic Categories",
        CLEAN_PARA,
        "1.3 Taxonomical Aids",
        "This entire paragraph about taxonomical aids got left out of storage by accident somehow.",
    ])
    result = core.process_chapter_pdf_bytes(pdf_bytes)
    sections = [dict(s) for s in result["sections"]]
    sections[-1]["text"] = ""  # simulate the last section's real text never making it into storage
    verify = v.verify_text_coverage(pdf_bytes, sections)
    check("a PDF paragraph missing from every section fails verification", verify["status"] == "FAIL", verify)
    check("the missing paragraph is reported", len(verify["missing_paragraphs"]) > 0, verify)
    check("no sentence mismatches are reported for this case", verify["sentence_mismatches"] == [], verify)


def test_short_missing_paragraph_does_not_fail():
    """A dropped paragraph of 8 words or fewer must not fail the check (the spec's own cutoff)."""
    pdf_bytes = make_pdf([
        "CHAPTER 1",
        "THE LIVING WORLD",
        "1.1 Diversity In The Living World",
        CLEAN_PARA,
        "1.2 Taxonomic Categories",
        "Only seven words long and nothing more.",
    ])
    result = core.process_chapter_pdf_bytes(pdf_bytes)
    sections = [dict(s) for s in result["sections"]]
    sections[-1]["text"] = ""
    verify = v.verify_text_coverage(pdf_bytes, sections)
    check("an 8-words-or-fewer dropped paragraph is not reported as missing", verify["missing_paragraphs"] == [], verify)
    check("and so the chapter still passes", verify["status"] == "PASS", verify)


def test_dropped_line_reasons():
    pdf_bytes = make_pdf([
        "UNIT I",
        "CHAPTER 1",
        "THE LIVING WORLD",
        "1.1 Diversity In The Living World",
        "THE LIVING WORLD 5",  # running header (ncert_clean_manual.RUNNING_HEADER_RES)
        "6",                     # page number
        "a",                     # stray-letter figure/diagram sub-label
        "ACTIVITY",              # box/sidebar marker (short all-caps, no digits)
        CLEAN_PARA,
        "1.2 Taxonomic Categories",
        CLEAN_PARA,
    ])
    result = core.process_chapter_pdf_bytes(pdf_bytes)
    verify = v.verify_text_coverage(pdf_bytes, result["sections"])
    reasons = {d["reason"] for d in verify["dropped_lines"]}
    check("unit intro detected", "unit intro" in reasons, reasons)
    check("running header detected", "running header" in reasons, reasons)
    check("page number detected", "page number" in reasons, reasons)
    check("figure caption or label detected", "figure caption or label" in reasons, reasons)
    check("box or sidebar detected", "box or sidebar" in reasons, reasons)
    check("every reason is one of the documented categories", reasons <= set(v.DROP_REASONS), reasons)
    check("dropped_by_reason counts match the dropped_lines list",
          sum(verify["dropped_by_reason"].values()) == verify["dropped_lines_count"], verify)


def test_repeated_contents_reason():
    """A heading that appears twice (contents listing, then the real body heading) -- the earlier
    occurrence is reported as a 'repeated contents' drop, matching dedupe_headings' own rule of
    keeping only the later one."""
    pdf_bytes = make_pdf([
        "CHAPTER 1",
        "THE LIVING WORLD",
        "1.1 Diversity In The Living World",
        "1.2 Taxonomic Categories",
        "1.1 Diversity In The Living World",
        CLEAN_PARA,
        "1.2 Taxonomic Categories",
        CLEAN_PARA,
    ])
    result = core.process_chapter_pdf_bytes(pdf_bytes)
    verify = v.verify_text_coverage(pdf_bytes, result["sections"])
    repeated = [d for d in verify["dropped_lines"] if d["reason"] == "repeated contents"]
    check("the earlier duplicate heading occurrences are reported as repeated contents", len(repeated) >= 1, verify["dropped_lines"])


def test_check_this_long_other_line():
    """A long (>25 word) dropped line that isn't a caption/header gets highlighted even when the
    'other' bucket as a whole stays under the 1% threshold. Front matter (before 'CHAPTER n') is
    unconditionally dropped regardless of is_junk_line, so a long ordinary sentence placed there
    is a clean way to land a >25-word line in the 'other' bucket."""
    long_other_line = (
        "This unusually long stray sentence sits in the front matter before the chapter marker "
        "and does not match any of the specific dropped-line categories at all whatsoever here now."
    )
    check("the test line is actually over 25 words", len(long_other_line.split()) > 25, long_other_line)
    # A real PDF line is page-width bound (rarely 25+ words); use a wide page so this one physical
    # line survives extraction intact instead of being clipped at the page edge.
    pdf_bytes = make_pdf([
        long_other_line,
        "CHAPTER 1",
        "THE LIVING WORLD",
        "1.1 Diversity In The Living World",
        CLEAN_PARA,
        "1.2 Taxonomic Categories",
        CLEAN_PARA,
        "1.3 Taxonomical Aids",
        CLEAN_PARA,
    ], page_width=3000)
    result = core.process_chapter_pdf_bytes(pdf_bytes)
    verify = v.verify_text_coverage(pdf_bytes, result["sections"])
    matches = [d for d in verify["dropped_lines"] if d["reason"] == "other" and d["words"] > 25]
    check("the long line was classified as 'other'", len(matches) == 1, verify["dropped_lines"])
    check("it is highlighted as check_this", matches and matches[0]["check_this"] is True, matches)


def test_check_this_other_bucket_threshold():
    """Short 'other' drops that individually stay under 25 words still get highlighted once the
    'other' bucket as a whole exceeds 1% of the chapter's total words."""
    pdf_bytes = make_pdf(
        ["CHAPTER 1", "THE LIVING WORLD", "1.1 Diversity In The Living World"]
        + ["Stray"] * 6  # six short junk-ish lines in a tiny chapter push "other" over 1%
        + [CLEAN_PARA, "1.2 Taxonomic Categories", CLEAN_PARA]
    )
    result = core.process_chapter_pdf_bytes(pdf_bytes)
    verify = v.verify_text_coverage(pdf_bytes, result["sections"])
    other_lines = [d for d in verify["dropped_lines"] if d["reason"] == "other"]
    check("at least one short 'other' line exists", len(other_lines) > 0, verify["dropped_lines"])
    check("every short 'other' line is flagged check_this once the bucket crosses 1%",
          other_lines and all(d["check_this"] for d in other_lines), other_lines)


def test_excerpts_capped_at_200_chars():
    pdf_bytes = good_chapter_pdf()
    result = core.process_chapter_pdf_bytes(pdf_bytes)
    sections = [dict(s) for s in result["sections"]]
    sections[0]["text"] = ("Z " * 300).strip()  # 600 chars, never found in the PDF -> mismatch
    verify = v.verify_text_coverage(pdf_bytes, sections)
    check("verification fails as expected", verify["status"] == "FAIL", verify)
    check("every sentence mismatch excerpt is capped at 200 chars",
          all(len(s) <= v.MAX_EXCERPT_CHARS for s in verify["sentence_mismatches"]),
          [len(s) for s in verify["sentence_mismatches"]])
    check("every dropped-line excerpt is capped at 200 chars",
          all(len(d["text"]) <= v.MAX_EXCERPT_CHARS for d in verify["dropped_lines"]),
          [len(d["text"]) for d in verify["dropped_lines"]])


def test_summary_is_a_plain_short_string():
    pdf_bytes = good_chapter_pdf()
    result = core.process_chapter_pdf_bytes(pdf_bytes)
    verify = v.verify_text_coverage(pdf_bytes, result["sections"])
    check("summary is a plain string", isinstance(verify["summary"], str), verify["summary"])
    check("summary is short", len(verify["summary"]) <= 500, verify["summary"])
    check("summary starts with the status", verify["summary"].startswith(verify["status"]), verify["summary"])


def main():
    test_pass_on_identical_text()
    test_coverage_calculation()
    test_fail_on_changed_sentence()
    test_fail_on_missing_paragraph()
    test_short_missing_paragraph_does_not_fail()
    test_dropped_line_reasons()
    test_repeated_contents_reason()
    test_check_this_long_other_line()
    test_check_this_other_bucket_threshold()
    test_excerpts_capped_at_200_chars()
    test_summary_is_a_plain_short_string()

    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
