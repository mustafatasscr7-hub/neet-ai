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


PARA_1 = (
    "How wonderful is the living world and the wide range of living types that are found in "
    "the forests and the oceans and the ponds and the rivers and the mountains of the earth."
)
PARA_2 = (
    "Taxonomic categories such as kingdom phylum class order family genus and species allow "
    "every living organism on earth to be placed into a clear hierarchy for study."
)
PARA_3 = (
    "Taxonomical aids like keys museums zoological parks herbaria and botanical gardens help "
    "both students and researchers identify and understand different organisms accurately."
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
        PARA_1,
        "1.2 Taxonomic Categories",
        PARA_2,
        "1.3 Taxonomical Aids",
        PARA_3,
    ])


def one_section_pdf(body_line, page_width=3000):
    # A wide page keeps body_line intact as a single physical line even when it runs long, so
    # the cosmetic differences below land inside one paragraph block, not split across two.
    return make_pdf([
        "CHAPTER 1",
        "THE LIVING WORLD",
        "1.1 Diversity In The Living World",
        body_line,
    ], page_width=page_width)


def by_type(differences, diff_type):
    return [d for d in differences if d["type"] == diff_type]


# ---------------- Baseline: identical content ----------------

def test_pass_on_identical_text():
    pdf_bytes = good_chapter_pdf()
    result = core.process_chapter_pdf_bytes(pdf_bytes)
    check("the synthetic chapter has no validate_chapter flags", result["flags"] == [], result["flags"])

    verify = v.verify_text_coverage(pdf_bytes, result["sections"])
    check("identical text passes", verify["status"] == "PASS", verify)
    check("no differences at all", verify["differences"] == [], verify["differences"])
    check("counts.real is 0", verify["counts"]["real"] == 0, verify["counts"])
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


# ---------------- Real differences ----------------

def test_missing_paragraph():
    pdf_bytes = good_chapter_pdf()
    result = core.process_chapter_pdf_bytes(pdf_bytes)
    sections = [dict(s) for s in result["sections"]]
    sections[2]["text"] = ""  # the whole "Taxonomical Aids" paragraph never made it into storage
    verify = v.verify_text_coverage(pdf_bytes, sections)
    check("a PDF paragraph missing from storage fails verification", verify["status"] == "FAIL", verify)
    missing = by_type(verify["differences"], "missing")
    check("exactly one missing difference", len(missing) == 1, verify["differences"])
    check("the missing difference is in the real category", missing and missing[0]["category"] == "real", missing)
    check("the missing difference names its section", missing and missing[0]["section"] == "1.3 Taxonomical Aids", missing)
    check("the missing difference has a PDF page number", missing and missing[0]["page"] is not None, missing)
    check("the missing difference has no stored excerpt", missing and missing[0]["stored_excerpt"] is None, missing)


def test_extra_paragraph():
    pdf_bytes = good_chapter_pdf()
    result = core.process_chapter_pdf_bytes(pdf_bytes)
    sections = [dict(s) for s in result["sections"]]
    sections[2]["text"] += "\nThis extra sentence about enzyme kinetics was invented and never appeared anywhere in the source PDF."
    verify = v.verify_text_coverage(pdf_bytes, sections)
    check("stored text not in the PDF fails verification", verify["status"] == "FAIL", verify)
    extra = by_type(verify["differences"], "extra")
    check("exactly one extra difference", len(extra) == 1, verify["differences"])
    check("the extra difference is in the real category", extra and extra[0]["category"] == "real", extra)
    check("the extra difference has no PDF excerpt", extra and extra[0]["pdf_excerpt"] is None, extra)
    check("the extra difference has a stored excerpt", extra and extra[0]["stored_excerpt"], extra)


def test_changed_word():
    pdf_bytes = good_chapter_pdf()
    result = core.process_chapter_pdf_bytes(pdf_bytes)
    sections = [dict(s) for s in result["sections"]]
    sections[1]["text"] = sections[1]["text"].replace("kingdom", "domain")
    verify = v.verify_text_coverage(pdf_bytes, sections)
    check("a one-word content change fails verification", verify["status"] == "FAIL", verify)
    changed = by_type(verify["differences"], "changed")
    check("exactly one changed difference", len(changed) == 1, verify["differences"])
    check("the changed difference carries word-level diff spans", changed and changed[0]["diff_spans"], changed)
    spans = changed[0]["diff_spans"] if changed else []
    removed_words = " ".join(s["text"] for s in spans if s["op"] == "removed")
    added_words = " ".join(s["text"] for s in spans if s["op"] == "added")
    check("the diff spans show 'kingdom' removed", "kingdom" in removed_words, spans)
    check("the diff spans show 'domain' added", "domain" in added_words, spans)


def test_heading_missing():
    pdf_bytes = good_chapter_pdf()
    result = core.process_chapter_pdf_bytes(pdf_bytes)
    sections = [dict(s) for s in result["sections"] if s["heading"] != "1.2 Taxonomic Categories"]
    verify = v.verify_text_coverage(pdf_bytes, sections)
    check("a heading absent from storage fails verification", verify["status"] == "FAIL", verify)
    missing = by_type(verify["differences"], "heading_missing")
    check("exactly one heading_missing difference", len(missing) == 1, verify["differences"])
    check("it names the missing heading", missing and "Taxonomic Categories" in missing[0]["pdf_excerpt"], missing)
    check("no stray paragraph-level diffs for the dropped heading's own text",
          not by_type(verify["differences"], "missing"), verify["differences"])


def test_heading_extra():
    pdf_bytes = good_chapter_pdf()
    result = core.process_chapter_pdf_bytes(pdf_bytes)
    sections = [dict(s) for s in result["sections"]]
    sections.append({
        "heading": "1.4 Extra Section Nobody Added",
        "text": "This whole section was added only to storage and was never present in the uploaded PDF chapter.",
    })
    verify = v.verify_text_coverage(pdf_bytes, sections)
    check("a heading absent from the PDF fails verification", verify["status"] == "FAIL", verify)
    extra = by_type(verify["differences"], "heading_extra")
    check("exactly one heading_extra difference", len(extra) == 1, verify["differences"])
    check("it names the extra heading", extra and "Extra Section" in extra[0]["stored_excerpt"], extra)
    check("the heading_extra difference has no PDF page", extra and extra[0]["page"] is None, extra)


def test_heading_reordered():
    pdf_bytes = good_chapter_pdf()
    result = core.process_chapter_pdf_bytes(pdf_bytes)
    sections = [dict(s) for s in result["sections"]]
    sections[1], sections[2] = sections[2], sections[1]  # swap 1.2 and 1.3 in storage
    verify = v.verify_text_coverage(pdf_bytes, sections)
    check("swapped section order fails verification", verify["status"] == "FAIL", verify)
    reordered = by_type(verify["differences"], "heading_reordered")
    check("at least one heading_reordered difference", len(reordered) >= 1, verify["differences"])


def test_word_count_mismatch():
    pdf_bytes = good_chapter_pdf()
    result = core.process_chapter_pdf_bytes(pdf_bytes)
    sections = [dict(s) for s in result["sections"]]
    words = sections[1]["text"].split()
    sections[1]["text"] = " ".join(words[: len(words) // 2])  # half the section's content is gone
    verify = v.verify_text_coverage(pdf_bytes, sections)
    check("a half-length section fails verification", verify["status"] == "FAIL", verify)
    mismatch = by_type(verify["differences"], "word_count_mismatch")
    check("exactly one word_count_mismatch difference", len(mismatch) == 1, verify["differences"])
    check("it names the affected section", mismatch and mismatch[0]["section"] == "1.2 Taxonomic Categories", mismatch)


# ---------------- Harmless differences ----------------

def test_harmless_whitespace():
    pdf_bytes = one_section_pdf(
        "The entire living world is wonderfully diverse across every single ecosystem that exists on this earth today."
    )
    result = core.process_chapter_pdf_bytes(pdf_bytes)
    sections = [dict(s) for s in result["sections"]]
    sections[0]["text"] = sections[0]["text"].replace("living world", "livingworld", 1)
    verify = v.verify_text_coverage(pdf_bytes, sections)
    check("a missing-space-only difference passes verification", verify["status"] == "PASS", verify)
    ws = by_type(verify["differences"], "extra or missing whitespace")
    check("exactly one whitespace difference, classified harmless", len(ws) == 1 and ws[0]["category"] == "harmless", verify["differences"])


def test_harmless_hyphenation():
    pdf_bytes = one_section_pdf(
        "The complete envi- ronment around every living organism changes constantly across each passing season of the year."
    )
    result = core.process_chapter_pdf_bytes(pdf_bytes)
    sections = [dict(s) for s in result["sections"]]
    sections[0]["text"] = sections[0]["text"].replace("envi- ronment", "environment", 1)
    verify = v.verify_text_coverage(pdf_bytes, sections)
    check("a line-break hyphenation difference passes verification", verify["status"] == "PASS", verify)
    hy = by_type(verify["differences"], "hyphenation or line-break join")
    check("exactly one hyphenation difference, classified harmless", len(hy) == 1 and hy[0]["category"] == "harmless", verify["differences"])


def test_harmless_ligature():
    pdf_bytes = one_section_pdf(
        "Scientific classification of every organism depends on careful comparison of all its structural features."
    )
    result = core.process_chapter_pdf_bytes(pdf_bytes)
    sections = [dict(s) for s in result["sections"]]
    sections[0]["text"] = sections[0]["text"].replace("classification", "classiﬁcation", 1)
    verify = v.verify_text_coverage(pdf_bytes, sections)
    check("a ligature-only difference passes verification", verify["status"] == "PASS", verify)
    lig = by_type(verify["differences"], "ligature or soft-hyphen difference")
    check("exactly one ligature difference, classified harmless", len(lig) == 1 and lig[0]["category"] == "harmless", verify["differences"])


def test_harmless_dash_quote():
    pdf_bytes = one_section_pdf(
        "The relationship between non-living and living components keeps every ecosystem balanced and fully functional over time."
    )
    result = core.process_chapter_pdf_bytes(pdf_bytes)
    sections = [dict(s) for s in result["sections"]]
    sections[0]["text"] = sections[0]["text"].replace("non-living", "non‑living", 1)
    verify = v.verify_text_coverage(pdf_bytes, sections)
    check("a dash-style-only difference passes verification", verify["status"] == "PASS", verify)
    dq = by_type(verify["differences"], "dash or quote style")
    check("exactly one dash/quote difference, classified harmless", len(dq) == 1 and dq[0]["category"] == "harmless", verify["differences"])


def test_harmless_running_header_and_page_number():
    pdf_bytes = make_pdf([
        "CHAPTER 1",
        "THE LIVING WORLD",
        "1.1 Diversity In The Living World",
        PARA_1,
        "THE LIVING WORLD 5",  # running header (ncert_clean_manual.RUNNING_HEADER_RES)
        "6",                     # lone page number
        "1.2 Taxonomic Categories",
        PARA_2,
    ])
    result = core.process_chapter_pdf_bytes(pdf_bytes)
    verify = v.verify_text_coverage(pdf_bytes, result["sections"])
    check("running headers/page numbers alone still pass verification", verify["status"] == "PASS", verify)
    rh = by_type(verify["differences"], "running header or page number")
    check("at least one running header/page number difference", len(rh) >= 1, verify["differences"])
    check("every one of them is classified harmless", all(d["category"] == "harmless" for d in rh), rh)


# ---------------- Cross-cutting checks ----------------

def test_every_difference_is_documented():
    pdf_bytes = good_chapter_pdf()
    result = core.process_chapter_pdf_bytes(pdf_bytes)
    sections = [dict(s) for s in result["sections"]]
    sections[0]["text"] = sections[0]["text"].replace("living world", "livingworld", 1)  # harmless
    sections[1]["text"] = sections[1]["text"].replace("kingdom", "domain")  # real: changed
    sections[2]["text"] = ""  # real: missing
    verify = v.verify_text_coverage(pdf_bytes, sections)
    for d in verify["differences"]:
        if d["category"] == "harmless":
            check(f"harmless type {d['type']!r} is documented", d["type"] in v.HARMLESS_TYPES, d)
        else:
            check(f"real type {d['type']!r} is documented", d["type"] in v.REAL_TYPES, d)
    check("counts.real + counts.harmless equals the number of differences",
          verify["counts"]["real"] + verify["counts"]["harmless"] == len(verify["differences"]), verify["counts"])
    check("counts.by_type adds up to the same total",
          sum(verify["counts"]["by_type"].values()) == len(verify["differences"]), verify["counts"])


def test_excerpts_capped_at_200_chars():
    pdf_bytes = good_chapter_pdf()
    result = core.process_chapter_pdf_bytes(pdf_bytes)
    sections = [dict(s) for s in result["sections"]]
    sections[0]["text"] = ("Z " * 300).strip()  # 600 chars, unrelated to the PDF -> missing + extra
    verify = v.verify_text_coverage(pdf_bytes, sections)
    check("verification fails as expected", verify["status"] == "FAIL", verify)
    for d in verify["differences"]:
        check(f"pdf_excerpt for {d['type']} is capped at 200 chars", not d.get("pdf_excerpt") or len(d["pdf_excerpt"]) <= v.MAX_EXCERPT_CHARS, d)
        check(f"stored_excerpt for {d['type']} is capped at 200 chars", not d.get("stored_excerpt") or len(d["stored_excerpt"]) <= v.MAX_EXCERPT_CHARS, d)


def test_summary_is_a_plain_short_string():
    pdf_bytes = good_chapter_pdf()
    result = core.process_chapter_pdf_bytes(pdf_bytes)
    verify = v.verify_text_coverage(pdf_bytes, result["sections"])
    check("summary is a plain string", isinstance(verify["summary"], str), verify["summary"])
    check("summary is short", len(verify["summary"]) <= 500, verify["summary"])
    check("summary starts with the status", verify["summary"].startswith(verify["status"]), verify["summary"])


def test_diff_key_is_a_hash_not_the_text():
    key1 = v._diff_key("changed", "some pdf text here", "some stored text here")
    key2 = v._diff_key("changed", "some pdf text here", "some stored text here")
    key3 = v._diff_key("changed", "some pdf text here", "a completely different stored text")
    check("diff_key is deterministic for the same inputs", key1 == key2, (key1, key2))
    check("diff_key changes when the excerpt changes", key1 != key3, (key1, key3))
    check("diff_key never contains the excerpt text itself", "pdf text" not in key1 and "stored text" not in key1, key1)
    check("diff_key looks like a hex digest", len(key1) == 24 and all(c in "0123456789abcdef" for c in key1), key1)


def main():
    test_pass_on_identical_text()
    test_coverage_calculation()
    test_missing_paragraph()
    test_extra_paragraph()
    test_changed_word()
    test_heading_missing()
    test_heading_extra()
    test_heading_reordered()
    test_word_count_mismatch()
    test_harmless_whitespace()
    test_harmless_hyphenation()
    test_harmless_ligature()
    test_harmless_dash_quote()
    test_harmless_running_header_and_page_number()
    test_every_difference_is_documented()
    test_excerpts_capped_at_200_chars()
    test_summary_is_a_plain_short_string()
    test_diff_key_is_a_hash_not_the_text()

    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
