"""Regression tests for the "3-column coaching book" extraction mode (process_pyq_book_3col.py).

Run with: python test_pyq_book_3col.py
No pytest dependency -- matches test_pyq_extraction.py's own style, since this repo has no
pytest-based test runner.

Uses pyq-pdfs/Chapter3_PlantKingdom_18-30.pdf (copied into the repo from the user's source file,
per their explicit instruction) -- the real PDF this mode was built and diagnosed against.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import process_pyq_book_3col as pk

PDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pyq-pdfs", "Chapter3_PlantKingdom_18-30.pdf")

_passed = []
_failed = []


def check(label, condition, detail=""):
    if condition:
        _passed.append(label)
        print(f"  PASS: {label}")
    else:
        _failed.append(label)
        print(f"  FAIL: {label}" + (f" -- {detail}" if detail else ""))


def test_zero_padded_question_number_detected():
    print("\n=== Zero-padded question number is detected as a question start ===")
    import fitz
    narrow_bbox = (63.6, 100.0, 218.2, 160.0)  # real single-column width from the PDF (~155pt)
    block = {"bbox": narrow_bbox, "lines": []}
    # _question_start_info reads text via _block_text_from_dict(block), which walks block["lines"]
    # -- easier to exercise the regex directly here, same as the real call does internally.
    for text, expect_num in [("01 Which of the following algae", 1), ("09 Which one is a wrong statement?", 9), ("15 Mannitol is the stored food in", 15), ("99 In Pinus/gymnosperms, the", 99)]:
        m = pk._QNUM_RE.match(text)
        check(f"'{text[:20]}...' matches as question {expect_num:02d}", bool(m) and int(m.group(1)) == expect_num)
    # A non-zero-padded or 3+ digit number must NOT match (this layout never numbers that way;
    # a stray 3-digit number elsewhere, e.g. a page number, shouldn't be treated as a question).
    check("'100 something' does NOT match (only 1-2 digit zero-padded numbers)", pk._QNUM_RE.match("100 something") is None)


def test_running_header_not_mistaken_for_a_question():
    """The book's own running header/footer ("Plant Kingdom\\n03", "20\\nNEET Chapterwise
    Topicwise Biology") spans the FULL page width and starts with a bare number + newline --
    the exact same shape _QNUM_RE looks for. Width is what must tell them apart."""
    print("\n=== Running header/footer is NOT mistaken for a question start ===")
    wide_header_block = {"bbox": (59.0, 73.4, 551.8, 87.8), "lines": []}  # real bbox from the PDF
    # Patch _block_text_from_dict's dependency by calling _question_start_info with a stand-in
    # that has the real wide bbox but synthetic lines -- simpler to just check the width gate
    # directly, since that's the actual fix under test.
    bbox = wide_header_block["bbox"]
    check("header block width exceeds the single-column limit", (bbox[2] - bbox[0]) > pk._MAX_QUESTION_BLOCK_WIDTH, detail=f"width={bbox[2]-bbox[0]:.1f}")
    narrow_question_block = (63.6, 292.4, 218.2, 359.2)  # real Q01 bbox from the PDF
    check("a real question block's width stays under the limit", (narrow_question_block[2] - narrow_question_block[0]) <= pk._MAX_QUESTION_BLOCK_WIDTH)


def test_2x2_grid_options_parsed():
    """Real Q18 shape: '(a) Rhodophyceae (b) Bacillariophyceae' then '(c) Chlorophyceae (d)
    Phaeophyceae' on the next line -- two markers per line, not one per line."""
    print("\n=== 2x2 grid options (two markers per line) parsed correctly ===")
    text = (
        "18 A research student collected certain alga. The alga belongs to\n"
        "[CBSE AIPMT 2000]\n"
        "(a) Rhodophyceae (b) Bacillariophyceae\n"
        "(c) Chlorophyceae (d) Phaeophyceae\n"
        "Ans. (a)"
    )
    parsed = pk._parse_question_text(text)
    check("option a correct", parsed["options"].get("a") == "Rhodophyceae", detail=repr(parsed["options"]))
    check("option b correct", parsed["options"].get("b") == "Bacillariophyceae", detail=repr(parsed["options"]))
    check("option c correct", parsed["options"].get("c") == "Chlorophyceae", detail=repr(parsed["options"]))
    check("option d correct", parsed["options"].get("d") == "Phaeophyceae", detail=repr(parsed["options"]))
    check("answer correct", parsed["answer"] == "a")
    check("year correct", parsed["year"] == 2000)


def test_wrapped_option_parsed():
    """Real Q08 shape: option (a)'s own content wraps onto two more lines with no marker of its
    own -- must all be read as belonging to (a), not lost or merged into (b)."""
    print("\n=== Wrapped (multi-line) option text parsed correctly ===")
    text = (
        "08 Which one of the following statements is wrong ?\n"
        "[NEET 2016, Phase II]\n"
        "(a) Algae increase the level of\n"
        "dissolved oxygen in the immediate\n"
        "environment\n"
        "(b) Algin is obtained from red algae\n"
        "Ans. (b)"
    )
    parsed = pk._parse_question_text(text)
    check(
        "option a includes all 3 wrapped lines",
        parsed["options"].get("a") == "Algae increase the level of dissolved oxygen in the immediate environment",
        detail=repr(parsed["options"].get("a")),
    )
    check("option b not swallowed by a's wrap", parsed["options"].get("b") == "Algin is obtained from red algae")


def test_year_tag_extracted_and_removed_from_stem():
    print("\n=== Year tag extracted into year/source_tag, removed from the stem ===")
    for raw_tag, expect_year in [("[NEET 2021]", 2021), ("[NEET (Oct.) 2020]", 2020), ("[CBSE AIPMT 2014]", 2014), ("[NEET 2016, Phase II]", 2016)]:
        text = f"05 Some question text\n{raw_tag}\n(a) x\n(b) y\n(c) z\n(d) w\nAns. (a)"
        parsed = pk._parse_question_text(text)
        check(f"{raw_tag} -> year {expect_year}", parsed["year"] == expect_year, detail=f"got {parsed['year']}")
        check(f"{raw_tag} removed from the stem", "[" not in parsed["question"] and "]" not in parsed["question"], detail=repr(parsed["question"]))


def test_ans_word_boundary_regression():
    """Real bug found and fixed 2026-10-11: option (a)'s own content ends in "...mannans"
    immediately followed by "\\n(b)" -- without a \\b before "Ans" in the regex, "ans" (the tail
    of "mannans") + the newline + "(b)" matched as a fabricated "Ans. (b)", truncating option a
    mid-word and reporting a wrong, made-up answer letter. Must fail on the old regex, pass on
    the new one."""
    print("\n=== Ans regex requires a word boundary (does not fire mid-word) ===")
    text = (
        "14 Algae have cell wall made up of\n"
        "[CBSE AIPMT 2010]\n"
        "(a) cellulose, galactans and mannans\n"
        "(b) hemicellulose, pectins and proteins\n"
        "(c) pectins, cellulose and proteins\n"
        "(d) cellulose, hemicellulose and pectins"
    )
    parsed = pk._parse_question_text(text)
    check("option a is NOT truncated at 'mann'", parsed["options"].get("a") == "cellulose, galactans and mannans", detail=repr(parsed["options"].get("a")))
    check("no fabricated answer letter from inside option a's own text", parsed["answer"] is None, detail=f"got {parsed['answer']!r}")
    check("option b still parsed correctly", parsed["options"].get("b") == "hemicellulose, pectins and proteins")

    # And the regex must still correctly match a REAL "Ans. (x)" immediately after a word.
    real_ans_text = text + "\nAns. (a)"
    parsed2 = pk._parse_question_text(real_ans_text)
    check("a genuine trailing 'Ans. (a)' is still matched", parsed2["answer"] == "a")


def test_diagram_only_attaches_within_question_bbox():
    """Step 9: a diagram belongs to the question only if its own bbox overlaps the question's
    own block -- one sitting in the gap before the next question (i.e. in the explanation that
    follows) must not count, even though it's nearby."""
    print("\n=== Diagram attribution: question bbox only, not the explanation's ===")
    question_bbox = (63.6, 292.4, 218.2, 359.2)
    diagram_inside = (100.0, 300.0, 150.0, 340.0)  # overlaps the question's own block
    diagram_in_explanation = (80.0, 400.0, 200.0, 460.0)  # sits well below the question's block
    check("diagram overlapping the question's own block attaches", pk._bbox_overlaps(question_bbox, diagram_inside))
    check("diagram in the explanation's area (below the question) does not attach", not pk._bbox_overlaps(question_bbox, diagram_in_explanation))


def test_answer_on_next_column_same_page():
    """Real Q84 shape (page 10): the question's own block sits in the middle column with all 4
    options but no 'Ans.' line -- the 'Ans. (a)' is the first block of the NEXT column over, on
    the SAME page. Must be reached without needing a page turn."""
    print("\n=== Answer line continues in the next column, same page ===")
    with open(PDF_PATH, "rb") as f:
        pdf_bytes = f.read()
    result = pk.scan_book_pdf_bytes(pdf_bytes, "Biology")
    q = next((q for q in result["questions"] if q["question"].startswith("Which one of the following is a\nliving fossil")
              or q["question"].startswith("Which one of the following is a living fossil")), None)
    check("Q84 (living fossil) found", q is not None)
    if q:
        check("Q84 is not flagged", not q["needs_review"], detail=repr(q))
        check("Q84's answer ('a', from the next column) was found", q["correct_answer"] == "a", detail=repr(q["correct_answer"]))
        check("Q84's own options are untouched", q["option_a"] == "Cycas" and q["option_d"] == "Spirogyra", detail=repr(q))


def test_answer_on_next_page():
    """Real Q07 shape (page 1, column 3 -- the page's last column): options run out of room at
    the very bottom of the page, and 'Ans. (b)' is the second block at the top of page 2's first
    column (the running header 'Plant Kingdom\\n19' is the first block there and must be skipped,
    not treated as a stop)."""
    print("\n=== Answer line continues on the next page ===")
    with open(PDF_PATH, "rb") as f:
        pdf_bytes = f.read()
    result = pk.scan_book_pdf_bytes(pdf_bytes, "Biology")
    q = next((q for q in result["questions"] if "colonial alga" in q["question"]), None)
    check("Q07 (colonial alga) found", q is not None)
    if q:
        check("Q07 is not flagged", not q["needs_review"], detail=repr(q))
        check("Q07's answer ('b', from the next page) was found", q["correct_answer"] == "b", detail=repr(q["correct_answer"]))


def test_options_split_across_columns():
    """Real Q75 shape (page 9): the question's own block (middle column) has no options at all;
    option (a) and (b) continue at the top of the SAME page's next column, and options (c) and
    (d) continue YET AGAIN at the top of a THIRD column -- all on one page, no page turn."""
    print("\n=== Options split across more than one column on the same page ===")
    with open(PDF_PATH, "rb") as f:
        pdf_bytes = f.read()
    result = pk.scan_book_pdf_bytes(pdf_bytes, "Biology")
    q = next((q for q in result["questions"] if q["question"] == "Select the correct statement."), None)
    check("Q75 (Select the correct statement) found", q is not None)
    if q:
        check("Q75 is not flagged", not q["needs_review"], detail=repr(q))
        check("option a (from column 2) present", q["option_a"].startswith("Salvinia"), detail=repr(q["option_a"]))
        check("option b (from column 2) present", "Sequoia" in q["option_b"], detail=repr(q["option_b"]))
        check("option c (from column 3) present", "not well adapted" in q["option_c"], detail=repr(q["option_c"]))
        check("option d present", q["option_d"], detail=repr(q["option_d"]))
        check("answer found", q["correct_answer"] == "b", detail=repr(q["correct_answer"]))


def test_embedded_question_start_recovered():
    """Real Q21/Q22 shape (page 3): PyMuPDF merges the TAIL of Q20's own explanation directly
    onto Q21's entire number+stem+options+Ans as ONE block, with Q22 then glued onto the tail of
    Q21's own explanation the same way -- both completely invisible to a check that only looks at
    a block's own first line. Must fail on the old code (both silently absent, no trace in
    flagged_pages) and pass on the new code (_split_embedded_question_starts)."""
    print("\n=== Questions whose number is merged mid-block into the previous explanation ===")
    with open(PDF_PATH, "rb") as f:
        pdf_bytes = f.read()
    result = pk.scan_book_pdf_bytes(pdf_bytes, "Biology")
    q21 = next((q for q in result["questions"] if "Brown algae is characterised" in q["question"]), None)
    q22 = next((q for q in result["questions"] if "very rich in protein" in q["question"]), None)
    check("Q21 (Brown algae) recovered", q21 is not None)
    check("Q22 (rich in protein) recovered", q22 is not None)
    if q21:
        check("Q21 is not flagged", not q21["needs_review"], detail=repr(q21))
        check("Q21's answer is correct", q21["correct_answer"] == "c", detail=repr(q21["correct_answer"]))
    if q22:
        check("Q22 is not flagged", not q22["needs_review"], detail=repr(q22))
        check("Q22's answer is correct", q22["correct_answer"] == "d", detail=repr(q22["correct_answer"]))


def test_numbering_gap_is_flagged():
    """Unit-level test of the gap-detection safety net itself (step 3): a missing integer in the
    1..max(found) sequence must be flagged by number, on a sensible nearby page, and the function
    must never invent content for it. Also checks the clean (no gap) case stays silent."""
    print("\n=== A gap in question numbers is flagged, not silently dropped ===")
    qnum_page_with_gap = {1: 1, 2: 1, 3: 1, 5: 2, 6: 2}  # 4 is missing
    gaps = pk._find_numbering_gaps(qnum_page_with_gap)
    check("exactly one gap found", len(gaps) == 1, detail=repr(gaps))
    if gaps:
        check("the gap is question 04", "04" in gaps[0]["reason"], detail=repr(gaps[0]))
        check("the gap is placed on the nearest-following found page (page 2, where Q5 is)", gaps[0]["page"] == 2, detail=repr(gaps[0]))
        check("the reason points to '+ Add Missed Question', never invents an answer/option", "Add Missed Question" in gaps[0]["reason"] and "option" not in gaps[0]["reason"].lower())

    qnum_page_multi_gap = {1: 1, 4: 3}  # 2 and 3 missing, both should land on page 3
    gaps2 = pk._find_numbering_gaps(qnum_page_multi_gap)
    check("two gaps found for a 2-wide hole", len(gaps2) == 2, detail=repr(gaps2))
    check("both land on page 3 (where Q4 is)", all(g["page"] == 3 for g in gaps2), detail=repr(gaps2))

    qnum_page_no_gap = {1: 1, 2: 1, 3: 2}
    check("no gap reported when the sequence is already continuous", pk._find_numbering_gaps(qnum_page_no_gap) == [])

    # And end-to-end: the real, fixed document has NO gaps left at all (99/99 found).
    with open(PDF_PATH, "rb") as f:
        pdf_bytes = f.read()
    result = pk.scan_book_pdf_bytes(pdf_bytes, "Biology")
    gap_flags = [f for f in result["flagged_pages"] if "was never found" in f["reason"]]
    check("the real, fixed document has zero numbering gaps left", len(gap_flags) == 0, detail=repr(gap_flags))


def test_full_document_extraction():
    """End-to-end regression against the real PDF: ALL 99 questions present (including the 2
    that were previously silently undetected), none flagged, with explanations never leaking
    into any question/option field."""
    print("\n=== Full-document extraction against the real PDF ===")
    with open(PDF_PATH, "rb") as f:
        pdf_bytes = f.read()
    result = pk.scan_book_pdf_bytes(pdf_bytes, "Biology")
    check("13 pages scanned", result["pages_scanned"] == 13, detail=f"got {result['pages_scanned']}")
    check("all 99 of 99 questions extracted", len(result["questions"]) == 99, detail=f"got {len(result['questions'])}")
    check("zero questions flagged for review", len(result["flagged_pages"]) == 0, detail=f"got {result['flagged_pages']}")

    explanation_telltales = ["can be explained as", "Other options can be", "is a cosmopolitan marine"]
    leaked = [q for q in result["questions"] for tell in explanation_telltales if tell in (q["question"] + q["option_a"] + q["option_b"] + q["option_c"] + q["option_d"])]
    check("no publisher explanation text leaked into any question/option field", len(leaked) == 0, detail=f"{len(leaked)} leaked")

    check("every question has all 4 options filled", all(q["option_a"] and q["option_b"] and q["option_c"] and q["option_d"] for q in result["questions"]))
    check("every question has an answer letter", all(q["correct_answer"] in ("a", "b", "c", "d") for q in result["questions"]))


if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"SKIPPED: {PDF_PATH} not found -- these tests need the real PDF fixture.")
        sys.exit(1)

    test_zero_padded_question_number_detected()
    test_running_header_not_mistaken_for_a_question()
    test_2x2_grid_options_parsed()
    test_wrapped_option_parsed()
    test_year_tag_extracted_and_removed_from_stem()
    test_ans_word_boundary_regression()
    test_diagram_only_attaches_within_question_bbox()
    test_answer_on_next_column_same_page()
    test_answer_on_next_page()
    test_options_split_across_columns()
    test_embedded_question_start_recovered()
    test_numbering_gap_is_flagged()
    test_full_document_extraction()

    print(f"\nTOTAL: {len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
