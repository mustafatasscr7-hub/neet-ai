"""Regression tests for the "Scanned pages (image-only)" extraction mode (process_pyq_scanned.py).

Run with: python test_pyq_scanned.py
No pytest dependency -- same plain pass/fail harness as test_pyq_book_3col.py.

No test here calls Gemini. Everything structural in this mode (column/band reading order,
continuation joining, numbering gaps, answer-key matching, caching, the page cap) is plain Python,
tested with synthetic items and a fake extract function. The DISHA end-to-end check replays the
REAL per-page Gemini responses captured in test-fixtures/scanned/disha-scanned-pages-real.json.
"""
import io
import json
import os
import re
import sys
import tempfile

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fitz
import process_pyq_scanned as ps

DISHA_FIXTURE = os.path.join(HERE, "test-fixtures", "scanned", "disha-scanned-pages-real.json")
DISHA_PDF = os.path.join(HERE, "pyq-pdfs", "DISHA_36_YEARS_BIOLOGY_11thTEST.pdf")

_passed = []
_failed = []


def check(label, condition, detail=""):
    if condition:
        _passed.append(label)
        print(f"  PASS: {label}")
    else:
        _failed.append(label)
        print(f"  FAIL: {label}" + (f" -- {detail}" if detail else ""))


def q(number, box, options="abcd", stem=None, **extra):
    return {"kind": "question", "number": number, "question": stem or f"Stem {number}",
            "options": {l: f"Q{number} option {l}" for l in options}, "box_2d": box, **extra}


def cont(box, text="", options=None):
    return {"kind": "continuation", "text": text, "options": options or {}, "box_2d": box}


def labels(items):
    return [it.get("number") if it["kind"] == "question" else it["kind"] for it in items]


# ---------------------------------------------------------------- column ordering

def test_two_column_reading_order():
    print("\n=== Two columns: whole left column top-to-bottom, then right ===")
    # Deliberately shuffled, the way a model might list them.
    items = [q(4, [100, 530, 200, 890]), q(2, [300, 95, 400, 460]), q(1, [100, 95, 290, 460]),
             q(5, [210, 530, 300, 890]), q(3, [410, 95, 500, 460])]
    ordered, geom = ps.order_page_items(items)
    check("geometry was used", geom)
    check("order is 1,2,3 (left) then 4,5 (right)", labels(ordered) == [1, 2, 3, 4, 5], detail=str(labels(ordered)))


def test_full_width_block_splits_page_into_bands():
    """DISHA page 4's shape: a 2-column question region, a full-width answer-key table, then a
    2-column solutions region. The right column ABOVE the key must be read before the key, and
    nothing below the key may be read before it."""
    print("\n=== Full-width block splits the page into bands (DISHA page 4 shape) ===")
    items = [
        {"kind": "answer_key", "entries": {}, "box_2d": [270, 130, 350, 970]},
        q(40, [400, 100, 600, 460]),                     # below the key, left
        cont([90, 560, 200, 975], text="(b) x (c) y"),   # above the key, right
        q(38, [180, 100, 260, 460]),                     # above the key, left
        {"kind": "solutions_start", "box_2d": [360, 345, 395, 755]},
        q(41, [420, 560, 600, 975]),                     # below the key, right
    ]
    ordered, _ = ps.order_page_items(items)
    check("band order: Q38, its right-column continuation, key, solutions heading, Q40, Q41",
          labels(ordered) == [38, "continuation", "answer_key", "solutions_start", 40, 41], detail=str(labels(ordered)))


def test_continuation_sorts_first_in_its_column_despite_bad_y():
    """Live DISHA page 4: Q38's box came back starting ABOVE Q37's continuation grid (y=89 vs 93)
    and reaching x=573. A plain y-sort glued Q37's options onto Q38."""
    print("\n=== Continuation sorts first in its column even when the model's y is off ===")
    items = [q(38, [89, 126, 235, 573]), cont([93, 169, 172, 389], options={"a": "(i) (iv) (iii) (ii)"})]
    ordered, _ = ps.order_page_items(items)
    check("box reaching x=573 is still a LEFT-column item, not full width", ps.item_column([89, 126, 235, 573]) == "left")
    check("continuation comes before Q38", labels(ordered) == ["continuation", 38], detail=str(labels(ordered)))


def test_column_classification():
    print("\n=== item_column classification ===")
    check("left column box", ps.item_column([0, 95, 100, 462]) == "left")
    check("right column box", ps.item_column([0, 530, 100, 976]) == "right")
    check("answer-key table spanning the page is full", ps.item_column([268, 128, 348, 970]) == "full")
    check("centred section heading is full", ps.item_column([358, 345, 396, 755]) == "full")


def test_nested_box_normalized_and_missing_box_falls_back():
    print("\n=== Nested [[...]] box is accepted; a missing box keeps model order and reports it ===")
    items = [q(2, [[300, 95, 400, 460]]), q(1, [100, 95, 200, 460])]
    ordered, geom = ps.order_page_items(items)
    check("nested box unwrapped and sorted geometrically", geom and labels(ordered) == [1, 2], detail=str(labels(ordered)))
    items = [q(2, [300, 95, 400, 460]), q(1, None)]
    ordered, geom = ps.order_page_items(items)
    check("missing box -> model order kept, geometry flagged as unused", not geom and labels(ordered) == [2, 1])


# ---------------------------------------------------------------- continuation joining

def _merge(pages):
    segments, page_flags = ps.merge_document_items([(p, ps.order_page_items(items)[0]) for p, items in pages])
    return segments, page_flags


def test_continuation_across_page_break():
    print("\n=== Question continuing across a PAGE break is joined, not split ===")
    pages = [
        (1, [q(8, [100, 530, 300, 890]), q(9, [800, 530, 880, 890], options="")]),
        (2, [cont([100, 150, 230, 540], text="(a) Kingdom (b) Order (c) Phylum (d) Class"), q(10, [230, 120, 290, 540])]),
    ]
    segments, page_flags = _merge(pages)
    qs = segments[0]["questions"]
    q9 = next(x for x in qs if x["number"] == 9)
    check("still exactly 3 questions (continuation did not become its own question)", [x["number"] for x in qs] == [8, 9, 10])
    check("Q9 got all four options from the next page", q9["options"] == {"a": "Kingdom", "b": "Order", "c": "Phylum", "d": "Class"}, detail=str(q9["options"]))
    check("Q9 stays attributed to the page it started on", q9["page"] == 1)
    check("no orphan flags", page_flags == [])


def test_continuation_across_column_break_with_markers_in_text():
    """Live DISHA page 2, Q15: the model put "(c) ... (d) ..." in the continuation's "text"
    instead of "options"; they must land in (c)/(d), not get glued onto (b)."""
    print("\n=== Question continuing across a COLUMN break; option markers inside 'text' ===")
    pages = [(2, [q(15, [770, 126, 890, 541], options="ab"),
                  cont([107, 603, 180, 976], text="(c) Felis tigris - The Indian tiger.\n(d) E.coli - a bacterium."),
                  q(16, [180, 561, 286, 976])])]
    segments, _ = _merge(pages)
    q15 = segments[0]["questions"][0]
    check("option (b) untouched", q15["options"]["b"] == "Q15 option b", detail=q15["options"]["b"])
    check("option (c) filled from the next column", q15["options"].get("c") == "Felis tigris - The Indian tiger.")
    check("option (d) filled from the next column", q15["options"].get("d") == "E.coli - a bacterium.")


def test_unmarked_leading_text_continues_stem_or_last_option():
    print("\n=== Unmarked leading text continues the stem, or the last started option ===")
    pages = [(1, [q(1, [100, 95, 200, 460], options="", stem="The living organisms can be"),
                  cont([100, 530, 200, 890], text="distinguished by their (a) w (b) x (c) y (d) z")])]
    segments, _ = _merge(pages)
    q1 = segments[0]["questions"][0]
    check("lead text appended to the stem", q1["question"] == "The living organisms can be distinguished by their", detail=q1["question"])
    check("then the options", q1["options"] == {"a": "w", "b": "x", "c": "y", "d": "z"})

    pages = [(4, [q(38, [180, 100, 260, 460], options="a"), cont([90, 560, 200, 975], text="for reference. (b) x (c) y (d) z")])]
    segments, _ = _merge(pages)
    q38 = segments[0]["questions"][0]
    check("lead text appended to option (a), the last one started", q38["options"]["a"] == "Q38 option a for reference.", detail=q38["options"]["a"])


def test_duplicate_option_letter_is_joined_and_flagged():
    print("\n=== An option letter seen on both sides of a break is joined AND flagged ===")
    pages = [(1, [q(5, [100, 95, 200, 460], options="ab"), cont([100, 530, 200, 890], options={"b": "more", "c": "y", "d": "z"})])]
    segments, _ = _merge(pages)
    q5 = segments[0]["questions"][0]
    check("both halves kept", q5["options"]["b"] == "Q5 option b more")
    check("flagged", any("option (b)" in i for i in q5["issues"]), detail=str(q5["issues"]))


def test_orphan_continuation_is_flagged_not_dropped():
    print("\n=== A continuation with nothing before it (scan starts mid-question) is flagged ===")
    segments, page_flags = _merge([(1, [cont([100, 95, 200, 460], text="(c) tail one (d) tail two"), q(12, [210, 95, 300, 460])])])
    check("the next real question is still extracted", [x["number"] for x in segments[0]["questions"]] == [12])
    check("orphan text reported on its page", len(page_flags) == 1 and page_flags[0]["page"] == 1 and "tail one" in page_flags[0]["reason"], detail=str(page_flags))


def test_answer_key_closes_open_question():
    print("\n=== Nothing after an answer key / solutions heading is joined onto an earlier question ===")
    pages = [(4, [q(38, [100, 95, 200, 460], options="abcd"),
                  {"kind": "answer_key", "entries": {"38": "b"}, "box_2d": [270, 130, 350, 970]},
                  cont([400, 95, 500, 460], text="(b) Basal metabolic rate is inversely proportional")])]
    segments, page_flags = _merge(pages)
    q38 = segments[0]["questions"][0]
    check("Q38 options untouched by text below the key", q38["options"]["b"] == "Q38 option b")
    check("the stray text is surfaced as an orphan flag", len(page_flags) == 1)


def test_solutions_section_is_not_extracted():
    print("\n=== Numbered entries after 'solutions_start' are never treated as questions ===")
    pages = [(4, [q(1, [100, 95, 200, 460]),
                  {"kind": "solutions_start", "box_2d": [360, 345, 395, 755]},
                  q(1, [400, 95, 500, 460], stem="(b) Basal metabolic rate...")])]
    segments, _ = _merge(pages)
    check("only the real Q1", [x["question"] for x in segments[0]["questions"]] == ["Stem 1"])


# ---------------------------------------------------------------- gap detection + flags

def test_find_numbering_issues():
    print("\n=== Numbering gaps and duplicates ===")
    check("no gap in 1..5", ps.find_numbering_issues([1, 2, 3, 4, 5], True) == ([], []))
    check("gap 3 and 6 found", ps.find_numbering_issues([1, 2, 4, 5, 7], True)[0] == [3, 6])
    check("mid-chapter scan starting at Q10 is NOT a gap", ps.find_numbering_issues([10, 11, 12], False) == ([], []))
    check("chapter-start scan starting at Q3 flags Q1, Q2", ps.find_numbering_issues([3, 4], True)[0] == [1, 2])
    check("duplicate number found", ps.find_numbering_issues([1, 2, 2, 3], False)[1] == [2])
    check("unreadable numbers ignored, not crashed on", ps.find_numbering_issues([1, None, 3], False) == ([2], []))


def test_finalize_flags_and_key_matching():
    print("\n=== Flags: <4 options, missing number, gap, missing key; keys only by number ===")
    pages = [
        (1, [{"kind": "chapter_start", "title": "The Living World", "box_2d": [50, 350, 240, 725]},
             q(1, [300, 95, 400, 460]),
             q(2, [410, 95, 500, 460], options="abc"),
             q(None, [510, 95, 600, 460]),
             q(4, [300, 530, 400, 890], year_tag="2003, 04")]),
        (2, [{"kind": "answer_key", "entries": {"1": "(b)", "2": "C", "4": "d", "9": "a", "x": "b", "3": "e"}, "box_2d": [100, 100, 200, 950]}]),
    ]
    segments, _ = _merge(pages)
    questions, flags = ps.finalize_segments(segments)
    by_num = {x["question_number"]: x for x in questions}
    reasons = " | ".join(f["reason"] for f in flags)
    check("every question kept (4 incl. the unnumbered one)", len(questions) == 4)
    check("key letters normalised and attached by number", by_num[1]["correct_answer"] == "b" and by_num[2]["correct_answer"] == "c" and by_num[4]["correct_answer"] == "d")
    check("unnumbered question gets NO key (never guessed)", by_num[None]["correct_answer"] == "")
    check("Q2 flagged for 3 of 4 options", by_num[2]["needs_review"] and "only 3 of 4 options" in by_num[2]["review_reason"])
    check("unnumbered question flagged", by_num[None]["needs_review"] and "no question number" in by_num[None]["review_reason"])
    check("gap Q3 flagged", "Q3 is missing from the numbering sequence" in reasons)
    check("missing key flagged", "No answer key found for 1 question(s)" in reasons, detail=reasons)
    check("key entries for numbers not in the scan are reported, not attached", "not in this scan" in reasons)
    check("year taken from the first 4-digit year in the tag, tag kept verbatim", by_num[4]["year"] == 2003 and by_num[4]["source_tag"] == "2003, 04")


def test_chapter_start_resets_numbering():
    print("\n=== A new chapter banner restarts numbering (no false duplicates/gaps across chapters) ===")
    pages = [(1, [q(1, [100, 95, 200, 460]), q(2, [210, 95, 300, 460])]),
             (2, [{"kind": "chapter_start", "title": "Ch 2", "box_2d": [50, 350, 240, 725]},
                  q(1, [300, 95, 400, 460]), {"kind": "answer_key", "entries": {"1": "a"}, "box_2d": [500, 100, 600, 950]}])]
    segments, _ = _merge(pages)
    questions, flags = ps.finalize_segments(segments)
    check("two segments", len(segments) == 2)
    check("chapter 2's key attaches to chapter 2's Q1 only", [x["correct_answer"] for x in questions] == ["", "", "a"])
    check("no duplicate-number flag", not any("more than once" in f["reason"] for f in flags))


def test_match_grid_options_expanded():
    print("\n=== Match-the-column grid options get their column labels back ===")
    pages = [(3, [q(37, [100, 95, 300, 460], options="", stem="Match Column I with Column II\n(A) Herbarium (i) x\n(B) Key (ii) y\n(C) Museum (iii) z\n(D) Catalogue (iv) w"),
                  cont([100, 530, 200, 890], options={"a": "(i) (iv) (iii) (ii)", "b": "(iii) (ii) (i) (iv)", "c": "(iii) (iv) (i) (ii)", "d": "(ii) (iv) (iii) (i)"})])]
    questions, _ = ps.finalize_segments(_merge(pages)[0])
    check("(a) rewritten", questions[0]["option_a"] == "(A)-(i), (B)-(iv), (C)-(iii), (D)-(ii)", detail=questions[0]["option_a"])
    check("column labels (A)-(D) in the stem were not mistaken for option markers", questions[0]["question"].startswith("Match Column I"))


# ---------------------------------------------------------------- entry point: cache, cap, cost

def _make_pdf(n_pages):
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page()
        page.insert_text((72, 72 + i * 20), f"Synthetic page {i + 1}")
    data = doc.tobytes()
    doc.close()
    return data


def test_cache_cap_and_cost():
    print("\n=== Cache avoids paying twice; page cap needs confirmation; cost reported ===")
    pdf = _make_pdf(3)
    calls = []
    key_to_page = {key: i + 1 for i, (_, key) in enumerate(ps.render_pdf_pages(pdf, "Biology"))}

    def fake_extract(image_bytes, subject):
        page = key_to_page[ps.page_cache_key(image_bytes, subject)]
        calls.append(page)
        return [q(page, [100, 95, 200, 460])], {"input_tokens": 1000, "output_tokens": 500}

    with tempfile.TemporaryDirectory() as cache_dir:
        est = ps.estimate_scanned_pdf_cost(pdf, "Biology", cache_dir=cache_dir)
        check("estimate before any run: 3 uncached pages, non-zero cost", est["uncached_pages"] == 3 and est["estimated_usd"] > 0, detail=str(est))

        r = ps.scan_scanned_pdf_bytes(pdf, "Biology", max_pages=2, cache_dir=cache_dir, extract_fn=fake_extract)
        check("over the cap without confirmation: refused, nothing called", r.get("needs_confirmation") and calls == [], detail=str(r))

        r = ps.scan_scanned_pdf_bytes(pdf, "Biology", max_pages=2, confirm_over_cap=True, cache_dir=cache_dir, extract_fn=fake_extract)
        check("confirmed: all 3 pages called once", sorted(calls) == [1, 2, 3])
        check("cost = 3 x (1000 in + 500 out)", r["cost"]["input_tokens"] == 3000 and r["cost"]["output_tokens"] == 1500 and abs(r["cost"]["usd"] - ps.tokens_to_usd(3000, 1500)) < 1e-9, detail=str(r["cost"]))
        check("3 questions extracted, in page order", [x["question_number"] for x in r["questions"]] == [1, 2, 3])

        r2 = ps.scan_scanned_pdf_bytes(pdf, "Biology", max_pages=2, confirm_over_cap=True, cache_dir=cache_dir, extract_fn=fake_extract)
        check("re-run: zero new calls, zero cost, same questions", len(calls) == 3 and r2["cost"]["usd"] == 0 and r2["cost"]["pages_cached"] == 3 and r2["questions"] == r["questions"])
        check("estimate after the run: everything cached, $0", ps.estimate_scanned_pdf_cost(pdf, "Biology", cache_dir=cache_dir)["estimated_usd"] == 0)

    def failing_extract(image_bytes, subject):
        raise RuntimeError("quota exceeded")

    with tempfile.TemporaryDirectory() as cache_dir:
        r = ps.scan_scanned_pdf_bytes(pdf, "Biology", cache_dir=cache_dir, extract_fn=failing_extract)
        check("a failed page call is flagged per page, not raised", len([f for f in r["flagged_pages"] if "quota exceeded" in f["reason"]]) == 3)
        check("failures are NOT cached (a retry will call again)", not os.listdir(cache_dir))


def test_rate_mirrors_server():
    print("\n=== Cost rate mirrors server.py's GEMINI_FLASH_LITE_RATE ===")
    src = open(os.path.join(HERE, "server.py"), encoding="utf-8").read()
    m = re.search(r'GEMINI_FLASH_LITE_RATE\s*=\s*\{"input":\s*([\d.]+),\s*"output":\s*([\d.]+)\}', src)
    check("server rate found", bool(m))
    if m:
        check("rates equal", (float(m.group(1)), float(m.group(2))) == (ps.GEMINI_RATE_PER_M["input"], ps.GEMINI_RATE_PER_M["output"]))


def test_parse_page_response():
    print("\n=== Model JSON parsing ===")
    items = ps.parse_page_response('```json\n[{"kind": "question", "number": 1, "question": "$\\mu_{0}$ and \\theta"}]\n```')
    check("fenced JSON with raw LaTeX backslashes parses", items and items[0]["question"] == "$\\mu_{0}$ and \\theta", detail=str(items))
    try:
        ps.parse_page_response("not json")
        check("garbage raises (so the page gets flagged, not silently emptied)", False)
    except ValueError:
        check("garbage raises (so the page gets flagged, not silently emptied)", True)


# ---------------------------------------------------------------- real DISHA replay

def test_disha_replay():
    print("\n=== DISHA 4-page PDF: replay of the real Gemini responses ===")
    fx = json.load(open(DISHA_FIXTURE, encoding="utf-8"))
    pages_items = [(p["page"], ps.order_page_items(p["items"])[0]) for p in fx["pages"]]
    segments, page_flags = ps.merge_document_items(pages_items)
    questions, flags = ps.finalize_segments(segments)
    printed_key = "b c a d b d b b d a a b a b b a a c d c c d d b a c b a b c b c a c d c c b".split()
    per_page = {}
    for x in questions:
        per_page[x["source_page"]] = per_page.get(x["source_page"], 0) + 1
    check("38 questions numbered 1..38 in reading order", [x["question_number"] for x in questions] == list(range(1, 39)))
    check("per page 9 / 13 / 15 / 1", per_page == {1: 9, 2: 13, 3: 15, 4: 1}, detail=str(per_page))
    check("every question has all 4 options", all(x[f"option_{l}"] for x in questions for l in "abcd"))
    check("all 38 keys match the printed answer key", [x["correct_answer"] for x in questions] == printed_key)
    by_num = {x["question_number"]: x for x in questions}
    check("Q9 (page 1 -> page 2) got its options from page 2", by_num[9]["option_a"].startswith("Kingdom, Class, Phylum"))
    check("Q15 (left -> right column) got (c)/(d)", by_num[15]["option_c"].startswith("Felis tigris") and by_num[15]["option_d"].startswith("E.coli"))
    check("Q15's (b) has no leaked '(c)'", "(c)" not in by_num[15]["option_b"])
    check("Q37 (page 3 -> page 4) got its grid, labelled", by_num[37]["option_a"] == "(A)-(i), (B)-(iv), (C)-(iii), (D)-(ii)", detail=by_num[37]["option_a"])
    check("Q38's options are clean statements (no grid leaked in)", "(iv)" not in by_num[38]["option_a"])
    check("nothing from Hints & Solutions leaked in", len(questions) == 38 and not page_flags)
    check("NCERT and year tags split", by_num[9]["ncert_tag"] == "Page-10 / N-8" and by_num[9]["year"] == 2022)
    if os.path.exists(DISHA_PDF):
        edges = ps.pdf_cropped_edges(open(DISHA_PDF, "rb").read())
        check("cropped right edge detected on pages 2 and 4 only", edges == [[], ["right"], [], ["right"]], detail=str(edges))
    else:
        print(f"  (skipped edge-crop check: {DISHA_PDF} not present)")


if __name__ == "__main__":
    test_two_column_reading_order()
    test_full_width_block_splits_page_into_bands()
    test_continuation_sorts_first_in_its_column_despite_bad_y()
    test_column_classification()
    test_nested_box_normalized_and_missing_box_falls_back()
    test_continuation_across_page_break()
    test_continuation_across_column_break_with_markers_in_text()
    test_unmarked_leading_text_continues_stem_or_last_option()
    test_duplicate_option_letter_is_joined_and_flagged()
    test_orphan_continuation_is_flagged_not_dropped()
    test_answer_key_closes_open_question()
    test_solutions_section_is_not_extracted()
    test_find_numbering_issues()
    test_finalize_flags_and_key_matching()
    test_chapter_start_resets_numbering()
    test_match_grid_options_expanded()
    test_cache_cap_and_cost()
    test_rate_mirrors_server()
    test_parse_page_response()
    test_disha_replay()

    print(f"\nTOTAL: {len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
