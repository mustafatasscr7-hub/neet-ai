"""Regression tests for the free "Scanned pages (free OCR)" mode (process_pyq_scanned_ocr.py).

Run with: python test_pyq_scanned_ocr.py
No pytest dependency, and no OCR engine needed: the layout rules are tested with synthetic OCR
lines, and the DISHA end-to-end check replays REAL RapidOCR output captured in
test-fixtures/scanned/disha-ocr-lines-real.json.
"""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fitz
import process_pyq_scanned as ps
import process_pyq_scanned_ocr as po

OCR_FIXTURE = os.path.join(HERE, "test-fixtures", "scanned", "disha-ocr-lines-real.json")
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


# Synthetic pages use a 1000x1000 "image", so pixel boxes are already in box_2d's 0-1000 units.
def L(text, x0, y, x1, score=1.0, h=18):
    return {"text": text, "score": score, "box": [x0, y, x1, y + h]}


def page(*lines):
    return {"width": 1000, "height": 1000, "lines": list(lines)}


def left_q(num, y, stem, opts="abcd", tag=None):
    """A left-column question: hanging number at x=50, text at x=90, right edge 460."""
    out = [L(f"{num}.", 50, y, 80), L(stem, 90, y, 400)]
    if tag:
        out.append(L(tag, 410, y, 460))
    for i, letter in enumerate(opts):
        out.append(L(f"({letter}) Q{num} option {letter}", 90, y + 20 * (i + 1), 400))
    return out


def right_q(num, y, stem, opts="abcd", x=560):
    out = [L(f"{num}.", x, y, x + 30), L(stem, x + 40, y, x + 380)]
    for i, letter in enumerate(opts):
        out.append(L(f"({letter}) Q{num} option {letter}", x + 40, y + 20 * (i + 1), x + 380))
    return out


def questions_of(items):
    return [it["number"] for it in items if it["kind"] == "question"]


def run_pages(pages, pdf=None):
    """Full mode with the given OCR output per page, on `pdf` (default: a blank N-page PDF)."""
    if pdf is None:
        doc = fitz.open()
        for _ in pages:
            doc.new_page()
        pdf = doc.tobytes()
        doc.close()
    return po.scan_ocr_pdf_bytes(pdf, "Biology", ocr_fn=lambda p: pages[p.number], figures_fn=lambda p: [])


# ---------------------------------------------------------------- column ordering

def test_left_column_then_right_column():
    print("\n=== Reading order: whole left column, then right column ===")
    items, _ = po.segment_page(page(*right_q(3, 100, "Third"), *left_q(1, 100, "First"), *left_q(2, 220, "Second"), *right_q(4, 220, "Fourth")))
    ordered, geom = ps.order_page_items(items)
    check("every OCR item has a usable box", geom)
    check("order 1, 2, 3, 4", questions_of(ordered) == [1, 2, 3, 4], detail=str(questions_of(ordered)))


def test_gutter_measured_per_page():
    """DISHA page 2 sits ~80 units right of pages 1/3: its LEFT-column year tags end at x=546,
    past where page 3's RIGHT column starts (481). A fixed gutter misfiles one or the other."""
    print("\n=== Column split is measured per page (shifted page 2 shape) ===")
    shifted = [L("14.", 125, 100, 158), L("The common characteristics between tomato", 170, 100, 546),
               L("2013", 500, 120, 546), L("(a) Genus", 170, 140, 268), L("(b) Family", 360, 140, 470),
               L("(c) Order", 170, 160, 265), L("(d) Division", 360, 160, 475),
               L("16.", 560, 100, 594), L("Which one of the following animals is", 600, 100, 977),
               L("(a) Tiger (b) Cuttle fish (c) Humans (d) Housefly", 603, 120, 977)]
    gutter = po.find_gutter(po._norm_lines(page(*shifted)))
    check("gutter lands between 546 and 560", 546 <= gutter <= 560, detail=f"gutter={gutter}")
    items, _ = po.segment_page(page(*shifted))
    q = {it["number"]: it for it in items if it["kind"] == "question"}
    check("Q14 and Q16 both found as their own questions", sorted(q) == [14, 16], detail=str(sorted(q)))
    check("Q14's year tag stays with Q14 (not pulled into the right column)", q.get(14, {}).get("year_tag") == "2013")
    check("Q16 got 4 options from its 2x2 row", len(q.get(16, {}).get("options", {})) == 4)


def test_full_width_answer_key_splits_bands():
    print("\n=== Full-width answer key splits the page into bands; solutions below are dropped ===")
    items, _ = po.segment_page(page(
        *left_q(37, 50, "Match", opts="ab"),
        L("(c) Q37 option c", 600, 50, 950), L("(d) Q37 option d", 600, 70, 950),
        L("ANSWER KEY", 450, 300, 600),
        L("37", 100, 330, 130), L("(c)", 140, 330, 170), L("38", 200, 330, 230), L("(b)", 240, 330, 270),
        L("Hints & Solutions", 350, 400, 750, h=40),
        L("1.", 50, 460, 80), L("(b) Basal metabolic rate is inversely proportional", 90, 460, 540),
    ))
    ordered, _ = ps.order_page_items(items)
    kinds = [it["kind"] if it["kind"] != "question" else it["number"] for it in ordered]
    check("order: Q37, its right-column continuation, key, solutions", kinds == [37, "continuation", "answer_key", "solutions_start"], detail=str(kinds))
    key = next(it for it in items if it["kind"] == "answer_key")
    check("key cells paired by row position", key["entries"] == {"37": "c", "38": "b"}, detail=str(key["entries"]))


# ---------------------------------------------------------------- continuation joining

def test_continuation_across_page_break():
    print("\n=== Question continuing across a PAGE break is joined, not split ===")
    p1 = page(*left_q(8, 100, "Static concept of species"), L("9.", 560, 800, 590), L("In the taxonomic categories which", 600, 800, 950))
    p2 = page(L("(a) Kingdom, Class, Phylum", 150, 60, 540), L("(b) Kingdom, Order, Class", 150, 80, 540),
              L("(c) Kingdom, Order, Phylum", 150, 100, 540), L("(d) Kingdom, Phylum, Class", 150, 120, 540),
              *left_q(10, 150, "Which one belongs to Muscidae?"))
    r = run_pages([p1, p2])
    by = {q["question_number"]: q for q in r["questions"]}
    check("three questions, the continuation did not become its own", sorted(by) == [8, 9, 10], detail=str(sorted(by)))
    check("Q9 got all four options from the next page", by[9]["option_a"] == "Kingdom, Class, Phylum" and by[9]["option_d"] == "Kingdom, Phylum, Class")
    check("Q9 stays on the page it started", by[9]["source_page"] == 1)
    check("cost is $0", r["cost"]["usd"] == 0)


def test_continuation_across_column_break():
    print("\n=== Question continuing across a COLUMN break (Q15 shape) is joined ===")
    p = page(*left_q(15, 700, "Which organism is correctly named?", opts="ab"),
             L("(c) Felis tigris - The Indian tiger", 600, 60, 977), L("(d) E.coli - a bacterium", 600, 80, 977),
             *right_q(16, 120, "Which animal is correctly matched?"))
    r = run_pages([p])
    by = {q["question_number"]: q for q in r["questions"]}
    check("Q15 option (b) untouched", by[15]["option_b"] == "Q15 option b", detail=by[15]["option_b"])
    check("Q15 (c)/(d) filled from the right column", by[15]["option_c"].startswith("Felis tigris") and by[15]["option_d"].startswith("E.coli"))
    check("Q16 is still its own question", by[16]["option_a"] == "Q16 option a")


def test_wrapped_stem_line_is_not_a_question_start():
    print("\n=== A wrapped line beginning with a number is NOT a new question ===")
    p = page(L("5.", 50, 100, 80), L("How many chromosomes are there after", 90, 100, 460),
             L("2. meiotic divisions of one cell?", 90, 120, 460),  # indented -> not hanging
             L("(a) 2 (b) 4", 90, 140, 460), L("(c) 8 (d) 16", 90, 160, 460))
    items, _ = po.segment_page(page(*p["lines"]))
    check("only Q5", questions_of(items) == [5], detail=str(questions_of(items)))


# ---------------------------------------------------------------- tags, key, zones, flags

def test_tags():
    print("\n=== Year / NCERT tags: separate box, merged into the stem box, misread ===")
    items, _ = po.segment_page(page(
        *left_q(1, 100, "Separate tag box", tag="2016"),
        L("2.", 50, 220, 80), L("Merged tag stem written by 1999", 90, 220, 460), L("(a) w (b) x (c) y (d) z", 90, 240, 400),
        L("3.", 50, 300, 80), L("NCERT tag on its own line", 90, 300, 300), L("NCERT | Page-8/N-6 | 1990,91,92,96", 250, 320, 460),
        L("(a) w (b) x (c) y (d) z", 90, 340, 400),
        L("4.", 50, 400, 80), L("Misread tag", 90, 400, 300), L("1661", 420, 400, 460), L("(a) w (b) x (c) y (d) z", 90, 420, 400),
        L("5.", 50, 500, 80), L("Year as an option's own text", 90, 500, 300), L("(a) 1990 (b) 1991", 90, 520, 300), L("(c)", 330, 540, 360), L("1992", 400, 540, 460),
    ))
    q = {it["number"]: it for it in items if it["kind"] == "question"}
    check("separate right-aligned box -> year_tag, removed from stem", q[1]["year_tag"] == "2016" and "2016" not in q[1]["question"])
    check("tag glued onto the stem box is split off", q[2]["year_tag"] == "1999" and q[2]["question"] == "Merged tag stem written by", detail=repr(q[2]["question"]))
    check("NCERT tag parsed and normalised", q[3]["ncert_tag"] == "Page-8 / N-6" and q[3]["year_tag"] == "1990,91,92,96", detail=str((q[3]["ncert_tag"], q[3]["year_tag"])))
    check("misread tag '1661': removed, year left blank, reported", q[4]["year_tag"] is None and "1661" not in q[4]["question"] and any("1661" in i for i in q[4]["_ocr_issues"]))
    check("a year that is an option's own text is NOT taken as a tag", q[5]["year_tag"] is None and q[5]["options"].get("c") == "1992", detail=str(q[5]["options"]))


def test_answer_key_never_guesses():
    """Real DISHA: the key's "1" cell (white on dark green) was never read; its "(b)" must stay
    unattached rather than being assumed to be Q1 from its grid position."""
    print("\n=== Answer key: a letter with no readable number beside it is never attached ===")
    entries, notes = po._parse_answer_key(po._norm_lines(page(
        L("(b)", 140, 300, 170), L("5", 200, 300, 220, score=0.59), L("(b)", 240, 300, 270),
        L("2", 100, 320, 120), L("(c)", 140, 320, 170), L("6", 200, 320, 220), L("(d)", 240, 320, 270),
    )))
    check("readable pairs kept", entries == {5: "b", 2: "c", 6: "d"}, detail=str(entries))
    check("unpaired cell reported", any("no readable question number" in n for n in notes))
    check("low-confidence cell reported", any("Q5" in n and "low OCR confidence" in n for n in notes))


def test_zones_and_furniture():
    print("\n=== Banner + trend table -> chapter_start; topic headings, headers, handle dropped ===")
    items, _ = po.segment_page(page(
        L("A-3", 850, 20, 895), L("Class-XI/Chapter-1/The Living World", 50, 25, 350),
        L("The Living World", 350, 60, 730, h=40), L("@sachasanatani", 495, 110, 625),
        L("Trend Analysis with Important Topics & Sub-Topics", 58, 150, 499),
        L("2023", 450, 170, 496), L("What is Living?", 400, 190, 600),
        L("LOD - Level of Difficulty", 97, 220, 273), L("Qns - No. of Questions", 710, 220, 878),
        L("Topic-2: Diversity in the Living World,", 60, 260, 440), L("Taxonomic Categories", 150, 278, 350),
        *left_q(9, 300, "In the taxonomic categories"),
    ))
    kinds = [it["kind"] for it in items]
    chapter = next((it for it in items if it["kind"] == "chapter_start"), None)
    check("chapter_start emitted with the banner's title", chapter is not None and chapter["title"] == "The Living World", detail=str(chapter))
    check("only Q9 besides the chapter start (no table/heading/header text leaked)", kinds == ["chapter_start", "question"] and questions_of(items) == [9], detail=str(kinds))
    q9 = next(it for it in items if it["kind"] == "question")
    check("2-line topic heading not glued onto anything", "Taxonomic" not in q9["question"])


def test_suspicious_text_flags():
    print("\n=== High-confidence OCR misreads that the score misses get flagged ===")
    check("accented letter flagged", po._suspicious_text("muscles in Îarge animals"))
    check("run-together words flagged", po._suspicious_text("is scientificallycorrectlynamed,correctly printed"))
    check("normal biology text not flagged", not po._suspicious_text("submicroscopic molecular level; Biosystematics; Entamoeba coli"))


def test_cost_and_missing_engine():
    print("\n=== $0, and a clear error when the OCR engine isn't installed ===")
    r = run_pages([page(*left_q(1, 100, "Only"))])
    check("cost reported as $0 with timing", r["cost"]["usd"] == 0 and "seconds_per_page" in r["cost"])
    real = po.ocr_available
    po.ocr_available = lambda: False
    try:
        r = po.scan_ocr_pdf_bytes(b"%PDF", "Biology")
        check("missing engine -> error, not a crash", "rapidocr" in r.get("error", ""))
    finally:
        po.ocr_available = real


# ---------------------------------------------------------------- real DISHA replay

def test_disha_replay():
    print("\n=== DISHA 4-page PDF: replay of the real OCR output ===")
    fx = json.load(open(OCR_FIXTURE, encoding="utf-8"))
    # On the real PDF when present, so the pixel-based crop check runs too.
    have_pdf = os.path.exists(DISHA_PDF)
    r = run_pages(fx["pages"], open(DISHA_PDF, "rb").read() if have_pdf else None)
    qs = r["questions"]
    by = {q["question_number"]: q for q in qs}
    printed_key = "b c a d b d b b d a a b a b b a a c d c c d d b a c b a b c b c a c d c c b".split()
    per_page = {}
    for q in qs:
        per_page[q["source_page"]] = per_page.get(q["source_page"], 0) + 1
    reasons = " | ".join(f["reason"] for f in r["flagged_pages"])
    check("38 questions numbered 1..38 in reading order", [q["question_number"] for q in qs] == list(range(1, 39)))
    check("per page 9 / 13 / 15 / 1", per_page == {1: 9, 2: 13, 3: 15, 4: 1}, detail=str(per_page))
    check("every question has 4 options", all(q[f"option_{l}"] for q in qs for l in "abcd"))
    check("Q2-Q38 keys all match the printed key", all(by[n]["correct_answer"] == printed_key[n - 1] for n in range(2, 39)))
    check("Q1's key cell was unreadable: left BLANK and flagged, not guessed", by[1]["correct_answer"] == "" and "No answer key found for 1 question(s) (Q1)" in reasons)
    check("Q9 (page 1 -> 2) joined", by[9]["option_a"].startswith("Kingdom, Class, Phylum"))
    check("Q15 (left -> right column) joined", by[15]["option_c"].startswith("Felis tigris") and "(c)" not in by[15]["option_b"])
    check("Q37 (page 3 -> 4) grid joined and labelled", by[37]["option_a"] == "(A)-(i), (B)-(iv), (C)-(iii), (D)-(ii)", detail=by[37]["option_a"])
    check("Q37's match table keeps one row per line", "\n(A) Herbarium (i)" in by[37]["question"] and by[37]["question_type"] == "match_column")
    check("Q35's statements keep one per line", "\n(ii) Key is analytical in nature." in by[35]["question"] and by[35]["question_type"] == "statement_based")
    check("nothing from the trend table, headings or solutions leaked", not any(s in q["question"] for q in qs for s in ("Trend", "Topic-", "Basal metabolic", "LOD")))
    check("cut-off words kept as printed, never completed (Q38)", "photograpl of" in by[38]["option_b"], detail=by[38]["option_b"])
    if have_pdf:
        check("crop flags on pages 2 and 4 say words were kept as printed",
              sorted(f["page"] for f in r["flagged_pages"] if "kept exactly as printed" in f["reason"]) == [2, 4] and "model may have" not in reasons)
    else:
        print(f"  (skipped crop-flag check: {DISHA_PDF} not present)")
    check("Q1 'Îarge' misread flagged", "unusual character" in by[1]["review_reason"])
    check("Q15 run-together words flagged", "run together" in by[15]["review_reason"])
    check("tags: Q23 multi-year, Q35 phase tag", by[23]["source_tag"] == "1990,91,92,96" and by[35]["source_tag"] == "Ph-II 2022" and by[35]["year"] == 2022)


if __name__ == "__main__":
    test_left_column_then_right_column()
    test_gutter_measured_per_page()
    test_full_width_answer_key_splits_bands()
    test_continuation_across_page_break()
    test_continuation_across_column_break()
    test_wrapped_stem_line_is_not_a_question_start()
    test_tags()
    test_answer_key_never_guesses()
    test_zones_and_furniture()
    test_suspicious_text_flags()
    test_cost_and_missing_engine()
    test_disha_replay()

    print(f"\nTOTAL: {len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
