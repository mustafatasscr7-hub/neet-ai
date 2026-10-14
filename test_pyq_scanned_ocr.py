"""Regression tests for the free "Scanned pages (free OCR)" mode (process_pyq_scanned_ocr.py).

Run with: python test_pyq_scanned_ocr.py
No pytest dependency, and no OCR engine needed: the layout rules are tested with synthetic OCR
lines, and the DISHA end-to-end check replays REAL RapidOCR output captured in
test-fixtures/scanned/disha-ocr-lines-real.json.
"""
import io
import json
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fitz
import process_pyq_scanned as ps
import process_pyq_scanned_ocr as po

OCR_FIXTURE = os.path.join(HERE, "test-fixtures", "scanned", "disha-ocr-lines-real.json")
BIOCLASS_FIXTURE = os.path.join(HERE, "test-fixtures", "scanned", "disha-bioclass-ocr-lines-real.json")
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
    lines = po._norm_lines(page(*shifted))
    gutter = po.find_gutter(lines)
    # The hanging "16." is left out of the measurement, so the gutter lands anywhere between the
    # left column's tags (end 546) and the right column's text (starts 600); "16." then goes with
    # its own text.
    check("gutter lands between 546 and 600", gutter is not None and 546 <= gutter <= 600, detail=f"gutter={gutter}")
    cols = {l["text"]: c for l, c in po.assign_columns(lines, gutter)}
    check("'16.' assigned to the right column", cols["16."] == "right")
    items, _ = po.segment_page(page(*shifted))
    q = {it["number"]: it for it in items if it["kind"] == "question"}
    check("Q14 and Q16 both found as their own questions", sorted(q) == [14, 16], detail=str(sorted(q)))
    check("Q14's year tag stays with Q14 (not pulled into the right column)", q.get(14, {}).get("year_tag") == "2013")
    check("Q16 got 4 options from its 2x2 row", len(q.get(16, {}).get("options", {})) == 4)


def test_hanging_numbers_near_midpoint():
    """DISHA Biological Classification page 1: the right column's "3." / "4." / "5." sit at
    x=505-531, just past the midpoint and nearer the left column's text (ends 487) than the gap to
    their own text (starts 549) is wide. Before the fix they were filed into the LEFT column, so
    Q3-Q5 never started and their text, tags and options were glued into Q2's options."""
    print("\n=== Right-column hanging numbers just past the midpoint stay in the right column ===")
    lines = [*left_q(1, 100, "First", tag="2022")[:1], L("Which of the following is a correct statement?", 107, 100, 487)]
    for i, letter in enumerate("abcd"):
        lines.append(L(f"({letter}) Q1 option {letter}", 107, 120 + 20 * i, 487))
    lines += [L("2.", 64, 220, 86), L("Five kingdom system is not based on:", 107, 220, 487),
              L("NCERT | Page-17 / N-11 |2014", 268, 240, 487),
              L("(a) Presence of nucleus.", 107, 260, 487), L("(b) Mode of reproduction.", 107, 280, 340),
              L("(c) Mode of nutrition.", 553, 60, 740), L("(d)Complexity of body organisation.", 553, 80, 860),
              L("3.", 505, 100, 531), L("In the five-kingdom system, which kingdom", 553, 100, 928),
              L("can include blue green algae?", 553, 120, 900),
              L("NCERT |Page-19 & 20 /N-13 | 1998, 2003", 631, 140, 928),
              L("(a) Fungi", 553, 160, 646), L("(b) Plantae", 743, 160, 850),
              L("(c) Protista", 553, 180, 659), L("(d) Monera", 744, 180, 852),
              L("4.", 505, 220, 531), L("In five kingdom system, the main basis of", 553, 220, 932),
              L("classification is", 553, 240, 682), L("NCERT|Page-17/ N-11 | 2002", 710, 240, 929),
              L("(a) structure of nucleus", 553, 260, 756), L("(b) mode of nutrition", 554, 280, 737),
              L("(c)", 558, 300, 592), L("structure of cell wall", 598, 300, 763),
              L("(d) asexual reproduction", 553, 320, 765),
              L("5.", 505, 360, 531), L("A system of classification in which a large", 553, 360, 932),
              L("number of traits are considered, is", 553, 380, 831), L("1999", 889, 380, 931),
              L("(a) artificial system (b) synthetic system", 553, 400, 932),
              L("(c) natural system (d) phylogenetic system", 553, 420, 932)]
    r = run_pages([page(*lines)])
    by = {q["question_number"]: q for q in r["questions"]}
    check("Q1-Q5 all found, in order", [q["question_number"] for q in r["questions"]] == [1, 2, 3, 4, 5], detail=str(list(by)))
    check("Q2 keeps exactly its own 4 options", [by[2][f"option_{l}"] for l in "abcd"] == ["Presence of nucleus.", "Mode of reproduction.", "Mode of nutrition.", "Complexity of body organisation."], detail=str([by[2][f"option_{l}"] for l in "abcd"]))
    check("Q3 tags parsed, not left in the text", by[3]["ncert_tag"] == "Page-19 & 20 / N-13" and by[3]["source_tag"] == "1998, 2003" and "NCERT" not in by[3]["question"], detail=str((by[3]["ncert_tag"], by[3]["source_tag"])))
    check("Q3's 2x2 grid read a, b, c, d", [by[3][f"option_{l}"] for l in "abcd"] == ["Fungi", "Plantae", "Protista", "Monera"])
    check("Q4 split '(c)' box joined to its text", by[4]["option_c"] == "structure of cell wall")
    check("Q5 year tag and 2x2 options", by[5]["source_tag"] == "1999" and by[5]["option_d"] == "phylogenetic system")
    check("no question flagged", not any(q["needs_review"] for q in r["questions"]), detail=str([q["review_reason"] for q in r["questions"] if q["needs_review"]]))


def test_off_centre_gutter_and_single_column():
    print("\n=== Gutter well off the midpoint; single-column page ===")
    # Left column 40-640, right column 680-980: a fixed midpoint split would cut the left column.
    p = page(L("1.", 40, 100, 60), L("Left question stem runs wide", 70, 100, 640),
             L("(a) w (b) x", 70, 120, 640), L("(c) y (d) z", 70, 140, 640),
             L("2.", 680, 100, 700), L("Right stem", 710, 100, 980),
             L("(a) p (b) q", 710, 120, 980), L("(c) r (d) s", 710, 140, 980))
    gutter = po.find_gutter(po._norm_lines(p))
    check("gutter found between 640 and 680", gutter is not None and 640 <= gutter <= 680, detail=f"gutter={gutter}")
    r = run_pages([p])
    check("both questions, left first", [q["question_number"] for q in r["questions"]] == [1, 2])
    check("left question's options intact", r["questions"][0]["option_d"] == "z")
    single = page(L("1.", 40, 100, 60), L("A single-column stem that runs right across the page", 70, 100, 960),
                  L("(a) one (b) two (c) three (d) four", 70, 120, 960),
                  L("2.", 40, 160, 60), L("Another stem across the whole page width here", 70, 160, 950),
                  L("(a) five", 70, 180, 300), L("(b) six", 520, 180, 700), L("(c) seven", 70, 200, 300), L("(d) eight", 520, 200, 700))
    check("single-column page -> no gutter", po.find_gutter(po._norm_lines(single)) is None)
    r = run_pages([single])
    by = {q["question_number"]: q for q in r["questions"]}
    check("single column: both questions with their own options", sorted(by) == [1, 2] and by[2]["option_b"] == "six" and by[1]["option_d"] == "four", detail=str({n: [q[f'option_{l}'] for l in 'abcd'] for n, q in by.items()}))


def test_tilted_column_hanging_numbers():
    """DISHA Biological Classification pages 3-4: the scan's left edge drifts from x=91 at the
    top to x=70 at the bottom, so the top numbers sat >20 units right of the column-wide minimum."""
    print("\n=== Tilted scan: hanging numbers judged against the nearby column edge ===")
    lines = []
    for i, n in enumerate(range(39, 44)):
        y = 100 + 170 * i
        x = 94 - 6 * i  # drifts 24 units left going down the page
        lines += [L(f"{n}.", x, y, x + 30), L(f"Stem {n}", x + 40, y, 480)]
        lines += [L(f"({l}) Q{n} {l}", x + 40, y + 20 * (k + 1), 480) for k, l in enumerate("abcd")]
    r = run_pages([page(*lines)])
    check("Q39-Q43 all start as questions", [q["question_number"] for q in r["questions"]] == [39, 40, 41, 42, 43], detail=str([q["question_number"] for q in r["questions"]]))


def test_grid_wrapped_cell():
    print("\n=== 2x2 grid whose (a) cell wraps onto its own line ===")
    items, _ = po.segment_page(page(
        L("7.", 50, 100, 80), L("Which pair is correct?", 90, 100, 460),
        L("(a) Fungi with chitin", 90, 120, 260), L("(b) Plantae", 280, 120, 400),
        L("cell walls", 110, 140, 220),
        L("(c) Protista", 90, 160, 220), L("(d) Monera", 280, 160, 400), L("cells", 300, 180, 360)))
    q = next(it for it in items if it["kind"] == "question")
    check("wrapped text stays in its own cell", q["options"] == {"a": "Fungi with chitin cell walls", "b": "Plantae", "c": "Protista", "d": "Monera cells"}, detail=str(q["options"]))


def test_polluted_option_split_and_flagged():
    print("\n=== Safety net: another question glued into an option is split out and flagged ===")
    q2 = {"number": 2, "question": "Five kingdom system is not based on:", "page": 1, "issues": [], "continued": False,
          "year_tag": "2014", "ncert_tag": "Page-17 / N-11", "question_type": "mcq", "has_diagram": False,
          "options": {"a": "Presence of nucleus.", "b": "Mode of reproduction.", "c": "Mode of nutrition.",
                      "d": "Complexity of body organisation. 3. Which single kingdom can include blue green algae? "
                           "NCERT |Page-19 & 20 /N-13 | 1998, 2003 (a) Fungi (b) Plantae (c) Protista (d) Monera"}}
    q4 = {**q2, "number": 4, "question": "Main basis", "issues": [], "year_tag": None, "ncert_tag": None,
          "options": {"a": "nucleus", "b": "nutrition 1998, 2003", "c": "cell wall",
                      "d": "asexual reproduction and a great deal of other text that does not belong here at all, glued on"}}
    seg = {"questions": [q2, q4], "answer_key": {}, "key_conflicts": [], "starts_at_chapter": False}
    po._resplit_polluted(seg)
    by = {q["number"]: q for q in seg["questions"]}
    check("Q3 split out between Q2 and Q4", [q["number"] for q in seg["questions"]] == [2, 3, 4])
    check("Q2 option (d) cleaned", by[2]["options"]["d"] == "Complexity of body organisation.", detail=by[2]["options"]["d"])
    check("Q3 stem, tags and options recovered", by[3]["question"] == "Which single kingdom can include blue green algae?"
          and by[3]["ncert_tag"] == "Page-19 & 20 / N-13" and by[3]["year_tag"] == "1998, 2003"
          and by[3]["options"] == {"a": "Fungi", "b": "Plantae", "c": "Protista", "d": "Monera"}, detail=str(by[3]))
    check("Q2 and Q3 both flagged", by[2]["issues"] and by[3]["issues"])
    check("year tag stripped from an option and given to the question", by[4]["options"]["b"] == "nutrition" and by[4]["year_tag"] == "1998, 2003")
    check("much-longer option flagged", any("much longer" in i for i in by[4]["issues"]))
    q5 = {**q2, "number": 5, "issues": [], "options": {"a": "1953, 1958", "b": "x", "c": "y", "d": "z 7. not next"}}
    seg = {"questions": [q5, {**q2, "number": 6, "issues": []}], "answer_key": {}, "key_conflicts": [], "starts_at_chapter": False}
    po._resplit_polluted(seg)
    check("years that ARE the option kept; a non-adjacent number not split, only flagged",
          len(seg["questions"]) == 2 and q5["options"]["a"] == "1953, 1958" and any("7." in i for i in q5["issues"]), detail=str(q5))


def test_glued_ocr_tokens():
    print("\n=== OCR dropped spaces: '123.Satellite', 'number(b) disappear' ===")
    items, _ = po.segment_page(page(
        L("123.Satellite RNAs are present in", 26, 100, 445),
        L("(a) Plant viruses (b) Viroids (c) Prions (d) Phages", 60, 120, 445),
        L("64.", 26, 160, 60), L("The vacuole will", 70, 160, 445),
        L("(a) increase in number(b) disappear", 70, 180, 445), L("(c) increase in size (d) decrease", 70, 200, 445)))
    q = {it["number"]: it for it in items if it["kind"] == "question"}
    check("'123.Satellite' starts Q123", q.get(123, {}).get("question") == "Satellite RNAs are present in", detail=str(list(q)))
    check("glued '(b)' split out", q.get(64, {}).get("options") == {"a": "increase in number", "b": "disappear", "c": "increase in size", "d": "decrease"}, detail=str(q.get(64, {}).get("options")))


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


def test_bioclass_replay():
    """DISHA Biological Classification (10 pages): before the column fix Q3-Q5 (page 1), Q39-41,
    Q43, Q49-51 (tilted pages 3-4), Q76 and Q123 never started and were glued into the previous
    question's options -- 137 of 149 found."""
    print("\n=== DISHA Biological Classification 10-page PDF: replay of the real OCR output ===")
    fx = json.load(open(BIOCLASS_FIXTURE, encoding="utf-8"))
    r = run_pages(fx["pages"])
    qs = r["questions"]
    by = {q["question_number"]: q for q in qs}
    check("149 questions numbered 1..149 in reading order", [q["question_number"] for q in qs] == list(range(1, 150)), detail=str(sorted(set(range(1, 150)) - set(by))))
    check("every question has 4 options", all(q[f"option_{l}"] for q in qs for l in "abcd"))
    check("Q2's options are exactly its own", [by[2][f"option_{l}"] for l in "abcd"] == [
        "Presence or absence of a well defined nucleus.", "Mode of reproduction.", "Mode of nutrition.", "Complexity of body organisation."])
    check("Q3 clean, tags parsed", by[3]["question"].startswith("In the five-kingdom system of classification, which single kingdom")
          and by[3]["question"].endswith("methanogenic archaebacteria?") and by[3]["source_tag"] == "1998, 2003"
          and by[3]["ncert_tag"] == "Page-19 & 20 / N-13" and [by[3][f"option_{l}"] for l in "abcd"] == ["Fungi", "Plantae", "Protista", "Monera"], detail=str(by[3]))
    check("Q4 clean", by[4]["source_tag"] == "2002" and [by[4][f"option_{l}"] for l in "abcd"] == [
        "structure of nucleus", "mode of nutrition", "structure of cell wall", "asexual reproduction"])
    check("Q5 clean", by[5]["source_tag"] == "1999" and [by[5][f"option_{l}"] for l in "abcd"] == [
        "artificial system", "synthetic system", "natural system", "phylogenetic system"])
    check("page 1 holds Q1-Q5, none flagged", [q["question_number"] for q in qs if q["source_page"] == 1] == [1, 2, 3, 4, 5]
          and not any(q["needs_review"] for q in qs if q["source_page"] == 1))
    check("no option holds an NCERT tag or a question number", not any(
        "NCERT" in q[f"option_{l}"] or re.search(r'(?:^|\s)\d{1,3}\.(?:\s|$|[A-Z])', q[f"option_{l}"]) for q in qs for l in "abcd"))
    check("tilted pages: Q39-41 and Q49-51 found as their own questions", all(by[n]["question"] for n in (39, 40, 41, 49, 50, 51)))


# ---------------------------------------------------------------- speed: pool, cache, diagram skip, jobs

# Module-level (picklable) so the page pool's worker processes can run them.
def slow_reversed_ocr(page):
    """Page i holds Q(i+1); earlier pages are made SLOWER, so a pool finishes them out of order."""
    import time as _t
    n = page.parent.page_count
    _t.sleep(0.15 * (n - page.number))
    return page_of_question(page.number + 1)


def page_of_question(num):
    return page(*left_q(num, 100, f"Question {num} stem"))


def no_figures(page):
    return []


def blank_pdf(n):
    doc = fitz.open()
    for _ in range(n):
        doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


def test_parallel_pages_keep_page_order():
    print("\n=== Page pool: results come back in page order, output identical to one-at-a-time ===")
    pdf = blank_pdf(5)
    seq_progress, par_progress = [], []
    seq = po.scan_ocr_pdf_bytes(pdf, "Biology", ocr_fn=slow_reversed_ocr, figures_fn=no_figures, workers=1,
                                progress=lambda d, t: seq_progress.append((d, t)))
    par = po.scan_ocr_pdf_bytes(pdf, "Biology", ocr_fn=slow_reversed_ocr, figures_fn=no_figures, workers=3,
                                progress=lambda d, t: par_progress.append((d, t)))
    for r in (seq, par):
        r["cost"].pop("seconds"), r["cost"].pop("seconds_per_page")
    check("Q1..Q5 in page order from the pool", [q["question_number"] for q in par["questions"]] == [1, 2, 3, 4, 5],
          detail=str([q["question_number"] for q in par["questions"]]))
    check("each question on its own page", [q["source_page"] for q in par["questions"]] == [1, 2, 3, 4, 5])
    check("pool output byte-identical to sequential", json.dumps(par, sort_keys=True) == json.dumps(seq, sort_keys=True))
    check("progress reported once per page, ending at 5/5", par_progress == [(i, 5) for i in range(1, 6)] and seq_progress == par_progress,
          detail=str(par_progress))
    check("worker count capped by env var and page count", _with_env("SCANNED_OCR_WORKERS", "2", lambda: po.ocr_worker_count(50)) == min(2, po._cpu_budget())
          and _with_env("SCANNED_OCR_WORKERS", "8", lambda: po.ocr_worker_count(1)) == 1)
    real_mem = po._memory_budget_workers
    po._memory_budget_workers = lambda: 1
    try:
        check("a container memory limit with room for 1 worker caps the pool at 1",
              _with_env("SCANNED_OCR_WORKERS", "8", lambda: po.ocr_worker_count(50)) == 1)
    finally:
        po._memory_budget_workers = real_mem
    r = po.read_all_pages(pdf, 5, ocr_fn=_raises_on_page_3, figures_fn=no_figures, workers=2)
    check("a failing page in the pool becomes that page's error, others unaffected",
          isinstance(r[2], str) and "boom" in r[2] and all(not isinstance(r[i], str) for i in (0, 1, 3, 4)))


def _raises_on_page_3(page):
    if page.number == 2:
        raise RuntimeError("boom")
    return page_of_question(page.number + 1)


def _with_env(name, value, fn):
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        return fn()
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


class _FakeOcrResult:
    def __init__(self, text):
        self.boxes = [[[10.0, 20.0], [110.5, 20.0], [110.5, 38.25], [10.0, 38.25]]]
        self.txts = (text,)
        self.scores = (0.9876543210123,)


class _FakeEngine:
    def __init__(self):
        self.calls = 0

    def __call__(self, img):
        self.calls += 1
        return _FakeOcrResult(f"call {self.calls}")


def test_ocr_cache_hit_miss_invalidation():
    print("\n=== OCR page cache: hit, miss, version invalidation, off switch ===")
    import tempfile
    real = (po._engine, po.OCR_CACHE_DIR, po._lib_version)
    tmp = tempfile.mkdtemp()
    po.OCR_CACHE_DIR = tmp
    po._engine = fake = _FakeEngine()
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "page one")
    doc.new_page().insert_text((72, 72), "page two")
    p1, p2 = doc[0], doc[1]  # adding a page invalidates earlier page handles
    try:
        a, hit_a = po._ocr_page_cached(p1)
        b, hit_b = po._ocr_page_cached(p1)
        check("first read is a miss and runs the engine", not hit_a and fake.calls == 1)
        check("second read of the same page is a hit, engine not run", hit_b and fake.calls == 1)
        check("cached value identical, floats exact", json.dumps(a) == json.dumps(b) and b["lines"][0]["score"] == 0.9876543210123)
        _, hit_c = po._ocr_page_cached(p2)
        check("a different page image misses", not hit_c and fake.calls == 2)
        po._lib_version = lambda name: "9.9.9" if name == "rapidocr" else real[2](name)
        _, hit_d = po._ocr_page_cached(p1)
        check("an OCR library version change invalidates the cache", not hit_d and fake.calls == 3)
        po._lib_version = real[2]
        _, hit_e = po._ocr_page_cached(p1)
        check("back on the original version, the original entry hits again", hit_e and fake.calls == 3)
        n = fake.calls
        _with_env("SCANNED_OCR_CACHE", "0", lambda: po._ocr_page_cached(p1))
        check("SCANNED_OCR_CACHE=0 bypasses the cache", fake.calls == n + 1)
    finally:
        po._engine, po.OCR_CACHE_DIR, po._lib_version = real
        doc.close()


def test_cached_rerun_skips_pool_and_pool_is_reused():
    print("\n=== Re-run of a cached PDF never reaches the page pool; the pool is reused across scans ===")
    import tempfile
    import numpy as np
    import process_pyq_vision as pv
    real = (po._engine, po.OCR_CACHE_DIR, pv._yolo_infer_figure_boxes, po._get_figure_session, po._acquire_pool)
    po.OCR_CACHE_DIR = tempfile.mkdtemp()
    po._engine = fake = _FakeEngine()
    pv._yolo_infer_figure_boxes = lambda *a: [(np.float32(1), np.float32(2), np.float32(3), np.float32(4))]
    po._get_figure_session = lambda: None
    doc = fitz.open()
    for i in range(3):
        doc.new_page().insert_text((72, 72), f"page {i}")
    pdf = doc.tobytes()
    doc.close()
    try:
        first = po.scan_ocr_pdf_bytes(pdf, "Biology", workers=1)
        calls = fake.calls

        def no_pool(*a, **k):
            raise AssertionError("pool used for a fully cached PDF")

        po._acquire_pool = no_pool
        second = po.scan_ocr_pdf_bytes(pdf, "Biology", workers=4)
        check("second run: every page from cache, engine not called, pool not touched",
              second["cost"]["pages_cached"] == 3 and fake.calls == calls, detail=str(second.get("cost")))
        for r in (first, second):
            for k in ("seconds", "seconds_per_page", "pages_cached"):
                r["cost"].pop(k)
        check("cached re-run output identical to the first run", json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True))
    finally:
        po._engine, po.OCR_CACHE_DIR, pv._yolo_infer_figure_boxes, po._get_figure_session, po._acquire_pool = real
    a = po._acquire_pool(2, warm=False)
    po._release_pool()
    b = po._acquire_pool(2, warm=False)
    po._release_pool()
    check("the same warm pool serves the next scan", a is b)
    po.shutdown_ocr_pool()
    check("idle shutdown frees it", po._pool is None)


def test_figure_cache_restores_float32():
    print("\n=== Diagram cache: restored boxes are the same float32 values the model returned ===")
    import tempfile
    import numpy as np
    import process_pyq_vision as pv
    real = (po.OCR_CACHE_DIR, pv._yolo_infer_figure_boxes, po._get_figure_session)
    po.OCR_CACHE_DIR = tempfile.mkdtemp()
    calls = []

    def fake_infer(session, arr, cv2, np_):
        calls.append(1)
        return [(np.float32(100.3), np.float32(200.7), np.float32(300.1), np.float32(400.9))]

    pv._yolo_infer_figure_boxes = fake_infer
    po._get_figure_session = lambda: None
    doc = fitz.open()
    pg = doc.new_page()
    try:
        first = po.figure_boxes_1000(pg)
        second = po.figure_boxes_1000(pg)
        check("second call served from cache", len(calls) == 1)
        check("same values", first == second, detail=f"{first} vs {second}")
        check("restored as numpy float32", all(type(v) is np.float32 for v in second[0]), detail=str([type(v) for v in second[0]]))
    finally:
        po.OCR_CACHE_DIR, pv._yolo_infer_figure_boxes, po._get_figure_session = real
        doc.close()


def test_diagram_detection_skipped_without_questions():
    print("\n=== Diagram detection only runs on pages with a question ===")
    key_page = page(L("ANSWER KEY", 400, 100, 600), L("1", 60, 130, 80), L("(b)", 100, 130, 130))
    q_page = page(*left_q(1, 100, "Which diagram shows this?"))
    calls = []

    def figs(p):
        calls.append(p.number)
        return [(90, 40, 220, 470)]  # covers Q1's whole box

    pages = [q_page, key_page]
    r = po.scan_ocr_pdf_bytes(blank_pdf(2), "Biology", ocr_fn=lambda p: pages[p.number], figures_fn=figs, workers=1)
    check("detection ran on the question page only", calls == [0], detail=str(calls))
    check("figure still sets has_diagram on the question", r["questions"][0]["has_diagram"] is True)
    items_with, _ = po.segment_page(key_page, [(0, 0, 1000, 1000)])
    items_without, _ = po.segment_page(key_page, ())
    check("on a question-less page, figures change nothing", items_with == items_without)


def test_job_progress():
    print("\n=== Background OCR job: queued -> running with progress -> done; errors; unknown id ===")
    import threading as _th
    import time as _t
    step, release = _th.Event(), _th.Event()

    def fake_scan(pdf_bytes, subject, progress=None):
        progress(1, 3)
        step.set()
        release.wait(5)
        progress(2, 3)
        progress(3, 3)
        return {"questions": [{"question_number": 1}], "pages_scanned": 3, "flagged_pages": [], "cost": {}}

    finished = []
    jid = po.start_ocr_job(blank_pdf(3), "Biology", on_finish=finished.append, scan_fn=fake_scan)
    step.wait(5)
    mid = po.get_ocr_job(jid)
    check("running, 1 of 3 pages done", mid == {"status": "running", "pages_done": 1, "pages_total": 3}, detail=str(mid))
    jid2 = po.start_ocr_job(blank_pdf(2), "Biology", scan_fn=lambda b, s, progress=None: {"error": "engine broke"})
    _t.sleep(0.2)
    check("a second job queues behind the running one", po.get_ocr_job(jid2)["status"] == "queued")
    release.set()
    done = _wait_for(lambda: po.get_ocr_job(jid)["status"] == "done")
    end = po.get_ocr_job(jid)
    check("done, 3 of 3, result attached", done and end["pages_done"] == 3 and end["result"]["pages_scanned"] == 3, detail=str(end))
    check("on_finish got the result", len(finished) == 1 and finished[0]["pages_scanned"] == 3)
    _wait_for(lambda: po.get_ocr_job(jid2)["status"] == "error")
    err = po.get_ocr_job(jid2)
    check("a failed scan reports status error with the message", err["status"] == "error" and err["error"] == "engine broke", detail=str(err))
    check("unknown job id -> None", po.get_ocr_job("nope") is None)


def _wait_for(cond, timeout=5):
    import time as _t
    end = _t.time() + timeout
    while _t.time() < end:
        if cond():
            return True
        _t.sleep(0.02)
    return False


if __name__ == "__main__":
    test_parallel_pages_keep_page_order()
    test_ocr_cache_hit_miss_invalidation()
    test_cached_rerun_skips_pool_and_pool_is_reused()
    test_figure_cache_restores_float32()
    test_diagram_detection_skipped_without_questions()
    test_job_progress()
    test_left_column_then_right_column()
    test_gutter_measured_per_page()
    test_hanging_numbers_near_midpoint()
    test_off_centre_gutter_and_single_column()
    test_tilted_column_hanging_numbers()
    test_grid_wrapped_cell()
    test_polluted_option_split_and_flagged()
    test_glued_ocr_tokens()
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
    test_bioclass_replay()

    print(f"\nTOTAL: {len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
