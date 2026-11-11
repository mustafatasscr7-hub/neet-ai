"""Regression test for process_mock_exam_math.py -- the deterministic (NO model call, ever) glyph-
geometry math reconstruction used by process_mock_exam_text.py's own question/option extraction.

Run with: python test_mock_exam_math.py
Same plain pass/fail harness as test_mock_exam_parser.py.

Covers the reconstruction shapes confirmed live on the real sample PDF (MockTest1.pdf, 2026-11-11):
  * superscript (a smaller, raised span) as ^{};
  * subscript (a smaller, lowered span) as _{};
  * a fraction whose only real signal is a vector-drawn horizontal bar (no "/" character anywhere
    in the text layer) as \\frac{}{};
  * a square root from a literal U+221A character already in the text layer, both glued onto its
    own radicand's span and as its own separate span, including a parenthesized radicand;
  * a vector arrow from the real "MT Extra" accent font (two different real-world internal PDF
    names for the same typeface, confirmed live on this machine -- see
    process_mock_exam_math._ACCENT_FONT_NAMES's own comment);
  * a 2x2 grid of options where two of the four cells each hold their own inline fraction, without
    one option's fraction bleeding into its neighbor's (requirement: never cross an option
    boundary);
  * the confidence rule itself: a small glyph whose own bbox straddles the midpoint between two
    options sharing one row falls back to the plain-text path (flagged by the existing gates)
    rather than being guessed into the wrong option.

A bold font is required on the "a."-"d." marker spans for every test here -- confirmed live,
process_mock_exam_math.build_option_zones only engages its own geometry-aware zone split when it
can find exactly one BOLD marker per letter; a non-bold marker (as test_mock_exam_parser.py's own
plain-text-focused fixtures use) falls back to the old text-only split, which never exercises any
of this module's own reconstruction at all.

Every fixture here is hand-built synthetic text/geometry via PyMuPDF, never real exam content.
"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fitz
import process_mock_exam_text as pmet
import process_mock_exam_math as pmm

_passed = []
_failed = []


def check(label, condition, detail=""):
    if condition:
        _passed.append(label)
        print(f"  PASS: {label}")
    else:
        _failed.append(label)
        print(f"  FAIL: {label}  {detail}")


_ARIAL = r"C:\Windows\Fonts\arial.ttf"
_MT_EXTRA = r"C:\Windows\Fonts\MTEXTRA.TTF"


def _new_page(doc, width=600, height=800):
    return doc.new_page(width=width, height=height)


def _header(page, subject, page_num, x0=50, x1=560):
    pad = " " * max(1, int((x1 - x0) / 4.3) - len(subject) - len(str(page_num)))
    page.insert_text((x0, 30), f"{subject}{pad}{page_num}", fontsize=10)


def _marker(page, x, y, letter, fontsize=10):
    """A bold "a."-"d." option marker -- see this module's own docstring for why bold is
    required for process_mock_exam_math's own zone split to engage at all."""
    page.insert_text((x, y), f"{letter}.", fontsize=fontsize, fontname="hebo")


def _pdf_bytes(doc):
    data = doc.tobytes()
    doc.close()
    return data


def _one_question(result, subject="Physics"):
    qs = [q for q in result["questions"] if q["subject"] == subject]
    check(f"exactly one {subject} question found", len(qs) == 1, qs)
    return qs[0]


# ---------------------------------------------------------------- superscript / subscript

def test_superscript_exponent():
    print("\n# A smaller, raised span is reconstructed as ^{} (requirement: superscripts)")
    doc = fitz.open()
    page = _new_page(doc)
    _header(page, "Physics", 2)
    page.insert_text((50, 80), "1. Superscript question?", fontsize=10)
    _marker(page, 50, 100, "a")
    page.insert_text((63, 100), "m", fontsize=10)
    page.insert_text((70, 94), "2", fontsize=6)  # raised + smaller -- exponent
    _marker(page, 50, 115, "b")
    page.insert_text((63, 115), "plain", fontsize=10)
    _marker(page, 50, 130, "c")
    page.insert_text((63, 130), "plain2", fontsize=10)
    _marker(page, 50, 145, "d")
    page.insert_text((63, 145), "plain3", fontsize=10)
    result = pmet.scan_mock_exam_pdf(_pdf_bytes(doc), "Synthetic Math 1")
    q = _one_question(result)
    check("not flagged", not q["needs_review"], q)
    check('option A reconstructed as "m^{2}"', q["option_a"].replace(" ", "") == "$m^{2}$", q["option_a"])


def test_subscript():
    print("\n# A smaller, lowered span is reconstructed as _{} (requirement: subscripts)")
    doc = fitz.open()
    page = _new_page(doc)
    _header(page, "Physics", 2)
    page.insert_text((50, 80), "2. Subscript question?", fontsize=10)
    _marker(page, 50, 100, "a")
    page.insert_text((63, 100), "V", fontsize=10)
    page.insert_text((70, 104), "A", fontsize=6)  # lowered + smaller -- subscript
    _marker(page, 50, 115, "b")
    page.insert_text((63, 115), "plain", fontsize=10)
    _marker(page, 50, 130, "c")
    page.insert_text((63, 130), "plain2", fontsize=10)
    _marker(page, 50, 145, "d")
    page.insert_text((63, 145), "plain3", fontsize=10)
    result = pmet.scan_mock_exam_pdf(_pdf_bytes(doc), "Synthetic Math 2")
    q = _one_question(result)
    check("not flagged", not q["needs_review"], q)
    check('option A reconstructed as "V_{A}"', q["option_a"].replace(" ", "") == "$V_{A}$", q["option_a"])


# ---------------------------------------------------------------- fraction

def test_stacked_fraction_from_drawn_bar():
    print("\n# A fraction whose ONLY signal is a drawn bar (no '/' character at all) -> \\frac{}{}")
    doc = fitz.open()
    page = _new_page(doc)
    _header(page, "Physics", 2)
    page.insert_text((50, 80), "3. Fraction question?", fontsize=10)
    _marker(page, 50, 100, "a")
    page.insert_text((70, 98), "1", fontsize=10)       # numerator
    page.draw_line((68, 102), (82, 102))               # the fraction bar itself
    page.insert_text((70, 112), "2", fontsize=10)      # denominator
    _marker(page, 50, 125, "b")
    page.insert_text((63, 125), "plain", fontsize=10)
    _marker(page, 50, 140, "c")
    page.insert_text((63, 140), "plain2", fontsize=10)
    _marker(page, 50, 155, "d")
    page.insert_text((63, 155), "plain3", fontsize=10)
    result = pmet.scan_mock_exam_pdf(_pdf_bytes(doc), "Synthetic Math 3")
    q = _one_question(result)
    check("not flagged", not q["needs_review"], q)
    check('option A reconstructed as "\\frac{1}{2}"', q["option_a"].replace(" ", "") == "$\\frac{1}{2}$", q["option_a"])


def test_2x2_grid_with_fraction_options_no_cross_contamination():
    print("\n# A 2x2 grid where TWO cells each hold their own fraction -- never bleeding into a neighbor")
    doc = fitz.open()
    page = _new_page(doc)
    _header(page, "Chemistry", 2)
    page.insert_text((50, 80), "4. Grid-with-fractions question?", fontsize=10)
    # Row 1: a (a plain fraction 1/2) and b (a plain fraction 3/4), side by side.
    _marker(page, 50, 100, "a")
    page.insert_text((70, 98), "1", fontsize=10)
    page.draw_line((68, 102), (82, 102))
    page.insert_text((70, 112), "2", fontsize=10)
    _marker(page, 200, 100, "b")
    page.insert_text((220, 98), "3", fontsize=10)
    page.draw_line((218, 102), (232, 102))
    page.insert_text((220, 112), "4", fontsize=10)
    # Row 2: c and d, plain short options -- far enough below row 1 not to overlap it. "sixteen"
    # (not "six") so the question's own union bbox -- which bounds every option zone's right edge
    # -- naturally reaches at least as far right as option b's own fraction bar does; a narrower
    # union bbox than that would clip b's own bar out of its own zone by a couple points, a test-
    # geometry artifact that has nothing to do with the reconstruction logic itself.
    _marker(page, 50, 140, "c")
    page.insert_text((63, 140), "five", fontsize=10)
    _marker(page, 200, 140, "d")
    page.insert_text((213, 140), "sixteen", fontsize=10)
    result = pmet.scan_mock_exam_pdf(_pdf_bytes(doc), "Synthetic Math 4")
    q = _one_question(result, subject="Chemistry")
    check("not flagged (both fractions resolved cleanly)", not q["needs_review"], q)
    check('option A is its OWN fraction "1/2", not b\'s', q["option_a"].replace(" ", "") == "$\\frac{1}{2}$", q["option_a"])
    check('option B is its OWN fraction "3/4", not a\'s', q["option_b"].replace(" ", "") == "$\\frac{3}{4}$", q["option_b"])
    check("option C untouched by either fraction", q["option_c"].strip() == "five", q["option_c"])
    check("option D untouched by either fraction", q["option_d"].strip() == "sixteen", q["option_d"])


# ---------------------------------------------------------------- root

def test_root_glued_to_its_own_span():
    print("\n# A literal U+221A glued onto the SAME span as its own radicand -> \\sqrt{}")
    doc = fitz.open()
    page = _new_page(doc)
    _header(page, "Physics", 2)
    page.insert_text((50, 80), "5. Root question?", fontsize=10)
    _marker(page, 50, 100, "a")
    page.insert_text((63, 100), "\u221a2", fontsize=10, fontfile=_ARIAL, fontname="F0")
    _marker(page, 50, 115, "b")
    page.insert_text((63, 115), "plain", fontsize=10)
    _marker(page, 50, 130, "c")
    page.insert_text((63, 130), "plain2", fontsize=10)
    _marker(page, 50, 145, "d")
    page.insert_text((63, 145), "plain3", fontsize=10)
    result = pmet.scan_mock_exam_pdf(_pdf_bytes(doc), "Synthetic Math 5")
    q = _one_question(result)
    check("not flagged", not q["needs_review"], q)
    check('option A reconstructed as "\\sqrt{2}"', q["option_a"].replace(" ", "") == "$\\sqrt{2}$", q["option_a"])


def test_root_with_parenthesized_radicand_across_spans():
    print("\n# A root's own radicand is a parenthesized group spanning SEVERAL separate spans")
    doc = fitz.open()
    page = _new_page(doc)
    _header(page, "Physics", 2)
    page.insert_text((50, 80), "6. Root-of-a-sum question?", fontsize=10)
    _marker(page, 50, 100, "a")
    page.insert_text((63, 100), "\u221a", fontsize=10, fontfile=_ARIAL, fontname="F0")
    page.insert_text((70, 100), "(", fontsize=10)
    page.insert_text((75, 100), "x", fontsize=10)
    page.insert_text((81, 100), "+", fontsize=10)
    page.insert_text((88, 100), "1", fontsize=10)
    page.insert_text((94, 100), ")", fontsize=10)
    _marker(page, 50, 115, "b")
    page.insert_text((63, 115), "plain", fontsize=10)
    _marker(page, 50, 130, "c")
    page.insert_text((63, 130), "plain2", fontsize=10)
    _marker(page, 50, 145, "d")
    page.insert_text((63, 145), "plain3", fontsize=10)
    result = pmet.scan_mock_exam_pdf(_pdf_bytes(doc), "Synthetic Math 6")
    q = _one_question(result)
    check("not flagged", not q["needs_review"], q)
    check('option A reconstructed as "\\sqrt{x+1}" (outer parens stripped)',
          q["option_a"].replace(" ", "") == "$\\sqrt{x+1}$", q["option_a"])


def test_bare_root_with_nothing_after_it_stays_literal():
    print("\n# A root with NOTHING immediately after it is never guessed -- stays literal, flagged by the existing gates")
    doc = fitz.open()
    page = _new_page(doc)
    _header(page, "Physics", 2)
    page.insert_text((50, 80), "7. Dangling root question?", fontsize=10)
    _marker(page, 50, 100, "a")
    page.insert_text((63, 100), "\u221a", fontsize=10, fontfile=_ARIAL, fontname="F0")
    page.insert_text((100, 100), "unrelated far-away text", fontsize=10)  # too far to be the radicand
    _marker(page, 50, 115, "b")
    page.insert_text((63, 115), "plain", fontsize=10)
    _marker(page, 50, 130, "c")
    page.insert_text((63, 130), "plain2", fontsize=10)
    _marker(page, 50, 145, "d")
    page.insert_text((63, 145), "plain3", fontsize=10)
    result = pmet.scan_mock_exam_pdf(_pdf_bytes(doc), "Synthetic Math 7")
    q = _one_question(result)
    check('the bare root glyph is still present as literal text (never silently dropped)',
          "\u221a" in q["option_a"], q["option_a"])
    check('never guessed into "\\sqrt{...}"', "\\sqrt" not in q["option_a"], q["option_a"])


# ---------------------------------------------------------------- vector arrow

def test_vector_arrow_from_mt_extra_font():
    print("\n# A narrow glyph in the real 'MT Extra' accent font, positioned over a base letter, -> \\vec{}")
    doc = fitz.open()
    page = _new_page(doc)
    _header(page, "Physics", 2)
    page.insert_text((50, 80), "8. Vector question?", fontsize=10)
    _marker(page, 50, 100, "a")
    # The arrow sits directly above the base letter "r", same shape confirmed live on the real
    # sample (a narrow MT Extra glyph a few points above its own base letter, same x-center).
    page.insert_text((63, 92), "r", fontsize=8, fontfile=_MT_EXTRA, fontname="F2")
    page.insert_text((63, 100), "r", fontsize=10, fontfile=_ARIAL, fontname="F0")
    _marker(page, 50, 115, "b")
    page.insert_text((63, 115), "plain", fontsize=10)
    _marker(page, 50, 130, "c")
    page.insert_text((63, 130), "plain2", fontsize=10)
    _marker(page, 50, 145, "d")
    page.insert_text((63, 145), "plain3", fontsize=10)
    result = pmet.scan_mock_exam_pdf(_pdf_bytes(doc), "Synthetic Math 8")
    q = _one_question(result)
    check("not flagged", not q["needs_review"], q)
    check('option A reconstructed as "\\vec{r}"', q["option_a"].replace(" ", "") == "$\\vec{r}$", q["option_a"])


# ---------------------------------------------------------------- confidence rule

def test_straddling_exponent_falls_back_instead_of_guessing():
    print("\n# requirement 4's confidence rule: a small glyph straddling the midpoint between two")
    print("# inline options is never silently assigned to either one -- build_option_zones bails")
    print("# out (its own documented None return) rather than guess which side it belongs to.")
    print("# Checked directly against build_option_zones, not through needs_review: falling back")
    print("# to the old plain-text path is a DIFFERENT code path with its own, unrelated gates --")
    print("# whether THAT path happens to also flag this particular text is not what this rule is")
    print("# about, and asserting on it would make this test depend on incidental old-path behavior.")
    doc = fitz.open()
    page = _new_page(doc)
    page.insert_text((50, 80), "9. Straddling-exponent question?", fontsize=10)
    # a and b share one row; a's own exponent is deliberately positioned to straddle the a/b
    # midpoint (same confirmed-live shape as a real unit like "W m^-2" overflowing into "b.").
    _marker(page, 50, 100, "a")
    page.insert_text((63, 100), "x", fontsize=10)
    mid_x = (50 + 120) / 2.0  # midpoint between this row's own "a." and "b." markers
    page.insert_text((mid_x - 2, 94), "2", fontsize=6)  # small, straddles mid_x by construction
    _marker(page, 120, 100, "b")
    page.insert_text((133, 100), "y", fontsize=10)
    _marker(page, 50, 115, "c")
    page.insert_text((63, 115), "plain2", fontsize=10)
    _marker(page, 50, 130, "d")
    page.insert_text((63, 130), "plain3", fontsize=10)

    d = page.get_text("dict")
    flat_spans = [s for b in d["blocks"] if b.get("type") == 0
                  for line in b["lines"] for s in line["spans"]]
    question_bbox = (45, 75, 200, 145)
    result = pmm.build_option_zones(flat_spans, question_bbox)
    check("build_option_zones bails out (returns None) instead of guessing which option the "
          "straddling exponent belongs to", result is None, result)
    doc.close()


def main():
    test_superscript_exponent()
    test_subscript()
    test_stacked_fraction_from_drawn_bar()
    test_2x2_grid_with_fraction_options_no_cross_contamination()
    test_root_glued_to_its_own_span()
    test_root_with_parenthesized_radicand_across_spans()
    test_bare_root_with_nothing_after_it_stays_literal()
    test_vector_arrow_from_mt_extra_font()
    test_straddling_exponent_falls_back_instead_of_guessing()

    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
