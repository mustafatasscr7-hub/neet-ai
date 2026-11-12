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
  * modulus/magnitude bars: "|x|" glued into one span, "|a + b + c|", a paired bar inside one
    option of an inline row (never bleeding into its neighbor), a vector norm's own double bar
    "||v||", and a bar with no character at all (two drawn vertical strokes, the same "no reliable
    glyph" shape this module's own fraction bar already handles);
  * the modulus-bar confidence rule, as fixed after a real regression (2026-11-11): a lone bar-like
    mark with no possible partner anywhere nearby is never flagged and never acted on (the fix for
    that regression - see process_mock_exam_math's own "Regression" comment), while a genuine
    multi-mark row that still cannot pair up cleanly is flagged needs_review (reason names
    "modulus bar") with the bar glyphs still literally present, never guessed into an expression;
    a thin filled rectangle (a table's own cell border/shading) and a drawn stroke inside an
    already-detected diagram region are both excluded before they are ever treated as a candidate
    bar at all, regardless of how a real bar's own width/height bounds would otherwise match them;
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


# ---------------------------------------------------------------- silent-error fixes (2026-11-13)

def test_fraction_bar_wider_than_narrow_zone_reclaimed():
    print("\n# A 4-inline-options row: option a's own fraction bar pokes past its own narrow "
          "zone's marker-midpoint boundary, and the denominator digit's own center naturally "
          "falls just past that SAME boundary into option b's own zone -- both the bar and the "
          "digit must resolve back to option a, confirmed live MockTest1.pdf Physics Q6/Q10.")
    doc = fitz.open()
    page = _new_page(doc)
    _header(page, "Physics", 2)
    page.insert_text((50, 80), "6. Narrow-zone fraction question?", fontsize=10)
    _marker(page, 50, 100, "a")
    page.insert_text((70, 98), "p", fontsize=10)
    page.draw_line((68, 102), (90, 102))          # bar wider than zone a's own narrow rectangle
    page.insert_text((72, 112), "4", fontsize=10)  # denominator -- its own center sits past the
    _marker(page, 95, 100, "b")                    # zone a/b boundary (midpoint of 50 and 95)
    page.insert_text((110, 100), "q", fontsize=10)
    _marker(page, 160, 100, "c")
    page.insert_text((175, 100), "r", fontsize=10)
    _marker(page, 220, 100, "d")
    page.insert_text((235, 100), "s", fontsize=10)
    result = pmet.scan_mock_exam_pdf(_pdf_bytes(doc), "Synthetic Math 5")
    q = _one_question(result)
    check("not flagged (bar and digit both correctly reclaimed)", not q["needs_review"], q)
    check('option A reconstructed as "\\frac{p}{4}"', q["option_a"].replace(" ", "") == "$\\frac{p}{4}$", q["option_a"])
    check('option B untouched (no stray "4" of its own)', q["option_b"].strip() == "q", q["option_b"])


def test_multiline_option_uses_its_own_row_baseline():
    print("\n# A multi-line option: a LATER physical line's own real exponent must classify "
          "against THAT line's own baseline, not an earlier, longer line's -- confirmed live "
          "MockTest1.pdf Physics Q18 (a stem's own later-line superscript read as a subscript "
          "against the zone-wide baseline, which an EARLIER, longer line dominated).")
    doc = fitz.open()
    page = _new_page(doc)
    _header(page, "Physics", 2)
    page.insert_text((50, 80), "7. Multi-line option question?", fontsize=10)
    _marker(page, 50, 100, "a")
    page.insert_text((70, 100), "first", fontsize=10)
    page.insert_text((100, 100), "line", fontsize=10)
    page.insert_text((70, 120), "x", fontsize=10)   # a second physical line, own baseline
    page.insert_text((76, 117), "2", fontsize=6)    # raised relative to THIS line, not the first
    _marker(page, 50, 140, "b")
    page.insert_text((70, 140), "other", fontsize=10)
    _marker(page, 50, 155, "c")
    page.insert_text((70, 155), "other2", fontsize=10)
    _marker(page, 50, 170, "d")
    page.insert_text((70, 170), "other3", fontsize=10)
    result = pmet.scan_mock_exam_pdf(_pdf_bytes(doc), "Synthetic Math 6")
    q = _one_question(result)
    check("not flagged", not q["needs_review"], q)
    check('option A\'s second line reads "x^{2}", not "x_{2}"',
          "^{2}" in q["option_a"].replace(" ", "") and "_{2}" not in q["option_a"].replace(" ", ""),
          q["option_a"])


def test_midsentence_fragment_not_absorbed_by_preceding_question():
    print("\n# A mid-sentence fragment's own bbox sits a couple points ABOVE the question-start "
          "block it actually belongs to -- a naive y-only sort would process it first and leak "
          "it into the PRECEDING question instead -- confirmed live MockTest1.pdf Physics "
          "Q29/Q30 (a degree-symbol value embedded mid-sentence).")
    doc = fitz.open()
    page = _new_page(doc)
    _header(page, "Physics", 2)
    page.insert_text((50, 80), "1. First question?", fontsize=10)
    _marker(page, 50, 94, "a")
    page.insert_text((63, 94), "a1", fontsize=10)
    _marker(page, 50, 108, "b")
    page.insert_text((63, 108), "b1", fontsize=10)
    _marker(page, 50, 122, "c")
    page.insert_text((63, 122), "c1", fontsize=10)
    _marker(page, 50, 136, "d")
    page.insert_text((63, 136), "d1", fontsize=10)
    page.insert_text((200, 158), "MIDVALUE", fontsize=9)
    page.insert_text((50, 160), "2. Second question with embedded value here:", fontsize=10)
    _marker(page, 50, 174, "a")
    page.insert_text((63, 174), "a2", fontsize=10)
    _marker(page, 50, 188, "b")
    page.insert_text((63, 188), "b2", fontsize=10)
    _marker(page, 50, 202, "c")
    page.insert_text((63, 202), "c2", fontsize=10)
    _marker(page, 50, 216, "d")
    page.insert_text((63, 216), "d2", fontsize=10)
    result = pmet.scan_mock_exam_pdf(_pdf_bytes(doc), "Synthetic Math 7")
    physics = [q for q in result["questions"] if q["subject"] == "Physics"]
    q1 = next(q for q in physics if q["printed_number"] == 1)
    q2 = next(q for q in physics if q["printed_number"] == 2)
    q1_fields = [q1["question"]] + [q1[f"option_{l}"] for l in "abcd"]
    check("question 1 not contaminated by question 2's own mid-sentence fragment",
          all("MIDVALUE" not in (f or "") for f in q1_fields), q1)
    check("question 2's own stem contains the fragment", "MIDVALUE" in q2["question"], q2["question"])


def test_fraction_bar_numerator_claimed_by_two_zones_flags_ambiguous():
    print("\n# A fraction bar wide enough that its own numerator-reach window catches REAL "
          "content from two different option zones at once -- a glyph genuinely claimed by two "
          "zones -- must never guess which one owns it; flagged instead (requirement 2).")
    doc = fitz.open()
    page = _new_page(doc)
    _header(page, "Physics", 2)
    page.insert_text((50, 80), "10. Ambiguous-bar question?", fontsize=10)
    _marker(page, 50, 100, "a")
    page.insert_text((70, 98), "p", fontsize=10)
    _marker(page, 150, 100, "b")
    page.insert_text((160, 98), "q", fontsize=10)
    page.draw_line((65, 102), (175, 102))  # one bar spanning across BOTH zones' own numerators
    page.insert_text((70, 112), "4", fontsize=10)
    page.insert_text((160, 112), "5", fontsize=10)
    _marker(page, 230, 100, "c")
    page.insert_text((244, 100), "r", fontsize=10)
    _marker(page, 280, 100, "d")
    page.insert_text((294, 100), "s", fontsize=10)
    result = pmet.scan_mock_exam_pdf(_pdf_bytes(doc), "Synthetic Math 8")
    q = _one_question(result)
    check("flagged", q["needs_review"], q)
    check("reason names the ambiguous fraction geometry",
          "claimed by more than one option" in (q["review_reason"] or ""), q["review_reason"])


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


# ---------------------------------------------------------------- modulus/magnitude bars

def test_modulus_single_variable_glued_in_one_span():
    print("\n# |x| - both bars glued into the SAME span as their own content (one PyMuPDF run,")
    print("# confirmed live: insert_text's own single call keeps a uniform-format run as one span)")
    doc = fitz.open()
    page = _new_page(doc)
    _header(page, "Physics", 2)
    page.insert_text((50, 80), "10. Modulus question?", fontsize=10)
    _marker(page, 50, 100, "a")
    page.insert_text((63, 100), "|x|", fontsize=10)
    _marker(page, 50, 115, "b")
    page.insert_text((63, 115), "plain", fontsize=10)
    _marker(page, 50, 130, "c")
    page.insert_text((63, 130), "plain2", fontsize=10)
    _marker(page, 50, 145, "d")
    page.insert_text((63, 145), "plain3", fontsize=10)
    result = pmet.scan_mock_exam_pdf(_pdf_bytes(doc), "Synthetic Math 10")
    q = _one_question(result)
    check("not flagged", not q["needs_review"], q)
    check('option A reconstructed as "|x|"', q["option_a"].strip() == "|x|", q["option_a"])


def test_modulus_multi_token_expression():
    print("\n# |a + b + c| - a longer expression between the pair, same baseline")
    doc = fitz.open()
    page = _new_page(doc)
    _header(page, "Physics", 2)
    page.insert_text((50, 80), "11. Modulus-of-a-sum question?", fontsize=10)
    _marker(page, 50, 100, "a")
    page.insert_text((63, 100), "|a + b + c|", fontsize=10)
    _marker(page, 50, 115, "b")
    page.insert_text((63, 115), "plain", fontsize=10)
    _marker(page, 50, 130, "c")
    page.insert_text((63, 130), "plain2", fontsize=10)
    _marker(page, 50, 145, "d")
    page.insert_text((63, 145), "plain3", fontsize=10)
    result = pmet.scan_mock_exam_pdf(_pdf_bytes(doc), "Synthetic Math 11")
    q = _one_question(result)
    check("not flagged", not q["needs_review"], q)
    check('option A reconstructed as "|a + b + c|"', q["option_a"].strip() == "|a + b + c|", q["option_a"])


def test_modulus_bar_inside_one_option_of_an_inline_row():
    print("\n# A paired bar inside ONE option of a 2-per-row inline layout - reconstructs cleanly")
    print("# and never bleeds into its neighbor's own zone (requirement: same option-zone geometry)")
    doc = fitz.open()
    page = _new_page(doc)
    _header(page, "Physics", 2)
    page.insert_text((50, 80), "12. Inline-row modulus question?", fontsize=10)
    _marker(page, 50, 100, "a")
    page.insert_text((63, 100), "|p|", fontsize=10)
    _marker(page, 120, 100, "b")
    page.insert_text((133, 100), "q", fontsize=10)
    _marker(page, 50, 115, "c")
    page.insert_text((63, 115), "plain2", fontsize=10)
    _marker(page, 50, 130, "d")
    page.insert_text((63, 130), "plain3", fontsize=10)
    result = pmet.scan_mock_exam_pdf(_pdf_bytes(doc), "Synthetic Math 12")
    q = _one_question(result)
    check("not flagged", not q["needs_review"], q)
    check('option A reconstructed as "|p|"', q["option_a"].strip() == "|p|", q["option_a"])
    check("option B untouched by option A's own bars", q["option_b"].strip() == "q", q["option_b"])


def test_double_bar_vector_norm():
    print('\n# ||v|| - a vector norm\'s own DOUBLE bar, rebuilt as "||" not two separate singles')
    doc = fitz.open()
    page = _new_page(doc)
    _header(page, "Physics", 2)
    page.insert_text((50, 80), "14. Vector-norm question?", fontsize=10)
    _marker(page, 50, 100, "a")
    page.insert_text((63, 100), "||v||", fontsize=10)
    _marker(page, 50, 115, "b")
    page.insert_text((63, 115), "plain", fontsize=10)
    _marker(page, 50, 130, "c")
    page.insert_text((63, 130), "plain2", fontsize=10)
    _marker(page, 50, 145, "d")
    page.insert_text((63, 145), "plain3", fontsize=10)
    result = pmet.scan_mock_exam_pdf(_pdf_bytes(doc), "Synthetic Math 14")
    q = _one_question(result)
    check("not flagged", not q["needs_review"], q)
    check('option A reconstructed as "||v||"', q["option_a"].strip() == "||v||", q["option_a"])


def test_drawn_vertical_stroke_as_modulus_bar():
    print("\n# A modulus bar with NO character at all - just two drawn vertical strokes, the same")
    print("# \"no reliable glyph\" shape this module's own fraction bar already handles (a vector-")
    print("# drawn line with no '/' character anywhere), transposed to the vertical axis.")
    doc = fitz.open()
    page = _new_page(doc)
    _header(page, "Physics", 2)
    page.insert_text((50, 80), "15. Drawn-stroke modulus question?", fontsize=10)
    _marker(page, 50, 100, "a")
    page.draw_line((66, 92), (66, 104))   # opening bar - a drawn stroke, no character
    page.insert_text((70, 100), "v", fontsize=10)
    page.draw_line((80, 92), (80, 104))   # closing bar
    _marker(page, 50, 115, "b")
    page.insert_text((63, 115), "plain", fontsize=10)
    _marker(page, 50, 130, "c")
    page.insert_text((63, 130), "plain2", fontsize=10)
    _marker(page, 50, 145, "d")
    page.insert_text((63, 145), "plain3", fontsize=10)
    result = pmet.scan_mock_exam_pdf(_pdf_bytes(doc), "Synthetic Math 15")
    q = _one_question(result)
    check("not flagged", not q["needs_review"], q)
    check('option A reconstructed as "|v|"', q["option_a"].strip() == "|v|", q["option_a"])


def test_lone_stray_bar_never_flagged():
    print("\n# Regression fix (2026-11-11): a LONE bar-like mark with no possible partner anywhere")
    print("# nearby - e.g. a chemistry double-bond character caught in running option text, the")
    print("# real false positive found on MockTest1.pdf - is never flagged and never acted on. It")
    print("# stays exactly as literal text, same as any other glyph this module does not")
    print("# confidently reconstruct (requirement: never bail a zone for a stray stroke elsewhere).")
    doc = fitz.open()
    page = _new_page(doc)
    _header(page, "Physics", 2)
    page.insert_text((50, 80), "13. Lone-bar question?", fontsize=10)
    _marker(page, 50, 100, "a")
    page.insert_text((63, 100), "|x", fontsize=10)
    _marker(page, 50, 115, "b")
    page.insert_text((63, 115), "plain", fontsize=10)
    _marker(page, 50, 130, "c")
    page.insert_text((63, 130), "plain2", fontsize=10)
    _marker(page, 50, 145, "d")
    page.insert_text((63, 145), "plain3", fontsize=10)
    result = pmet.scan_mock_exam_pdf(_pdf_bytes(doc), "Synthetic Math 13")
    q = _one_question(result)
    check("NOT flagged (a lone mark is never ambiguous, never acted on)", not q["needs_review"], q)
    check("the bar glyph is still literally present, completely untouched",
          q["option_a"].strip() == "|x", q["option_a"])


def test_two_bars_with_nothing_real_between_flags_as_ambiguous():
    print("\n# Two bar-like marks DO sit on the same baseline - a genuine candidate pair - but with")
    print("# nothing real between them (the gap a chemistry double bond's own two near-parallel")
    print("# strokes leave behind). This is the \"ambiguous\" shape that must still flag needs_review")
    print("# (reason names \"modulus bar\"), never guessed into an empty \"||\" expression.")
    doc = fitz.open()
    page = _new_page(doc)
    _header(page, "Physics", 2)
    page.insert_text((50, 80), "16. Empty-between-bars question?", fontsize=10)
    _marker(page, 50, 100, "a")
    page.insert_text((63, 100), "|", fontsize=10)
    page.insert_text((80, 100), "|", fontsize=10)  # same row, far enough apart not to merge,
    _marker(page, 50, 115, "b")                     # but nothing real sits between the two bars
    page.insert_text((63, 115), "plain", fontsize=10)
    _marker(page, 50, 130, "c")
    page.insert_text((63, 130), "plain2", fontsize=10)
    _marker(page, 50, 145, "d")
    page.insert_text((63, 145), "plain3", fontsize=10)
    result = pmet.scan_mock_exam_pdf(_pdf_bytes(doc), "Synthetic Math 16")
    q = _one_question(result)
    check("flagged needs_review (a genuine pair that still can't be trusted)", q["needs_review"], q)
    check('reason names "modulus bar"', "modulus bar" in (q["review_reason"] or ""), q["review_reason"])
    check("both bar glyphs are still literally present (never silently dropped/guessed)",
          q["option_a"].count("|") == 2, q["option_a"])


def test_filled_rectangle_not_treated_as_modulus_bar():
    print("\n# Regression fix (2026-11-11): a THIN FILLED RECTANGLE (a Match-the-Following table's")
    print("# own cell border/shading on the real sample, PyMuPDF drawing type \"f\" with a \"re\"")
    print("# item) passes every width/height bound a real bar's drawn stroke does, but is a wholly")
    print("# different drawing primitive - never treated as a bar candidate at all, even though a")
    print("# real pair sitting exactly here, around exactly this content, would have reconstructed.")
    doc = fitz.open()
    page = _new_page(doc)
    _header(page, "Physics", 2)
    page.insert_text((50, 80), "17. Filled-rectangle question?", fontsize=10)
    _marker(page, 50, 100, "a")
    page.insert_text((63, 100), "x", fontsize=10)
    page.draw_rect(fitz.Rect(72, 92, 72.4, 104), color=None, fill=(0, 0, 0))
    page.insert_text((76, 100), "y", fontsize=10)
    page.draw_rect(fitz.Rect(84, 92, 84.4, 104), color=None, fill=(0, 0, 0))
    page.insert_text((88, 100), "z", fontsize=10)
    _marker(page, 50, 115, "b")
    page.insert_text((63, 115), "plain", fontsize=10)
    _marker(page, 50, 130, "c")
    page.insert_text((63, 130), "plain2", fontsize=10)
    _marker(page, 50, 145, "d")
    page.insert_text((63, 145), "plain3", fontsize=10)
    result = pmet.scan_mock_exam_pdf(_pdf_bytes(doc), "Synthetic Math 17")
    q = _one_question(result)
    check("not flagged", not q["needs_review"], q)
    check("no bar characters inserted around the filled rectangles",
          "|" not in q["option_a"], q["option_a"])
    check('option A keeps its own real content ("x y z")',
          q["option_a"].replace(" ", "") == "xyz", q["option_a"])


def test_drawn_stroke_inside_diagram_excluded():
    print("\n# Regression fix (2026-11-11): a drawn stroke that sits inside an already-detected")
    print("# diagram region (a circuit's own battery symbol, a structural formula's own ring bond,")
    print("# confirmed live on the real sample) is excluded before it is ever treated as a bar")
    print("# candidate, even though it is a genuine stroke that would otherwise qualify.")
    doc = fitz.open()
    page = _new_page(doc)
    _header(page, "Physics", 2)
    page.insert_text((50, 80), "18. Diagram-stroke question?", fontsize=10)
    _marker(page, 50, 100, "a")
    page.insert_text((63, 100), "x", fontsize=10)
    # A small synthetic raster image -- _get_diagram_bboxes's own raster pass needs no YOLO model
    # (unlike its vector pass), so this is the deterministic way to mark a region as "a diagram"
    # in a test, matching _MIN_DIAGRAM_DIM's own 20pt floor.
    pix = fitz.Pixmap(fitz.csGRAY, (0, 0, 40, 40), False)
    pix.set_rect(pix.irect, (200,))
    page.insert_image(fitz.Rect(100, 85, 140, 125), stream=pix.tobytes("png"))
    page.draw_line((110, 98), (110, 110))
    page.insert_text((114, 106), "y", fontsize=10)
    page.draw_line((124, 98), (124, 110))
    page.insert_text((128, 100), "z", fontsize=10)
    _marker(page, 50, 115, "b")
    page.insert_text((63, 115), "plain", fontsize=10)
    _marker(page, 50, 130, "c")
    page.insert_text((63, 130), "plain2", fontsize=10)
    _marker(page, 50, 145, "d")
    page.insert_text((63, 145), "plain3", fontsize=10)
    result = pmet.scan_mock_exam_pdf(_pdf_bytes(doc), "Synthetic Math 18")
    q = _one_question(result)
    check("not flagged", not q["needs_review"], q)
    check("no bar characters inserted around the diagram's own strokes",
          "|" not in q["option_a"], q["option_a"])


# ---------------------------------------------------------------- confidence rule

def test_straddling_exponent_resolved_by_marker_sequence():
    print("\n# Reconstruct-before-splitting requirement: a small glyph (superscript-sized) whose")
    print("# own bbox straddles the midpoint between two options sharing one row is no longer a")
    print("# reason to bail -- assign_spans_to_zones resolves it by marker-sequence adjacency")
    print("# instead (see that function's own docstring), the same real shape as the live sample's")
    print("# own Q1 (several options' own unit exponents packed on one shared line).")
    doc = fitz.open()
    page = _new_page(doc)
    _header(page, "Physics", 2)
    page.insert_text((50, 80), "9. Straddling-exponent question?", fontsize=10)
    # a and b share one row; a's own exponent is deliberately positioned to straddle the a/b
    # midpoint (same confirmed-live shape as a real unit like "W m^-2" overflowing into "b.").
    _marker(page, 50, 100, "a")
    page.insert_text((75, 100), "x", fontsize=10)
    mid_x = (50 + 120) / 2.0  # midpoint between this row's own "a." and "b." markers
    page.insert_text((mid_x - 2, 94), "2", fontsize=6)  # small, straddles mid_x by construction
    _marker(page, 120, 100, "b")
    page.insert_text((133, 100), "y", fontsize=10)
    _marker(page, 50, 115, "c")
    page.insert_text((63, 115), "plain2", fontsize=10)
    _marker(page, 50, 130, "d")
    page.insert_text((63, 130), "plain3", fontsize=10)
    result = pmet.scan_mock_exam_pdf(_pdf_bytes(doc), "Synthetic Math 9")
    q = _one_question(result)
    check("not flagged", not q["needs_review"], q)
    check('option A keeps both its own base "x" and its own exponent ("^{2}"), not severed to option B',
          "x" in q["option_a"] and "^{2}" in q["option_a"], q["option_a"])
    check('option B stays just "y", never contaminated by option A\'s own exponent',
          q["option_b"].replace(" ", "") == "y", q["option_b"])


def test_normal_sized_content_past_marker_midpoint_still_resolved():
    print("\n# The SAME requirement, for BASE-sized (not just small/superscript) content: a row's")
    print("# own marker-midpoint rectangle can fall INSIDE one option's own real content whenever")
    print("# that option's own content is simply wider than half its own gap to the next marker")
    print("# (confirmed live, the real sample's own Q6 -- option A's own base-sized denominator")
    print("# letter sits past the straight a/b midpoint, purely because option A's own content is")
    print("# short and option B's marker follows close behind). Marker-sequence assignment has no")
    print("# such failure mode: whatever comes after marker A and before marker B stays correct no")
    print("# matter how far right it runs.")
    doc = fitz.open()
    page = _new_page(doc)
    _header(page, "Physics", 2)
    page.insert_text((50, 80), "10. Wide-option question?", fontsize=10)
    # "a." and "b." sit close together, but option A's own content ("wideword") is wide enough to
    # run past the raw a/b midpoint -- by construction, well into where a naive rectangle split
    # would hand it to option B instead.
    _marker(page, 50, 100, "a")
    page.insert_text((63, 100), "wideword", fontsize=10)
    _marker(page, 130, 100, "b")
    page.insert_text((143, 100), "y", fontsize=10)
    _marker(page, 50, 115, "c")
    page.insert_text((63, 115), "plain2", fontsize=10)
    _marker(page, 50, 130, "d")
    page.insert_text((63, 130), "plain3", fontsize=10)
    result = pmet.scan_mock_exam_pdf(_pdf_bytes(doc), "Synthetic Math 10")
    q = _one_question(result)
    check("not flagged", not q["needs_review"], q)
    check('option A keeps its own full "wideword", not severed at the marker midpoint',
          q["option_a"] == "wideword", q["option_a"])
    check('option B stays just "y"', q["option_b"] == "y", q["option_b"])


# ---------------------------------------------------------------- font-aware symbol table

def _span(text, x0, y0, x1=None, y1=None, font="Symbol", size=10.0):
    """A hand-built span dict in the same shape PyMuPDF's own get_text("dict") spans use --
    process_mock_exam_math._collapse_symbol_font_glyphs is a pure function over this shape, and a
    Private Use Area codepoint (this section's own font-aware table) has no real glyph in any
    ordinary installed font, so round-tripping one through a real PDF's own text rendering is not
    reliable on an arbitrary machine (confirmed live while writing these tests: a font with no
    glyph for the codepoint silently substitutes something else instead). These tests build the
    exact span shape directly instead, the same thing a real PDF's own text layer would already
    have handed this function."""
    if x1 is None:
        x1 = x0 + max(1.0, size * 0.6 * len(text))
    if y1 is None:
        y1 = y0 + size
    return {"text": text, "font": font, "size": size, "bbox": (x0, y0, x1, y1), "origin": (x0, y1)}


def test_new_symbol_commands_mapped():
    print("\n# Font-aware symbol table requirement: the additional Unicode symbols confirmed live")
    print("# on the real sample (Rightarrow, therefore, angle, cdot, prime, and the Symbol font's")
    print("# own U+2206/U+2126 lookalikes for Delta/Omega) map to their own LaTeX commands, the")
    print("# same font-independent mechanism the existing Greek letters already use.")
    text, replaced = pmm._sub_symbols("⇒ ∴ ∠ ⋅ ′ ∆ Ω")
    check("replaced something", replaced, text)
    for expect in ("\\Rightarrow", "\\therefore", "\\angle", "\\cdot", "\\prime", "\\Delta", "\\Omega"):
        check(f'"{expect}" present', expect in text, text)


def test_symbol_font_bracket_pieces_collapse_to_parens():
    print("\n# Font-aware symbol table requirement: a SHORT tall-bracket construct's own pieces,")
    print("# glued into one span's own text (confirmed live shape), collapse to a single literal")
    print("# '(' / ')' each -- never one paren character per piece glyph.")
    left_pieces = chr(0xF8EB) + chr(0xF8EC) + chr(0xF8ED)   # top/ext/bottom of a tall "("
    right_pieces = chr(0xF8F6) + chr(0xF8F7) + chr(0xF8F8)  # top/ext/bottom of a tall ")"
    spans = [
        _span("x", 60, 100, font="TimesNewRoman"),
        _span(left_pieces, 70, 100),
        _span("y", 85, 100, font="TimesNewRoman"),
        _span(right_pieces, 95, 100),
    ]
    new_spans, issue = pmm._collapse_symbol_font_glyphs(spans)
    check("no issue reported", issue is None, issue)
    joined = "".join(s["text"] for s in new_spans)
    check('collapsed to "x(y)"', joined == "x(y)", repr(joined))


def test_symbol_font_arrow_shaft_dropped_before_real_arrowhead():
    print("\n# Font-aware symbol table requirement: the arrow-shaft-extender glyph contributes")
    print("# nothing once it leads into a real arrowhead character -- the arrowhead alone already")
    print("# carries the whole arrow's own meaning.")
    shaft = chr(0xF8E7) * 3
    spans = [
        _span("A", 50, 100, font="TimesNewRoman"),
        _span(shaft, 60, 100),
        _span(chr(0x2192), 80, 100),
        _span("B", 90, 100, font="TimesNewRoman"),
    ]
    new_spans, issue = pmm._collapse_symbol_font_glyphs(spans)
    check("no issue reported", issue is None, issue)
    joined = "".join(s["text"] for s in new_spans)
    check('shaft removed, arrowhead kept ("A->B")', joined == "A" + chr(0x2192) + "B", repr(joined))


def test_symbol_font_arrow_shaft_with_no_arrowhead_flags_question():
    print("\n# Font-aware symbol table requirement: an arrow-shaft run that does NOT lead into a")
    print("# real arrowhead anywhere on its own row is an unknown shape, not guessed into an arrow")
    print("# that was never really there.")
    spans = [
        _span("A", 50, 100, font="TimesNewRoman"),
        _span(chr(0xF8E7) * 3, 60, 100),
        _span("B", 90, 100, font="TimesNewRoman"),
    ]
    new_spans, issue = pmm._collapse_symbol_font_glyphs(spans)
    check("an issue is reported", issue is not None, issue)
    check("U+F8E7 named in the issue", "F8E7" in (issue or ""), issue)
    joined = "".join(s["text"] for s in new_spans)
    check("the shaft glyphs are still present literally (never silently dropped)",
          chr(0xF8E7) in joined, repr(joined))


def test_unrecognized_symbol_font_glyph_flags_question():
    print("\n# Font-aware symbol table requirement: a Symbol-font Private Use Area character this")
    print("# table does NOT recognize is left as literal text AND reported as an issue -- never")
    print('# silently reaching the output as an un-renderable character ("unknown glyphs still')
    print('# flag the question").')
    spans = [
        _span("before", 50, 100, font="TimesNewRoman"),
        _span(chr(0xF8FC), 90, 100),  # not in the recognized piece/shaft sets
        _span("after", 100, 100, font="TimesNewRoman"),
    ]
    new_spans, issue = pmm._collapse_symbol_font_glyphs(spans)
    check("an issue is reported", issue is not None, issue)
    check("U+F8FC named in the issue", "F8FC" in (issue or ""), issue)
    joined = "".join(s["text"] for s in new_spans)
    check("the unknown glyph is still present literally (never silently dropped)",
          chr(0xF8FC) in joined, repr(joined))


def test_tall_bracket_across_multiple_spans_flags_instead_of_guessing():
    print("\n# Font-aware symbol table requirement: a TALL bracket construct whose own pieces sit on")
    print("# DIFFERENT baselines (hence different spans, confirmed live -- see this module's own")
    print("# docstring, real sample Q40) is never collapsed by guessing how many '(' it should")
    print("# become -- it is flagged instead, same confidence rule as everywhere else in this")
    print("# module.")
    spans = [
        _span(chr(0xF8EB), 50, 90),   # top piece, its own row
        _span(chr(0xF8EC), 50, 102),  # extension piece, a DIFFERENT row
        _span(chr(0xF8ED), 50, 114),  # bottom piece, a THIRD row
    ]
    new_spans, issue = pmm._collapse_symbol_font_glyphs(spans)
    check("flagged rather than collapsed into multiple '('", issue is not None, issue)
    check('reason names "multiple lines"', "multiple lines" in (issue or ""), issue)


# ---------------------------------------------------------------- silent-error (order-mismatch) check

def test_order_mismatch_detects_scrambled_result():
    print("\n# Silent-error check (requirement 3): an order-sensitive comparison catches a result")
    print("# whose own final text is reordered relative to the page's own independent reading")
    print("# order, even though every individual WORD is still present (a plain word-overlap gate,")
    print("# like process_mock_exam_vision.py's own, would miss this entirely).")
    correct = pmet._content_tokens("alpha beta gamma delta epsilon zeta eta theta")
    scrambled = pmet._content_tokens("theta eta zeta epsilon delta gamma beta alpha")
    import difflib
    ratio = difflib.SequenceMatcher(None, correct, scrambled).ratio()
    check("a fully-reversed sequence scores well below the 0.80 cutoff",
          ratio < pmet._ORDER_MISMATCH_MIN_RATIO, ratio)
    same_order = pmet._content_tokens("alpha beta gamma delta epsilon zeta eta theta")
    ratio_same = difflib.SequenceMatcher(None, correct, same_order).ratio()
    check("the SAME order scores a perfect match", ratio_same == 1.0, ratio_same)


def test_order_mismatch_ignores_math_symbols_and_whitespace():
    print("\n# Silent-error check (requirement 3): LaTeX markup and math symbols are stripped before")
    print("# comparing -- two representations of the SAME content that merely mark it up")
    print("# differently (one side wraps a value in \\frac{}{}, the other leaves it as plain")
    print('# adjacent tokens) must still read as "the same order", not a false mismatch.')
    latex_form = pmet._content_tokens("Let $\\vec{a}$ be a vector of length $\\frac{3}{4}$")
    plain_form = pmet._content_tokens("Let a be a vector of length 3 4")
    check("both sides reduce to the identical token sequence", latex_form == plain_form,
          (latex_form, plain_form))


def test_order_mismatch_grid_reading_order_not_a_false_positive():
    print("\n# Silent-error check (requirement 3): a 2x2 grid's own TRUE page reading order genuinely")
    print("# reads row-by-row (every row-1 numerator, then every row-1 denominator) rather than")
    print('# "option A\'s own numerator+denominator, then option B\'s" -- a real structural')
    print("# difference from the stem+a+b+c+d concatenation order, not a reconstruction mistake.")
    print("# _question_reading_order_tokens must reuse the SAME zone buckets (not raw whole-page")
    print("# reading order) so this never reads as a false order-mismatch -- covered end-to-end by")
    print("# test_2x2_grid_with_fraction_options_no_cross_contamination's own \"not flagged\" check;")
    print("# this test asserts the mechanism directly.")
    doc = fitz.open()
    page = _new_page(doc)
    _header(page, "Chemistry", 2)
    page.insert_text((50, 80), "4. Grid-with-fractions question?", fontsize=10)
    _marker(page, 50, 100, "a")
    page.insert_text((70, 98), "1", fontsize=10)
    page.draw_line((68, 102), (82, 102))
    page.insert_text((70, 112), "2", fontsize=10)
    _marker(page, 120, 100, "b")
    page.insert_text((140, 98), "3", fontsize=10)
    page.draw_line((138, 102), (152, 102))
    page.insert_text((140, 112), "4", fontsize=10)
    _marker(page, 50, 125, "c")
    page.insert_text((63, 125), "five", fontsize=10)
    _marker(page, 120, 125, "d")
    page.insert_text((133, 125), "sixteen", fontsize=10)
    result = pmet.scan_mock_exam_pdf(_pdf_bytes(doc), "Synthetic Math Order")
    q = _one_question(result, subject="Chemistry")
    check("not flagged (raw page reading order genuinely differs from a-b-c-d order here, "
          "but the bucket-based reference accounts for that)", not q["needs_review"], q)


def main():
    test_superscript_exponent()
    test_subscript()
    test_stacked_fraction_from_drawn_bar()
    test_2x2_grid_with_fraction_options_no_cross_contamination()
    test_fraction_bar_wider_than_narrow_zone_reclaimed()
    test_multiline_option_uses_its_own_row_baseline()
    test_midsentence_fragment_not_absorbed_by_preceding_question()
    test_fraction_bar_numerator_claimed_by_two_zones_flags_ambiguous()
    test_root_glued_to_its_own_span()
    test_root_with_parenthesized_radicand_across_spans()
    test_bare_root_with_nothing_after_it_stays_literal()
    test_vector_arrow_from_mt_extra_font()
    test_modulus_single_variable_glued_in_one_span()
    test_modulus_multi_token_expression()
    test_modulus_bar_inside_one_option_of_an_inline_row()
    test_double_bar_vector_norm()
    test_drawn_vertical_stroke_as_modulus_bar()
    test_lone_stray_bar_never_flagged()
    test_two_bars_with_nothing_real_between_flags_as_ambiguous()
    test_filled_rectangle_not_treated_as_modulus_bar()
    test_drawn_stroke_inside_diagram_excluded()
    test_straddling_exponent_resolved_by_marker_sequence()
    test_normal_sized_content_past_marker_midpoint_still_resolved()
    test_new_symbol_commands_mapped()
    test_symbol_font_bracket_pieces_collapse_to_parens()
    test_symbol_font_arrow_shaft_dropped_before_real_arrowhead()
    test_symbol_font_arrow_shaft_with_no_arrowhead_flags_question()
    test_unrecognized_symbol_font_glyph_flags_question()
    test_tall_bracket_across_multiple_spans_flags_instead_of_guessing()
    test_order_mismatch_detects_scrambled_result()
    test_order_mismatch_ignores_math_symbols_and_whitespace()
    test_order_mismatch_grid_reading_order_not_a_false_positive()

    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
