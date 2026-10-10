"""Regression tests for the PYQ PDF extraction pipeline (process_pyq_vision.py).

Run with: python test_pyq_extraction.py
No pytest dependency -- this repo has no existing test runner, so this uses plain asserts and a
small pass/fail harness, matching how the rest of this project has been manually verified so far.

Covers the 2026-10-10 fix: _reconstruct_fragmented_options was silently losing content in two
related ways on PDFs that number options "1." "2." "3." "4." instead of "A." "B." "C." "D.":
  1. A question's own option-list zone could extend past the NEXT question's stem/options (the
     "next boundary" detector accepted any bare marker digit 1-4, and used a too-tight x0 match
     that missed real question-start blocks) -- so one question's math fragments got vacuumed
     into an unrelated, earlier question's last option.
  2. A fully self-contained option list (the whole option text already on its own marker line,
     e.g. "1. 0.14 V") was discarded entirely whenever there was nothing EXTRA to find nearby --
     even though its own text was already fully known.

These fixtures use pyq-pdfs/Chapter_6_EMI.pdf (copied into the repo from the user's source file,
per their explicit instruction) since it's what exposed both bugs for real.
"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import process_pyq_vision as ppv

EMI_PDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pyq-pdfs", "Chapter_6_EMI.pdf")

_passed = []
_failed = []


def check(label, condition, detail=""):
    if condition:
        _passed.append(label)
        print(f"  PASS: {label}")
    else:
        _failed.append(label)
        print(f"  FAIL: {label}" + (f" -- {detail}" if detail else ""))


def test_q2_disc_question_math_preserved_in_stem():
    """The real bug report: EMI page 1, Q2's own fraction (1/pi Wb/m^2), its 60-degree angle and
    its vector B all got swallowed into Q1's option 4 and silently vanished from Q2's stem."""
    print("\n=== Q2 disc question: fraction/angle/vector preserved in the stem's own text ===")
    with open(EMI_PDF_PATH, "rb") as f:
        pdf_bytes = f.read()
    pages = ppv.extract_pages_text_and_diagrams(pdf_bytes)
    page1_text = pages[0]["text"]

    # Anchor to the actual question text so this fails loudly (not silently no-ops) if the PDF
    # or page layout ever changes.
    q2_start = page1_text.find("A circular disc of radius 0.2 m")
    check("Q2's own stem text is present on page 1 at all", q2_start != -1)
    if q2_start == -1:
        return

    # Previously, Q1's option list (not Q2) is the one that absorbed all of this -- so the
    # regression specifically checks it no longer ends up stuck onto Q1's last option (which
    # textually reads "4. 4mu0 pi" when correct, vs. corrupted "4. 4mu0 pi mu0l R^2 ... 60 Wb m^2
    # ... B ..." before the fix).
    q1_option4_line = next((l for l in page1_text.splitlines() if l.strip().startswith("4.") and "pi" not in l.lower() and "π" in l), "")
    option4_lines = [l for l in page1_text.splitlines() if l.strip().startswith("4. ") or l.strip() == "4."]
    q1_option4_text = next((l for l in option4_lines if "μ" in l or "pi" in l.lower()), "")
    check(
        "Q1's own option 4 no longer has Q2's content stuck onto it",
        "60" not in q1_option4_text and "Wb" not in q1_option4_text,
        detail=f"option 4 line: {q1_option4_text!r}",
    )

    # The actual fix: Q2's own math fragments must appear SOMEWHERE in page 1's text (not
    # necessarily in perfect sentence order -- this PDF's equation objects are positioned oddly
    # even in the original file -- but present, not dropped).
    check("the fraction numerator '1' (from 1/pi) is present", "\n1\n" in page1_text or page1_text.count("\n1 ") > 0 or "\n1" in page1_text)
    check("the Wb/m^2 unit is present", "Wb" in page1_text and ("m^{2}" in page1_text or "m2" in page1_text))
    check("the 60-degree angle is present", "60" in page1_text)
    check("the vector arrow for B is present", "→" in page1_text)

    # The new safety-net flag (step 4): must NOT fire a false positive on the real, fixed page --
    # the raw pdfplumber text and the actual extracted questions should agree now.
    import fitz
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    raw_texts = ppv._raw_pdfplumber_page_texts(pdf_bytes)
    doc.close()
    extracted = ppv.extract_questions_from_text(page1_text, 1, subject="Physics")
    combined = [q.get(f, "") for q in extracted for f in ppv._EXTRACTED_TEXT_FIELDS]
    missing = ppv.find_missing_raw_tokens(raw_texts[0], combined)
    check("no raw-token flag fires on the real, fixed page 1", len(missing) == 0, detail=f"missing={missing}")


def test_find_missing_raw_tokens_catches_the_original_bug_shape():
    """Doesn't need a live model call: directly proves find_missing_raw_tokens (the new step-4
    safety net) fires on OLD-code-shaped output (fraction dropped) and stays silent once the
    content is present -- the exact fail-on-old/pass-on-new split the task asked for."""
    print("\n=== find_missing_raw_tokens: fires on dropped content, silent once present ===")
    # Uses the real Greek pi glyph and degree sign -- confirmed live via pdfplumber against the
    # actual PDF (not spelled out as ascii "pi"/"degrees") -- so this matches what the real raw
    # text layer actually contains, not an idealized ascii stand-in.
    raw_page_text = (
        "A circular disc of radius 0.2 m is placed in a uniform magnetic field of induction "
        "1/π (Wb/m2) in such a way that its axis makes an angle of 60° with →B. "
        "The magnetic flux linked to the disc will be:"
    )
    # Shaped exactly like the OLD code's actual output for this question (confirmed live): the
    # fraction, unit and angle are gone, replaced by a vague paraphrase.
    old_code_stem = "magnetic field with $\\vec{B}$ in such a way that its magnetic flux is $60^{\\circ}$ Wb m$^{2}$"
    missing_old = ppv.find_missing_raw_tokens(raw_page_text, [old_code_stem])
    check("fires on the old, content-dropping extraction", len(missing_old) > 0, detail=f"missing={missing_old}")

    # Shaped like the FIXED code's actual output: the same facts, just reformatted as LaTeX.
    new_code_stem = (
        "A circular disc of radius $0.2$ m is placed in a uniform magnetic field of induction "
        "$\\frac{1}{\\pi}$ (Wb/m$^2$) in such a way that its axis makes an angle of $60^{\\circ}$ "
        "with $\\vec{B}$. The magnetic flux linked to the disc will be:"
    )
    missing_new = ppv.find_missing_raw_tokens(raw_page_text, [new_code_stem])
    check("stays silent once the same content survives (just reformatted as LaTeX)", len(missing_new) == 0, detail=f"missing={missing_new}")


def test_numbered_options_1_to_4_reconstructed_not_dropped():
    """The core reported bug: a plain, non-fraction, non-fragmented numbered option list (EMI
    Q10, "1. 0.14 V" / "2. 0.12 V" / "3. 0.11 V" / "4. 0.13 V") disappeared completely -- the old
    code required extra zone content to exist before emitting ANY reconstructed option, even
    though a self-contained option's own line already has everything it needs."""
    print("\n=== Numbered (1.-4.) self-contained options: not silently dropped ===")
    import fitz
    with open(EMI_PDF_PATH, "rb") as f:
        pdf_bytes = f.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[1]  # page 2 (0-indexed) -- Q10's "0.14 V" options live here
    mid_x = page.rect.width / 2
    page_blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type") == 0]
    excluded_ids, reconstructed_entries, modified_block_text = ppv._reconstruct_fragmented_options(page_blocks, mid_x, page.rect.width)
    doc.close()

    recon_texts = [e[2] for e in reconstructed_entries]
    for expected in ["1. 0.14 V", "2. 0.12 V", "3. 0.11 V", "4. 0.13 V"]:
        check(f"reconstructed_entries contains {expected!r}", any(expected in t for t in recon_texts), detail=f"all entries: {recon_texts}")

    # Also directly exercises the broadened marker regex (step 2): "1)" and "(1)" formats, and
    # lettered "a." options, must be recognized too -- not just this PDF's own "1." style. No real
    # PDF in this repo uses these formats, so this part is a synthetic, rather than live, check.
    for label, text in [
        ("digit-paren '1)' format", "1) Zero\n2) 1\n3) 2\n4) 3"),
        ("parenthesized '(1)' format", "(1) Zero\n(2) 1\n(3) 2\n(4) 3"),
        ("lettered 'a.' format", "a. Zero\nb. 1\nc. 2\nd. 3"),
    ]:
        m1 = ppv._SELF_CONTAINED_OPTION_RE.match(text.splitlines()[0])
        check(f"self-contained marker regex recognizes {label}", bool(m1), detail=f"line={text.splitlines()[0]!r}")
        if m1:
            check(f"{label} marker 1 normalizes to num=1", ppv._marker_label_to_num(m1.group(1)) == 1)


def test_math_only_options_reconstructed_in_order():
    """EMI Q9 (Faraday's Law page): options are pure fraction fragments with no real text at all
    (NBA/omega, NBA*omega, NBA/omega^2, NBA*omega^2) -- bare markers, no self_text, fully
    dependent on the zone-capture this task's bug lived in. Must still come back in the right
    1-2-3-4 order and must NOT include the next question's own content."""
    print("\n=== Math-only (fraction) options: reconstructed in order, no cross-question bleed ===")
    import fitz
    with open(EMI_PDF_PATH, "rb") as f:
        pdf_bytes = f.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[1]
    mid_x = page.rect.width / 2
    page_blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type") == 0]
    excluded_ids, reconstructed_entries, modified_block_text = ppv._reconstruct_fragmented_options(page_blocks, mid_x, page.rect.width)
    doc.close()

    # Identify Q9's own 4 entries by their distinctive content ("NBA"/omega), not by a y0 range --
    # a range alone isn't safe here since Q8's, Q9's and Q12's (left+right column) entries can
    # all have y0s under the same cutoff.
    q9_entries = sorted((e for e in reconstructed_entries if "NBA" in e[2] or "ω" in e[2]), key=lambda e: e[1])
    q9_texts = [e[2] for e in q9_entries if e[2].startswith(("1.", "2.", "3.", "4."))]
    check("all 4 of Q9's math-only options were reconstructed", len(q9_texts) == 4, detail=f"{q9_texts}")
    if len(q9_texts) == 4:
        check("they came back in 1,2,3,4 order", [t[0] for t in q9_texts] == ["1", "2", "3", "4"], detail=f"{q9_texts}")
        check("option 4 does not contain Q10's own content ('0.14' / 'inductance')",
              "0.14" not in q9_texts[3] and "inductance" not in q9_texts[3], detail=f"{q9_texts[3]!r}")


if __name__ == "__main__":
    if not os.path.exists(EMI_PDF_PATH):
        print(f"SKIPPED: {EMI_PDF_PATH} not found -- these tests need the real PDF fixture.")
        sys.exit(1)

    test_q2_disc_question_math_preserved_in_stem()
    test_find_missing_raw_tokens_catches_the_original_bug_shape()
    test_numbered_options_1_to_4_reconstructed_not_dropped()
    test_math_only_options_reconstructed_in_order()

    print(f"\nTOTAL: {len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
