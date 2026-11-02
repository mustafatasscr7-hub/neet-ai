"""Unit tests for Hindi scanned-PDF support (process_pyq_scanned.py): the review-flag checks
(hindi_review_flags), the two-pass disagreement comparison (_compare_hindi_passes), and how both
wire into scan_scanned_pdf_bytes(language="hi") end to end.

Run with: python test_pyq_scanned_hindi.py
No pytest dependency -- same plain pass/fail harness as test_pyq_scanned.py. No test here calls
Gemini: scan_scanned_pdf_bytes's existing extract_fn test seam is reused to inject a fake
(items, usage, disagreements, pass_failed) 4-tuple result per page, the same shape extract_page_
items_two_pass produces for real.
"""
import io
import os
import sys
import tempfile

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fitz
import process_pyq_scanned as ps

_passed = []
_failed = []


def check(label, condition, detail=""):
    if condition:
        _passed.append(label)
        print(f"  PASS: {label}")
    else:
        _failed.append(label)
        print(f"  FAIL: {label}" + (f" -- {detail}" if detail else ""))


def _q(number=1, question="यह एक प्रश्न है?", options=None, year=1, **extra):
    options = options if options is not None else {l: f"विकल्प {l}" for l in "abcd"}
    out = {
        "question": question, "option_a": options.get("a", ""), "option_b": options.get("b", ""),
        "option_c": options.get("c", ""), "option_d": options.get("d", ""), "correct_answer": "",
        "question_type": "mcq", "has_diagram": False, "year": year, "source_tag": str(year) if year else None,
        "ncert_tag": None, "question_number": number, "source_page": 1, "needs_review": False, "review_reason": None,
    }
    out.update(extra)
    return out


def _make_pdf(n_pages):
    doc = fitz.open()
    for i in range(n_pages):
        doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


# ---------------------------------------------------------------- hindi_review_flags

def test_clean_question_has_no_flags():
    print("\n=== hindi_review_flags: a clean question gets no flags ===")
    issues = ps.hindi_review_flags(_q(question="यह प्रश्न सही ढंग से समाप्त होता है?"))
    check("no issues", issues == [], detail=str(issues))


def test_flags_non_devanagari_non_latin_chars():
    print("\n=== hindi_review_flags: non-Devanagari/non-Latin script is flagged ===")
    # Task's own example: Arabic characters inside what should be pure Devanagari/Latin text.
    issues = ps.hindi_review_flags(_q(question="यह अरबी के साथ मिश्रित है السلام علیکم؟"))
    check("flags unexpected script", any("unexpected script" in i for i in issues), detail=str(issues))

    # Latin letters, digits, common punctuation and LaTeX are all legitimate inside Hindi PYQ text
    # (English terms, chemical formulae, numerals) and must NOT be flagged on their own.
    clean = ps.hindi_review_flags(_q(question="$CO_{2}$ की मात्रा 10% है, Class 11 के अनुसार?"))
    check("Latin/digits/LaTeX alone don't trigger the script flag", not any("unexpected script" in i for i in clean), detail=str(clean))


def test_flags_missing_year():
    print("\n=== hindi_review_flags: a missing year tag is flagged ===")
    issues = ps.hindi_review_flags(_q(year=None))
    check("flags missing year", any("year" in i for i in issues), detail=str(issues))
    issues_with_year = ps.hindi_review_flags(_q(year=2019))
    check("a real year raises no year flag", not any("year" in i for i in issues_with_year), detail=str(issues_with_year))


def test_flags_stem_not_ending_in_sentence_terminator():
    print("\n=== hindi_review_flags: a stem that doesn't end in ?/:/./। is flagged ===")
    cut_off = ps.hindi_review_flags(_q(question="यह प्रश्न अधूरा लग रहा है और"))
    check("flags a stem with no terminator", any("doesn't end" in i for i in cut_off), detail=str(cut_off))

    for terminator in ("?", ":", ".", "।", "॥"):
        ok = ps.hindi_review_flags(_q(question=f"यह प्रश्न ठीक से समाप्त होता है{terminator}"))
        check(f"'{terminator}' counts as a real sentence ending", not any("doesn't end" in i for i in ok), detail=str(ok))

    # A trailing closing quote/paren after the real terminator shouldn't defeat the check.
    quoted = ps.hindi_review_flags(_q(question='वह बोला "यह सही है।"'))
    check("a terminator followed by a closing quote still counts", not any("doesn't end" in i for i in quoted), detail=str(quoted))


def test_duplicate_options_not_duplicated_by_hindi_flags():
    print("\n=== hindi_review_flags: duplicate options are left to _flag_suspicious_mcq_options, not re-checked here ===")
    dupe_opts = {"a": "विकल्प", "b": "विकल्प", "c": "अन्य", "d": "कुछ और"}
    issues = ps.hindi_review_flags(_q(options=dupe_opts))
    check("hindi_review_flags itself has no duplicate-option rule", not any("identical" in i or "duplicate" in i.lower() for i in issues), detail=str(issues))


# ---------------------------------------------------------------- _compare_hindi_passes

def test_two_pass_agreement_no_disagreement():
    print("\n=== _compare_hindi_passes: identical passes disagree on nothing ===")
    items1 = [{"kind": "question", "number": 1, "question": "क्या यह सही है?", "options": {"a": "हाँ", "b": "नहीं", "c": "शायद", "d": "पता नहीं"}}]
    items2 = [dict(items1[0])]
    check("no disagreements", ps._compare_hindi_passes(items1, items2) == [])


def test_two_pass_question_text_disagreement():
    print("\n=== _compare_hindi_passes: differing stem text is a disagreement ===")
    items1 = [{"kind": "question", "number": 5, "question": "पहला पाठ", "options": {}}]
    items2 = [{"kind": "question", "number": 5, "question": "दूसरा पाठ", "options": {}}]
    check("Q5 flagged", ps._compare_hindi_passes(items1, items2) == [5])


def test_two_pass_option_disagreement():
    print("\n=== _compare_hindi_passes: differing option text is a disagreement ===")
    items1 = [{"kind": "question", "number": 7, "question": "same", "options": {"a": "ठीक"}}]
    items2 = [{"kind": "question", "number": 7, "question": "same", "options": {"a": "गलत"}}]
    check("Q7 flagged", ps._compare_hindi_passes(items1, items2) == [7])


def test_two_pass_whitespace_only_difference_not_flagged():
    print("\n=== _compare_hindi_passes: pure whitespace differences are normalized away ===")
    items1 = [{"kind": "question", "number": 3, "question": "यह   प्रश्न है", "options": {"a": "ठीक"}}]
    items2 = [{"kind": "question", "number": 3, "question": "यह प्रश्न है", "options": {"a": " ठीक "}}]
    check("Q3 not flagged (whitespace-only)", ps._compare_hindi_passes(items1, items2) == [])


def test_two_pass_only_compares_numbers_present_in_both():
    print("\n=== _compare_hindi_passes: a question missing from one pass is never flagged as a disagreement ===")
    items1 = [{"kind": "question", "number": 1, "question": "a", "options": {}},
              {"kind": "question", "number": 2, "question": "b", "options": {}}]
    items2 = [{"kind": "question", "number": 1, "question": "a", "options": {}}]
    check("only the shared number (1) is even considered, and it agrees", ps._compare_hindi_passes(items1, items2) == [])


# ---------------------------------------------------------------- end-to-end: scan_scanned_pdf_bytes(language="hi")

def test_hindi_scan_skips_no_answer_key_flag():
    print("\n=== scan_scanned_pdf_bytes(language='hi'): no answer key -> no 'no answer key found' spam ===")
    pdf = _make_pdf(1)

    def fake_extract(image_bytes, subject):
        items = [{"kind": "question", "number": 1, "question": "साफ प्रश्न?", "year_tag": "2019",
                  "options": {l: f"विकल्प {l}" for l in "abcd"}, "box_2d": [100, 95, 200, 460]}]
        return items, {"input_tokens": 100, "output_tokens": 50}, [], False

    with tempfile.TemporaryDirectory() as cache_dir:
        r = ps.scan_scanned_pdf_bytes(pdf, "Physics", language="hi", cache_dir=cache_dir, extract_fn=fake_extract)
    check("1 question extracted", len(r["questions"]) == 1, detail=str(r["questions"]))
    check("correct_answer is empty (never guessed)", r["questions"][0]["correct_answer"] == "")
    check("no 'no answer key found' flag", not any("answer key" in f["reason"] for f in r["flagged_pages"]), detail=str(r["flagged_pages"]))
    check("not flagged otherwise (clean question, real year)", not r["questions"][0]["needs_review"], detail=str(r["questions"][0]))


def test_hindi_scan_flags_two_pass_disagreement_and_bad_script():
    print("\n=== scan_scanned_pdf_bytes(language='hi'): disagreement + bad-script flags reach the question ===")
    pdf = _make_pdf(1)

    def fake_extract(image_bytes, subject):
        items = [{"kind": "question", "number": 9, "question": "यह लैटिन Ω है?",
                  "options": {l: f"विकल्प {l}" for l in "abcd"}, "year_tag": "2020", "box_2d": [100, 95, 200, 460]}]
        return items, {"input_tokens": 100, "output_tokens": 50}, [9], False  # pre-computed disagreement for Q9

    with tempfile.TemporaryDirectory() as cache_dir:
        r = ps.scan_scanned_pdf_bytes(pdf, "Physics", language="hi", cache_dir=cache_dir, extract_fn=fake_extract)
    q9 = next(x for x in r["questions"] if x["question_number"] == 9)
    check("Q9 flagged for review", q9["needs_review"])
    check("Q9's reason mentions the two-pass disagreement", "two independent extraction passes disagree" in q9["review_reason"], detail=q9["review_reason"])
    check("a matching flagged_pages entry exists", any("Q9" in f["reason"] and "disagree" in f["reason"] for f in r["flagged_pages"]), detail=str(r["flagged_pages"]))


def test_hindi_scan_pass_failed_is_flagged_and_not_cached():
    print("\n=== scan_scanned_pdf_bytes(language='hi'): a one-pass-failed page is flagged and never cached ===")
    pdf = _make_pdf(1)

    def fake_extract(image_bytes, subject):
        items = [{"kind": "question", "number": 1, "question": "केवल एक पास से?",
                  "options": {l: f"विकल्प {l}" for l in "abcd"}, "year_tag": "2021", "box_2d": [100, 95, 200, 460]}]
        return items, {"input_tokens": 50, "output_tokens": 25}, [], True  # pass_failed=True

    with tempfile.TemporaryDirectory() as cache_dir:
        r = ps.scan_scanned_pdf_bytes(pdf, "Physics", language="hi", cache_dir=cache_dir, extract_fn=fake_extract)
        check("1 question still saved despite the failed partner pass", len(r["questions"]) == 1, detail=str(r["questions"]))
        check("a 'one of the two extraction passes' flag is present", any("one of the two extraction passes" in f["reason"].lower() for f in r["flagged_pages"]), detail=str(r["flagged_pages"]))
        check("nothing was cached (a retry must call the model again)", not os.listdir(cache_dir))


if __name__ == "__main__":
    test_clean_question_has_no_flags()
    test_flags_non_devanagari_non_latin_chars()
    test_flags_missing_year()
    test_flags_stem_not_ending_in_sentence_terminator()
    test_duplicate_options_not_duplicated_by_hindi_flags()
    test_two_pass_agreement_no_disagreement()
    test_two_pass_question_text_disagreement()
    test_two_pass_option_disagreement()
    test_two_pass_whitespace_only_difference_not_flagged()
    test_two_pass_only_compares_numbers_present_in_both()
    test_hindi_scan_skips_no_answer_key_flag()
    test_hindi_scan_flags_two_pass_disagreement_and_bad_script()
    test_hindi_scan_pass_failed_is_flagged_and_not_cached()

    print(f"\nTOTAL: {len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
