"""Small, targeted unit tests for auto_check() and the approve-all count guard -- no network, no
Supabase. Run with dummy credentials, e.g.:
  OPENAI_KEY=dummy ANTHROPIC_KEY=dummy SUPABASE_URL=https://example.supabase.co SUPABASE_KEY=dummy \
  SUPABASE_SERVICE_KEY=dummy DEEPSEEK_KEY=dummy QWEN_API_KEY=dummy GEMINI_API_KEY=dummy \
  ADMIN_PASSWORD=dummy python -m pytest test_pyq_auto_approve.py
"""
import server

GOOD_ROW = {
    "id": "11111111-1111-1111-1111-111111111111",
    "subject": "Biology",
    "chapter": "Cell Structure and Function",
    "question": "Which of the following organelles is primarily responsible for ATP production in eukaryotic cells?",
    "option_a": "Mitochondria",
    "option_b": "Ribosome",
    "option_c": "Golgi apparatus",
    "option_d": "Lysosome",
    "correct_answer": "A",
    "review_status": "pending",
    "has_diagram": False,
    "diagram_url": None,
    "option_a_diagram_url": None,
    "option_b_diagram_url": None,
    "option_c_diagram_url": None,
    "option_d_diagram_url": None,
}


def _row(**overrides):
    return {**GOOD_ROW, **overrides}


def test_pass():
    ok, reasons = server.auto_check(_row())
    assert ok is True
    assert reasons == []


def test_fewer_than_four_options():
    ok, reasons = server.auto_check(_row(option_d=None))
    assert ok is False
    assert "options" in reasons


def test_duplicate_options():
    # option_c repeats option_b's text -- only 3 distinct option values, not 4.
    ok, reasons = server.auto_check(_row(option_c="Ribosome"))
    assert ok is False
    assert "options" in reasons


def test_literal_backslash_n():
    ok, reasons = server.auto_check(_row(question=GOOD_ROW["question"] + "\\nNote: see diagram."))
    assert ok is False
    assert "broken_text" in reasons


def test_too_short():
    ok, reasons = server.auto_check(_row(question="Too short?"))
    assert ok is False
    assert "broken_text" in reasons


def test_excluded_type_diagram():
    ok, reasons = server.auto_check(_row(has_diagram=True))
    assert ok is False
    assert "diagram" in reasons


def test_excluded_type_match_the_following():
    ok, reasons = server.auto_check(_row(
        question="Match Column I with Column II and choose the correct combination."
    ))
    assert ok is False
    assert "match_following" in reasons


def test_excluded_type_math_or_latex_formula():
    ok, reasons = server.auto_check(_row(
        question=GOOD_ROW["question"] + " Calculate the net value using the given formula."
    ))
    assert ok is False
    assert "math_formula" in reasons


def test_excluded_type_chemical_formula():
    ok, reasons = server.auto_check(_row(
        question="Balance the following reaction: CH4 and C2H6 and C3H8 are burnt completely in oxygen."
    ))
    assert ok is False
    assert "chemical_formula" in reasons


def test_rejected_row_never_passes():
    ok, reasons = server.auto_check(_row(review_status="rejected"))
    assert ok is False
    assert reasons == ["rejected"]


def test_duplicate_against_corpus():
    # Same tokenizer the Duplicates tool (/admin/pyq-duplicates) uses -- a corpus entry with
    # identical question+option text is a 100% overlap, well above the 0.5 threshold.
    tokens = server._tokenize_for_dup_check(server._auto_row_text(GOOD_ROW))
    corpus = [("22222222-2222-2222-2222-222222222222", tokens)]
    ok, reasons = server.auto_check(_row(), dup_corpus=corpus)
    assert ok is False
    assert "duplicate" in reasons


def test_approve_all_count_guard():
    assert server._auto_approve_all_count_ok(5, 5) is True
    assert server._auto_approve_all_count_ok(5, 6) is False
    assert server._auto_approve_all_count_ok(0, 0) is True


if __name__ == "__main__":
    test_pass()
    test_fewer_than_four_options()
    test_duplicate_options()
    test_literal_backslash_n()
    test_too_short()
    test_excluded_type_diagram()
    test_excluded_type_match_the_following()
    test_excluded_type_math_or_latex_formula()
    test_excluded_type_chemical_formula()
    test_rejected_row_never_passes()
    test_duplicate_against_corpus()
    test_approve_all_count_guard()
    print("OK")
