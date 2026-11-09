"""Small, targeted unit tests for auto_check() and the approve-all count guard -- no network, no
Supabase. Run with dummy credentials, e.g.:
  OPENAI_KEY=dummy SUPABASE_URL=https://example.supabase.co SUPABASE_KEY=dummy \
  SUPABASE_SERVICE_KEY=dummy DEEPSEEK_KEY=dummy QWEN_API_KEY=dummy GEMINI_API_KEY=dummy \
  ADMIN_PASSWORD=dummy python -m pytest test_pyq_auto_approve.py
"""
import asyncio

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


def test_formula_word_mention_without_latex_passes():
    # Just the word "formula"/"calculate" with no actual LaTeX span -- nothing for the formula
    # gate to parse, so it's not held on that basis alone (see _auto_formula_check).
    ok, reasons = server.auto_check(_row(
        question=GOOD_ROW["question"] + " Calculate the net value using the given formula."
    ))
    assert ok is True
    assert reasons == []


def test_formula_with_valid_latex_passes():
    ok, reasons = server.auto_check(_row(
        question="A particle starts from rest. Using $v^2 = u^2 + 2as$, what is its final velocity?"
    ))
    assert ok is True
    assert reasons == []


def test_formula_unbalanced_braces_fails():
    ok, reasons = server.auto_check(_row(
        question="What is the value of $\\frac{1}{2mv^2$ for this particle?"
    ))
    assert ok is False
    assert "formula_unbalanced" in reasons


def test_formula_text_corruption_fails():
    # "\text{Ans}" with its leading backslash-t dropped, e.g. by a lossy PDF extraction.
    ok, reasons = server.auto_check(_row(
        question="Evaluate $ext{Ans} = 5$ for the following reaction."
    ))
    assert ok is False
    assert "formula_text_corruption" in reasons


def test_formula_duplicate_options_after_normalizing_fails():
    # Raw strings differ only by whitespace inside the math span -- _auto_options_ok (raw string
    # compare) would miss this, which is exactly why formula questions get their own normalized
    # distinctness check.
    ok, reasons = server.auto_check(_row(
        question="What is the molecular formula of water, written as $H_2O$?",
        option_a="$H_2O$", option_b="$ H_2 O $", option_c="$CO_2$", option_d="$NH_3$",
    ))
    assert ok is False
    assert "formula_duplicate_options" in reasons


def test_formula_unknown_command_fails():
    ok, reasons = server.auto_check(_row(
        question="What is the value of $\\bogus{x}$ in this expression?"
    ))
    assert ok is False
    assert "formula_unknown_command" in reasons


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


def test_recheck_flips_stale_fail_to_pass_without_approving():
    # Simulates a pending row that was checked before _auto_formula_check existed: it carries a
    # stale auto_pass=false/math_formula verdict even though its formula is actually clean.
    # allow_approve=False is what /admin/pyq-auto-check passes in recheck mode -- confirms the
    # fresh verdict overwrites the stale one and that review_status/auto_approved are never set,
    # regardless of AUTO_APPROVE_ENABLED.
    row = _row(
        question="A particle starts from rest. Using $v^2 = u^2 + 2as$, what is its final velocity?",
        auto_pass=False, auto_reasons="math_formula",
    )
    fields = asyncio.run(server._auto_check_row(row, {}, "2026-11-05T00:00:00+00:00", allow_approve=False))
    assert fields["auto_pass"] is True
    assert fields["auto_reasons"] is None
    assert "review_status" not in fields
    assert "auto_approved" not in fields


def test_recheck_never_touches_approved_row():
    # An already-approved row should never be re-checked by the real sweep (it only ever fetches
    # review_status=pending rows), but _auto_check_row itself must still refuse to move it even
    # if it were ever called on one -- allow_approve=False is an unconditional guard, not just a
    # filter applied earlier.
    row = _row(review_status="approved")
    fields = asyncio.run(server._auto_check_row(row, {}, "2026-11-05T00:00:00+00:00", allow_approve=False))
    assert "review_status" not in fields
    assert "auto_approved" not in fields


if __name__ == "__main__":
    test_pass()
    test_fewer_than_four_options()
    test_duplicate_options()
    test_literal_backslash_n()
    test_too_short()
    test_excluded_type_diagram()
    test_excluded_type_match_the_following()
    test_formula_word_mention_without_latex_passes()
    test_formula_with_valid_latex_passes()
    test_formula_unbalanced_braces_fails()
    test_formula_text_corruption_fails()
    test_formula_duplicate_options_after_normalizing_fails()
    test_formula_unknown_command_fails()
    test_excluded_type_chemical_formula()
    test_rejected_row_never_passes()
    test_duplicate_against_corpus()
    test_approve_all_count_guard()
    test_recheck_flips_stale_fail_to_pass_without_approving()
    test_recheck_never_touches_approved_row()
    print("OK")
