"""Small, targeted unit tests for the contractor payout math (PYQ_PRICE_USD_PER_Q /
PYQ_DUP_DEDUCT_FROM) -- no network, no Supabase. Run with dummy credentials, e.g.:
  OPENAI_KEY=dummy ANTHROPIC_KEY=dummy SUPABASE_URL=https://example.supabase.co SUPABASE_KEY=dummy \
  SUPABASE_SERVICE_KEY=dummy DEEPSEEK_KEY=dummy QWEN_API_KEY=dummy GEMINI_API_KEY=dummy \
  ADMIN_PASSWORD=dummy python -m pytest test_pyq_payout.py
"""
from datetime import timedelta

import server


def test_error_rejection_is_deducted():
    assert server._pyq_reject_classify("Garbled text: OCR ate half the options", None) == "error"
    assert server._pyq_reject_classify("Wrong answer key", server.PYQ_DUP_DEDUCT_FROM) == "error"


def test_duplicate_before_cutoff_is_free():
    before = server.PYQ_DUP_DEDUCT_FROM - timedelta(days=1)
    assert server._pyq_reject_classify("Duplicate", before) == "duplicate_free"
    assert server._pyq_reject_classify("Duplicate: same as question abc123", before) == "duplicate_free"


def test_duplicate_on_cutoff_day_is_deducted():
    # Midday on the cutoff day itself, and the exact cutoff instant -- both "on or after", so
    # both deducted, not free.
    midday = server.PYQ_DUP_DEDUCT_FROM + timedelta(hours=12)
    assert server._pyq_reject_classify("Duplicate", midday) == "duplicate_deducted"
    assert server._pyq_reject_classify("Duplicate", server.PYQ_DUP_DEDUCT_FROM) == "duplicate_deducted"


def test_duplicate_after_cutoff_is_deducted():
    after = server.PYQ_DUP_DEDUCT_FROM + timedelta(days=1)
    assert server._pyq_reject_classify("Duplicate", after) == "duplicate_deducted"


def test_payable_and_amount_rounding():
    # 1000 uploaded, 50 errors, 30 deducted duplicates -> 920 payable * $0.02 = $18.40.
    result = server._pyq_payable_amount(1000, 50, 30)
    assert result == {"payable": 920, "amount_usd": 18.40}

    # Free duplicates never reduce payable -- only rejected_errors/duplicates_deducted do.
    result = server._pyq_payable_amount(1000, 0, 0)
    assert result == {"payable": 1000, "amount_usd": 20.00}

    # Fractional-cent rounding: 7 payable * $0.02 = $0.14 exactly, no float drift surfacing.
    result = server._pyq_payable_amount(7, 0, 0)
    assert result == {"payable": 7, "amount_usd": 0.14}

    # Can't go negative even if deductions somehow exceed uploaded (stale log, manual edit, etc).
    result = server._pyq_payable_amount(5, 10, 0)
    assert result == {"payable": 0, "amount_usd": 0.0}


def test_zero_uploads():
    result = server._pyq_payable_amount(0, 0, 0)
    assert result == {"payable": 0, "amount_usd": 0.0}


if __name__ == "__main__":
    test_error_rejection_is_deducted()
    test_duplicate_before_cutoff_is_free()
    test_duplicate_on_cutoff_day_is_deducted()
    test_duplicate_after_cutoff_is_deducted()
    test_payable_and_amount_rounding()
    test_zero_uploads()
    print("OK")
