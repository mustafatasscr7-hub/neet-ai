"""Small, targeted unit tests for the Reject (error) flow -- no network, no Supabase.
Run with dummy credentials, e.g.:
  OPENAI_KEY=dummy SUPABASE_URL=https://example.supabase.co SUPABASE_KEY=dummy \
  SUPABASE_SERVICE_KEY=dummy DEEPSEEK_KEY=dummy QWEN_API_KEY=dummy GEMINI_API_KEY=dummy \
  ADMIN_PASSWORD=dummy python -m pytest test_pyq_reject.py
"""
import server


def _payable(uploaded: int, rejected: int) -> int:
    return max(0, uploaded - rejected)


def test_payable_calculation():
    assert _payable(1000, 50) == 950
    assert _payable(10, 0) == 10
    # Can't go negative even if rejected somehow exceeds uploaded (stale log, manual edit, etc).
    assert _payable(5, 10) == 0
    assert _payable(0, 0) == 0


def test_rejected_rows_excluded_from_student_reads():
    # PYQ_STUDENT_VISIBLE is the one filter every student-facing pyq query adds. It requires
    # review_status == 'approved' exactly, so a 'rejected' row (a new, distinct status value)
    # is excluded by construction -- no extra exclusion logic needed anywhere a student reads.
    assert server.PYQ_STUDENT_VISIBLE["review_status"] == "eq.approved"
    assert "rejected" in server.PYQ_REVIEW_STATUSES
    assert "rejected" != server.PYQ_STUDENT_VISIBLE["review_status"].split(".", 1)[1]


if __name__ == "__main__":
    test_payable_calculation()
    test_rejected_rows_excluded_from_student_reads()
    print("OK")
