"""Small, targeted unit tests for the Duplicates "Reject (duplicate)" bulk action and the
contractor-tally uploader filter -- no network, no Supabase.
Run with dummy credentials, e.g.:
  OPENAI_KEY=dummy SUPABASE_URL=https://example.supabase.co SUPABASE_KEY=dummy \
  SUPABASE_SERVICE_KEY=dummy DEEPSEEK_KEY=dummy QWEN_API_KEY=dummy GEMINI_API_KEY=dummy \
  ADMIN_PASSWORD=dummy python -m pytest test_pyq_duplicate_reject.py
"""
import server


def test_contractor_tally_excludes_non_contractors():
    # PYQ_UPLOADER_NAMES ("Mr. Md Moniruzzaman", "Admin") is every legitimate uploader, but the
    # tally is a payout sheet -- only the contractor should ever be iterated over.
    assert "Admin" in server.PYQ_UPLOADER_NAMES
    assert server.PYQ_CONTRACTOR_NAMES == (server.CONTRACTOR_UPLOADER_NAME,)
    assert "Admin" not in server.PYQ_CONTRACTOR_NAMES


def test_bulk_reject_keeps_one_copy_per_group():
    # Two groups of exact-duplicate text (A/B and C/D), plus one unrelated singleton (E).
    rows = [
        {"id": "A", "question": "What is the powerhouse of the cell?", "review_status": "pending"},
        {"id": "B", "question": "What is the powerhouse of the cell?", "review_status": "approved"},
        {"id": "C", "question": "Define Ohm's law.", "review_status": "pending"},
        {"id": "D", "question": "Define   Ohm's Law.  ", "review_status": "needs_fix"},  # same, normalized
        {"id": "E", "question": "Unrelated question text.", "review_status": "pending"},
    ]
    plan = server._pyq_bulk_reject_plan(rows)
    assert plan["already_rejected"] == []
    # Every group of 2+ keeps exactly one member out of to_reject.
    assert len(plan["kept_for_safety"]) == 2
    rejected = set(plan["to_reject"])
    kept = set(plan["kept_for_safety"])
    assert rejected | kept | set(plan["already_rejected"]) == {"A", "B", "C", "D", "E"}
    assert rejected.isdisjoint(kept)
    # The singleton with no duplicate in the batch is always rejected, never held back.
    assert "E" in rejected
    # Exactly one of each pair survives.
    assert len({"A", "B"} & kept) == 1
    assert len({"C", "D"} & kept) == 1


def test_bulk_reject_skips_already_rejected_rows():
    rows = [
        {"id": "A", "question": "Same text", "review_status": "rejected"},
        {"id": "B", "question": "Same text", "review_status": "pending"},
    ]
    plan = server._pyq_bulk_reject_plan(rows)
    # "A" is already rejected, so it's left alone -- "B" is the only unrejected copy of this
    # text in the batch, so it's a singleton group of one and gets rejected normally (the
    # "other copy" isn't in a reject-eligible state any more, so there's nothing left to protect).
    assert plan["already_rejected"] == ["A"]
    assert plan["to_reject"] == ["B"]
    assert plan["kept_for_safety"] == []


def test_bulk_reject_blank_question_text_never_merges():
    # Diagram-only questions (no question text) must never be treated as duplicates of each
    # other just because they're both blank.
    rows = [
        {"id": "A", "question": "", "review_status": "pending"},
        {"id": "B", "question": None, "review_status": "pending"},
    ]
    plan = server._pyq_bulk_reject_plan(rows)
    assert set(plan["to_reject"]) == {"A", "B"}
    assert plan["kept_for_safety"] == []


if __name__ == "__main__":
    test_contractor_tally_excludes_non_contractors()
    test_bulk_reject_keeps_one_copy_per_group()
    test_bulk_reject_skips_already_rejected_rows()
    test_bulk_reject_blank_question_text_never_merges()
    print("OK")
