"""Small, targeted unit tests for the "Duplicate" pill cross-check panel's pure logic
(_pyq_build_dup_matches / _pyq_match_status / _pyq_dup_pill_flagged) -- no network, no Supabase.
Covers the bug where /admin/pyq-matches' corpus excluded rejected/deactivated rows while the
"Duplicate" pill (auto_reasons) is a stale, point-in-time snapshot that doesn't un-set itself.
Run with dummy credentials, e.g.:
  OPENAI_KEY=dummy SUPABASE_URL=https://example.supabase.co SUPABASE_KEY=dummy \
  SUPABASE_SERVICE_KEY=dummy DEEPSEEK_KEY=dummy QWEN_API_KEY=dummy GEMINI_API_KEY=dummy \
  ADMIN_PASSWORD=dummy python -m pytest test_pyq_dup_match_panel.py
"""
import server

TARGET = {
    "id": "376e0722-73c1-4a1f-ba56-85c6ecec55c7",
    "subject": "Biology",
    "language": "en",
    "auto_reasons": "duplicate",
    "question": "Infection of Ascaris usually occurs by",
    "option_a": "drinking water containing egg of Ascaris",
    "option_b": "eating imperfectly cooked pork",
    "option_c": "tse-tse fly",
    "option_d": "mosquito bite",
}

REJECTED_MATCH = {
    "id": "0ba3c058-c0b5-4f57-b2ad-e3fb1f7e0f90",
    "is_active": True,
    "review_status": "rejected",
    "question": "Infection of Ascaris usually occurs by :",
    "option_a": "drinking water containing egg of Ascaris",
    "option_b": "eating imperfectly cooked pork",
    "option_c": "tse-tse fly",
    "option_d": "mosquito bite",
}


def _corpus_entry(row):
    return (row["id"], server._pyq_dup_tokens(row), row)


def test_flagged_row_finds_its_rejected_match():
    corpus = [_corpus_entry(REJECTED_MATCH)]
    result = server._pyq_build_dup_matches(TARGET, corpus)
    assert result["stale_flag"] is False
    assert len(result["matches"]) == 1
    match = result["matches"][0]
    assert match["id"] == REJECTED_MATCH["id"]
    assert match["status"] == "rejected"


def test_flagged_row_with_no_match_is_reported_stale():
    result = server._pyq_build_dup_matches(TARGET, [])
    assert result["matches"] == []
    assert result["total_matches"] == 0
    assert result["stale_flag"] is True


def test_unflagged_row_with_no_match_is_not_stale():
    unflagged = {**TARGET, "auto_reasons": None}
    result = server._pyq_build_dup_matches(unflagged, [])
    assert result["matches"] == []
    assert result["stale_flag"] is False


def test_deactivated_match_is_labelled_deleted_not_dropped():
    deactivated = {**REJECTED_MATCH, "review_status": "pending", "is_active": False}
    corpus = [_corpus_entry(deactivated)]
    result = server._pyq_build_dup_matches(TARGET, corpus)
    assert len(result["matches"]) == 1
    assert result["matches"][0]["status"] == "deleted"
    assert result["stale_flag"] is False
