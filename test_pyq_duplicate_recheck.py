"""Small, targeted unit tests for the bounded post-action duplicate-flag recheck -- no network,
no Supabase. Run with dummy credentials, e.g.:
  OPENAI_KEY=dummy ANTHROPIC_KEY=dummy SUPABASE_URL=https://example.supabase.co SUPABASE_KEY=dummy \
  SUPABASE_SERVICE_KEY=dummy DEEPSEEK_KEY=dummy QWEN_API_KEY=dummy GEMINI_API_KEY=dummy \
  ADMIN_PASSWORD=dummy python -m pytest test_pyq_duplicate_recheck.py
"""
import asyncio

import server

OTHER_ROW = {
    "id": "22222222-2222-2222-2222-222222222222",
    "subject": "Biology",
    "chapter": "Cell Structure and Function",
    "question": "Which of the following organelles is primarily responsible for ATP production in eukaryotic cells?",
    "option_a": "Mitochondria",
    "option_b": "Ribosome",
    "option_c": "Golgi apparatus",
    "option_d": "Lysosome",
}

PENDING_ROW = {
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
    return {**PENDING_ROW, **overrides}


def _corpus(*rows):
    return [(r["id"], server._tokenize_for_dup_check(server._auto_row_text(r))) for r in rows]


# ---------- _pyq_duplicate_flagged ----------

def test_duplicate_flagged_true():
    assert server._pyq_duplicate_flagged({"auto_reasons": "diagram,duplicate"}) is True


def test_duplicate_flagged_false_without_reason():
    assert server._pyq_duplicate_flagged({"auto_reasons": "diagram"}) is False


def test_duplicate_flagged_false_when_null():
    assert server._pyq_duplicate_flagged({"auto_reasons": None}) is False


# ---------- _pyq_rows_for_duplicate_recheck (what the GET -> PATCH pipeline is bounded to) ----------

def test_rows_for_recheck_excludes_approved_rows():
    rows = [
        _row(id="a", review_status="pending", auto_reasons="duplicate"),
        _row(id="b", review_status="approved", auto_reasons="duplicate"),
    ]
    out = {r["id"] for r in server._pyq_rows_for_duplicate_recheck(rows, "en")}
    assert out == {"a"}


def test_rows_for_recheck_excludes_other_language():
    rows = [
        _row(id="a", language="en", auto_reasons="duplicate"),
        _row(id="b", language="hi", auto_reasons="duplicate"),
    ]
    assert {r["id"] for r in server._pyq_rows_for_duplicate_recheck(rows, "en")} == {"a"}
    assert {r["id"] for r in server._pyq_rows_for_duplicate_recheck(rows, "hi")} == {"b"}


def test_rows_for_recheck_excludes_rows_without_duplicate_flag():
    rows = [
        _row(id="a", auto_reasons="duplicate"),
        _row(id="b", auto_reasons="diagram"),
    ]
    assert {r["id"] for r in server._pyq_rows_for_duplicate_recheck(rows, "en")} == {"a"}


# ---------- flag lost / restored / kept, against auto_check() directly ----------

def test_flag_lost_when_only_match_is_excluded_from_corpus():
    # Before: OTHER_ROW is live, so PENDING_ROW is flagged duplicate.
    ok, reasons = server.auto_check(_row(), dup_corpus=_corpus(OTHER_ROW))
    assert ok is False
    assert "duplicate" in reasons
    # After OTHER_ROW is rejected/deleted, it's no longer in the live corpus passed in --
    # the row becomes auto-eligible (every other rule still passes).
    ok, reasons = server.auto_check(_row(), dup_corpus=[])
    assert ok is True
    assert reasons == []


def test_restoring_the_match_reflags_it():
    # Match already excluded (rejected) -- row passes clean.
    ok, reasons = server.auto_check(_row(), dup_corpus=[])
    assert ok is True
    # Restoring (unreject) puts it back in the live corpus -- the flag returns.
    ok, reasons = server.auto_check(_row(), dup_corpus=_corpus(OTHER_ROW))
    assert ok is False
    assert "duplicate" in reasons


def test_row_with_another_live_match_keeps_the_flag():
    other_2 = {**OTHER_ROW, "id": "33333333-3333-3333-3333-333333333333"}
    # Both OTHER_ROW and other_2 currently match. Reject/delete OTHER_ROW only --
    # other_2 is still live, so the flag must survive.
    ok, reasons = server.auto_check(_row(), dup_corpus=_corpus(other_2))
    assert ok is False
    assert "duplicate" in reasons


# ---------- becomes auto-eligible only through the existing, gated auto-approve path ----------

def test_cleared_row_auto_approves_when_switch_is_on(monkeypatch):
    monkeypatch.setattr(server, "AUTO_APPROVE_ENABLED", True)
    monkeypatch.setattr(server, "AUTO_APPROVE_SAMPLE_RATE", 0.0)  # never hold back for sampling
    fields = asyncio.run(server._auto_check_row(
        _row(auto_pass=False, auto_reasons="duplicate"), {}, "2026-11-06T00:00:00+00:00", allow_approve=True
    ))
    assert fields["auto_pass"] is True
    assert fields["auto_reasons"] is None
    assert fields["review_status"] == "approved"
    assert fields["auto_approved"] is True


def test_cleared_row_stays_pending_when_switch_is_off(monkeypatch):
    # Production default (AUTO_APPROVE_ENABLED=false): losing the duplicate flag must only ever
    # flip auto_pass/auto_reasons -- review_status is never touched by this path on its own.
    monkeypatch.setattr(server, "AUTO_APPROVE_ENABLED", False)
    fields = asyncio.run(server._auto_check_row(
        _row(auto_pass=False, auto_reasons="duplicate"), {}, "2026-11-06T00:00:00+00:00", allow_approve=True
    ))
    assert fields["auto_pass"] is True
    assert "review_status" not in fields
    assert "auto_approved" not in fields


if __name__ == "__main__":
    test_duplicate_flagged_true()
    test_duplicate_flagged_false_without_reason()
    test_duplicate_flagged_false_when_null()
    test_rows_for_recheck_excludes_approved_rows()
    test_rows_for_recheck_excludes_other_language()
    test_rows_for_recheck_excludes_rows_without_duplicate_flag()
    test_flag_lost_when_only_match_is_excluded_from_corpus()
    test_restoring_the_match_reflags_it()
    test_row_with_another_live_match_keeps_the_flag()
    print("OK (monkeypatch-dependent tests skipped in __main__; run via pytest)")
