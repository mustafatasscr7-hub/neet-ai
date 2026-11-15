"""Tests for chapter-level keyword (re)generation: POST /admin/ncert-content/keywords in
server.py, plus the manual add/edit endpoint's stale-keyword fix (admin_ncert_content_section).

Fully offline, same style as test_admin_ncert_content.py / test_ncert_editable_preview.py: Supabase
is an in-memory fake table, DeepSeek is a fake client (no real network/model calls, no real spend).

Run with: python test_ncert_keywords_generation.py
"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import server
import test_admin_ncert_content as base  # reuses FakeAsyncClient/FakeDeepSeekClient/setup

ADMIN_KEY = base.ADMIN_KEY
CONTRACTOR_KEY = base.CONTRACTOR_KEY
setup = base.setup

_passed = []
_failed = []


def check(label, condition, detail=""):
    if condition:
        _passed.append(label)
        print(f"  PASS: {label}")
    else:
        _failed.append(label)
        print(f"  FAIL: {label}" + (f" -- {detail}" if detail else ""))


def seed_chapter(fake, rows):
    """rows: [{"heading":, "text":, "keywords": None|str}, ...] in order -- ordinal is the index."""
    out = []
    for i, r in enumerate(rows):
        out.append(fake.table.insert({
            "class": "11", "subject": "Biology", "chapter": "Animal Kingdom",
            "heading": r["heading"], "ordinal": i, "text": r["text"],
            "word_count": len(r["text"].split()), "keywords": r.get("keywords"),
        })[0])
    return out


def call_keywords(tc, mode="pending", chapter="Animal Kingdom", cls="11", subject="Biology", key=ADMIN_KEY):
    headers = {"X-Admin-Key": key} if key else None
    return tc.post(
        "/admin/ncert-content/keywords",
        json={"class": cls, "subject": subject, "chapter": chapter, "mode": mode},
        headers=headers,
    )


def test_admin_only_contractor_rejected():
    print("\n# /admin/ncert-content/keywords is admin-only, contractor key rejected")
    tc, fake = setup()
    seed_chapter(fake, [{"heading": "7.1 X", "text": base.CLEAN_PARA}])

    server._admin_login_attempts.clear()
    res = call_keywords(tc, key=CONTRACTOR_KEY)
    check("contractor key refused (401)", res.status_code == 401, res.status_code)
    server._admin_login_attempts.clear()
    res2 = call_keywords(tc, key=None)
    check("no key refused (401)", res2.status_code == 401, res2.status_code)
    server._admin_login_attempts.clear()
    res3 = call_keywords(tc, key=ADMIN_KEY)
    check("admin key accepted", res3.status_code == 200 and not res3.json().get("error"), res3.json())


def test_pending_only_touches_only_empty_sections():
    print("\n# mode=pending only calls DeepSeek for sections with empty keywords")
    tc, fake = setup()
    seed_chapter(fake, [
        {"heading": "7.1 Already Done", "text": base.CLEAN_PARA, "keywords": "old, keywords, here"},
        {"heading": "7.2 Still Pending", "text": base.CLEAN_PARA, "keywords": None},
    ])
    data = call_keywords(tc, mode="pending").json()
    check("no error", not data.get("error"), data)
    check("exactly one call made (only the pending section)", data.get("calls") == 1, data)
    check("one succeeded", data.get("succeeded") == 1, data)
    check("results has exactly one entry, for the pending section", data.get("results") == [{"heading": "7.2 Still Pending", "ok": True}], data.get("results"))

    rows = fake.table.get({"class": "eq.11", "subject": "eq.Biology", "chapter": "eq.Animal Kingdom", "order": "ordinal"})
    check("the already-done section's keywords are untouched", rows[0]["keywords"] == "old, keywords, here", rows[0])
    check("the pending section now has keywords", bool(rows[1]["keywords"]), rows[1])


def test_pending_mode_with_nothing_pending_is_a_no_op():
    print("\n# mode=pending with zero pending sections makes no DeepSeek calls")
    tc, fake = setup()
    seed_chapter(fake, [{"heading": "7.1 X", "text": base.CLEAN_PARA, "keywords": "already, has, keywords"}])
    data = call_keywords(tc, mode="pending").json()
    check("no error", not data.get("error"), data)
    check("zero calls made", data.get("calls") == 0, data)
    check("no DeepSeek calls were actually made", server.deepseek_client.messages.calls == 0, server.deepseek_client.messages.calls)
    check("no log row written (nothing happened)", len(fake.chapter_log.rows) == 0, fake.chapter_log.rows)


def test_all_mode_regenerates_every_section():
    print("\n# mode=all regenerates every section, even ones that already have keywords")
    tc, fake = setup()
    seed_chapter(fake, [
        {"heading": "7.1 Has Keywords", "text": base.CLEAN_PARA, "keywords": "old, keywords, here"},
        {"heading": "7.2 Also Has Keywords", "text": base.CLEAN_PARA, "keywords": "more, old, ones"},
    ])
    data = call_keywords(tc, mode="all").json()
    check("no error", not data.get("error"), data)
    check("both sections were called", data.get("calls") == 2, data)
    check("both succeeded", data.get("succeeded") == 2, data)

    rows = fake.table.get({"class": "eq.11", "subject": "eq.Biology", "chapter": "eq.Animal Kingdom", "order": "ordinal"})
    check("both rows now carry the freshly-generated keywords, not the old ones",
          all(r["keywords"] == "diversity, taxonomy, classification" for r in rows), rows)


def test_partial_failure_keeps_failed_section_pending_and_continues():
    print("\n# A failing section stays pending and does not stop the rest")
    tc, fake = setup()
    seed_chapter(fake, [
        {"heading": "7.1 Will Fail", "text": base.CLEAN_PARA, "keywords": None},
        {"heading": "7.2 Will Succeed", "text": base.CLEAN_PARA, "keywords": None},
    ])
    orig_gen = server.ncert_ingest_core_mod.generate_keywords_with_cost

    def flaky_gen(client, heading, text, is_peak):
        if "Will Fail" in heading:
            raise RuntimeError("simulated DeepSeek failure")
        return orig_gen(client, heading, text, is_peak)

    server.ncert_ingest_core_mod.generate_keywords_with_cost = flaky_gen
    try:
        data = call_keywords(tc, mode="pending").json()
    finally:
        server.ncert_ingest_core_mod.generate_keywords_with_cost = orig_gen

    check("no top-level error (a section failure isn't a request failure)", not data.get("error"), data)
    check("both sections were attempted", data.get("calls") == 2, data)
    check("exactly one succeeded", data.get("succeeded") == 1, data)
    check("exactly one failed", data.get("failed") == 1, data)
    check("results record success/failure per section",
          sorted(data.get("results", []), key=lambda r: r["heading"]) == [
              {"heading": "7.1 Will Fail", "ok": False}, {"heading": "7.2 Will Succeed", "ok": True},
          ], data.get("results"))

    rows = fake.table.get({"class": "eq.11", "subject": "eq.Biology", "chapter": "eq.Animal Kingdom", "order": "ordinal"})
    check("the failed section's keywords are still empty (stayed pending)", not rows[0]["keywords"], rows[0])
    check("the succeeded section got its keywords", bool(rows[1]["keywords"]), rows[1])


def test_section_count_cap_enforced():
    print(f"\n# Section-count cap: at most server.NCERT_CONTENT_KEYWORDS_SECTION_CAP sections per request")
    tc, fake = setup()
    seed_chapter(fake, [{"heading": f"7.{i} Section {i}", "text": base.CLEAN_PARA, "keywords": None} for i in range(4)])

    orig_cap = server.NCERT_CONTENT_KEYWORDS_SECTION_CAP
    server.NCERT_CONTENT_KEYWORDS_SECTION_CAP = 2
    try:
        data = call_keywords(tc, mode="all").json()
    finally:
        server.NCERT_CONTENT_KEYWORDS_SECTION_CAP = orig_cap

    check("no error", not data.get("error"), data)
    check("only the capped number of sections were attempted", data.get("calls") == 2, data)
    check("capped is reported true", data.get("capped") is True, data)
    check("2 sections remain for a follow-up request", data.get("remaining") == 2, data)


def test_dollar_cost_cap_enforced():
    print("\n# Dollar cost cap: a request stops once NCERT_CONTENT_KEYWORDS_COST_CAP is reached")
    # ~200k input tokens/call at the (forced) peak rate costs ~$0.06/call -- 2 calls cross the
    # $0.10 cap, so a 3rd pending section must be left untouched this request.
    tc, fake = setup(deepseek_input_tokens=200_000)
    seed_chapter(fake, [{"heading": f"7.{i} Section {i}", "text": base.CLEAN_PARA, "keywords": None} for i in range(3)])
    data = call_keywords(tc, mode="all").json()
    check("no error", not data.get("error"), data)
    check("fewer than all 3 sections were attempted (cost cap reached)", 0 < data.get("calls", 0) < 3, data)
    check("capped is reported true", data.get("capped") is True, data)
    check("cost stayed near the $0.10 cap, not the full 3-section cost", data.get("cost", 0) < 0.15, data.get("cost"))


def test_cost_and_model_calls_logged_to_chapter_log():
    print("\n# A successful run logs cost/model_calls to ncert_chapter_log (source=manual)")
    tc, fake = setup()
    seed_chapter(fake, [{"heading": "7.1 X", "text": base.CLEAN_PARA, "keywords": None}])
    data = call_keywords(tc, mode="pending").json()
    check("no error", not data.get("error"), data)
    check("exactly one log row written", len(fake.chapter_log.rows) == 1, fake.chapter_log.rows)
    row = fake.chapter_log.rows[0]
    # The ncert_chapter_log.source check constraint only allows 'pdf'/'manual' -- keyword
    # regeneration already logs under 'manual' elsewhere (admin_ncert_content_section), so this
    # follows that same convention.
    check("source is 'manual'", row.get("source") == "manual", row)
    check("model_calls matches succeeded count", row.get("model_calls") == data.get("succeeded"), (row, data))
    check("cost matches the response's cost", row.get("keywords_cost_usd") == data.get("cost"), (row, data))
    check("model is recorded", row.get("model") == server.DEEPSEEK_MODEL, row)


def test_no_such_chapter_and_bad_inputs_are_plain_string_errors():
    print("\n# No such chapter / invalid class/subject/mode -- all plain string errors, never [object Object]")
    tc, fake = setup()
    seed_chapter(fake, [{"heading": "7.1 X", "text": base.CLEAN_PARA, "keywords": None}])

    missing = call_keywords(tc, chapter="Nonexistent Chapter").json()
    check("plain string error for a missing chapter", isinstance(missing.get("error"), str), missing)

    bad_class = call_keywords(tc, cls="99").json()
    check("plain string error for an invalid class", isinstance(bad_class.get("error"), str), bad_class)

    bad_subject = call_keywords(tc, subject="History").json()
    check("plain string error for an invalid subject", isinstance(bad_subject.get("error"), str), bad_subject)

    bad_mode = call_keywords(tc, mode="everything").json()
    check("plain string error for an invalid mode", isinstance(bad_mode.get("error"), str), bad_mode)


def test_manual_edit_clears_stale_keywords_regen_overwrites_them():
    print("\n# Manual add/edit form: saving heading/text clears stale keywords unless regenerating")
    tc, fake = setup()
    body = {"class": "12", "subject": "Chemistry", "chapter": "Solutions", "heading": "2.1 Types Of Solutions", "ordinal": 0, "text": base.CLEAN_PARA}
    tc.post("/admin/ncert-content/section", json=body, headers={"X-Admin-Key": ADMIN_KEY})
    fake.table.update({"class": "eq.12", "subject": "eq.Chemistry", "chapter": "eq.Solutions", "ordinal": "eq.0"}, {"keywords": "stale, old, keywords"})
    check("setup: the section has keywords before the edit", fake.table.rows[0]["keywords"] == "stale, old, keywords", fake.table.rows[0])

    edited = dict(body, text=base.CLEAN_PARA + " A hand-edited sentence.")
    tc.post("/admin/ncert-content/section", json=edited, headers={"X-Admin-Key": ADMIN_KEY})
    check("editing heading/text without regenerate_keywords clears the stale keywords", fake.table.rows[0]["keywords"] is None, fake.table.rows[0])

    regen = dict(edited, regenerate_keywords=True)
    tc.post("/admin/ncert-content/section", json=regen, headers={"X-Admin-Key": ADMIN_KEY})
    check("regenerate_keywords=True writes fresh keywords in the same save", bool(fake.table.rows[0]["keywords"]), fake.table.rows[0])


def main():
    test_admin_only_contractor_rejected()
    test_pending_only_touches_only_empty_sections()
    test_pending_mode_with_nothing_pending_is_a_no_op()
    test_all_mode_regenerates_every_section()
    test_partial_failure_keeps_failed_section_pending_and_continues()
    test_section_count_cap_enforced()
    test_dollar_cost_cap_enforced()
    test_cost_and_model_calls_logged_to_chapter_log()
    test_no_such_chapter_and_bad_inputs_are_plain_string_errors()
    test_manual_edit_clears_stale_keywords_regen_overwrites_them()

    print(f"\nTOTAL: {len(_passed)} passed, {len(_failed)} failed")
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
