"""Tests for the "Save all" batch endpoint: POST /admin/ncert-content/chapter/save-all in
server.py. Used by admin-ncert-content.html's loaded-chapter section editor, which no longer saves
one section at a time -- it batches every edited section into one write.

Fully offline, same style as test_admin_ncert_content.py: Supabase is an in-memory fake table, no
real network/model calls (this feature makes none).

Run with: python test_ncert_save_all.py
"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import server
import test_admin_ncert_content as base  # reuses FakeAsyncClient/setup/CLEAN_PARA

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
    for i, r in enumerate(rows):
        fake.table.insert({
            "class": "11", "subject": "Biology", "chapter": "Save All Test",
            "heading": r["heading"], "ordinal": i, "text": r["text"],
            "word_count": len(r["text"].split()), "keywords": r.get("keywords"),
        })


def save_all(tc, edits, chapter="Save All Test", cls="11", subject="Biology", key=ADMIN_KEY):
    headers = {"X-Admin-Key": key} if key else None
    return tc.post(
        "/admin/ncert-content/chapter/save-all",
        json={"class": cls, "subject": subject, "chapter": chapter, "edits": edits},
        headers=headers,
    )


def test_three_edits_write_in_one_request():
    print("\n# Editing 3 sections then Save all writes all 3 in one safe-replace cycle")
    tc, fake = setup()
    seed_chapter(fake, [
        {"heading": "9.1 One", "text": base.CLEAN_PARA, "keywords": "a, b, c"},
        {"heading": "9.2 Two", "text": base.CLEAN_PARA, "keywords": "d, e, f"},
        {"heading": "9.3 Three", "text": base.CLEAN_PARA, "keywords": "g, h, i"},
    ])
    fake.calls.clear()
    edits = [
        {"ordinal": 0, "heading": "9.1 One (edited)", "text": base.CLEAN_PARA + " more one."},
        {"ordinal": 1, "heading": "9.2 Two (edited)", "text": base.CLEAN_PARA + " more two."},
        {"ordinal": 2, "heading": "9.3 Three (edited)", "text": base.CLEAN_PARA + " more three."},
    ]
    data = save_all(tc, edits).json()
    check("no error", not data.get("error"), data)
    check("all 3 results report ok", all(r["ok"] for r in data.get("results", [])), data.get("results"))
    check("3 sections written", data.get("section_count") == 3, data)

    rows = fake.table.get({"class": "eq.11", "subject": "eq.Biology", "chapter": "eq.Save All Test", "order": "ordinal"})
    check("all 3 headings actually updated", [r["heading"] for r in rows] == [
        "9.1 One (edited)", "9.2 Two (edited)", "9.3 Three (edited)",
    ], rows)

    # Exactly one safe-replace cycle: one POST (insert under temp label) + one DELETE (old rows) +
    # one PATCH (rename temp label back) to ncert_sections -- not 3 of each.
    sections_posts = [c for c in fake.calls if c[0] == "POST" and "ncert_sections" in c[1]]
    sections_deletes = [c for c in fake.calls if c[0] == "DELETE" and "ncert_sections" in c[1]]
    sections_patches = [c for c in fake.calls if c[0] == "PATCH" and "ncert_sections" in c[1]]
    check("exactly one insert call for the whole batch", len(sections_posts) == 1, sections_posts)
    check("exactly one delete call for the whole batch", len(sections_deletes) == 1, sections_deletes)
    check("exactly one rename (patch) call for the whole batch", len(sections_patches) == 1, sections_patches)


def test_partial_failure_keeps_the_rest():
    print("\n# A bad entry in the batch (unknown ordinal, blank heading) fails only that entry")
    tc, fake = setup()
    seed_chapter(fake, [
        {"heading": "9.1 One", "text": base.CLEAN_PARA},
        {"heading": "9.2 Two", "text": base.CLEAN_PARA},
    ])
    edits = [
        {"ordinal": 0, "heading": "9.1 One (edited)", "text": base.CLEAN_PARA + " more."},
        {"ordinal": 1, "heading": "   ", "text": "blank heading should fail"},
        {"ordinal": 5, "heading": "9.9 Does not exist", "text": "bad ordinal should fail"},
    ]
    data = save_all(tc, edits).json()
    check("no top-level error (one bad entry isn't a request failure)", not data.get("error"), data)
    results_by_ordinal = {r["ordinal"]: r for r in data.get("results", [])}
    check("ordinal 0 succeeded", results_by_ordinal.get(0, {}).get("ok") is True, results_by_ordinal)
    check("ordinal 1 (blank heading) failed with a plain string error", results_by_ordinal.get(1, {}).get("ok") is False and isinstance(results_by_ordinal[1].get("error"), str), results_by_ordinal)
    check("ordinal 5 (unknown) failed with a plain string error", results_by_ordinal.get(5, {}).get("ok") is False and isinstance(results_by_ordinal[5].get("error"), str), results_by_ordinal)

    rows = fake.table.get({"class": "eq.11", "subject": "eq.Biology", "chapter": "eq.Save All Test", "order": "ordinal"})
    check("section 0's edit still applied", rows[0]["heading"] == "9.1 One (edited)", rows[0])
    check("section 1 is unchanged (its bad edit never applied)", rows[1]["heading"] == "9.2 Two", rows[1])


def test_keywords_cleared_only_for_sections_that_actually_changed():
    print("\n# Keywords are cleared only for sections whose heading or text actually changed")
    tc, fake = setup()
    seed_chapter(fake, [
        {"heading": "9.1 One", "text": base.CLEAN_PARA, "keywords": "kept, keywords"},
        {"heading": "9.2 Two", "text": base.CLEAN_PARA, "keywords": "stale, keywords"},
    ])
    edits = [
        {"ordinal": 0, "heading": "9.1 One", "text": base.CLEAN_PARA},  # identical to stored -- no-op
        {"ordinal": 1, "heading": "9.2 Two (edited)", "text": base.CLEAN_PARA},  # heading actually changed
    ]
    data = save_all(tc, edits).json()
    check("no error", not data.get("error"), data)

    rows = fake.table.get({"class": "eq.11", "subject": "eq.Biology", "chapter": "eq.Save All Test", "order": "ordinal"})
    check("unchanged section keeps its keywords", rows[0]["keywords"] == "kept, keywords", rows[0])
    check("actually-changed section's keywords are cleared", not rows[1]["keywords"], rows[1])


def test_structural_flags_computed_once_over_whole_batch():
    print("\n# Structural validation (duplicate heading) is caught across the whole batch")
    tc, fake = setup()
    seed_chapter(fake, [
        {"heading": "9.1 One", "text": base.CLEAN_PARA},
        {"heading": "9.2 Two", "text": base.CLEAN_PARA},
    ])
    edits = [{"ordinal": 1, "heading": "9.1 One", "text": base.CLEAN_PARA}]  # now duplicates ordinal 0
    data = save_all(tc, edits).json()
    check("no error", not data.get("error"), data)
    check("duplicate heading flag is raised", any("duplicate" in f.lower() for f in data.get("flags", [])), data.get("flags"))


def test_contractor_key_rejected():
    print("\n# Contractor key is rejected on the new Save-all endpoint")
    tc, fake = setup()
    seed_chapter(fake, [{"heading": "9.1 One", "text": base.CLEAN_PARA}])
    server._admin_login_attempts.clear()
    res = save_all(tc, [{"ordinal": 0, "heading": "x", "text": "y"}], key=CONTRACTOR_KEY)
    check("contractor key refused (401)", res.status_code == 401, res.status_code)
    server._admin_login_attempts.clear()
    res2 = save_all(tc, [{"ordinal": 0, "heading": "x", "text": "y"}], key=None)
    check("no key refused (401)", res2.status_code == 401, res2.status_code)


def test_no_such_chapter_and_bad_body_are_plain_string_errors():
    print("\n# No such chapter / empty edits list -- plain string errors, never [object Object]")
    tc, fake = setup()
    seed_chapter(fake, [{"heading": "9.1 One", "text": base.CLEAN_PARA}])

    missing = save_all(tc, [{"ordinal": 0, "heading": "x", "text": "y"}], chapter="Nonexistent").json()
    check("plain string error for a missing chapter", isinstance(missing.get("error"), str), missing)

    empty = save_all(tc, []).json()
    check("plain string error for an empty edits list", isinstance(empty.get("error"), str), empty)

    not_a_list = tc.post(
        "/admin/ncert-content/chapter/save-all",
        json={"class": "11", "subject": "Biology", "chapter": "Save All Test", "edits": "not a list"},
        headers={"X-Admin-Key": ADMIN_KEY},
    ).json()
    check("plain string error when edits is not a list", isinstance(not_a_list.get("error"), str), not_a_list)


def test_one_log_row_for_the_whole_batch():
    print("\n# Save-all writes exactly one ncert_chapter_log row for the whole batch, not one per edit")
    tc, fake = setup()
    seed_chapter(fake, [
        {"heading": "9.1 One", "text": base.CLEAN_PARA},
        {"heading": "9.2 Two", "text": base.CLEAN_PARA},
        {"heading": "9.3 Three", "text": base.CLEAN_PARA},
    ])
    edits = [
        {"ordinal": 0, "heading": "9.1 One (edited)", "text": base.CLEAN_PARA},
        {"ordinal": 1, "heading": "9.2 Two (edited)", "text": base.CLEAN_PARA},
        {"ordinal": 2, "heading": "9.3 Three (edited)", "text": base.CLEAN_PARA},
    ]
    data = save_all(tc, edits).json()
    check("no error", not data.get("error"), data)
    check("exactly one log row for 3 edited sections", len(fake.chapter_log.rows) == 1, fake.chapter_log.rows)
    check("the log row's source is 'manual'", fake.chapter_log.rows[0].get("source") == "manual", fake.chapter_log.rows[0])


def main():
    test_three_edits_write_in_one_request()
    test_partial_failure_keeps_the_rest()
    test_keywords_cleared_only_for_sections_that_actually_changed()
    test_structural_flags_computed_once_over_whole_batch()
    test_contractor_key_rejected()
    test_no_such_chapter_and_bad_body_are_plain_string_errors()
    test_one_log_row_for_the_whole_batch()

    print(f"\nTOTAL: {len(_passed)} passed, {len(_failed)} failed")
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
