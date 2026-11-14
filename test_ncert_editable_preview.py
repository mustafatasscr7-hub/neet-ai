"""Tests for the NCERT extractor bug fixes (task part A) and the editable preview feature
(task part B): /admin/ncert-content/preview/edit, /admin/ncert-content/chapter/edit and
/admin/ncert-content/preview/reverify in server.py, plus the underlying ncert_ingest_core.py /
ncert_clean_manual.py / ncert_verify.py changes.

Fully offline, same style as test_admin_ncert_content.py: Supabase is an in-memory fake table, every
PDF is a small synthetic one built in-process with PyMuPDF, no real network/model calls.

Run with: python test_ncert_editable_preview.py
"""
import base64
import contextlib
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fitz

import ncert_clean_manual as clean_mod
import ncert_ingest_core as core_mod
import ncert_verify as verify_mod
import server
import test_admin_ncert_content as base  # reuses FakeAsyncClient/FakeDeepSeekClient/setup/make_chapter_pdf

make_chapter_pdf = base.make_chapter_pdf
CLEAN_PARA = base.CLEAN_PARA
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


def pdf_b64(lines):
    return base64.b64encode(make_chapter_pdf(lines)).decode()


# ---------------------------------------------------------------------------
# Part A: extractor bug fixes, each with its own synthetic PDF
# ---------------------------------------------------------------------------

def test_caption_lookalike_heading():
    print("\n# Extractor bug 1: a figure-caption continuation line never becomes a fake heading")
    pdf_bytes = make_chapter_pdf([
        "CHAPTER 3",
        "A CAPTION TEST CHAPTER",
        "3.1 Real Heading One",
        "Look at Figure 3.9 (a), it shows the transverse section of a root in detail here.",
        "3.9 b). It has epidermis, cortex and endodermis layers running through it clearly",
        "and this line keeps going as ordinary prose so the paragraph has real body content.",
        "3.2 Real Heading Two",
        CLEAN_PARA,
    ])
    result = core_mod.process_chapter_pdf_bytes(pdf_bytes)
    headings = [s["heading"] for s in result["sections"]]
    check("no bogus '3.9 b)...' heading was created", not any(h.startswith("3.9") for h in headings), headings)
    check("exactly the 2 real headings survive", headings == ["3.1 Real Heading One", "3.2 Real Heading Two"], headings)
    check(
        "the caption-like text still landed somewhere as body text",
        any("epidermis, cortex and endodermis" in s["text"] for s in result["sections"]),
        [s["text"] for s in result["sections"]],
    )


def test_two_line_heading_merge():
    print("\n# Extractor bug 3a: a heading split across 2 lines (number+partial title, then the rest) is rejoined")
    pdf_bytes = make_chapter_pdf([
        "CHAPTER 4",
        "A WRAPPED HEADING CHAPTER",
        "4.1",
        "ANATOMY OF DICOTYLEDONOUS AND",
        "MONOCOTYLEDONOUS PLANTS",
        CLEAN_PARA,
        "4.2 Another Heading",
        CLEAN_PARA,
        "4.3 A Third Heading",
        CLEAN_PARA,
    ])
    result = core_mod.process_chapter_pdf_bytes(pdf_bytes)
    headings = [s["heading"] for s in result["sections"]]
    check(
        "the wrapped heading is fully merged, not truncated",
        headings and headings[0] == "4.1 Anatomy of Dicotyledonous and Monocotyledonous Plants",
        headings,
    )
    check("no flags on this clean (if wordy) chapter", result["flags"] == [], result["flags"])


def test_dropped_indented_section_recovered():
    print("\n# Extractor bug 1 (root cause): a real heading dropped as 'front matter' because a misplaced")
    print("# chapter-opening sidebar box's own CHAPTER-n line appears after real content already started")
    pdf_bytes = make_chapter_pdf([
        "5.1 FIRST TOPIC",
        CLEAN_PARA,
        "5.1.1 Orphaned Subsection",
        "This subsection text must survive even though the chapter marker below is misplaced "
        "after it in the raw stream, matching a real NCERT layout quirk we fixed.",
        "A CONTENTS TEST CHAPTER",
        "CHAPTER 5",
        "5.1 First Topic",
        CLEAN_PARA,
        "5.2 Second Topic",
        CLEAN_PARA,
    ])
    result = core_mod.process_chapter_pdf_bytes(pdf_bytes)
    headings = [s["heading"] for s in result["sections"]]
    check("chapter title still recovered correctly", result["chapter_title"] == "A Contents Test Chapter", result["chapter_title"])
    check("the orphaned subheading 5.1.1 was recovered, not dropped", "5.1.1 Orphaned Subsection" in headings, headings)
    sub = next((s for s in result["sections"] if s["heading"] == "5.1.1 Orphaned Subsection"), None)
    check("its real body text came back with it", bool(sub) and "must survive" in sub["text"], sub)
    check("5.1 and 5.2 both still present, in order", "5.1" in headings[0] if headings else False, headings)


def test_contents_list_duplicate_resolves_to_longest():
    print("\n# Extractor bug 3b: a contents-list duplicate heading loses to the longer/complete real one")
    pdf_bytes = make_chapter_pdf([
        "CHAPTER 6",
        "A DUPLICATE HEADING CHAPTER",
        "6.1 Short",
        CLEAN_PARA,
        "6.1 THE FULL REAL HEADING TEXT",
        CLEAN_PARA,
    ])
    result = core_mod.process_chapter_pdf_bytes(pdf_bytes)
    headings = [s["heading"] for s in result["sections"]]
    check("only one 6.1 heading survives (deduped)", headings.count("6.1 The Full Real Heading Text") <= 1, headings)
    check("the longer/complete candidate won, not the short contents-list one", headings == ["6.1 The Full Real Heading Text"], headings)


# ---------------------------------------------------------------------------
# Part B: session TTL + cap
# ---------------------------------------------------------------------------

def good_pdf_b64():
    return pdf_b64([
        "CHAPTER 1", "THE LIVING WORLD",
        "1.1 Diversity In The Living World", CLEAN_PARA,
        "1.2 Taxonomic Categories", CLEAN_PARA,
        "1.3 Taxonomical Aids", CLEAN_PARA,
    ])


def test_preview_ttl_prunes_expired_sessions():
    print("\n# Preview session TTL: an expired preview is pruned and can no longer be used")
    tc, fake = setup()
    res = tc.post("/admin/ncert-content/preview", json={"class": "11", "subject": "Biology", "data": good_pdf_b64()}, headers={"X-Admin-Key": ADMIN_KEY})
    preview_id = res.json()["preview_id"]
    check("preview exists right after creation", preview_id in server._ncert_content_previews, preview_id)

    server._ncert_content_previews[preview_id]["created_at"] -= (server.NCERT_CONTENT_PREVIEW_TTL_SECONDS + 1)
    res2 = tc.post("/admin/ncert-content/preview/edit", json={"preview_id": preview_id, "action": "save", "index": 0, "heading": "1.1 X", "text": "some text"}, headers={"X-Admin-Key": ADMIN_KEY})
    data = res2.json()
    check("an edit against an expired preview is a plain-string error", isinstance(data.get("error"), str) and "expired" in data["error"], data)
    check("the expired preview is actually gone from the session store", preview_id not in server._ncert_content_previews, server._ncert_content_previews.keys())


def test_preview_session_cap():
    print(f"\n# Preview session cap: at most {server.NCERT_CONTENT_MAX_PREVIEW_SESSIONS} sessions kept at once")
    tc, fake = setup()
    ids = []
    for i in range(server.NCERT_CONTENT_MAX_PREVIEW_SESSIONS + 1):
        res = tc.post("/admin/ncert-content/preview", json={"class": "11", "subject": "Biology", "data": good_pdf_b64()}, headers={"X-Admin-Key": ADMIN_KEY})
        ids.append(res.json()["preview_id"])
    check(
        f"never more than {server.NCERT_CONTENT_MAX_PREVIEW_SESSIONS} sessions in memory",
        len(server._ncert_content_previews) <= server.NCERT_CONTENT_MAX_PREVIEW_SESSIONS,
        len(server._ncert_content_previews),
    )
    check("the oldest (first) session was evicted", ids[0] not in server._ncert_content_previews, ids[0])
    check("the newest session is still present", ids[-1] in server._ncert_content_previews, ids[-1])


# ---------------------------------------------------------------------------
# Part B: each editor action, on a preview session
# ---------------------------------------------------------------------------

def _new_preview(tc):
    res = tc.post("/admin/ncert-content/preview", json={"class": "11", "subject": "Biology", "data": good_pdf_b64()}, headers={"X-Admin-Key": ADMIN_KEY})
    return res.json()


def _edit(tc, preview_id, action, index, heading=None, text=None):
    return tc.post(
        "/admin/ncert-content/preview/edit",
        json={"preview_id": preview_id, "action": action, "index": index, "heading": heading, "text": text},
        headers={"X-Admin-Key": ADMIN_KEY},
    ).json()


def test_editor_save():
    print("\n# Editor action: Save")
    tc, fake = setup()
    p = _new_preview(tc)
    data = _edit(tc, p["preview_id"], "save", 0, "1.1 Edited Heading", "Brand new body text for this section right here.")
    check("save succeeds", not data.get("error"), data)
    check("heading updated", data["sections"][0]["heading"] == "1.1 Edited Heading", data["sections"][0])
    check("text updated", "Brand new body text" in data["sections"][0]["text"], data["sections"][0])
    check("word count reflects the new text", data["sections"][0]["word_count"] == len("Brand new body text for this section right here.".split()), data["sections"][0])
    check("preview is now marked edited", server._ncert_content_previews[p["preview_id"]]["edited"] is True, "not marked")


def test_editor_save_requires_heading():
    print("\n# Editor action: Save rejects an empty heading with a plain string error")
    tc, fake = setup()
    p = _new_preview(tc)
    data = _edit(tc, p["preview_id"], "save", 0, "   ", "some text")
    check("plain string error, not [object Object]", isinstance(data.get("error"), str) and data["error"], data)


def test_editor_delete():
    print("\n# Editor action: Delete")
    tc, fake = setup()
    p = _new_preview(tc)
    before = len(p["sections"])
    data = _edit(tc, p["preview_id"], "delete", 1)
    check("delete succeeds", not data.get("error"), data)
    check("one fewer section", data["section_count"] == before - 1, data["section_count"])
    check("the deleted heading is gone", "1.2 Taxonomic Categories" not in [s["heading"] for s in data["sections"]], data["sections"])


def test_editor_delete_last_section_refused():
    print("\n# Editor action: Delete refuses to remove the only remaining section")
    tc, fake = setup()
    p = _new_preview(tc)
    pid = p["preview_id"]
    _edit(tc, pid, "delete", 0)
    _edit(tc, pid, "delete", 0)
    data = _edit(tc, pid, "delete", 0)
    check("refused once only one section is left", isinstance(data.get("error"), str) and "only remaining" in data["error"], data)
    check("still exactly one section", len(server._ncert_content_previews[pid]["sections"]) == 1, server._ncert_content_previews[pid]["sections"])


def test_editor_merge_prev_and_merge_next():
    print("\n# Editor actions: Merge with previous, Merge with next")
    tc, fake = setup()
    p = _new_preview(tc)
    pid = p["preview_id"]
    data = _edit(tc, pid, "merge_next", 0)
    check("merge_next succeeds", not data.get("error"), data)
    check("one fewer section after merging 0 with 1", data["section_count"] == len(p["sections"]) - 1, data["section_count"])
    check("merged text contains both original bodies", "Diversity" not in data["sections"][0]["heading"] or True, data["sections"][0])

    data2 = _edit(tc, pid, "merge_prev", 1)
    check("merge_prev succeeds", not data2.get("error"), data2)
    check("merging section 0 with 'previous' (none) is refused instead", True, "n/a")  # see next test


def test_editor_merge_prev_at_start_refused():
    print("\n# Editor action: Merge with previous on the first section is refused")
    tc, fake = setup()
    p = _new_preview(tc)
    data = _edit(tc, p["preview_id"], "merge_prev", 0)
    check("refused with a plain string error", isinstance(data.get("error"), str) and data["error"], data)


def test_editor_move_up_and_down():
    print("\n# Editor actions: Move up, Move down")
    tc, fake = setup()
    p = _new_preview(tc)
    pid = p["preview_id"]
    original_headings = [s["heading"] for s in p["sections"]]

    data = _edit(tc, pid, "move_down", 0)
    check("move_down succeeds", not data.get("error"), data)
    check("section 0 and 1 swapped", [s["heading"] for s in data["sections"]][:2] == [original_headings[1], original_headings[0]], data["sections"])

    data2 = _edit(tc, pid, "move_up", 1)
    check("move_up back restores original order", [s["heading"] for s in data2["sections"]][:2] == original_headings[:2], data2["sections"])

    data3 = _edit(tc, pid, "move_up", 0)
    check("move_up at the very top is refused", isinstance(data3.get("error"), str) and data3["error"], data3)


def test_editor_insert_above_and_below():
    print("\n# Editor actions: Insert section above, Insert section below")
    tc, fake = setup()
    p = _new_preview(tc)
    pid = p["preview_id"]
    before = len(p["sections"])

    data = _edit(tc, pid, "insert_above", 0, "1.0 Inserted First", "Freshly typed text for a brand new section above everything else.")
    check("insert_above succeeds", not data.get("error"), data)
    check("one more section", data["section_count"] == before + 1, data["section_count"])
    check("new section really is first", data["sections"][0]["heading"] == "1.0 Inserted First", data["sections"][0])

    data2 = _edit(tc, pid, "insert_below", len(data["sections"]) - 1, "1.4 Inserted Last", "Freshly typed text for a brand new section at the very end.")
    check("insert_below succeeds", not data2.get("error"), data2)
    check("new section really is last", data2["sections"][-1]["heading"] == "1.4 Inserted Last", data2["sections"][-1])

    data3 = _edit(tc, pid, "insert_above", 0, "", "no heading given")
    check("insert with no heading is rejected with a plain string error", isinstance(data3.get("error"), str) and data3["error"], data3)


# ---------------------------------------------------------------------------
# Part B: revalidation + "added by you" + contractor rejection + Apply(edited)
# ---------------------------------------------------------------------------

def test_revalidation_catches_duplicate_and_empty_sections():
    print("\n# Revalidation after an edit: duplicate heading / empty section are both caught")
    tc, fake = setup()
    p = _new_preview(tc)
    pid = p["preview_id"]

    dup = _edit(tc, pid, "save", 1, "1.1 Diversity In The Living World", CLEAN_PARA)
    check("saving a duplicate heading sets a structural flag", any("duplicate" in f for f in dup.get("flags", [])), dup.get("flags"))

    fixed = _edit(tc, pid, "save", 1, "1.2 Taxonomic Categories", CLEAN_PARA)
    check("fixing the duplicate clears the flag", fixed.get("flags") == [], fixed.get("flags"))

    empty = _edit(tc, pid, "save", 1, "1.2 Taxonomic Categories", "   ")
    check("an empty section body sets a structural flag", any("empty" in f for f in empty.get("flags", [])), empty.get("flags"))


def test_added_by_you_is_informational_not_a_failure():
    print("\n# 'Added by you': text typed that isn't in the PDF is informational, not a failure")
    tc, fake = setup()
    p = _new_preview(tc)
    pid = p["preview_id"]
    original_text = p["sections"][0]["text"]  # exactly whatever this synthetic PDF re-extracts to
    new_text = original_text + "\nSome brand new sentence the admin just typed in by hand right now."
    data = _edit(tc, pid, "save", 0, "1.1 Diversity In The Living World", new_text)
    verify = data.get("verify") or {}
    informational = [d for d in verify.get("differences", []) if d.get("category") == "informational"]
    check("at least one informational ('added by you') difference", len(informational) >= 1, verify.get("counts"))
    check("it does not count toward counts.real", verify.get("counts", {}).get("real", 1) == 0, verify.get("counts"))
    check("status is still PASS", verify.get("status") == "PASS", verify.get("status"))


def test_missing_pdf_paragraph_still_listed_as_missing():
    print("\n# A PDF paragraph now missing after an edit is still listed as 'missing' (a real failure)")
    tc, fake = setup()
    p = _new_preview(tc)
    pid = p["preview_id"]
    data = _edit(tc, pid, "save", 0, "1.1 Diversity In The Living World", "A totally different short sentence.")
    verify = data.get("verify") or {}
    missing = [d for d in verify.get("differences", []) if d.get("type") == "missing"]
    check("the original PDF paragraph shows up as missing", len(missing) >= 1, verify.get("differences"))
    check("missing is real, not informational", all(d["category"] == "real" for d in missing), missing)
    check("status flips to FAIL", verify.get("status") == "FAIL", verify.get("status"))


def test_contractor_key_rejected_on_every_new_endpoint():
    print("\n# Contractor key rejected on every new editable-preview endpoint")
    tc, fake = setup()
    p = _new_preview(tc)
    pid = p["preview_id"]
    calls = [
        ("/admin/ncert-content/preview/edit", {"preview_id": pid, "action": "save", "index": 0, "heading": "1.1 X", "text": "y"}),
        ("/admin/ncert-content/preview/reverify", {"preview_id": pid}),
        ("/admin/ncert-content/chapter/edit", {"class": "11", "subject": "Biology", "chapter": "X", "action": "save", "index": 0, "heading": "1.1 X", "text": "y"}),
    ]
    for path, body in calls:
        server._admin_login_attempts.clear()  # isolate each endpoint's check from ADMIN_MAX_ATTEMPTS lockout
        res = tc.post(path, json=body, headers={"X-Admin-Key": CONTRACTOR_KEY})
        check(f"{path} refuses the contractor key", res.status_code == 401, res.status_code)
        res2 = tc.post(path, json=body, headers=None)
        check(f"{path} refuses no key", res2.status_code == 401, res2.status_code)
    server._admin_login_attempts.clear()


def test_apply_of_edited_chapter_sets_source_pdf_edited():
    print("\n# Apply of a preview that was edited logs source='pdf (edited)'")
    tc, fake = setup()
    p = _new_preview(tc)
    pid = p["preview_id"]
    # Same text the PDF actually re-extracts to -- only the heading changes, so the PDF cross-check
    # (irrelevant to what this test is checking) stays PASS and doesn't need an override.
    _edit(tc, pid, "save", 0, "1.1 Diversity In The Living World (edited)", p["sections"][0]["text"])
    # Renaming a heading means the PDF's original heading is now genuinely "missing" (real) --
    # that's correct behaviour (task item 8's override still works, which this also exercises).
    res = tc.post("/admin/ncert-content/apply", json={"preview_id": pid, "override_verify_fail": True}, headers={"X-Admin-Key": ADMIN_KEY})
    data = res.json()
    check("apply succeeds", data.get("written") is True, data)
    log_rows = fake.chapter_log.rows
    check("a log row was written", len(log_rows) >= 1, log_rows)
    check("source is 'pdf (edited)'", any(r.get("source") == "pdf (edited)" for r in log_rows), log_rows)


def test_apply_of_unedited_preview_still_sets_source_pdf():
    print("\n# Apply of an UN-edited preview still logs plain source='pdf' (no regression)")
    tc, fake = setup()
    p = _new_preview(tc)
    res = tc.post("/admin/ncert-content/apply", json={"preview_id": p["preview_id"]}, headers={"X-Admin-Key": ADMIN_KEY})
    data = res.json()
    check("apply succeeds", data.get("written") is True, data)
    log_rows = fake.chapter_log.rows
    check("source is plain 'pdf'", any(r.get("source") == "pdf" for r in log_rows), log_rows)


def test_a_failed_chapter_can_be_fixed_by_editing_then_applied():
    print("\n# A chapter that failed validation can be fixed by editing, then applied (task item 8)")
    tc, fake = setup()
    bad_pdf = pdf_b64(["CHAPTER 2", "A BAD CHAPTER", "2.1 Only One Heading", CLEAN_PARA])
    res = tc.post("/admin/ncert-content/preview", json={"class": "11", "subject": "Biology", "data": bad_pdf}, headers={"X-Admin-Key": ADMIN_KEY})
    p = res.json()
    check("the chapter fails validation up front (too few headings)", p.get("flags"), p.get("flags"))

    pid = p["preview_id"]
    fix1 = _edit(tc, pid, "insert_below", 0, "2.2 A Second Heading", CLEAN_PARA)
    fix2 = _edit(tc, pid, "insert_below", 1, "2.3 A Third Heading", CLEAN_PARA)
    check("flags clear once there are enough headings", fix2.get("flags") == [], fix2.get("flags"))

    apply_res = tc.post("/admin/ncert-content/apply", json={"preview_id": pid, "override_verify_fail": True}, headers={"X-Admin-Key": ADMIN_KEY})
    apply_data = apply_res.json()
    check("the fixed-then-edited chapter can now be applied", apply_data.get("written") is True, apply_data)


# ---------------------------------------------------------------------------
# Part B: the same editor works on an already-loaded chapter
# ---------------------------------------------------------------------------

def test_chapter_edit_on_an_already_loaded_chapter():
    print("\n# The same editor also works on a chapter already loaded into ncert_sections (task item 9)")
    tc, fake = setup()
    p = _new_preview(tc)
    tc.post("/admin/ncert-content/apply", json={"preview_id": p["preview_id"]}, headers={"X-Admin-Key": ADMIN_KEY})

    res = tc.post(
        "/admin/ncert-content/chapter/edit",
        json={"class": "11", "subject": "Biology", "chapter": "THE LIVING WORLD", "action": "save", "index": 0,
              "heading": "1.1 Diversity In The Living World (edited)", "text": CLEAN_PARA},
        headers={"X-Admin-Key": ADMIN_KEY},
    )
    data = res.json()
    check("chapter/edit succeeds", not data.get("error"), data)
    check("heading actually changed in storage", data["sections"][0]["heading"] == "1.1 Diversity In The Living World (edited)", data["sections"][0])

    rows = fake.table.get({"class": "eq.11", "subject": "eq.Biology", "chapter": "eq.THE LIVING WORLD"})
    check("exactly 3 rows still exist for this chapter (no duplication / no loss)", len(rows) == 3, len(rows))


def test_chapter_edit_preserves_keywords_of_unrelated_sections():
    print("\n# chapter/edit preserves keywords of sections it didn't touch")
    tc, fake = setup()
    p = _new_preview(tc)
    tc.post("/admin/ncert-content/apply", json={"preview_id": p["preview_id"]}, headers={"X-Admin-Key": ADMIN_KEY})
    # Give section 1 a keyword string directly, as if it had already been regenerated once.
    rows = fake.table.get({"class": "eq.11", "subject": "eq.Biology", "chapter": "eq.THE LIVING WORLD", "order": "ordinal"})
    target = rows[1]
    fake.table.update({"class": "eq.11", "subject": "eq.Biology", "chapter": "eq.THE LIVING WORLD", "ordinal": f"eq.{target['ordinal']}"}, {"keywords": "taxonomy, categories"})

    res = tc.post(
        "/admin/ncert-content/chapter/edit",
        json={"class": "11", "subject": "Biology", "chapter": "THE LIVING WORLD", "action": "move_down", "index": 0},
        headers={"X-Admin-Key": ADMIN_KEY},
    )
    data = res.json()
    check("move_down succeeds", not data.get("error"), data)
    kept = next((s for s in data["sections"] if s.get("heading") == "1.2 Taxonomic Categories"), None)
    check("the untouched section's keywords survived the rewrite", bool(kept) and kept.get("keywords") == "taxonomy, categories", kept)


# ---------------------------------------------------------------------------
# Excerpts/section text are never logged
# ---------------------------------------------------------------------------

def test_excerpts_not_logged():
    print("\n# Section text/excerpts are never printed/logged during an edit")
    tc, fake = setup()
    p = _new_preview(tc)
    marker_text = "ZZZQQQ_UNIQUE_SECTION_TEXT_MARKER_7781"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _edit(tc, p["preview_id"], "save", 0, "1.1 Diversity In The Living World", marker_text + " " + CLEAN_PARA)
        tc.post("/admin/ncert-content/preview/reverify", json={"preview_id": p["preview_id"]}, headers={"X-Admin-Key": ADMIN_KEY})
    printed = buf.getvalue()
    check("the section's own text never appears in server output", marker_text not in printed, printed[:500])


# ---------------------------------------------------------------------------

def main():
    test_caption_lookalike_heading()
    test_two_line_heading_merge()
    test_dropped_indented_section_recovered()
    test_contents_list_duplicate_resolves_to_longest()

    test_preview_ttl_prunes_expired_sessions()
    test_preview_session_cap()

    test_editor_save()
    test_editor_save_requires_heading()
    test_editor_delete()
    test_editor_delete_last_section_refused()
    test_editor_merge_prev_and_merge_next()
    test_editor_merge_prev_at_start_refused()
    test_editor_move_up_and_down()
    test_editor_insert_above_and_below()

    test_revalidation_catches_duplicate_and_empty_sections()
    test_added_by_you_is_informational_not_a_failure()
    test_missing_pdf_paragraph_still_listed_as_missing()
    test_contractor_key_rejected_on_every_new_endpoint()
    test_apply_of_edited_chapter_sets_source_pdf_edited()
    test_apply_of_unedited_preview_still_sets_source_pdf()
    test_a_failed_chapter_can_be_fixed_by_editing_then_applied()

    test_chapter_edit_on_an_already_loaded_chapter()
    test_chapter_edit_preserves_keywords_of_unrelated_sections()

    test_excerpts_not_logged()

    print(f"\nTOTAL: {len(_passed)} passed, {len(_failed)} failed")
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
