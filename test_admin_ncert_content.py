"""Tests for the admin NCERT Content pipeline (admin-ncert-content.html + the
/admin/ncert-content* endpoints in server.py). Fully offline: Supabase is an in-memory recording
fake table (no real network call, no SQL executed), DeepSeek is a fake client (no real API call,
no real spend) and every PDF is a small synthetic one built in-process with PyMuPDF -- same style
as test_pyq_scanned.py's synthetic pages.

Covers the verification list from the task: endpoint auth (admin ok, contractor key rejected, no
key rejected), preview does not write, apply with overwrite replaces safely, a validation failure
blocks apply, the size cap, manual section add/edit, and delete with confirm.

Run with: python test_admin_ncert_content.py
Plain pass/fail harness like test_admin_dashboard.py.
"""
import base64
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fitz
import server
from fastapi.testclient import TestClient

_passed = []
_failed = []


def check(label, condition, detail=""):
    if condition:
        _passed.append(label)
        print(f"  PASS: {label}")
    else:
        _failed.append(label)
        print(f"  FAIL: {label}" + (f" -- {detail}" if detail else ""))


ADMIN_KEY = "test-admin-key"
CONTRACTOR_KEY = "test-contractor-key"

CLEAN_PARA = (
    "How wonderful is the living world and the wide range of living types that are found in "
    "the forests and the oceans and the ponds and the rivers and the mountains of the earth."
)


def make_chapter_pdf(lines):
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in lines:
        page.insert_text((72, y), line, fontsize=11)
        y += 18
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def good_chapter_pdf_b64():
    pdf_bytes = make_chapter_pdf([
        "CHAPTER 1",
        "THE LIVING WORLD",
        "1.1 Diversity In The Living World",
        CLEAN_PARA,
        "1.2 Taxonomic Categories",
        CLEAN_PARA,
        "1.3 Taxonomical Aids",
        CLEAN_PARA,
    ])
    return base64.b64encode(pdf_bytes).decode()


def bad_chapter_pdf_b64():
    """Only one heading -- fails validate_chapter's MIN_HEADINGS check."""
    pdf_bytes = make_chapter_pdf([
        "CHAPTER 2",
        "A BAD CHAPTER",
        "2.1 Only One Heading",
        CLEAN_PARA,
    ])
    return base64.b64encode(pdf_bytes).decode()


class FakeResp:
    def __init__(self, status_code=200, data=None, text=""):
        self.status_code = status_code
        self._data = data if data is not None else []
        self.text = text or ("ok" if status_code < 400 else "error")

    def json(self):
        return self._data


class FakeNcertTable:
    def __init__(self):
        self.rows = []
        self.next_id = 1

    def _matches(self, row, params):
        for k, v in (params or {}).items():
            if k in ("select", "order", "limit", "on_conflict"):
                continue
            if not isinstance(v, str):
                continue
            if v.startswith("eq."):
                if str(row.get(k)) != v[3:]:
                    return False
            elif v == "is.null":
                if row.get(k) is not None:
                    return False
            elif v.startswith("gt."):
                try:
                    if not (float(row.get(k) or 0) > float(v[3:])):
                        return False
                except Exception:
                    return False
        return True

    def get(self, params):
        matched = [r for r in self.rows if self._matches(r, params)]
        order = (params or {}).get("order")
        if order:
            keys = order.split(",")
            matched = sorted(matched, key=lambda r: tuple(r.get(k) or 0 for k in keys))
        limit = (params or {}).get("limit")
        if limit:
            matched = matched[: int(limit)]
        select = (params or {}).get("select")
        if select and select != "*":
            fields = select.split(",")
            matched = [{f: r.get(f) for f in fields} for r in matched]
        return [dict(r) for r in matched]

    def insert(self, json_body, on_conflict=None, merge=False):
        items = json_body if isinstance(json_body, list) else [json_body]
        out = []
        for item in items:
            if merge and on_conflict:
                keys = on_conflict.split(",")
                existing = next((r for r in self.rows if all(r.get(k) == item.get(k) for k in keys)), None)
                if existing:
                    existing.update(item)
                    out.append(dict(existing))
                    continue
            row = dict(item)
            row.setdefault("id", self.next_id)
            self.next_id = max(self.next_id, row["id"] + 1)
            row.setdefault("created_at", "2026-11-12T00:00:00Z")
            row.setdefault("keywords", None)
            self.rows.append(row)
            out.append(dict(row))
        return out

    def update(self, params, json_body):
        matched = [r for r in self.rows if self._matches(r, params)]
        for r in matched:
            r.update(json_body)
        return [dict(r) for r in matched]

    def delete(self, params):
        matched = [r for r in self.rows if self._matches(r, params)]
        self.rows = [r for r in self.rows if r not in matched]
        return [dict(r) for r in matched]


class FakeAsyncClient:
    def __init__(self):
        self.table = FakeNcertTable()
        self.chapter_log = FakeNcertTable()
        self.calls = []

    async def get(self, url, headers=None, params=None, **kw):
        self.calls.append(("GET", url, params))
        if "ncert_chapter_log" in url:
            return FakeResp(200, self.chapter_log.get(params))
        if "ncert_sections" in url:
            return FakeResp(200, self.table.get(params))
        return FakeResp(200, [])

    async def post(self, url, headers=None, params=None, json=None, **kw):
        self.calls.append(("POST", url, params, json))
        if "ncert_chapter_log" in url:
            return FakeResp(200, self.chapter_log.insert(json))
        if "ncert_sections" in url:
            prefer = (headers or {}).get("Prefer", "")
            on_conflict = (params or {}).get("on_conflict")
            merge = "merge-duplicates" in prefer
            return FakeResp(200, self.table.insert(json, on_conflict=on_conflict, merge=merge))
        if "provider_usage_log" in url:
            return FakeResp(200, [])
        return FakeResp(404, [], text="unknown table")

    async def patch(self, url, headers=None, params=None, json=None, **kw):
        self.calls.append(("PATCH", url, params, json))
        if "ncert_chapter_log" in url:
            return FakeResp(200, self.chapter_log.update(params, json))
        if "ncert_sections" in url:
            return FakeResp(200, self.table.update(params, json))
        return FakeResp(404, [])

    async def delete(self, url, headers=None, params=None, **kw):
        self.calls.append(("DELETE", url, params))
        if "ncert_sections" in url:
            return FakeResp(200, self.table.delete(params))
        return FakeResp(404, [])


class FakeContentBlock:
    def __init__(self, text):
        self.text = text


class FakeUsage:
    def __init__(self, input_tokens=50, cache_creation=0, cache_read=0, output_tokens=20):
        self.input_tokens = input_tokens
        self.cache_creation_input_tokens = cache_creation
        self.cache_read_input_tokens = cache_read
        self.output_tokens = output_tokens


class FakeMessages:
    def __init__(self, keyword_text, input_tokens):
        self.keyword_text = keyword_text
        self.input_tokens = input_tokens
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        resp = FakeResp.__new__(FakeResp)  # unused, just avoiding a stray import
        return _FakeDeepSeekResponse(self.keyword_text, self.input_tokens)


class _FakeDeepSeekResponse:
    def __init__(self, text, input_tokens):
        self.content = [FakeContentBlock(text)]
        self.usage = FakeUsage(input_tokens=input_tokens)


class FakeDeepSeekClient:
    def __init__(self, keyword_text="diversity, taxonomy, classification", input_tokens=50):
        self.messages = FakeMessages(keyword_text, input_tokens)


def setup(deepseek_input_tokens=50):
    server.ADMIN_PASSWORD = ADMIN_KEY
    server.CONTRACTOR_PDF_KEY = CONTRACTOR_KEY
    server._admin_login_attempts.clear()
    server._rate_buckets.clear()
    server._rate_windows.clear()
    server._ncert_content_previews.clear()
    server._ncert_sections_cache["by_id"] = {}
    server._ncert_sections_cache["index_lines"] = []
    server._ncert_sections_cache["loaded_at"] = 0.0
    server._is_deepseek_peak_hour = lambda: True  # deterministic rate, not real wall-clock time
    fake = FakeAsyncClient()
    server.async_client = fake
    server.deepseek_client = FakeDeepSeekClient(input_tokens=deepseek_input_tokens)
    return TestClient(server.app), fake  # no `with`: skips startup hooks, nothing here needs them


def test_admin_auth():
    print("\n# Admin auth -- admin ok, contractor key rejected, no key rejected")
    tc, _ = setup()
    good_pdf = good_chapter_pdf_b64()

    calls = [
        ("GET", "/admin/ncert-content", None),
        ("POST", "/admin/ncert-content/preview", {"class": "11", "subject": "Biology", "data": good_pdf}),
        ("POST", "/admin/ncert-content/apply", {"preview_id": "nonexistent"}),
        ("POST", "/admin/ncert-content/section", {"class": "11", "subject": "Biology", "chapter": "X", "heading": "1.1 X", "ordinal": 0, "text": "some text here"}),
        ("DELETE", "/admin/ncert-content/chapter?class=11&subject=Biology&chapter=X", None),
    ]
    for method, path, body in calls:
        req = lambda headers=None: (
            tc.get(path, headers=headers) if method == "GET"
            else tc.delete(path, headers=headers) if method == "DELETE"
            else tc.post(path, json=body, headers=headers)
        )
        res = req(None)
        check(f"{method} {path.split('?')[0]} refuses without a key", res.status_code == 401, res.status_code)
        res = req({"X-Admin-Key": CONTRACTOR_KEY})
        check(f"{method} {path.split('?')[0]} refuses the contractor key", res.status_code == 401, res.status_code)
        res = req({"X-Admin-Key": ADMIN_KEY})
        check(f"{method} {path.split('?')[0]} accepts the real admin key", res.status_code == 200, (res.status_code, res.text[:200]))


def test_preview_does_not_write():
    print("\n# Preview: extracts/validates but writes nothing")
    tc, fake = setup()
    res = tc.post("/admin/ncert-content/preview", json={"class": "11", "subject": "Biology", "data": good_chapter_pdf_b64()}, headers={"X-Admin-Key": ADMIN_KEY})
    data = res.json()
    check("preview succeeds", res.status_code == 200 and not data.get("error"), data)
    check("chapter title extracted", "LIVING WORLD" in (data.get("chapter_title") or ""), data.get("chapter_title"))
    check("3 sections found", data.get("section_count") == 3, data.get("section_count"))
    check("no validation flags on a clean chapter", data.get("flags") == [], data.get("flags"))
    check("exists is False (nothing loaded yet)", data.get("exists") is False)
    check("a preview_id is returned", bool(data.get("preview_id")))
    check("nothing was written to ncert_sections", fake.table.rows == [], fake.table.rows)
    write_calls = [c for c in fake.calls if c[0] in ("POST", "PATCH", "DELETE") and "ncert_sections" in c[1]]
    check("no write call touched ncert_sections during preview", write_calls == [], write_calls)


def test_size_cap():
    print("\n# PDF upload size cap (15 MB decoded)")
    tc, fake = setup()
    # Just over the 15 MB decoded cap, but still comfortably under the request body-size
    # middleware's own raw-body cap (21 MB, sized for base64 inflation + JSON envelope slack) --
    # this exercises the endpoint's OWN decoded-size check, not the earlier middleware layer.
    oversized = base64.b64encode(b"0" * 15_800_000).decode()
    res = tc.post("/admin/ncert-content/preview", json={"class": "11", "subject": "Biology", "data": oversized}, headers={"X-Admin-Key": ADMIN_KEY})
    check("the request isn't rejected by the body-size middleware itself", res.status_code == 200, res.status_code)
    data = res.json()
    check("oversized PDF is rejected with a plain string error", isinstance(data.get("error"), str) and "MB" in data["error"], data)
    check("nothing written on an oversized upload", fake.table.rows == [])


def test_validation_failure_blocks_apply():
    print("\n# A chapter that fails validation cannot be applied from the PDF route")
    tc, fake = setup()
    preview_res = tc.post("/admin/ncert-content/preview", json={"class": "11", "subject": "Biology", "data": bad_chapter_pdf_b64()}, headers={"X-Admin-Key": ADMIN_KEY})
    preview_data = preview_res.json()
    check("preview flags the chapter (only 1 heading)", any("heading" in f for f in preview_data.get("flags", [])), preview_data.get("flags"))
    preview_id = preview_data.get("preview_id")

    apply_res = tc.post("/admin/ncert-content/apply", json={"preview_id": preview_id}, headers={"X-Admin-Key": ADMIN_KEY})
    apply_data = apply_res.json()
    check("apply refuses a chapter that failed validation", isinstance(apply_data.get("error"), str) and "validation" in apply_data["error"].lower(), apply_data)
    check("nothing was written", fake.table.rows == [], fake.table.rows)


def test_apply_writes_and_overwrite_replaces_safely():
    print("\n# Apply writes a fresh chapter, then overwrite replaces it safely")
    tc, fake = setup()
    pdf = good_chapter_pdf_b64()

    preview_res = tc.post("/admin/ncert-content/preview", json={"class": "11", "subject": "Biology", "data": pdf}, headers={"X-Admin-Key": ADMIN_KEY})
    preview_id = preview_res.json()["preview_id"]

    apply_res = tc.post("/admin/ncert-content/apply", json={"preview_id": preview_id}, headers={"X-Admin-Key": ADMIN_KEY})
    apply_data = apply_res.json()
    check("fresh apply succeeds", apply_data.get("written") is True, apply_data)
    check("3 sections written", apply_data.get("sections_written") == 3, apply_data)
    check("3 rows now in the table", len(fake.table.rows) == 3, len(fake.table.rows))
    first_ids = sorted(r["id"] for r in fake.table.rows)

    # Preview again: now it should report the chapter already exists.
    preview_res2 = tc.post("/admin/ncert-content/preview", json={"class": "11", "subject": "Biology", "data": pdf}, headers={"X-Admin-Key": ADMIN_KEY})
    preview_data2 = preview_res2.json()
    check("second preview reports the chapter already exists", preview_data2.get("exists") is True, preview_data2)
    preview_id2 = preview_data2["preview_id"]

    # Apply without overwrite must refuse.
    apply_res2 = tc.post("/admin/ncert-content/apply", json={"preview_id": preview_id2}, headers={"X-Admin-Key": ADMIN_KEY})
    apply_data2 = apply_res2.json()
    check("apply without overwrite refuses an existing chapter", isinstance(apply_data2.get("error"), str) and "already exists" in apply_data2["error"], apply_data2)
    check("table unchanged after the refused apply", len(fake.table.rows) == 3, len(fake.table.rows))

    # Apply with overwrite=true replaces it safely.
    apply_res3 = tc.post("/admin/ncert-content/apply", json={"preview_id": preview_id2, "overwrite": True}, headers={"X-Admin-Key": ADMIN_KEY})
    apply_data3 = apply_res3.json()
    check("overwrite apply succeeds", apply_data3.get("written") is True, apply_data3)
    check("still exactly 3 rows after overwrite (old rows replaced, not duplicated)", len(fake.table.rows) == 3, len(fake.table.rows))
    chapters_present = {r["chapter"] for r in fake.table.rows}
    check("no leftover temp-labeled chapter rows remain", chapters_present == {"THE LIVING WORLD"}, chapters_present)
    new_ids = sorted(r["id"] for r in fake.table.rows)
    check("the old rows were actually replaced (new ids, not the same rows reused)", new_ids != first_ids, (first_ids, new_ids))


def test_apply_keyword_cost_cap():
    print("\n# DeepSeek keyword spend on Apply is capped at $0.10/request")
    # 200k input tokens/call at the (forced) peak rate (0.30/1M) costs ~$0.06/call -- 2 calls
    # cross the $0.10 cap, so the 3rd section's keywords must be skipped this request.
    tc, fake = setup(deepseek_input_tokens=200_000)
    pdf = good_chapter_pdf_b64()  # 3 sections
    preview_id = tc.post("/admin/ncert-content/preview", json={"class": "11", "subject": "Biology", "data": pdf}, headers={"X-Admin-Key": ADMIN_KEY}).json()["preview_id"]
    apply_data = tc.post("/admin/ncert-content/apply", json={"preview_id": preview_id}, headers={"X-Admin-Key": ADMIN_KEY}).json()
    check("apply still succeeds (writing isn't blocked by the keyword cap)", apply_data.get("written") is True, apply_data)
    check("cost stayed near the $0.10 cap, not the full 3-section cost", apply_data.get("cost", 0) < 0.15, apply_data.get("cost"))
    check("fewer than all 3 sections got keywords (cap reached)", 0 < apply_data.get("keywords_generated", 0) < 3, apply_data)
    check("keywords_pending is true", apply_data.get("keywords_pending") is True, apply_data)


def test_manual_add_or_edit():
    print("\n# Manual section add/edit")
    tc, fake = setup()
    body = {"class": "12", "subject": "Chemistry", "chapter": "Solutions", "heading": "2.1 Types Of Solutions", "ordinal": 0, "text": CLEAN_PARA}
    res = tc.post("/admin/ncert-content/section", json=body, headers={"X-Admin-Key": ADMIN_KEY})
    data = res.json()
    check("manual add succeeds", data.get("saved") is True, data)
    check("one row written", len(fake.table.rows) == 1, len(fake.table.rows))
    check("word_count computed server-side", fake.table.rows[0]["word_count"] == len(CLEAN_PARA.split()), fake.table.rows[0])

    edited_text = CLEAN_PARA + " Extra sentence added during the edit."
    body2 = dict(body, text=edited_text)
    res2 = tc.post("/admin/ncert-content/section", json=body2, headers={"X-Admin-Key": ADMIN_KEY})
    data2 = res2.json()
    check("manual edit (same class/subject/chapter/ordinal) succeeds", data2.get("saved") is True, data2)
    check("still exactly one row (edited in place, not duplicated)", len(fake.table.rows) == 1, len(fake.table.rows))
    check("the row's text was actually updated", fake.table.rows[0]["text"] == edited_text, fake.table.rows[0]["text"])

    bad_res = tc.post("/admin/ncert-content/section", json={"class": "99", "subject": "Chemistry", "chapter": "X", "heading": "h", "ordinal": 0, "text": "x"}, headers={"X-Admin-Key": ADMIN_KEY})
    check("an invalid class is rejected with a plain string error", isinstance(bad_res.json().get("error"), str), bad_res.json())


def test_delete_confirm():
    print("\n# Delete requires confirm=true")
    tc, fake = setup()
    fake.table.insert({"class": "11", "subject": "Physics", "chapter": "Motion", "heading": "1.1 X", "ordinal": 0, "text": "x", "word_count": 1})
    fake.table.insert({"class": "11", "subject": "Physics", "chapter": "Motion", "heading": "1.2 Y", "ordinal": 1, "text": "y", "word_count": 1})

    res = tc.delete("/admin/ncert-content/chapter?class=11&subject=Physics&chapter=Motion", headers={"X-Admin-Key": ADMIN_KEY})
    data = res.json()
    check("delete without confirm is refused with a plain string error", isinstance(data.get("error"), str) and "confirm" in data["error"].lower(), data)
    check("nothing deleted without confirm", len(fake.table.rows) == 2, len(fake.table.rows))

    res2 = tc.delete("/admin/ncert-content/chapter?class=11&subject=Physics&chapter=Motion&confirm=true", headers={"X-Admin-Key": ADMIN_KEY})
    data2 = res2.json()
    check("delete with confirm=true succeeds", data2.get("deleted") == 2, data2)
    check("both rows actually removed", fake.table.rows == [], fake.table.rows)


def test_chapter_log_written_on_apply():
    print("\n# ncert_chapter_log: a row is written on Apply (source=pdf)")
    tc, fake = setup()
    pdf = good_chapter_pdf_b64()
    preview_id = tc.post("/admin/ncert-content/preview", json={"class": "11", "subject": "Biology", "data": pdf}, headers={"X-Admin-Key": ADMIN_KEY}).json()["preview_id"]
    apply_data = tc.post("/admin/ncert-content/apply", json={"preview_id": preview_id}, headers={"X-Admin-Key": ADMIN_KEY}).json()

    check("exactly one log row written", len(fake.chapter_log.rows) == 1, fake.chapter_log.rows)
    log_row = fake.chapter_log.rows[0]
    check("log row's class/subject/chapter match the applied chapter",
          (log_row["class"], log_row["subject"], log_row["chapter"]) == ("11", "Biology", "THE LIVING WORLD"), log_row)
    check("source is 'pdf'", log_row["source"] == "pdf", log_row)
    check("model_calls matches keywords_generated", log_row["model_calls"] == apply_data["keywords_generated"], (log_row, apply_data))
    check("model is recorded", log_row["model"] == server.DEEPSEEK_MODEL, log_row)
    check("cost matches the apply response's cost", log_row["keywords_cost_usd"] == apply_data["cost"], (log_row, apply_data))
    check("pages recorded from the PDF extraction", log_row["pages"] == 1, log_row)
    check("extract_seconds recorded (not null)", log_row["extract_seconds"] is not None, log_row)
    check("sections/words recorded", log_row["sections"] == 3 and log_row["words"] > 0, log_row)


def test_chapter_log_written_on_manual_add_and_regen():
    print("\n# ncert_chapter_log: rows written on manual add and on keyword regeneration (source=manual)")
    tc, fake = setup()
    body = {"class": "12", "subject": "Chemistry", "chapter": "Solutions", "heading": "2.1 Types Of Solutions", "ordinal": 0, "text": CLEAN_PARA}
    tc.post("/admin/ncert-content/section", json=body, headers={"X-Admin-Key": ADMIN_KEY})

    check("one log row after a plain manual add", len(fake.chapter_log.rows) == 1, fake.chapter_log.rows)
    first = fake.chapter_log.rows[0]
    check("source is 'manual'", first["source"] == "manual", first)
    check("no keyword call means zero cost and zero model_calls", first["keywords_cost_usd"] == 0 and first["model_calls"] == 0 and first["model"] is None, first)

    body2 = dict(body, regenerate_keywords=True)
    tc.post("/admin/ncert-content/section", json=body2, headers={"X-Admin-Key": ADMIN_KEY})
    check("a second log row is added for the regeneration", len(fake.chapter_log.rows) == 2, fake.chapter_log.rows)
    second = fake.chapter_log.rows[1]
    check("regeneration row has one model call and a real model name", second["model_calls"] == 1 and second["model"] == server.DEEPSEEK_MODEL, second)
    check("regeneration row has a positive cost", second["keywords_cost_usd"] > 0, second)


def test_older_chapters_show_unknown_fields():
    print("\n# A chapter with no ncert_chapter_log history reports null cost/source/model/pages fields")
    tc, fake = setup()
    fake.table.insert({"class": "11", "subject": "Physics", "chapter": "Motion", "heading": "1.1 X", "ordinal": 0, "text": "x", "word_count": 5})

    data = tc.get("/admin/ncert-content", headers={"X-Admin-Key": ADMIN_KEY}).json()
    check("one chapter listed", len(data.get("chapters", [])) == 1, data)
    c = data["chapters"][0]
    check("source/cost/model/pages/extract_seconds are all null for a pre-existing chapter",
          c.get("source") is None and c.get("cost") is None and c.get("model") is None
          and c.get("pages") is None and c.get("extract_seconds") is None, c)
    check("totals cost fields are null when nothing has known cost", data["totals"].get("keywords_cost_usd") is None and data["totals"].get("avg_cost_usd") is None, data["totals"])


def test_totals_cost_summary():
    print("\n# Totals: total keywords cost and average cost per chapter, across listed chapters")
    tc, fake = setup()
    pdf = good_chapter_pdf_b64()
    preview_id = tc.post("/admin/ncert-content/preview", json={"class": "11", "subject": "Biology", "data": pdf}, headers={"X-Admin-Key": ADMIN_KEY}).json()["preview_id"]
    apply_data = tc.post("/admin/ncert-content/apply", json={"preview_id": preview_id}, headers={"X-Admin-Key": ADMIN_KEY}).json()

    # A second, pre-existing-style chapter with no log history at all.
    fake.table.insert({"class": "11", "subject": "Physics", "chapter": "Motion", "heading": "1.1 X", "ordinal": 0, "text": "x", "word_count": 5})

    data = tc.get("/admin/ncert-content", headers={"X-Admin-Key": ADMIN_KEY}).json()
    check("two chapters listed", len(data.get("chapters", [])) == 2, data)
    totals = data["totals"]
    check("total keywords cost counts only the chapter with known cost", totals.get("keywords_cost_usd") == apply_data["cost"], (totals, apply_data))
    check("average cost per chapter is computed over chapters with known cost only (not divided by 2)",
          totals.get("avg_cost_usd") == apply_data["cost"], (totals, apply_data))


def test_preview_estimated_cost_and_apply_real_cost():
    print("\n# Preview reports an estimated keywords cost; Apply reports the real cost")
    tc, fake = setup()
    pdf = good_chapter_pdf_b64()
    preview_data = tc.post("/admin/ncert-content/preview", json={"class": "11", "subject": "Biology", "data": pdf}, headers={"X-Admin-Key": ADMIN_KEY}).json()
    check("preview includes an estimated_cost", isinstance(preview_data.get("estimated_cost"), (int, float)), preview_data)
    check("fallback estimate uses the known $0.002/section baseline (no log history yet)",
          preview_data["estimated_cost"] == round(preview_data["section_count"] * server.NCERT_CONTENT_FALLBACK_SECTION_COST, 4), preview_data)

    apply_data = tc.post("/admin/ncert-content/apply", json={"preview_id": preview_data["preview_id"]}, headers={"X-Admin-Key": ADMIN_KEY}).json()
    check("apply reports a real cost", isinstance(apply_data.get("cost"), (int, float)) and apply_data["cost"] > 0, apply_data)


def test_delete_marks_log_rows_deleted_not_removed():
    print("\n# Delete marks the chapter's log rows deleted_at instead of removing them")
    tc, fake = setup()
    pdf = good_chapter_pdf_b64()
    preview_id = tc.post("/admin/ncert-content/preview", json={"class": "11", "subject": "Biology", "data": pdf}, headers={"X-Admin-Key": ADMIN_KEY}).json()["preview_id"]
    tc.post("/admin/ncert-content/apply", json={"preview_id": preview_id}, headers={"X-Admin-Key": ADMIN_KEY})
    check("one log row exists before delete", len(fake.chapter_log.rows) == 1, fake.chapter_log.rows)

    tc.delete("/admin/ncert-content/chapter?class=11&subject=Biology&chapter=THE%20LIVING%20WORLD&confirm=true", headers={"X-Admin-Key": ADMIN_KEY})
    check("the log row still exists after delete (not removed)", len(fake.chapter_log.rows) == 1, fake.chapter_log.rows)
    check("the log row's deleted_at is now set", fake.chapter_log.rows[0].get("deleted_at") is not None, fake.chapter_log.rows[0])

    data = tc.get("/admin/ncert-content", headers={"X-Admin-Key": ADMIN_KEY}).json()
    check("the deleted chapter no longer appears in the list", data.get("chapters") == [], data)


def test_contractor_response_has_no_cost_fields():
    print("\n# Contractor key never reaches a response that could carry cost data")
    tc, fake = setup()
    res = tc.get("/admin/ncert-content", headers={"X-Admin-Key": CONTRACTOR_KEY})
    check("contractor key is refused outright (401)", res.status_code == 401, res.status_code)
    body_text = res.text
    check("the refusal body carries no cost fields", "keywords_cost_usd" not in body_text and "estimated_cost" not in body_text, body_text)


def main():
    test_admin_auth()
    test_preview_does_not_write()
    test_size_cap()
    test_validation_failure_blocks_apply()
    test_apply_writes_and_overwrite_replaces_safely()
    test_apply_keyword_cost_cap()
    test_manual_add_or_edit()
    test_delete_confirm()
    test_chapter_log_written_on_apply()
    test_chapter_log_written_on_manual_add_and_regen()
    test_older_chapters_show_unknown_fields()
    test_totals_cost_summary()
    test_preview_estimated_cost_and_apply_real_cost()
    test_delete_marks_log_rows_deleted_not_removed()
    test_contractor_response_has_no_cost_fields()

    print(f"\nTOTAL: {len(_passed)} passed, {len(_failed)} failed")
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
