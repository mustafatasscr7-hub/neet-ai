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
import ncert_ingest_core as ncert_core_mod
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
        self.verify_decisions = FakeNcertTable()
        self.calls = []

    async def get(self, url, headers=None, params=None, **kw):
        self.calls.append(("GET", url, params))
        if "ncert_chapter_log" in url:
            return FakeResp(200, self.chapter_log.get(params))
        if "ncert_verify_decisions" in url:
            return FakeResp(200, self.verify_decisions.get(params))
        if "ncert_sections" in url:
            return FakeResp(200, self.table.get(params))
        return FakeResp(200, [])

    async def post(self, url, headers=None, params=None, json=None, **kw):
        self.calls.append(("POST", url, params, json))
        if "ncert_chapter_log" in url:
            return FakeResp(200, self.chapter_log.insert(json))
        if "ncert_verify_decisions" in url:
            prefer = (headers or {}).get("Prefer", "")
            on_conflict = (params or {}).get("on_conflict")
            merge = "merge-duplicates" in prefer
            return FakeResp(200, self.verify_decisions.insert(json, on_conflict=on_conflict, merge=merge))
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
        ("POST", "/admin/ncert-content/verify", {"class": "11", "subject": "Biology", "chapter": "X", "data": good_pdf}),
        ("POST", "/admin/ncert-content/verify/mark-ok", {"class": "11", "subject": "Biology", "chapter": "X", "diff_key": "abc123"}),
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


def test_preview_includes_verify():
    print("\n# Preview includes the verify-against-PDF classification (real vs harmless differences)")
    tc, fake = setup()
    res = tc.post("/admin/ncert-content/preview", json={"class": "11", "subject": "Biology", "data": good_chapter_pdf_b64()}, headers={"X-Admin-Key": ADMIN_KEY})
    data = res.json()
    verify = data.get("verify") or {}
    check("verify block is present", bool(verify), data)
    check("a clean round-tripped chapter passes the cross-check", verify.get("status") == "PASS", verify)
    check("no differences at all", verify.get("differences") == [], verify)
    check("counts.real is 0", (verify.get("counts") or {}).get("real") == 0, verify)
    check("coverage_pct is a positive number", isinstance(verify.get("coverage_pct"), (int, float)) and verify["coverage_pct"] > 0, verify)
    check("verify is not run for a chapter that already failed validate_chapter",
          tc.post("/admin/ncert-content/preview", json={"class": "11", "subject": "Biology", "data": bad_chapter_pdf_b64()}, headers={"X-Admin-Key": ADMIN_KEY}).json().get("verify") is None)


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


def test_apply_blocked_on_verify_fail_and_override_allows_it():
    print("\n# Apply is blocked when the text cross-check fails, unless overridden")
    tc, fake = setup()
    pdf = good_chapter_pdf_b64()

    fake_fail = {
        "status": "FAIL", "summary": "FAIL - 42.0% coverage, 1 real difference(s), 0 harmless difference(s)",
        "total_words": 100, "kept_words": 42, "coverage_pct": 42.0, "word_diff": 58, "word_diff_pct": 58.0,
        "dropped_lines_count": 1,
        "differences": [{
            "type": "extra", "category": "real", "section": "1.1 Diversity In The Living World", "page": None,
            "pdf_excerpt": None, "stored_excerpt": "a sentence not found", "diff_spans": None, "diff_key": "fakekey123",
        }],
        "counts": {"real": 1, "harmless": 0, "by_type": {"extra": 1}},
    }
    orig_verify = server.ncert_verify_mod.verify_text_coverage
    server.ncert_verify_mod.verify_text_coverage = lambda pdf_bytes, sections: dict(fake_fail)
    try:
        preview_id = tc.post("/admin/ncert-content/preview", json={"class": "11", "subject": "Biology", "data": pdf}, headers={"X-Admin-Key": ADMIN_KEY}).json()["preview_id"]
    finally:
        server.ncert_verify_mod.verify_text_coverage = orig_verify

    apply_res = tc.post("/admin/ncert-content/apply", json={"preview_id": preview_id}, headers={"X-Admin-Key": ADMIN_KEY})
    apply_data = apply_res.json()
    check("apply refuses a chapter that failed the text cross-check",
          isinstance(apply_data.get("error"), str) and "cross-check" in apply_data["error"].lower(), apply_data)
    check("nothing was written", fake.table.rows == [], fake.table.rows)

    apply_res2 = tc.post("/admin/ncert-content/apply", json={"preview_id": preview_id, "override_verify_fail": True}, headers={"X-Admin-Key": ADMIN_KEY})
    apply_data2 = apply_res2.json()
    check("apply succeeds with the override", apply_data2.get("written") is True, apply_data2)
    check("3 sections written despite the FAIL", len(fake.table.rows) == 3, fake.table.rows)

    check("exactly one log row (the blocked attempt never logged anything)", len(fake.chapter_log.rows) == 1, fake.chapter_log.rows)
    log_row = fake.chapter_log.rows[0]
    check("the log row records the FAIL status", log_row.get("verify_status") == "FAIL", log_row)
    check("the log row's summary notes the override was used", "override" in (log_row.get("verify_summary") or "").lower(), log_row)
    check("source_words/kept_words/coverage_pct are stored from the verify result",
          log_row.get("source_words") == 100 and log_row.get("kept_words") == 42 and log_row.get("coverage_pct") == 42.0, log_row)
    check("differences and their excerpts are never stored in the log row",
          "differences" not in log_row and "counts" not in log_row, log_row)


def test_verify_endpoint_checks_existing_chapter_and_stamps_log():
    print("\n# 'Verify against PDF' checks an already-loaded chapter and stamps the log row")
    tc, fake = setup()
    pdf = good_chapter_pdf_b64()
    preview_id = tc.post("/admin/ncert-content/preview", json={"class": "11", "subject": "Biology", "data": pdf}, headers={"X-Admin-Key": ADMIN_KEY}).json()["preview_id"]
    tc.post("/admin/ncert-content/apply", json={"preview_id": preview_id}, headers={"X-Admin-Key": ADMIN_KEY})
    check("coverage_pct was stored by Apply", fake.chapter_log.rows[0].get("coverage_pct") is not None, fake.chapter_log.rows)
    original_coverage = fake.chapter_log.rows[0]["coverage_pct"]

    res = tc.post("/admin/ncert-content/verify", json={"class": "11", "subject": "Biology", "chapter": "THE LIVING WORLD", "data": pdf}, headers={"X-Admin-Key": ADMIN_KEY})
    data = res.json()
    verify = data.get("verify") or {}
    check("verify succeeds against the stored chapter", verify.get("status") == "PASS", data)
    check("still exactly 3 rows in ncert_sections (no writes)", len(fake.table.rows) == 3, fake.table.rows)

    log_row = fake.chapter_log.rows[0]
    check("the existing log row's verify_status is stamped", log_row.get("verify_status") == "PASS", log_row)
    check("verified_at is stamped", log_row.get("verified_at") is not None, log_row)
    check("coverage_pct from the original Apply is untouched by Verify", log_row.get("coverage_pct") == original_coverage, (log_row, original_coverage))


def test_verify_endpoint_no_such_chapter():
    print("\n# 'Verify against PDF' refuses when the chapter doesn't exist")
    tc, fake = setup()
    res = tc.post("/admin/ncert-content/verify", json={"class": "11", "subject": "Biology", "chapter": "Nonexistent", "data": good_chapter_pdf_b64()}, headers={"X-Admin-Key": ADMIN_KEY})
    data = res.json()
    check("a plain string error for a missing chapter", isinstance(data.get("error"), str), data)
    check("nothing written", fake.table.rows == [], fake.table.rows)


def test_verify_endpoint_inserts_fallback_log_row_for_legacy_chapter():
    print("\n# 'Verify against PDF' writes a fresh log row when the chapter predates ncert_chapter_log")
    tc, fake = setup()
    fake.table.insert({"class": "11", "subject": "Physics", "chapter": "Motion", "heading": "1.1 X", "ordinal": 0, "text": CLEAN_PARA, "word_count": len(CLEAN_PARA.split())})
    check("no log rows exist yet for this legacy chapter", fake.chapter_log.rows == [])

    pdf_bytes = make_chapter_pdf(["CHAPTER 1", "MOTION", "1.1 X", CLEAN_PARA])
    res = tc.post("/admin/ncert-content/verify", json={"class": "11", "subject": "Physics", "chapter": "Motion", "data": base64.b64encode(pdf_bytes).decode()}, headers={"X-Admin-Key": ADMIN_KEY})
    data = res.json()
    check("verify still runs", "verify" in data, data)
    check("a fallback log row was written", len(fake.chapter_log.rows) == 1, fake.chapter_log.rows)
    row = fake.chapter_log.rows[0]
    check("the fallback row carries the verify stamp", row.get("verify_status") in ("PASS", "FAIL") and row.get("verified_at") is not None, row)
    check("the fallback row has no coverage data (this never went through the PDF-route Apply)", row.get("coverage_pct") is None, row)


def test_verify_endpoint_no_writes_to_sections_even_on_fail():
    print("\n# 'Verify against PDF' never writes to ncert_sections, even on a FAIL")
    tc, fake = setup()
    pdf = good_chapter_pdf_b64()
    preview_id = tc.post("/admin/ncert-content/preview", json={"class": "11", "subject": "Biology", "data": pdf}, headers={"X-Admin-Key": ADMIN_KEY}).json()["preview_id"]
    tc.post("/admin/ncert-content/apply", json={"preview_id": preview_id}, headers={"X-Admin-Key": ADMIN_KEY})

    mismatched_pdf = base64.b64encode(make_chapter_pdf([
        "CHAPTER 1", "THE LIVING WORLD", "1.1 Diversity In The Living World",
        "A completely different sentence that was never stored anywhere before this.",
    ])).decode()
    calls_before = len(fake.calls)
    res = tc.post("/admin/ncert-content/verify", json={"class": "11", "subject": "Biology", "chapter": "THE LIVING WORLD", "data": mismatched_pdf}, headers={"X-Admin-Key": ADMIN_KEY})
    data = res.json()
    check("this mismatched PDF fails verification", (data.get("verify") or {}).get("status") == "FAIL", data)
    check("still exactly 3 rows (Verify never writes sections, even on FAIL)", len(fake.table.rows) == 3, fake.table.rows)
    calls_during_verify = fake.calls[calls_before:]
    write_calls = [c for c in calls_during_verify if c[0] in ("POST", "PATCH", "DELETE") and "ncert_sections" in c[1]]
    check("no write call touched ncert_sections during verify", write_calls == [], write_calls)


def test_mark_ok_persists_and_only_resolves_its_own_difference():
    print("\n# 'Mark as OK' persists a decision, resolves only its own diff_key, and PASS follows the verdict rule")
    tc, fake = setup()
    pdf = good_chapter_pdf_b64()
    preview_id = tc.post("/admin/ncert-content/preview", json={"class": "11", "subject": "Biology", "data": pdf}, headers={"X-Admin-Key": ADMIN_KEY}).json()["preview_id"]
    tc.post("/admin/ncert-content/apply", json={"preview_id": preview_id}, headers={"X-Admin-Key": ADMIN_KEY})

    # A second PDF for the same stored chapter with exactly two real differences: a one-word
    # change in 1.2 ("changed"), and 1.3's whole paragraph missing from this PDF ("extra" -- the
    # stored text has it, this PDF doesn't).
    mismatched_pdf = base64.b64encode(make_chapter_pdf([
        "CHAPTER 1", "THE LIVING WORLD", "1.1 Diversity In The Living World", CLEAN_PARA,
        "1.2 Taxonomic Categories", CLEAN_PARA.replace("wonderful", "amazing"),
        "1.3 Taxonomical Aids",
    ])).decode()

    def run_verify():
        res = tc.post("/admin/ncert-content/verify", json={"class": "11", "subject": "Biology", "chapter": "THE LIVING WORLD", "data": mismatched_pdf}, headers={"X-Admin-Key": ADMIN_KEY})
        return res.json()["verify"]

    verify1 = run_verify()
    check("exactly 2 real differences, 0 harmless", verify1["counts"]["real"] == 2 and verify1["counts"]["harmless"] == 0, verify1)
    check("FAILs with both unresolved", verify1["status"] == "FAIL", verify1)
    real_diffs = [d for d in verify1["differences"] if d["category"] == "real"]
    check("neither difference is marked_ok yet", not any(d["marked_ok"] for d in real_diffs), real_diffs)
    changed_key = next(d["diff_key"] for d in real_diffs if d["type"] == "changed")
    extra_key = next(d["diff_key"] for d in real_diffs if d["type"] == "extra")

    mark_res = tc.post("/admin/ncert-content/verify/mark-ok", json={"class": "11", "subject": "Biology", "chapter": "THE LIVING WORLD", "diff_key": changed_key}, headers={"X-Admin-Key": ADMIN_KEY})
    check("mark-ok succeeds", mark_res.status_code == 200 and mark_res.json().get("ok") is True, mark_res.json())
    check("exactly one decision row stored", len(fake.verify_decisions.rows) == 1, fake.verify_decisions.rows)

    tc.post("/admin/ncert-content/verify/mark-ok", json={"class": "11", "subject": "Biology", "chapter": "THE LIVING WORLD", "diff_key": changed_key}, headers={"X-Admin-Key": ADMIN_KEY})
    check("marking the same difference twice stays one row (upsert, not a duplicate)", len(fake.verify_decisions.rows) == 1, fake.verify_decisions.rows)

    verify2 = run_verify()
    check("still FAIL -- the other difference remains unresolved", verify2["status"] == "FAIL", verify2)
    marked = next(d for d in verify2["differences"] if d["diff_key"] == changed_key)
    other = next(d for d in verify2["differences"] if d["diff_key"] == extra_key)
    check("the marked difference now reports marked_ok", marked["marked_ok"] is True, marked)
    check("marking one difference does not mark/hide the other one", other["marked_ok"] is False, other)
    log_row_after_one_mark = fake.chapter_log.rows[0]
    check("the log row's verify_status still reflects FAIL after only one mark", log_row_after_one_mark.get("verify_status") == "FAIL", log_row_after_one_mark)

    tc.post("/admin/ncert-content/verify/mark-ok", json={"class": "11", "subject": "Biology", "chapter": "THE LIVING WORLD", "diff_key": extra_key}, headers={"X-Admin-Key": ADMIN_KEY})
    verify3 = run_verify()
    check("PASS once every real difference is marked OK", verify3["status"] == "PASS", verify3)
    check("counts.real still reports the raw total (decisions don't erase the raw count)", verify3["counts"]["real"] == 2, verify3)
    log_row_after_both_marks = fake.chapter_log.rows[0]
    check("the log row's verify_status now reflects the final PASS verdict", log_row_after_both_marks.get("verify_status") == "PASS", log_row_after_both_marks)

    bad_res = tc.post("/admin/ncert-content/verify/mark-ok", json={"class": "11", "subject": "Biology", "chapter": "THE LIVING WORLD"}, headers={"X-Admin-Key": ADMIN_KEY})
    check("mark-ok rejects a request missing diff_key", isinstance(bad_res.json().get("error"), str), bad_res.json())

    check("ncert_sections was never touched by any verify or mark-ok call (still exactly 3 rows)", len(fake.table.rows) == 3, fake.table.rows)
    check("exactly 2 decision rows total -- nothing else got written to that table", len(fake.verify_decisions.rows) == 2, fake.verify_decisions.rows)
    check("exactly one chapter_log row -- every verify call stamped the SAME row, nothing extra", len(fake.chapter_log.rows) == 1, fake.chapter_log.rows)


def test_list_surfaces_coverage_and_verify_status():
    print("\n# The chapters list surfaces coverage_pct, verify_status and verified_at")
    tc, fake = setup()
    pdf = good_chapter_pdf_b64()
    preview_id = tc.post("/admin/ncert-content/preview", json={"class": "11", "subject": "Biology", "data": pdf}, headers={"X-Admin-Key": ADMIN_KEY}).json()["preview_id"]
    tc.post("/admin/ncert-content/apply", json={"preview_id": preview_id}, headers={"X-Admin-Key": ADMIN_KEY})

    data = tc.get("/admin/ncert-content", headers={"X-Admin-Key": ADMIN_KEY}).json()
    chapter = data["chapters"][0]
    check("coverage_pct is surfaced", isinstance(chapter.get("coverage_pct"), (int, float)), chapter)
    check("verify_status is PASS", chapter.get("verify_status") == "PASS", chapter)
    check("verified_at is set", chapter.get("verified_at") is not None, chapter)

    tc.post("/admin/ncert-content/verify", json={"class": "11", "subject": "Biology", "chapter": "THE LIVING WORLD", "data": pdf}, headers={"X-Admin-Key": ADMIN_KEY})
    data2 = tc.get("/admin/ncert-content", headers={"X-Admin-Key": ADMIN_KEY}).json()
    chapter2 = data2["chapters"][0]
    check("coverage_pct is unchanged by a later Verify call", chapter2.get("coverage_pct") == chapter.get("coverage_pct"), (chapter, chapter2))

    fake.table.insert({"class": "11", "subject": "Physics", "chapter": "Motion", "heading": "1.1 X", "ordinal": 0, "text": "x", "word_count": 5})
    data3 = tc.get("/admin/ncert-content", headers={"X-Admin-Key": ADMIN_KEY}).json()
    legacy = next(c for c in data3["chapters"] if c["chapter"] == "Motion")
    check("a never-verified chapter has null verify_status/verified_at/coverage_pct",
          legacy.get("verify_status") is None and legacy.get("verified_at") is None and legacy.get("coverage_pct") is None, legacy)


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
    test_preview_includes_verify()
    test_size_cap()
    test_validation_failure_blocks_apply()
    test_apply_writes_and_overwrite_replaces_safely()
    test_apply_keyword_cost_cap()
    test_apply_blocked_on_verify_fail_and_override_allows_it()
    test_verify_endpoint_checks_existing_chapter_and_stamps_log()
    test_verify_endpoint_no_such_chapter()
    test_verify_endpoint_inserts_fallback_log_row_for_legacy_chapter()
    test_verify_endpoint_no_writes_to_sections_even_on_fail()
    test_mark_ok_persists_and_only_resolves_its_own_difference()
    test_list_surfaces_coverage_and_verify_status()
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
