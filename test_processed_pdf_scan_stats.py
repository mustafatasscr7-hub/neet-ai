"""Regression test for scan-cost tracking on processed_pdfs (mock_scan_stats.sql):
_write_processed_pdf_scan_stats (server.py), wired into /admin/scan-pdf (every layout) and
/admin/scan-pdf/job, plus the contractor-blind fields on GET /admin/processed-pdfs.

Run with: python test_processed_pdf_scan_stats.py

No real model call anywhere in this file: scan_scanned_pdf_bytes, scan_ocr_pdf_bytes,
scan_pdf_bytes, scan_mock_exam_pdf and apply_vision_to_flagged_questions are all monkeypatched to
fakes returning fixed "cost" dicts in the exact shapes those modules already produce (see
process_pyq_scanned.py, process_pyq_scanned_ocr.py, process_mock_exam_vision.py). Same offline
FakeSupabase/FakeSyncRequests approach as test_processed_pdfs_list.py -- reused directly from
there rather than duplicated.

Covers:
  * a "scanned" (Gemini OCR) layout scan writes scan_cost_usd/model_calls/models/pages/scan_seconds;
  * a mock-exam scan with the Vision re-read pass also writes q_found/q_flagged/q_model_read;
  * a "scanned_ocr" (free OCR) scan writes pages/scan_seconds but leaves cost/calls/models null
    (no LLM involved -- the popup shows "no model calls", not a wrong $0 model count);
  * a "standard" layout scan (no cost tracking of its own) leaves scan_cost_usd/model_calls/models
    null too, but still gets pages/scan_seconds from this file's own timing;
  * the same stats get written via the background job path (/admin/scan-pdf/job), not just the
    old synchronous endpoint;
  * a record from before this migration (every new column absent) comes back as None, not an
    error -- the popup's "-" fallback;
  * GET /admin/processed-pdfs for the contractor key never carries any of the 8 new fields, while
    the admin key sees them all.

Deliberately NOT covered here (frontend-only, checked instead via Playwright against the real
page): the per-row muted cost line's exact wording and the popup's filtered-list cost summary.
"""
import io
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fitz
import server
from fastapi.testclient import TestClient
from test_processed_pdfs_list import FakeSupabase, FakeSyncRequests, pdf_row, ADMIN, CONTRACTOR, ADMIN_KEY, CONTRACTOR_KEY

_passed = []
_failed = []


def check(label, condition, detail=""):
    if condition:
        _passed.append(label)
        print(f"  PASS: {label}")
    else:
        _failed.append(label)
        print(f"  FAIL: {label}  {detail}")


def make_pdf_b64(tag="x"):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), tag)
    data = doc.tobytes()
    doc.close()
    import base64
    return base64.b64encode(data).decode()


def fake_question(q=1, needs_review=False, model_read=False):
    return {"question": f"Q{q}", "option_a": "a", "option_b": "b", "option_c": "c", "option_d": "d",
            "correct_answer": "", "question_type": "mcq", "has_diagram": False, "year": None,
            "source_tag": None, "source_page": 1, "needs_review": needs_review, "review_reason": None,
            "model_read": model_read}


def setup(fake):
    server.async_client = __import__("httpx").AsyncClient(transport=__import__("httpx").MockTransport(fake.handle))
    server.http_requests = FakeSyncRequests(fake)
    server.ADMIN_PASSWORD = ADMIN_KEY
    server.CONTRACTOR_PDF_KEY = CONTRACTOR_KEY
    server._pdf_live_jobs.clear()
    server._scan_jobs.clear()
    server._PROCESSED_PDFS_CACHE.update({"at": 0.0, "base_rows": None, "gaps": None, "stats_map": None})


def get_pdf_row(fake, processed_pdf_id):
    return next(r for r in fake.tables["processed_pdfs"] if r["id"] == processed_pdf_id)


def test_scanned_layout_writes_cost(client, fake):
    print("\n# /admin/scan-pdf, layout=scanned: writes cost/calls/model/pages/seconds from its own cost dict")
    fake.tables["processed_pdfs"] = []

    def fake_scan_scanned(pdf_bytes, subject, confirm_over_cap=False, language="en"):
        return {"questions": [fake_question(1)], "pages_scanned": 24, "flagged_pages": [],
                "cost": {"model": "gemini-3.5-flash-lite", "pages_called": 2, "pages_cached": 22,
                         "input_tokens": 1000, "output_tokens": 500, "usd": 0.021}}
    server.scan_scanned_pdf_bytes = fake_scan_scanned

    r = client.post("/admin/scan-pdf", headers=ADMIN,
                     json={"subject": "Biology", "data": make_pdf_b64("sc1"), "filename": "Scanned.pdf", "layout": "scanned"})
    check("200", r.status_code == 200, r.text)
    pid = r.json()["processed_pdf_id"]
    check("processed_pdf_id set", pid is not None)
    row = get_pdf_row(fake, pid)
    check("scan_cost_usd matches the cost dict's usd", row["scan_cost_usd"] == 0.021, row)
    check("model_calls uses pages_called (the real model-call count for this layout)", row["model_calls"] == 2, row)
    check("models carries the model name", row["models"] == "gemini-3.5-flash-lite", row)
    check("pages comes from pages_scanned, not pages_called", row["pages"] == 24, row)
    check("scan_seconds is a non-negative number (timed by the caller)", isinstance(row["scan_seconds"], (int, float)) and row["scan_seconds"] >= 0, row)
    check("no mock-only fields on a real PYQ scan", row.get("q_found") is None and row.get("q_flagged") is None and row.get("q_model_read") is None, row)


def test_mock_exam_with_vision_writes_question_counts(client, fake):
    print("\n# /admin/scan-pdf, layout=mock_exam (+ Vision re-read): writes cost AND q_found/q_flagged/q_model_read")
    fake.tables["processed_pdfs"] = []
    original_flag = server.MOCK_VISION_FLAGGED_ENABLED

    def fake_scan_mock(pdf_bytes, mock_name):
        return {"questions": [fake_question(1), fake_question(2, needs_review=True), fake_question(3, needs_review=True)],
                "pages_scanned": 50, "flagged_pages": [], "completeness": {}, "mock_name": mock_name}

    def fake_apply_vision(result, **kwargs):
        # Simulate the real pass fixing one of the two flagged questions, same shape as
        # process_mock_exam_vision.apply_vision_to_flagged_questions's own result["cost"].
        for q in result["questions"]:
            if q["question"] == "Q2":
                q["needs_review"] = False
                q["model_read"] = True
        result["cost"] = {"usd": 0.002, "calls": 2, "attempted": 2, "model_read": 1, "gate_failed": 1,
                           "retried": 0, "skipped_over_cap": 0, "model": "gemini-3.5-flash-lite"}
        return result

    server.scan_mock_exam_pdf = fake_scan_mock
    server.apply_vision_to_flagged_questions = fake_apply_vision
    server.MOCK_VISION_FLAGGED_ENABLED = True
    try:
        r = client.post("/admin/scan-pdf", headers=ADMIN,
                         json={"data": make_pdf_b64("mk1"), "filename": "Mock.pdf", "layout": "mock_exam", "mock_name": "Mock Test 1"})
        check("200", r.status_code == 200, r.text)
        pid = r.json()["processed_pdf_id"]
        row = get_pdf_row(fake, pid)
        check("scan_cost_usd from the Vision pass's cost dict", row["scan_cost_usd"] == 0.002, row)
        check("model_calls uses cost['calls'] for this layout, not pages_called", row["model_calls"] == 2, row)
        check("models carries the Vision model name", row["models"] == "gemini-3.5-flash-lite", row)
        check("q_found counts every question, flagged or not", row["q_found"] == 3, row)
        check("q_flagged counts only the one still needs_review after the Vision pass", row["q_flagged"] == 1, row)
        check("q_model_read counts the one the Vision pass actually fixed", row["q_model_read"] == 1, row)
    finally:
        server.MOCK_VISION_FLAGGED_ENABLED = original_flag


def test_free_ocr_leaves_model_fields_null(client, fake):
    print("\n# /admin/scan-pdf, layout=scanned_ocr: real pages/seconds, but no model was ever called")
    fake.tables["processed_pdfs"] = []

    def fake_scan_ocr(pdf_bytes, subject, progress=None):
        return {"questions": [fake_question(1)], "pages_scanned": 10, "flagged_pages": [],
                "cost": {"engine": "rapidocr", "usd": 0.0, "pages_called": 10, "pages_cached": 0,
                         "seconds": 5.0, "seconds_per_page": 0.5}}
    server.scan_ocr_pdf_bytes = fake_scan_ocr

    r = client.post("/admin/scan-pdf", headers=ADMIN,
                     json={"subject": "Physics", "data": make_pdf_b64("ocr1"), "filename": "FreeOcr.pdf", "layout": "scanned_ocr"})
    check("200", r.status_code == 200, r.text)
    row = get_pdf_row(fake, r.json()["processed_pdf_id"])
    check("scan_cost_usd is the real $0 (not null -- cost WAS computed, it's just zero)", row["scan_cost_usd"] == 0.0, row)
    check("model_calls stays null -- free OCR never calls a model (cost dict has no 'model'/'calls' key)", row["model_calls"] is None, row)
    check("models stays null for the same reason", row["models"] is None, row)
    check("pages still recorded", row["pages"] == 10, row)


def test_standard_layout_has_no_cost_tracking(client, fake):
    print("\n# /admin/scan-pdf, layout=standard: this layout computes no cost dict at all today --")
    print("# scan_cost_usd/model_calls/models stay null, pages/seconds still get recorded")
    fake.tables["processed_pdfs"] = []

    def fake_scan_standard(pdf_bytes, subject, progress=None, max_workers=4):
        return {"questions": [fake_question(1)], "pages_scanned": 7, "flagged_pages": []}
    server.scan_pdf_bytes = fake_scan_standard

    r = client.post("/admin/scan-pdf", headers=ADMIN,
                     json={"subject": "Chemistry", "data": make_pdf_b64("std1"), "filename": "Standard.pdf", "layout": "standard"})
    check("200", r.status_code == 200, r.text)
    row = get_pdf_row(fake, r.json()["processed_pdf_id"])
    check("scan_cost_usd null (no cost dict to read)", row["scan_cost_usd"] is None, row)
    check("model_calls null", row["model_calls"] is None, row)
    check("models null", row["models"] is None, row)
    check("pages still recorded from pages_scanned", row["pages"] == 7, row)
    check("scan_seconds still recorded (timed independently of any per-layout cost dict)",
          isinstance(row["scan_seconds"], (int, float)) and row["scan_seconds"] >= 0, row)


def test_job_path_also_writes_stats(client, fake):
    print("\n# /admin/scan-pdf/job (background job, standard layout): same stats end up on the row")
    fake.tables["processed_pdfs"] = []

    def fake_scan_standard(pdf_bytes, subject, progress=None):
        return {"questions": [fake_question(1)], "pages_scanned": 3, "flagged_pages": []}
    server.scan_pdf_bytes = fake_scan_standard

    r = client.post("/admin/scan-pdf/job", headers=ADMIN,
                     json={"subject": "Biology", "data": make_pdf_b64("job1"), "filename": "JobScan.pdf", "layout": "standard"})
    check("job start 200", r.status_code == 200, r.text)
    job_id = r.json()["job_id"]
    pid = r.json()["processed_pdf_id"]
    check("processed_pdf_id set up front", pid is not None)

    deadline = time.time() + 5
    status = None
    while time.time() < deadline:
        status = client.get(f"/admin/scan-pdf/job/{job_id}", headers=ADMIN).json().get("status")
        if status in ("done", "error"):
            break
        time.sleep(0.05)
    check("job reaches done", status == "done", status)

    row = get_pdf_row(fake, pid)
    check("pages recorded via the job path too", row["pages"] == 3, row)
    check("scan_seconds recorded via the job path too (timed around start_scan_job, including any queue wait)",
          isinstance(row["scan_seconds"], (int, float)) and row["scan_seconds"] >= 0, row)


def test_old_record_has_null_stats(client, fake):
    print("\n# A pre-migration record (scanned before mock_scan_stats.sql existed): None, not an error")
    fake.tables["processed_pdfs"] = [pdf_row(50, "OldScan.pdf", "completed")]
    fake.tables["pyq"] = []
    server._PROCESSED_PDFS_CACHE.update({"at": 0.0, "base_rows": None, "gaps": None, "stats_map": None})

    data = client.get("/admin/processed-pdfs", headers=ADMIN).json()
    row = next(p for p in data["pdfs"] if p["id"] == 50)
    for field in ("scan_cost_usd", "model_calls", "models", "pages", "scan_seconds", "q_found", "q_flagged", "q_model_read"):
        check(f"old record's {field} is None (popup shows '-')", row.get(field) is None, row)


def test_contractor_never_sees_cost_fields(client, fake):
    print("\n# GET /admin/processed-pdfs: contractor key gets every row but none of the 8 cost fields")
    fake.tables["processed_pdfs"] = [{
        **pdf_row(60, "WithCost.pdf", "completed"),
        "scan_cost_usd": 0.05, "model_calls": 4, "models": "gemini-3.5-flash-lite",
        "pages": 12, "scan_seconds": 33.0, "q_found": None, "q_flagged": None, "q_model_read": None,
    }]
    fake.tables["pyq"] = []
    server._PROCESSED_PDFS_CACHE.update({"at": 0.0, "base_rows": None, "gaps": None, "stats_map": None})

    cost_fields = ("scan_cost_usd", "model_calls", "models", "pages", "scan_seconds", "q_found", "q_flagged", "q_model_read")

    admin_data = client.get("/admin/processed-pdfs", headers=ADMIN).json()
    admin_row = next(p for p in admin_data["pdfs"] if p["id"] == 60)
    check("admin DOES see the cost fields", admin_row.get("scan_cost_usd") == 0.05 and admin_row.get("model_calls") == 4, admin_row)

    server._PROCESSED_PDFS_CACHE.update({"at": 0.0, "base_rows": None, "gaps": None, "stats_map": None})
    contractor_data = client.get("/admin/processed-pdfs", headers=CONTRACTOR).json()
    contractor_row = next(p for p in contractor_data["pdfs"] if p["id"] == 60)
    check("contractor response carries none of the 8 cost fields at all (not even as null)",
          all(field not in contractor_row for field in cost_fields), contractor_row)
    check("contractor still sees everything else (filename, status, counts)",
          contractor_row.get("filename") == "WithCost.pdf" and contractor_row.get("status") == "completed", contractor_row)

    server._PROCESSED_PDFS_CACHE.update({"at": 0.0, "base_rows": None, "gaps": None, "stats_map": None})


def main():
    fake = FakeSupabase()
    setup(fake)
    client = TestClient(server.app)

    test_scanned_layout_writes_cost(client, fake)
    test_mock_exam_with_vision_writes_question_counts(client, fake)
    test_free_ocr_leaves_model_fields_null(client, fake)
    test_standard_layout_has_no_cost_tracking(client, fake)
    test_job_path_also_writes_stats(client, fake)
    test_old_record_has_null_stats(client, fake)
    test_contractor_never_sees_cost_fields(client, fake)

    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
