"""Mock test for /admin/scan-pdf/job and /admin/scan-pdf/job/{job_id} -- the background-job
version of /admin/scan-pdf for the "standard", "3col_book" and "mock_exam" layouts (see server.py's
start_scan_job/get_scan_job and admin-pdf-review.html / contractor-pdf-review.html's runScanJob).

Run with: .venv/Scripts/python.exe test_scan_job.py

No real model call and no real Supabase write anywhere in this file: server.scan_pdf_bytes,
server.scan_book_pdf_bytes, server.scan_mock_exam_pdf and server.apply_vision_to_flagged_questions
are all monkeypatched to fake, instant-or-controllable functions, and no request here sets
`filename`, so _create_processed_pdf_row (the only Supabase write scan-pdf ever does) never fires.

Covers:
  * a job goes queued -> running -> done, with the same result shape /admin/scan-pdf returns;
  * a mock_exam job runs the deterministic scan then the vision pass, in that order, before done;
  * a mock_exam job with no mock_name is refused up front, same as the synchronous endpoint;
  * polling with the wrong key role, or an unknown job_id, is refused (both come back as the same
    "unknown or expired" error -- a caller can't tell the two apart);
  * two jobs run at once and a third queues behind them (the semaphore(2) slot limit);
  * a scan that errors ends with status "error" and a readable message;
  * finished jobs are pruned after their TTL;
  * the old /admin/scan-pdf endpoint still works, unchanged, for the same input;
  * starting a job is fast even though the scan itself is slow (no request in this flow blocks
    for more than a moment).
"""
import io
import os
import sys
import threading
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fitz
import server
from fastapi.testclient import TestClient
from test_pyq_review_gate import ADMIN, CONTRACTOR, ADMIN_KEY, CONTRACTOR_KEY

_passed = []
_failed = []


def check(label, condition, detail=""):
    if condition:
        _passed.append(label)
        print(f"  PASS: {label}")
    else:
        _failed.append(label)
        print(f"  FAIL: {label}  {detail}")


def make_pdf_b64(pages=2, tag="x"):
    """A tiny real one-or-more-page PDF (fitz needs to open it to count pages), base64-encoded.
    `tag` just keeps different calls' bytes distinct so a fake scan_fn can tell jobs apart."""
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"{tag}-{i}")
    data = doc.tobytes()
    doc.close()
    import base64
    return base64.b64encode(data).decode()


def fake_scan_pdf_bytes(pdf_bytes, subject, progress=None, max_workers=4):
    if progress:
        progress(0, 2)
        progress(1, 2)
    gate = _GATES.get(pdf_bytes)
    if gate:
        gate.wait(timeout=5)
    if progress:
        progress(2, 2)
    if pdf_bytes in _ERROR_ON:
        raise RuntimeError("simulated extraction failure")
    return {"questions": [{"question": f"fake standard Q for {subject}", "option_a": "a", "option_b": "b",
                            "option_c": "c", "option_d": "d", "correct_answer": "", "question_type": "mcq",
                            "has_diagram": False, "year": None, "source_tag": None, "source_page": 1,
                            "needs_review": False, "review_reason": None}],
            "pages_scanned": 2, "flagged_pages": []}


def fake_scan_book_pdf_bytes(pdf_bytes, subject):
    gate = _GATES.get(pdf_bytes)
    if gate:
        gate.wait(timeout=5)
    return {"questions": [{"question": f"fake 3col_book Q for {subject}", "option_a": "a", "option_b": "b",
                            "option_c": "c", "option_d": "d", "correct_answer": "", "question_type": "mcq",
                            "has_diagram": False, "year": None, "source_tag": None, "source_page": 1,
                            "needs_review": False, "review_reason": None}],
            "pages_scanned": 1, "flagged_pages": []}


def fake_scan_mock_exam_pdf(pdf_bytes, mock_name, progress=None):
    if progress:
        progress(1, 2)
        progress(2, 2)
    flagged = {"question": "fake flagged mock Q", "option_a": "a", "option_b": "b", "option_c": "c",
               "option_d": "d", "correct_answer": "", "question_type": "mcq", "has_diagram": False,
               "year": None, "source_tag": None, "source_page": 1, "needs_review": True,
               "review_reason": "fake reason", "subject": "Physics", "source": "mock",
               "mock_name": mock_name, "mock_position": 1, "printed_number": 1,
               "crop_image_b64": None, "_text_layer_hint": "hint"}
    clean = {"question": f"fake clean mock Q for {mock_name}", "option_a": "a", "option_b": "b",
             "option_c": "c", "option_d": "d", "correct_answer": "", "question_type": "mcq",
             "has_diagram": False, "year": None, "source_tag": None, "source_page": 1,
             "needs_review": False, "review_reason": None, "subject": "Physics", "source": "mock",
             "mock_name": mock_name, "mock_position": 2, "printed_number": 2}
    return {"questions": [flagged, clean], "pages_scanned": 2, "flagged_pages": [{"page": 1, "reason": "fake reason"}],
            "completeness": {"Physics": {"expected": 45, "found_count": 2, "missing": [], "duplicates": [],
                                          "out_of_range": [], "coverage_pct": 100.0, "red_flag_count": 1}}}


def fake_apply_vision(result, progress=None, **kwargs):
    attempted = [q for q in result["questions"] if q.get("needs_review")]
    if progress:
        progress(1, max(1, len(attempted)))
    for q in attempted:
        q["needs_review"] = False
        q["model_read"] = True
    for q in result["questions"]:
        q.pop("_text_layer_hint", None)
    result["cost"] = {"usd": 0.0001, "calls": 1, "attempted": len(attempted), "model_read": len(attempted),
                       "gate_failed": 0, "retried": 0, "skipped_over_cap": 0, "model": "fake-model"}
    return result


_GATES = {}   # pdf_bytes -> threading.Event the fake scan waits on, if any
_ERROR_ON = set()  # pdf_bytes that should raise instead of returning


def poll(client, headers, job_id):
    return client.get(f"/admin/scan-pdf/job/{job_id}", headers=headers).json()


def poll_until(client, headers, job_id, want_statuses, timeout=5):
    deadline = time.time() + timeout
    job = None
    while time.time() < deadline:
        job = poll(client, headers, job_id)
        if job.get("status") in want_statuses:
            return job
        time.sleep(0.05)
    return job


def main():
    server.scan_pdf_bytes = fake_scan_pdf_bytes
    server.scan_book_pdf_bytes = fake_scan_book_pdf_bytes
    server.scan_mock_exam_pdf = fake_scan_mock_exam_pdf
    server.apply_vision_to_flagged_questions = fake_apply_vision
    server.MOCK_VISION_FLAGGED_ENABLED = True
    server.ADMIN_PASSWORD = ADMIN_KEY
    server.CONTRACTOR_PDF_KEY = CONTRACTOR_KEY
    client = TestClient(server.app)

    print("\n# A standard-layout job goes queued/running -> done, with the usual result shape")
    b64 = make_pdf_b64(tag="std1")
    r = client.post("/admin/scan-pdf/job", json={"subject": "Biology", "data": b64, "layout": "standard"}, headers=ADMIN)
    check("job start returns 200", r.status_code == 200, r.text)
    start = r.json()
    check("job start carries a job_id and pages_total", bool(start.get("job_id")) and start.get("pages_total") == 2, start)
    job = poll_until(client, ADMIN, start["job_id"], {"done", "error"})
    check("job reaches done", job.get("status") == "done", job)
    result = job.get("result", {})
    check("result has the same shape /admin/scan-pdf returns",
          result.get("pages_scanned") == 2 and len(result.get("questions", [])) == 1
          and result["questions"][0]["question"] == "fake standard Q for Biology", result)
    check("result carries processed_pdf_id (null here since no filename was sent)",
          "processed_pdf_id" in result and result["processed_pdf_id"] is None, result)

    print("\n# A 3col_book-layout job also runs end to end")
    b64b = make_pdf_b64(tag="book1")
    r = client.post("/admin/scan-pdf/job", json={"subject": "Physics", "data": b64b, "layout": "3col_book"}, headers=ADMIN)
    start_b = r.json()
    job_b = poll_until(client, ADMIN, start_b["job_id"], {"done", "error"})
    check("3col_book job reaches done", job_b.get("status") == "done", job_b)
    check("3col_book result shape", job_b.get("result", {}).get("questions", [{}])[0].get("question") == "fake 3col_book Q for Physics", job_b)

    print("\n# Polling with the wrong key role is refused, same as an unknown job_id")
    r = client.post("/admin/scan-pdf/job", json={"subject": "Biology", "data": make_pdf_b64(tag="role1"), "layout": "standard"}, headers=CONTRACTOR)
    cjob_id = r.json()["job_id"]
    wrong_role = poll(client, ADMIN, cjob_id)
    check("admin key can't read a contractor-started job", "error" in wrong_role and "status" not in wrong_role, wrong_role)
    unknown = poll(client, ADMIN, "not-a-real-job-id")
    check("unknown job_id gives the same shape of refusal", "error" in unknown and "status" not in unknown, unknown)
    check("the two refusals read identically (role mismatch isn't distinguishable from unknown)",
          wrong_role == unknown, (wrong_role, unknown))
    own_role = poll_until(client, CONTRACTOR, cjob_id, {"done", "error"})
    check("the contractor key that started it CAN read it", own_role.get("status") == "done", own_role)

    print("\n# Two jobs run at once; a third queues behind them")
    gate = threading.Event()
    b64_a, b64_c, b64_d = make_pdf_b64(tag="conc-a"), make_pdf_b64(tag="conc-c"), make_pdf_b64(tag="conc-d")
    import base64 as _b64
    for b in (b64_a, b64_c, b64_d):
        _GATES[_b64.b64decode(b)] = gate
    ids = []
    for b in (b64_a, b64_c, b64_d):
        r = client.post("/admin/scan-pdf/job", json={"subject": "Biology", "data": b, "layout": "standard"}, headers=ADMIN)
        ids.append(r.json()["job_id"])
    time.sleep(0.3)  # let the 2 slot-holders reach "running" and block on the gate
    statuses = [poll(client, ADMIN, jid).get("status") for jid in ids]
    check("exactly 2 of the 3 jobs are running, 1 is queued",
          sorted(statuses) == ["queued", "running", "running"], statuses)
    gate.set()  # release all three
    for jid in ids:
        job = poll_until(client, ADMIN, jid, {"done", "error"})
        check(f"job {jid[:8]} eventually finishes once released", job.get("status") == "done", job)

    print("\n# A scan that errors ends with status 'error' and a readable message")
    b64_err = make_pdf_b64(tag="err1")
    _ERROR_ON.add(_b64.b64decode(b64_err))
    r = client.post("/admin/scan-pdf/job", json={"subject": "Biology", "data": b64_err, "layout": "standard"}, headers=ADMIN)
    err_job = poll_until(client, ADMIN, r.json()["job_id"], {"done", "error"})
    check("errored job ends with status error", err_job.get("status") == "error", err_job)
    check("error message is readable, not a stack trace", err_job.get("error") == "simulated extraction failure", err_job)

    print("\n# Finished jobs are pruned after their TTL")
    old_ttl = server._SCAN_JOB_TTL_SECONDS
    server._SCAN_JOB_TTL_SECONDS = 0.05
    try:
        b64_ttl = make_pdf_b64(tag="ttl1")
        r = client.post("/admin/scan-pdf/job", json={"subject": "Biology", "data": b64_ttl, "layout": "standard"}, headers=ADMIN)
        ttl_job_id = r.json()["job_id"]
        poll_until(client, ADMIN, ttl_job_id, {"done", "error"})
        check("job exists right after finishing", ttl_job_id in server._scan_jobs, list(server._scan_jobs))
        time.sleep(0.15)
        # _prune_scan_jobs only runs when a new job starts -- trigger it with a throwaway job.
        client.post("/admin/scan-pdf/job", json={"subject": "Biology", "data": make_pdf_b64(tag="ttl-trigger"), "layout": "standard"}, headers=ADMIN)
        check("the finished job was pruned once its TTL passed", ttl_job_id not in server._scan_jobs, list(server._scan_jobs))
    finally:
        server._SCAN_JOB_TTL_SECONDS = old_ttl

    print("\n# The old /admin/scan-pdf endpoint still works exactly as before, same input/output")
    b64_old = make_pdf_b64(tag="old1")
    r = client.post("/admin/scan-pdf", json={"subject": "Biology", "data": b64_old, "layout": "standard"}, headers=ADMIN)
    check("old endpoint still returns 200", r.status_code == 200, r.text)
    old_result = r.json()
    check("old endpoint's result shape is unchanged",
          old_result.get("pages_scanned") == 2 and old_result["questions"][0]["question"] == "fake standard Q for Biology", old_result)

    print("\n# Starting a job is fast even though the underlying scan is slow")
    slow_gate = threading.Event()
    b64_slow = make_pdf_b64(tag="slow1")
    _GATES[_b64.b64decode(b64_slow)] = slow_gate
    t0 = time.time()
    r = client.post("/admin/scan-pdf/job", json={"subject": "Biology", "data": b64_slow, "layout": "standard"}, headers=ADMIN)
    elapsed = time.time() - t0
    check("POST /admin/scan-pdf/job returns almost immediately, not after the scan finishes",
          elapsed < 1.0, elapsed)
    slow_job_id = r.json()["job_id"]
    t1 = time.time()
    status_poll = poll(client, ADMIN, slow_job_id)
    poll_elapsed = time.time() - t1
    check("GET /admin/scan-pdf/job/{id} (poll) also returns almost immediately",
          poll_elapsed < 1.0, poll_elapsed)
    check("...and it reports the job as still running/queued while the gate is held",
          status_poll.get("status") in ("running", "queued"), status_poll)
    slow_gate.set()
    poll_until(client, ADMIN, slow_job_id, {"done", "error"})

    # Run last, not before the TTL-pruning section above: that section's own timing margin
    # (0.05s TTL / 0.15s sleep) is tight enough that extra work running ahead of it under load
    # can flake it, confirmed live when this block briefly sat earlier in this same function.
    # Also reset the rate limiter here -- /admin/scan-pdf/job is capped at 10 requests/300s per
    # IP, and every job test above (all from the SAME TestClient "IP") has already spent most of
    # that quota by this point in the file; confirmed live (a real 429, not a flaky timing issue)
    # once this block's own 2 requests became the 10th and 11th.
    server._rate_buckets.clear()
    print("\n# A mock_exam-layout job runs the deterministic scan, then the vision pass, then done")
    b64m = make_pdf_b64(tag="mock1")
    r = client.post("/admin/scan-pdf/job", json={"data": b64m, "layout": "mock_exam", "mock_name": "Job Test Mock"}, headers=ADMIN)
    check("job start returns 200", r.status_code == 200, r.text)
    start_m = r.json()
    check("job start carries a job_id and pages_total", bool(start_m.get("job_id")) and start_m.get("pages_total") == 2, start_m)
    job_m = poll_until(client, ADMIN, start_m["job_id"], {"done", "error"})
    check("mock_exam job reaches done", job_m.get("status") == "done", job_m)
    result_m = job_m.get("result", {})
    check("result has scan_mock_exam_pdf's own shape", result_m.get("pages_scanned") == 2 and len(result_m.get("questions", [])) == 2, result_m)
    check("the vision pass ran and fixed the flagged question",
          result_m["questions"][0]["needs_review"] is False and result_m["questions"][0].get("model_read") is True, result_m)
    check("the clean question was never touched by the vision pass",
          result_m["questions"][1]["needs_review"] is False and not result_m["questions"][1].get("model_read"), result_m)
    check("cost dict from the vision pass is attached", result_m.get("cost", {}).get("model") == "fake-model", result_m)

    print("\n# A mock_exam job without a mock_name is refused up front, same as the sync endpoint")
    r = client.post("/admin/scan-pdf/job", json={"data": make_pdf_b64(tag="mock-noname"), "layout": "mock_exam"}, headers=ADMIN)
    check("missing mock_name is refused", "error" in r.json() and "job_id" not in r.json(), r.json())

    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
