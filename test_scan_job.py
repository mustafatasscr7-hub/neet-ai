"""Mock test for /admin/scan-pdf/job and /admin/scan-pdf/job/{job_id} -- the background-job
version of /admin/scan-pdf for the "standard" and "3col_book" layouts (see server.py's
start_scan_job/get_scan_job and admin-pdf-review.html / contractor-pdf-review.html's runScanJob).

Run with: .venv/Scripts/python.exe test_scan_job.py

No real model call and no real Supabase write anywhere in this file: server.scan_pdf_bytes and
server.scan_book_pdf_bytes are monkeypatched to fake, instant-or-controllable functions, and no
request here sets `filename`, so _create_processed_pdf_row (the only Supabase write scan-pdf ever
does) never fires.

Covers:
  * a job goes queued -> running -> done, with the same result shape /admin/scan-pdf returns;
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

    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
