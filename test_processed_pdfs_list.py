"""Regression test for the enriched Processed PDFs list (GET /admin/processed-pdfs), the
admin-only Remove record endpoint (DELETE /admin/processed-pdfs/{id}), the startup repair of
stuck "processing" rows, the per-subject chapter gap summary, and the Contractor Queue's new
source_pdf_filename filter on /admin/pyq-search.

Run with: python test_processed_pdfs_list.py
Same plain pass/fail harness and offline fake-Supabase approach as test_pyq_review_gate.py:
server.async_client is swapped for an httpx client backed by an in-memory PostgREST stand-in, so
the real endpoint code (including every count query it fires) runs without touching a database.
"""
import asyncio
import copy
import io
import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import httpx
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
        print(f"  FAIL: {label}  {detail}")


ADMIN_KEY = "test-admin-key"
CONTRACTOR_KEY = "test-contractor-key"
ADMIN = {"X-Admin-Key": ADMIN_KEY}
CONTRACTOR = {"X-Admin-Key": CONTRACTOR_KEY}


def _parse_time(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------- fake Supabase
class FakeSupabase:
    def __init__(self):
        self.tables = {"pyq": [], "processed_pdfs": []}
        self.next_id = 1000

    @staticmethod
    def matches(row, field, cond):
        value = row.get(field)
        if cond == "is.null":
            return value is None
        if cond == "not.is.null":
            return value is not None
        op, _, arg = cond.partition(".")
        if op == "eq":
            if isinstance(value, bool):
                return str(value).lower() == arg.lower()
            return value is not None and str(value) == arg
        if op == "neq":
            return value is not None and str(value) != arg
        if op in ("gte", "lt"):
            a, b = _parse_time(value), _parse_time(arg)
            if a is None or b is None:
                return False
            return a >= b if op == "gte" else a < b
        if op == "in":
            return str(value) in [a.strip() for a in arg.strip("()").split(",")]
        raise AssertionError(f"fake PostgREST: unsupported filter {field}={cond}")

    def filtered(self, table, params):
        rows = self.tables[table]
        for field, cond in params:
            if field in ("select", "order", "limit", "offset"):
                continue
            if field == "or":
                # (col1.neq.,col2.neq.,...) -- true if ANY listed column is non-empty, same
                # shape _PYQ_HAS_DIAGRAM_OR uses.
                parts = cond.strip("()").split(",")
                def any_col(r, parts=parts):
                    for part in parts:
                        col, _, rest = part.partition(".")
                        if self.matches(r, col, rest):
                            return True
                    return False
                rows = [r for r in rows if any_col(r)]
                continue
            rows = [r for r in rows if self.matches(r, field, cond)]
        return rows

    def handle(self, request):
        path = request.url.path
        assert path.startswith("/rest/v1/"), path
        table = path[len("/rest/v1/"):]
        method = request.method
        params = list(request.url.params.multi_items())
        p = dict(params)
        headers = {}
        if method == "GET":
            rows = self.filtered(table, params)
            if p.get("order"):
                col, _, direction = p["order"].partition(".")
                rows = sorted(rows, key=lambda r: (r.get(col) is None, str(r.get(col))), reverse=direction == "desc")
            total = len(rows)
            if p.get("limit"):
                rows = rows[: int(p["limit"])]
            if "count=exact" in request.headers.get("prefer", ""):
                headers["content-range"] = f"0-{max(len(rows) - 1, 0)}/{total}"
            sel = p.get("select", "*")
            if sel != "*":
                cols = sel.split(",")
                rows = [{c: r.get(c) for c in cols} for r in rows]
            return httpx.Response(200, json=copy.deepcopy(rows), headers=headers)
        body = json.loads(request.content or b"null")
        rep = "return=representation" in request.headers.get("prefer", "")
        if method == "POST":
            out = []
            for item in body if isinstance(body, list) else [body]:
                row = dict(item)
                row.setdefault("id", str(uuid.uuid4()) if table == "pyq" else self.next_id)
                if table != "pyq":
                    self.next_id += 1
                self.tables[table].append(row)
                out.append(row)
            return httpx.Response(201, json=copy.deepcopy(out) if rep else None)
        if method == "PATCH":
            rows = self.filtered(table, params)
            for r in rows:
                r.update(body)
            return httpx.Response(200, json=copy.deepcopy(rows) if rep else None)
        if method == "DELETE":
            rows = self.filtered(table, params)
            self.tables[table] = [r for r in self.tables[table] if r not in rows]
            return httpx.Response(200, json=copy.deepcopy(rows) if rep else None)
        raise AssertionError(f"fake PostgREST: unsupported {method}")


def pdf_row(id, filename, status, *, subject="Biology", age_minutes=0, error_message=None, questions_extracted=0):
    upload_date = datetime.now(timezone.utc) - timedelta(minutes=age_minutes)
    return {"id": id, "filename": filename, "subject": subject, "status": status,
            "questions_extracted": questions_extracted, "error_message": error_message,
            "upload_date": upload_date.isoformat(), "processed_at": None}


def pyq_row(filename, *, review_status="approved", active=True, has_diagram=False, uploader_name=None, subject="Biology"):
    return {"id": str(uuid.uuid4()), "subject": subject, "source_pdf_filename": filename,
            "review_status": review_status, "is_active": active, "uploader_name": uploader_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "diagram_url": "https://x/d.png" if has_diagram else None,
            "option_a_diagram_url": None, "option_b_diagram_url": None,
            "option_c_diagram_url": None, "option_d_diagram_url": None}


def setup(fake):
    server.async_client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handle))
    server.ADMIN_PASSWORD = ADMIN_KEY
    server.CONTRACTOR_PDF_KEY = CONTRACTOR_KEY
    server._pdf_live_jobs.clear()
    server._scan_jobs.clear()
    server._PROCESSED_PDFS_CACHE.update({"at": 0.0, "pdfs": None, "gaps": None})


# ---------------------------------------------------------------- unit tests (no HTTP)
def test_effective_status():
    print("\n# _effective_pdf_status: processing/failed -> interrupted only when actually stuck")
    fresh_processing = pdf_row(1, "a.pdf", "processing", age_minutes=5)
    check("fresh 'processing' (5 min) stays processing", server._effective_pdf_status(fresh_processing) == "processing")

    stale_no_job = pdf_row(2, "b.pdf", "processing", age_minutes=45)
    check("stale 'processing' (45 min), no live job -> interrupted", server._effective_pdf_status(stale_no_job) == "interrupted")

    stale_with_job = pdf_row(3, "c.pdf", "processing", age_minutes=45)
    server._pdf_live_jobs[3] = ("scan", "job-3")
    server._scan_jobs["job-3"] = {"status": "running"}
    check("stale 'processing' WITH a live job -> stays processing", server._effective_pdf_status(stale_with_job) == "processing")
    server._scan_jobs["job-3"] = {"status": "done"}
    check("same row once its job finished -> interrupted", server._effective_pdf_status(stale_with_job) == "interrupted")
    check("_pdf_job_is_live prunes the finished entry", 3 not in server._pdf_live_jobs)

    marked_failed = pdf_row(4, "d.pdf", "failed", error_message="Interrupted: the server restarted while this PDF was still processing.")
    check("startup-repair-written 'failed' shows as interrupted", server._effective_pdf_status(marked_failed) == "interrupted")

    real_failure = pdf_row(5, "e.pdf", "failed", error_message="Claude Vision returned malformed JSON")
    check("a genuine scan failure stays failed", server._effective_pdf_status(real_failure) == "failed")

    completed = pdf_row(6, "f.pdf", "completed")
    check("completed stays completed", server._effective_pdf_status(completed) == "completed")


def test_repair_stuck_processed_pdfs():
    print("\n# _repair_stuck_processed_pdfs: only stale 'processing' rows are touched")
    fake = FakeSupabase()
    setup(fake)
    fake.tables["processed_pdfs"] = [
        pdf_row(10, "recent.pdf", "processing", age_minutes=2),
        pdf_row(11, "stale.pdf", "processing", age_minutes=45),
        pdf_row(12, "done.pdf", "completed", age_minutes=120),
    ]
    asyncio.run(server._repair_stuck_processed_pdfs())
    by_id = {r["id"]: r for r in fake.tables["processed_pdfs"]}
    check("recent processing row untouched", by_id[10]["status"] == "processing")
    check("stale processing row marked failed", by_id[11]["status"] == "failed")
    check("stale row's error_message carries the Interrupted marker", (by_id[11]["error_message"] or "").startswith("Interrupted:"))
    check("already-completed row untouched", by_id[12]["status"] == "completed")


def test_gap_summary():
    print("\n# _processed_pdfs_gap_summary: missing chapters 1-5 and 36-38 found -> 'Missing: 6-35'")
    pdfs = (
        [{"status": "completed", "subject": "Biology", "filename": f"Chapter {n} - X.pdf"} for n in [1, 2, 3, 4, 5, 36, 37, 38]]
        + [{"status": "completed", "subject": "Biology", "filename": "Full Mock Test.pdf"}]
        + [{"status": "processing", "subject": "Biology", "filename": "Chapter 10 - Y.pdf"}]  # not completed -> doesn't fill the gap
    )
    gaps = server._processed_pdfs_gap_summary(pdfs)
    check("found lists 1-5, 36-38", gaps["Biology"]["found"] == "1-5, 36-38", gaps["Biology"])
    check("missing is exactly 6-35 (a still-processing Chapter 10 doesn't count)", gaps["Biology"]["missing"] == "6-35", gaps["Biology"])
    check("unnumbered filename listed separately", gaps["Biology"]["unnumbered"] == ["Full Mock Test.pdf"])


# ---------------------------------------------------------------- HTTP-level tests
def test_list_counts_and_roles(client, fake):
    print("\n# GET /admin/processed-pdfs: live counts, uploader, deleted=unknown, role gating")
    fake.tables["processed_pdfs"] = [pdf_row(100, "Chapter 5 - Cell.pdf", "completed", subject="Biology", questions_extracted=3)]
    fake.tables["pyq"] = [
        pyq_row("Chapter 5 - Cell.pdf", review_status="approved", has_diagram=True, uploader_name="Admin"),
        pyq_row("Chapter 5 - Cell.pdf", review_status="pending"),
        pyq_row("Chapter 5 - Cell.pdf", review_status="needs_fix"),
        pyq_row("Chapter 5 - Cell.pdf", review_status="approved", active=False),  # soft-inactive: excluded from "total"
        pyq_row("Other Chapter.pdf", review_status="approved"),  # different PDF -- must not leak in
    ]
    server._PROCESSED_PDFS_CACHE.update({"at": 0.0, "pdfs": None, "gaps": None})

    r_admin = client.get("/admin/processed-pdfs", headers=ADMIN)
    check("admin: 200", r_admin.status_code == 200, r_admin.text)
    data = r_admin.json()
    check("can_remove true for admin", data["can_remove"] is True)
    row = data["pdfs"][0]
    c = row["counts"]
    check("total counts only this PDF's active rows (3, not the 4th inactive or the other file)", c["total"] == 3, c)
    check("approved/pending/needs_fix are also scoped to active rows (the 4th, inactive approved row doesn't count)",
          (c["approved"], c["pending"], c["needs_fix"]) == (1, 1, 1), c)
    check("with/without diagram split correctly", (c["with_diagram"], c["without_diagram"]) == (1, 2), c)
    check("deleted is unknown by design, with a reason", c["deleted"] is None and bool(row["deleted_reason"]))
    check("uploader_name sampled from a real row", row["uploader_name"] in ("Admin", None, "") or row["uploader_name"] is not None)

    r_contractor = client.get("/admin/processed-pdfs", headers=CONTRACTOR)
    check("contractor: 200 too (read access)", r_contractor.status_code == 200)
    check("can_remove false for contractor", r_contractor.json()["can_remove"] is False)

    check("wrong key -> 401", client.get("/admin/processed-pdfs", headers={"X-Admin-Key": "nope"}).status_code == 401)


def test_zero_and_cache(client, fake):
    print("\n# A PDF with zero saved questions shows all-zero counts; list is cached for 30s")
    fake.tables["processed_pdfs"] = [pdf_row(200, "Untouched.pdf", "processing", age_minutes=1, subject="Physics")]
    fake.tables["pyq"] = []
    server._PROCESSED_PDFS_CACHE.update({"at": 0.0, "pdfs": None, "gaps": None})
    data = client.get("/admin/processed-pdfs", headers=ADMIN).json()
    c = data["pdfs"][0]["counts"]
    check("zero rows -> every count is 0", all(c[k] == 0 for k in ("total", "approved", "pending", "needs_fix", "with_diagram", "without_diagram")), c)

    # Mutate the underlying table directly (bypassing the endpoint) -- a cached response must
    # keep serving the old snapshot until force=true or the TTL elapses.
    fake.tables["processed_pdfs"].append(pdf_row(201, "New.pdf", "completed"))
    cached = client.get("/admin/processed-pdfs", headers=ADMIN).json()
    check("still cached: new row not yet visible", len(cached["pdfs"]) == 1, cached["pdfs"])
    forced = client.get("/admin/processed-pdfs?force=true", headers=ADMIN).json()
    check("force=true bypasses the cache", len(forced["pdfs"]) == 2, forced["pdfs"])


def test_remove_record(client, fake):
    print("\n# DELETE /admin/processed-pdfs/{id}: admin-only, interrupted/failed only, removes only the record")
    fake.tables["processed_pdfs"] = [
        pdf_row(300, "completed.pdf", "completed"),
        pdf_row(301, "failed.pdf", "failed", error_message="boom"),
        pdf_row(302, "stale.pdf", "processing", age_minutes=45),
    ]
    fake.tables["pyq"] = [pyq_row("failed.pdf", review_status="approved")]

    check("contractor key -> 403 (cannot remove)", client.delete("/admin/processed-pdfs/301", headers=CONTRACTOR).status_code == 403)
    check("record still present after the 403", any(r["id"] == 301 for r in fake.tables["processed_pdfs"]))

    check("admin cannot remove a Completed record", client.delete("/admin/processed-pdfs/300", headers=ADMIN).json().get("error"))
    check("completed record untouched", any(r["id"] == 300 for r in fake.tables["processed_pdfs"]))

    r = client.delete("/admin/processed-pdfs/301", headers=ADMIN)
    check("admin removes a Failed record: success", r.json().get("success") is True, r.text)
    check("that record is gone", not any(r["id"] == 301 for r in fake.tables["processed_pdfs"]))
    check("its questions were never touched", any(q["source_pdf_filename"] == "failed.pdf" for q in fake.tables["pyq"]))

    r2 = client.delete("/admin/processed-pdfs/302", headers=ADMIN)
    check("admin removes a stale/Interrupted record too: success", r2.json().get("success") is True, r2.text)

    check("unknown id -> error, not a 500", "error" in client.delete("/admin/processed-pdfs/9999", headers=ADMIN).json())


def test_pyq_search_filename_filter(client, fake):
    print("\n# GET /admin/pyq-search?source_pdf_filename=...: Contractor Queue 'Showing: <file>' filter")
    fake.tables["pyq"] = [pyq_row("A.pdf"), pyq_row("A.pdf"), pyq_row("B.pdf")]
    r = client.get("/admin/pyq-search?source_pdf_filename=A.pdf&full=true", headers=ADMIN)
    check("200", r.status_code == 200, r.text)
    rows = r.json()["rows"]
    check("only A.pdf's 2 rows come back", len(rows) == 2 and all(row["source_pdf_filename"] == "A.pdf" for row in rows), rows)


def main():
    test_effective_status()
    test_gap_summary()
    test_repair_stuck_processed_pdfs()

    fake = FakeSupabase()
    setup(fake)
    client = TestClient(server.app)  # no `with`: skips startup hooks, nothing here needs them

    test_list_counts_and_roles(client, fake)
    test_zero_and_cache(client, fake)
    test_remove_record(client, fake)
    test_pyq_search_filename_filter(client, fake)

    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
