"""Regression test for the Contractor Queue's new Approved/Needs fix/Deleted counts and the
pyq_delete_log table behind "Deleted".

Run with: python test_pyq_delete_log.py
Same offline fake-Supabase approach as test_pyq_review_gate.py: server.async_client is swapped
for an httpx client backed by an in-memory PostgREST stand-in. /admin/pyq-delete and
/admin/pyq-contractor-queue-counts are called directly as coroutines (not through TestClient)
so the test can control the event loop and await the fire-and-forget log-insert task before
asserting on it -- TestClient tears its own loop down before a created_task is guaranteed to run.
The one exception is the admin-auth check, which fails before any async body runs.

Covers:
  * deleting a contractor question logs one row with the contractor's name;
  * deleting an Admin or null-uploader question logs a row with that name/null, but it's never
    counted in the contractor's "Deleted" total;
  * a failing log insert, and a missing pyq_delete_log table, never stop the delete itself;
  * the counts endpoint's new approved/needs_fix/deleted numbers, including an all-zero case;
  * the existing with_diagram/without_diagram numbers are unchanged;
  * the endpoint still refuses a request without admin auth.
"""
import asyncio
import copy
import io
import json
import os
import sys
import uuid
from datetime import datetime, timezone

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
ADMIN = {"X-Admin-Key": ADMIN_KEY}
DIAGRAM_COLS = ("diagram_url", "option_a_diagram_url", "option_b_diagram_url", "option_c_diagram_url", "option_d_diagram_url")


class FakeSupabase:
    """Minimal PostgREST stand-in: just enough eq/neq/or filtering, count=exact and
    return=representation for the two endpoints under test, plus a way to simulate a table
    that doesn't exist yet (every request on it comes back 404, like a real missing relation)."""

    def __init__(self):
        self.tables = {"pyq": [], "pyq_delete_log": []}
        self.missing_tables = set()
        self.fail_next = {}

    @staticmethod
    def matches(row, field, cond):
        value = row.get(field)
        op, _, arg = cond.partition(".")
        if op == "eq":
            return value is not None and str(value) == arg
        if op == "neq":
            return value is not None and str(value) != arg
        raise AssertionError(f"fake PostgREST: unsupported filter {field}={cond}")

    def filtered(self, table, params):
        rows = self.tables.get(table, [])
        for field, cond in params:
            if field in ("select", "order", "limit", "offset"):
                continue
            if field == "or":
                parts = cond.strip("()").split(",")

                def any_match(r):
                    for part in parts:
                        col, _, rest = part.partition(".")
                        if self.matches(r, col, rest):
                            return True
                    return False
                rows = [r for r in rows if any_match(r)]
                continue
            rows = [r for r in rows if self.matches(r, field, cond)]
        return rows

    def handle(self, request):
        path = request.url.path
        assert path.startswith("/rest/v1/"), path
        table = path[len("/rest/v1/"):]
        method = request.method
        if table in self.missing_tables:
            return httpx.Response(404, json={"message": f"relation \"{table}\" does not exist"})
        status = self.fail_next.pop((method, table), None)
        if status:
            return httpx.Response(status, json={"message": "injected failure"})
        params = list(request.url.params.multi_items())
        p = dict(params)
        if method == "GET":
            rows = self.filtered(table, params)
            total = len(rows)
            headers = {}
            if "count=exact" in request.headers.get("prefer", ""):
                headers["content-range"] = f"0-{max(len(rows) - 1, 0)}/{total}"
            if p.get("limit"):
                rows = rows[: int(p["limit"])]
            return httpx.Response(200, json=copy.deepcopy(rows), headers=headers)
        body = json.loads(request.content or b"null")
        rep = "return=representation" in request.headers.get("prefer", "")
        if method == "POST":
            out = []
            for item in body if isinstance(body, list) else [body]:
                row = dict(item)
                row.setdefault("id", str(uuid.uuid4()))
                row.setdefault("created_at", datetime.now(timezone.utc).isoformat())
                self.tables.setdefault(table, []).append(row)
                out.append(row)
            return httpx.Response(201, json=copy.deepcopy(out) if rep else None)
        if method == "DELETE":
            rows = self.filtered(table, params)
            self.tables[table] = [r for r in self.tables[table] if r not in rows]
            return httpx.Response(200, json=copy.deepcopy(rows) if rep else None)
        raise AssertionError(f"fake PostgREST: unsupported {method}")


def pyq_row(uploader_name, review_status, **extra):
    row = {
        "id": str(uuid.uuid4()), "uploader_name": uploader_name, "review_status": review_status,
        "subject": "Biology", "question": "q?", "is_active": True,
        **{c: None for c in DIAGRAM_COLS},
    }
    row.update(extra)
    return row


async def main():
    fake = FakeSupabase()
    server.async_client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handle))
    server.ADMIN_PASSWORD = ADMIN_KEY
    CONTRACTOR = server.CONTRACTOR_UPLOADER_NAME

    pending_no_diagram = pyq_row(CONTRACTOR, "pending")
    approved = pyq_row(CONTRACTOR, "approved")
    needs_fix = pyq_row(CONTRACTOR, "needs_fix")
    admin_row = pyq_row("Admin", "pending")
    null_uploader_row = pyq_row(None, "approved")
    fake.tables["pyq"] = [pending_no_diagram, approved, needs_fix, admin_row, null_uploader_row]

    print("\n# Counts before any delete")
    counts = await server.admin_pyq_contractor_queue_counts(None)
    check("with_diagram/without_diagram unchanged by this change (1 pending, no diagram)",
          counts.get("with_diagram") == 0 and counts.get("without_diagram") == 1, counts)
    check("approved count for the contractor", counts.get("approved") == 1, counts)
    check("needs_fix count for the contractor", counts.get("needs_fix") == 1, counts)
    check("deleted count starts at 0", counts.get("deleted") == 0, counts)

    print("\n# Deleting a contractor question logs one row with the contractor's name")
    r = await server.admin_pyq_delete(pending_no_diagram["id"], None)
    await asyncio.sleep(0)  # let the fire-and-forget log-insert task post
    check("delete succeeds", r.get("success") is True, r)
    log_rows = [l for l in fake.tables["pyq_delete_log"] if l.get("pyq_id") == pending_no_diagram["id"]]
    check("exactly one log row for the deleted question", len(log_rows) == 1, fake.tables["pyq_delete_log"])
    check("logged with the contractor's name", log_rows and log_rows[0].get("uploader_name") == CONTRACTOR, log_rows)

    print("\n# Deleting an Admin or null-uploader question logs it, but it's never counted for the contractor")
    r = await server.admin_pyq_delete(admin_row["id"], None)
    await asyncio.sleep(0)
    check("admin-row delete succeeds", r.get("success") is True, r)
    r = await server.admin_pyq_delete(null_uploader_row["id"], None)
    await asyncio.sleep(0)
    check("null-uploader delete succeeds", r.get("success") is True, r)
    admin_log = [l for l in fake.tables["pyq_delete_log"] if l.get("pyq_id") == admin_row["id"]]
    null_log = [l for l in fake.tables["pyq_delete_log"] if l.get("pyq_id") == null_uploader_row["id"]]
    check("admin delete logged with uploader_name='Admin'", admin_log and admin_log[0].get("uploader_name") == "Admin", admin_log)
    check("null-uploader delete logged with uploader_name=None", null_log and null_log[0].get("uploader_name") is None, null_log)
    counts = await server.admin_pyq_contractor_queue_counts(None)
    check("contractor's deleted count only counts the contractor's own row (still 1)", counts.get("deleted") == 1, counts)

    print("\n# A failing log insert never stops the delete")
    fake.fail_next[("POST", "pyq_delete_log")] = 500
    before = len(fake.tables["pyq_delete_log"])
    r = await server.admin_pyq_delete(approved["id"], None)
    await asyncio.sleep(0)
    check("delete still succeeds when the log insert fails", r.get("success") is True, r)
    check("no error surfaced to the admin despite the failed log insert", "error" not in r, r)
    check("the failed insert left no new log row", len(fake.tables["pyq_delete_log"]) == before, fake.tables["pyq_delete_log"])

    print("\n# A missing pyq_delete_log table never stops the delete, and counts as 0")
    fake.missing_tables.add("pyq_delete_log")
    r = await server.admin_pyq_delete(needs_fix["id"], None)
    await asyncio.sleep(0)
    check("delete succeeds even though the log table doesn't exist", r.get("success") is True, r)
    counts = await server.admin_pyq_contractor_queue_counts(None)
    check("deleted count comes back 0, not an error, when the table is missing", counts.get("deleted") == 0 and "error" not in counts, counts)
    fake.missing_tables.discard("pyq_delete_log")

    print("\n# Counts endpoint refuses a request without admin auth")
    client = TestClient(server.app)  # no `with`: skips startup hooks, nothing here needs them
    resp = client.get("/admin/pyq-contractor-queue-counts")
    check("no key -> 401", resp.status_code == 401, resp.status_code)
    resp = client.get("/admin/pyq-contractor-queue-counts", headers={"X-Admin-Key": "wrong"})
    check("wrong key -> 401", resp.status_code == 401, resp.status_code)

    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
