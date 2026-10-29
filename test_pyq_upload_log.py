"""Regression test for the Contractor Queue's permanent "Uploaded so far" totals and the
pyq_upload_log table behind them (pyq_upload_log.sql + server.py).

Run with: python test_pyq_upload_log.py
Same offline fake-Supabase approach as test_pyq_delete_log.py: server.async_client is swapped
for an httpx client backed by an in-memory PostgREST stand-in. /admin/pyq-bulk-create and
/admin/pyq-contractor-queue-counts are called directly as coroutines (not through TestClient) so
the test can control the event loop and await the fire-and-forget log-insert task before
asserting on it -- TestClient tears its own loop down before a created_task is guaranteed to
run. The one exception is the admin-auth check, which fails before any async body runs.
_embed_pyq_row_background (a separate fire-and-forget task on every create, calling a real
embedding API) is stubbed out so this stays fully offline.

Covers:
  * a new contractor-uploaded question (with or without a diagram) logs exactly one row, counted
    correctly by /admin/pyq-contractor-queue-counts;
  * restoring the same question (same pyq_id, the undo-delete path) never logs a second row;
  * an Admin-named or null-named creation is never counted for the contractor;
  * a contractor save with no uploader_name at all (stores null) logs nothing for the contractor;
  * an edit (/admin/pyq-update), a needs_fix re-save, an approve (/admin/pyq-review) and a delete
    (/admin/pyq-delete) never change the uploaded totals;
  * a failing log insert, and a missing pyq_upload_log table, never stop the bulk-create save,
    and the counts come back 0 cleanly;
  * the counts endpoint returns the two new fields with every existing field unchanged;
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
UNIQUE_COLS = {"pyq_upload_log": "pyq_id"}


class FakeSupabase:
    """Minimal PostgREST stand-in: just enough eq/neq/or filtering, count=exact,
    return=representation and resolution=ignore-duplicates (via UNIQUE_COLS) for the endpoints
    under test, plus a way to simulate a table that doesn't exist yet (every request on it comes
    back 404, like a real missing relation)."""

    def __init__(self):
        self.tables = {"pyq": [], "pyq_upload_log": []}
        self.missing_tables = set()
        self.fail_next = {}

    @staticmethod
    def matches(row, field, cond):
        value = row.get(field)
        op, _, arg = cond.partition(".")
        if isinstance(value, bool):
            value_str = str(value).lower()
        else:
            value_str = None if value is None else str(value)
        if op == "eq":
            return value_str is not None and value_str == arg
        if op == "neq":
            return value_str is not None and value_str != arg
        if op == "in":
            return value_str is not None and value_str in arg.strip("()").split(",")
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
        ignore_dup = "resolution=ignore-duplicates" in request.headers.get("prefer", "")
        if method == "POST":
            out = []
            unique_col = UNIQUE_COLS.get(table)
            existing = {r.get(unique_col) for r in self.tables.setdefault(table, [])} if unique_col else set()
            for item in body if isinstance(body, list) else [body]:
                row = dict(item)
                row.setdefault("id", str(uuid.uuid4()))
                row.setdefault("created_at", datetime.now(timezone.utc).isoformat())
                if unique_col and ignore_dup and row.get(unique_col) in existing:
                    continue  # ON CONFLICT DO NOTHING
                self.tables[table].append(row)
                if unique_col:
                    existing.add(row.get(unique_col))
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


def question(text, **extra):
    base = {
        "subject": "Biology", "chapter": "1", "question": text,
        "option_a": "a", "option_b": "b", "option_c": "c", "option_d": "d",
    }
    base.update(extra)
    return server.PyqQuestionCreate(**base)


async def _noop_embed(*args, **kwargs):
    pass


async def main():
    fake = FakeSupabase()
    server.async_client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handle))
    server._embed_pyq_row_background = _noop_embed
    server.ADMIN_PASSWORD = ADMIN_KEY
    CONTRACTOR = server.CONTRACTOR_UPLOADER_NAME

    async def counts():
        return await server.admin_pyq_contractor_queue_counts(None)

    async def create(role, questions, uploader_name=None, **extra):
        body = server.PyqBulkCreate(questions=questions, uploader_name=uploader_name, **extra)
        r = await server.admin_pyq_bulk_create(body, role)
        await asyncio.sleep(0)  # let the fire-and-forget log-insert task post
        return r

    print("\n# A new contractor question without a diagram logs one row")
    r = await create("contractor", [question("Upload log: no diagram")], uploader_name=CONTRACTOR)
    check("create succeeds", "created" in r, r)
    row_no_diagram = r["created"][0]
    log = [l for l in fake.tables["pyq_upload_log"] if l.get("pyq_id") == row_no_diagram["id"]]
    check("exactly one log row", len(log) == 1, fake.tables["pyq_upload_log"])
    check("has_diagram is False", log and log[0].get("has_diagram") is False, log)

    print("\n# A new contractor question with a diagram logs one row, has_diagram True")
    r = await create("contractor", [question("Upload log: with diagram", diagram_url="https://x/d.png")], uploader_name=CONTRACTOR)
    row_with_diagram = r["created"][0]
    log = [l for l in fake.tables["pyq_upload_log"] if l.get("pyq_id") == row_with_diagram["id"]]
    check("exactly one log row", len(log) == 1, fake.tables["pyq_upload_log"])
    check("has_diagram is True", log and log[0].get("has_diagram") is True, log)

    c = await counts()
    check("counts endpoint reflects both uploads", c.get("uploaded_with_diagram") == 1 and c.get("uploaded_without_diagram") == 1, c)

    print("\n# Restoring the same question (same pyq_id) never logs a second row")
    r = await create("admin", [question("Upload log: no diagram", id=row_no_diagram["id"], reviewed=True)], uploader_name=CONTRACTOR)
    check("restore create succeeds", "created" in r, r)
    log = [l for l in fake.tables["pyq_upload_log"] if l.get("pyq_id") == row_no_diagram["id"]]
    check("still exactly one log row after the restore", len(log) == 1, fake.tables["pyq_upload_log"])
    c = await counts()
    check("restore doesn't change the counted total", c.get("uploaded_without_diagram") == 1, c)

    print("\n# Admin-named and null-named creations are never counted for the contractor")
    r = await create("admin", [question("Upload log: admin name")], uploader_name="Admin")
    admin_row = r["created"][0]
    r = await create("admin", [question("Upload log: null name")])
    null_row = r["created"][0]
    log_ids = {l.get("pyq_id") for l in fake.tables["pyq_upload_log"]}
    check("admin-named row never logged", admin_row["id"] not in log_ids, fake.tables["pyq_upload_log"])
    check("null-named row never logged", null_row["id"] not in log_ids, fake.tables["pyq_upload_log"])
    c = await counts()
    check("counts unaffected by admin/null creations", c.get("uploaded_with_diagram") == 1 and c.get("uploaded_without_diagram") == 1, c)

    print("\n# A contractor save with no uploader_name at all logs nothing for the contractor")
    r = await create("contractor", [question("Upload log: contractor no name")])
    no_name_row = r["created"][0]
    check("row stored with null uploader_name", no_name_row.get("uploader_name") is None, no_name_row)
    check("not logged", no_name_row["id"] not in {l.get("pyq_id") for l in fake.tables["pyq_upload_log"]}, fake.tables["pyq_upload_log"])

    print("\n# An edit, a needs_fix re-save, an approve and a delete never change the totals")
    before = await counts()
    r = await server.admin_pyq_update(row_with_diagram["id"], server.AdminPyqUpdate(question="edited text"), None)
    check("edit call succeeds", "error" not in r, r)
    r = await server.admin_pyq_review(server.PyqReviewRequest(ids=[row_no_diagram["id"]], status="needs_fix", note="fix it"), None)
    check("needs_fix review call succeeds", "error" not in r, r)
    r = await server.admin_pyq_review(server.PyqReviewRequest(ids=[row_with_diagram["id"]], status="approved"), None)
    check("approve review call succeeds", "error" not in r, r)
    r = await server.admin_pyq_delete(admin_row["id"], None)
    check("delete succeeds", r.get("success") is True or "error" not in r, r)
    after = await counts()
    check("uploaded totals unchanged by edit/needs_fix/approve/delete",
          after.get("uploaded_with_diagram") == before.get("uploaded_with_diagram")
          and after.get("uploaded_without_diagram") == before.get("uploaded_without_diagram"), (before, after))

    print("\n# A failing log insert never stops the bulk-create save")
    fake.fail_next[("POST", "pyq_upload_log")] = 500
    before_rows = len(fake.tables["pyq_upload_log"])
    r = await create("contractor", [question("Upload log: failing insert")], uploader_name=CONTRACTOR)
    check("save still succeeds when the log insert fails", "created" in r and "error" not in r, r)
    check("the failed insert left no new log row", len(fake.tables["pyq_upload_log"]) == before_rows, fake.tables["pyq_upload_log"])

    print("\n# A missing pyq_upload_log table never stops the save, and counts as 0")
    fake.missing_tables.add("pyq_upload_log")
    r = await create("contractor", [question("Upload log: missing table")], uploader_name=CONTRACTOR)
    check("save still succeeds when the table doesn't exist", "created" in r and "error" not in r, r)
    c = await counts()
    check("uploaded counts come back 0, not an error, when the table is missing",
          c.get("uploaded_with_diagram") == 0 and c.get("uploaded_without_diagram") == 0 and "error" not in c, c)
    fake.missing_tables.discard("pyq_upload_log")

    print("\n# Counts endpoint: new fields present, existing fields unchanged")
    c = await counts()
    for key in ("with_diagram", "without_diagram", "approved", "needs_fix", "deleted",
                "uploaded_with_diagram", "uploaded_without_diagram"):
        check(f"counts has '{key}'", key in c, c)

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
