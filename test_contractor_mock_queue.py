"""Regression test for the Mock exams option added to the contractor review queue
(contractor-pdf-review.html + /contractor/pyq-needs-fix, /contractor/pyq-rejected,
/admin/contractor-tally, /admin/pyq-contractor-queue-counts).

Run with: python test_contractor_mock_queue.py
Same offline fake-Supabase approach as test_pyq_upload_log.py / test_pyq_review_gate.py:
server.async_client is swapped for an httpx client backed by an in-memory PostgREST stand-in,
and endpoint coroutines are awaited directly. _embed_pyq_row_background is stubbed out (same
reasoning as test_pyq_upload_log.py) so a real bulk-create call stays fully offline.

Covers:
  * /contractor/pyq-needs-fix and /contractor/pyq-rejected default to source="pyq" and never
    return a mock row; source="mock" returns only mock rows, grouped/orderable by mock_name and
    mock_position, and never a PYQ row (requirement 4: a view never leaks the other source);
  * a mock row saved through /admin/pyq-bulk-create never appears in a source="pyq" duplicate
    corpus read -- the pre-existing pyq/mock isolation still holds after this change;
  * admin_contractor_tally and /admin/pyq-contractor-queue-counts never count a mock row toward
    the PYQ uploaded/approved/payable/target figures, and report mock_approved/mock_amount_usd
    (at PYQ_PRICE_USD_PER_Q per approved mock question) as their own separate numbers;
  * CONTRACTOR_PDF_KEY is still rejected on /admin/mock-exams-status and /admin/contractor-tally.
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
CONTRACTOR_KEY = "test-contractor-key"


def _parse_time(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


class FakeSupabase:
    """Minimal PostgREST stand-in -- same shape as test_pyq_upload_log.py's, plus ilike/not./
    gte/lt so /contractor/pyq-rejected's reject-breakdown filters (_pyq_reject_breakdown) work."""

    def __init__(self):
        self.tables = {"pyq": [], "pyq_upload_log": [], "pyq_delete_log": []}
        self.missing_tables = set()

    @staticmethod
    def matches(row, field, cond):
        value = row.get(field)
        negate = False
        if cond.startswith("not."):
            negate = True
            cond = cond[len("not."):]
        op, _, arg = cond.partition(".")
        if op == "is":
            result = (value is None) if arg == "null" else (value is not None)
        elif op in ("eq", "neq"):
            value_str = None if value is None else (str(value).lower() if isinstance(value, bool) else str(value))
            matched = value_str is not None and value_str == arg
            result = matched if op == "eq" else (value_str is not None and not matched)
        elif op == "in":
            value_str = None if value is None else str(value)
            result = value_str is not None and value_str in arg.strip("()").split(",")
        elif op == "ilike":
            value_str = "" if value is None else str(value)
            pattern = arg.lower()
            result = value_str.lower().startswith(pattern[:-1]) if pattern.endswith("*") else value_str.lower() == pattern
        elif op in ("gte", "lt"):
            a, b = _parse_time(value), _parse_time(arg)
            result = False if (a is None or b is None) else (a >= b if op == "gte" else a < b)
        else:
            raise AssertionError(f"fake PostgREST: unsupported filter {field}={cond}")
        return (not result) if negate else result

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
        params = list(request.url.params.multi_items())
        p = dict(params)
        if method == "GET":
            rows = self.filtered(table, params)
            total = len(rows)
            headers = {}
            if "count=exact" in request.headers.get("prefer", ""):
                headers["content-range"] = f"0-{max(total - 1, 0)}/{total}"
            if p.get("order"):
                col, _, direction = p["order"].partition(".")
                rows = sorted(rows, key=lambda r: (r.get(col) is None, str(r.get(col))), reverse=direction == "desc")
            offset = int(p.get("offset") or 0)
            rows = rows[offset:]
            if p.get("limit"):
                rows = rows[: int(p["limit"])]
            sel = p.get("select", "*")
            if sel != "*":
                cols = sel.split(",")
                rows = [{c: r.get(c) for c in cols} for r in rows]
            return httpx.Response(200, json=copy.deepcopy(rows), headers=headers)
        body = json.loads(request.content or b"null")
        rep = "return=representation" in request.headers.get("prefer", "")
        ignore_dup = "resolution=ignore-duplicates" in request.headers.get("prefer", "")
        if method == "POST":
            out = []
            unique_col = "pyq_id" if table == "pyq_upload_log" else None
            existing = {r.get(unique_col) for r in self.tables.setdefault(table, [])} if unique_col else set()
            for item in body if isinstance(body, list) else [body]:
                row = dict(item)
                row.setdefault("id", str(uuid.uuid4()))
                row.setdefault("created_at", datetime.now(timezone.utc).isoformat())
                if table == "pyq":
                    row.setdefault("source", "pyq")  # DB column default (mock_exam_columns.sql)
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
        raise AssertionError(f"fake PostgREST: unsupported {method}")


def pyq_row(status="approved", *, source="pyq", mock_name=None, mock_position=None,
            uploader_name=None, question="Q", reject_reason=None, **extra):
    now = datetime.now(timezone.utc).isoformat()
    row = {
        "id": str(uuid.uuid4()), "subject": "Physics", "chapter": "1", "class": 12, "year": 2020,
        "question": question, "option_a": "a", "option_b": "b", "option_c": "c", "option_d": "d",
        "correct_answer": "a", "has_diagram": False, "diagram_url": None,
        "option_a_diagram_url": None, "option_b_diagram_url": None, "option_c_diagram_url": None,
        "option_d_diagram_url": None, "is_active": True, "review_status": status,
        "review_note": "Fix the wording" if status == "needs_fix" else None,
        "uploaded_by": "contractor", "uploaded_at": now, "batch_id": None,
        "reviewed_at": now, "reviewed_by": "Admin", "uploader_name": uploader_name,
        "reject_reason": reject_reason, "rejected_at": now if status == "rejected" else None,
        "created_at": now, "source": source, "mock_name": mock_name, "mock_position": mock_position,
        "source_pdf_filename": None,
    }
    row.update(extra)
    return row


def question(text, **extra):
    base = {
        "subject": "Physics", "chapter": "1", "question": text,
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
    server.CONTRACTOR_PDF_KEY = CONTRACTOR_KEY
    CONTRACTOR = server.CONTRACTOR_UPLOADER_NAME

    async def create(role, questions, uploader_name=None):
        body = server.PyqBulkCreate(questions=questions, uploader_name=uploader_name)
        r = await server.admin_pyq_bulk_create(body, role)
        await asyncio.sleep(0)  # let fire-and-forget tasks (upload-log, auto-check) post
        return r

    # ---- /contractor/pyq-needs-fix: source filter, mock_name/mock_position carried through ----
    fake.tables["pyq"] = [
        pyq_row("needs_fix", source="pyq", uploader_name=CONTRACTOR, question="pyq needs fix"),
        pyq_row("needs_fix", source="mock", mock_name="Mock Test 1", mock_position=5,
                uploader_name=CONTRACTOR, question="mock needs fix"),
    ]
    data_pyq = await server.contractor_pyq_needs_fix(_=None)
    check("default source (pyq) returns only the pyq row",
          [r["question"] for r in data_pyq["rows"]] == ["pyq needs fix"], data_pyq["rows"])
    data_mock = await server.contractor_pyq_needs_fix(source="mock", _=None)
    check("source=mock returns only the mock row",
          [r["question"] for r in data_mock["rows"]] == ["mock needs fix"], data_mock["rows"])
    check("mock row carries mock_name/mock_position",
          data_mock["rows"][0].get("mock_name") == "Mock Test 1" and data_mock["rows"][0].get("mock_position") == 5,
          data_mock["rows"])
    check("an unrecognized source value falls back to pyq (never widens access)",
          [r["question"] for r in (await server.contractor_pyq_needs_fix(source="bogus", _=None))["rows"]] == ["pyq needs fix"])

    # ---- /contractor/pyq-rejected: source filter, mock pay reported separately ----
    fake.tables["pyq"] = [
        pyq_row("rejected", source="pyq", uploader_name=CONTRACTOR, question="pyq rejected", reject_reason="Wrong answer"),
        pyq_row("rejected", source="mock", mock_name="Mock Test 1", mock_position=9,
                uploader_name=CONTRACTOR, question="mock rejected", reject_reason="Wrong answer"),
    ]
    fake.tables["pyq_upload_log"] = []
    data_pyq = await server.contractor_pyq_rejected(_=None)
    check("default source (pyq) returns only the pyq rejected row",
          [r["question"] for r in data_pyq["rows"]] == ["pyq rejected"], data_pyq["rows"])
    data_mock = await server.contractor_pyq_rejected(source="mock", _=None)
    check("source=mock returns only the mock rejected row",
          [r["question"] for r in data_mock["rows"]] == ["mock rejected"], data_mock["rows"])
    check("PYQ rejection is deducted from the PYQ errors count",
          data_pyq["rejected_errors"] == 1, data_pyq)
    check("pay response always carries both a PYQ target and a separate mock figure",
          data_pyq["target"] == server.PYQ_PAYOUT_TARGET and "mock_approved" in data_pyq, data_pyq)

    # Isolated case: ONLY a mock rejection exists (no co-existing PYQ rejection) -- it must not
    # deduct from the PYQ errors count at all.
    fake.tables["pyq"] = [
        pyq_row("rejected", source="mock", mock_name="Mock Test 1", mock_position=9,
                uploader_name=CONTRACTOR, question="mock rejected only", reject_reason="Wrong answer"),
    ]
    data_mock_only = await server.contractor_pyq_rejected(source="mock", _=None)
    check("a mock-only rejection never deducts from the PYQ errors count",
          data_mock_only["rejected_errors"] == 0, data_mock_only)

    # ---- A mock row saved via /admin/pyq-bulk-create never enters a PYQ-scoped read ----
    server._pyq_dup_corpus_cache.clear()
    fake.tables["pyq"] = []
    r = await create("contractor",
                      [question("Isolation check mock question", source="mock", mock_name="Mock Test 2", mock_position=3)],
                      uploader_name=CONTRACTOR)
    check("bulk-create succeeds", "created" in r, r)
    created_row = r["created"][0]
    check("mock row saved with source='mock'", created_row.get("source") == "mock", created_row)
    check("a contractor mock row still starts pending (never auto-approved)",
          created_row.get("review_status") == "pending", created_row)
    pyq_scoped_corpus = await server._pyq_dup_corpus("Physics", source="pyq")
    check("the new mock row never appears in a source='pyq' duplicate corpus read",
          all(row_id != created_row["id"] for row_id, _tokens, _row in (pyq_scoped_corpus or [])),
          pyq_scoped_corpus)
    mock_scoped_corpus = await server._pyq_dup_corpus("Physics", source="mock")
    check("...but it does appear in the source='mock' corpus",
          any(row_id == created_row["id"] for row_id, _tokens, _row in (mock_scoped_corpus or [])),
          mock_scoped_corpus)

    # An admin-role save of a mock row is also forced back to pending (final-say override).
    r = await create("admin",
                      [question("Admin mock question", source="mock", mock_name="Mock Test 2",
                                mock_position=4, review_status="approved")])
    admin_mock_row = r["created"][0]
    check("even an admin-key mock save never comes back pre-approved",
          admin_mock_row.get("review_status") == "pending", admin_mock_row)

    # ---- admin_contractor_tally: mock is never merged into the PYQ uploaded/payable/target ----
    fake.tables["pyq_upload_log"] = [
        {"id": str(uuid.uuid4()), "pyq_id": str(uuid.uuid4()), "uploader_name": CONTRACTOR,
         "has_diagram": False, "created_at": datetime.now(timezone.utc).isoformat()}
        for _ in range(5)
    ]
    fake.tables["pyq"] = [
        pyq_row("approved", source="pyq", uploader_name=CONTRACTOR, question="p1"),
        pyq_row("approved", source="pyq", uploader_name=CONTRACTOR, question="p2"),
        pyq_row("pending", source="pyq", uploader_name=CONTRACTOR, question="p3"),
        pyq_row("approved", source="mock", mock_name="Mock Test 1", mock_position=1, uploader_name=CONTRACTOR, question="m1"),
        pyq_row("pending", source="mock", mock_name="Mock Test 1", mock_position=2, uploader_name=CONTRACTOR, question="m2"),
    ]
    data = await server.admin_contractor_tally(_=None)
    row = next(r for r in data["rows"] if r["uploader_name"] == CONTRACTOR)
    check("uploaded excludes the 2 live mock rows from the 5 logged uploads (5-2=3)", row["uploaded"] == 3, row)
    check("approved counts only the pyq-source approved rows", row["approved"] == 2, row)
    check("payable is PYQ-only and unaffected by the mock rows", row["payable"] == 3, row)
    check("target stays the 5,000 PYQ constant, never adjusted for mock", row["target"] == server.PYQ_PAYOUT_TARGET, row)
    check("mock_approved counts only the approved mock row", row["mock_approved"] == 1, row)
    check("mock_amount_usd is PYQ_PRICE_USD_PER_Q per approved mock question",
          row["mock_amount_usd"] == round(1 * server.PYQ_PRICE_USD_PER_Q, 2), row)

    # ---- /admin/pyq-contractor-queue-counts: same mock exclusion, same separate field ----
    counts = await server.admin_pyq_contractor_queue_counts(_=None)
    check("approved excludes mock here too", counts["approved"] == 2, counts)
    check("mock_approved reported separately on this panel too", counts["mock_approved"] == 1, counts)

    # ---- Security: CONTRACTOR_PDF_KEY still rejected on admin-only endpoints ----
    client = TestClient(server.app)
    resp = client.get("/admin/mock-exams-status", headers={"X-Admin-Key": CONTRACTOR_KEY})
    check("contractor key rejected on /admin/mock-exams-status", resp.status_code == 401, resp.status_code)
    resp = client.get("/admin/contractor-tally", headers={"X-Admin-Key": CONTRACTOR_KEY})
    check("contractor key rejected on /admin/contractor-tally", resp.status_code == 401, resp.status_code)
    resp = client.get("/admin/pyq-contractor-queue-counts", headers={"X-Admin-Key": CONTRACTOR_KEY})
    check("contractor key rejected on /admin/pyq-contractor-queue-counts", resp.status_code == 401, resp.status_code)
    # The checks above deliberately fail auth a few times from this same test IP -- clear the
    # brute-force lockout counter (shared across every verify_* function by IP) before the next
    # check, which is a legitimate request and shouldn't be caught by it.
    server._admin_login_attempts.clear()
    resp = client.get("/contractor/pyq-needs-fix?source=mock", headers={"X-Admin-Key": CONTRACTOR_KEY})
    check("contractor key still works on its own needs-fix route", resp.status_code == 200, resp.status_code)

    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    if _failed:
        print("FAILED:")
        for label in _failed:
            print(f"  - {label}")
    return len(_failed) == 0


if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
