"""Regression test for "Add question above/below" on the mock question review (mock-question-add.js
+ admin-pyq-preview.html + contractor-pdf-review.html's Needs fix queue), backed by
GET /admin/mock-question-positions and POST /admin/mock-question-add + /contractor/mock-question-add.

Run with: python test_mock_question_add.py
Same offline fake-Supabase approach as test_contractor_mock_queue.py / test_mock_exams.py:
server.async_client is swapped for an httpx client backed by an in-memory PostgREST stand-in, and
endpoint coroutines are awaited directly (no real network, no model call either -- this feature
never makes one).

Covers:
  * adding at a free position works, for both the admin and the contractor endpoint;
  * an already-used position, and anything outside 1-180, is rejected with a plain-string error;
  * subject is always derived from position server-side (1-45 Physics, 46-90 Chemistry,
    91-180 Biology), never taken from the request;
  * the created row is source='mock', review_status='pending' -- never auto-approved, regardless
    of who added it;
  * a contractor-added row only counts toward mock pay (admin_contractor_tally's
    mock_approved/mock_amount_usd) once an admin approves it via /admin/pyq-review;
  * CONTRACTOR_PDF_KEY is rejected on the admin add endpoint (and every other admin-only mock
    endpoint), while still working on the two endpoints actually meant for it.
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


class FakeSupabase:
    """Minimal PostgREST stand-in -- same shape as test_contractor_mock_queue.py's."""

    def __init__(self):
        self.tables = {"pyq": [], "pyq_upload_log": [], "pyq_delete_log": []}

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
                    return any(self.matches(r, *part.partition(".")[::2]) for part in parts)
                rows = [r for r in rows if any_match(r)]
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
        if method == "POST":
            out = []
            for item in body if isinstance(body, list) else [body]:
                row = dict(item)
                row.setdefault("id", str(uuid.uuid4()))
                row.setdefault("created_at", datetime.now(timezone.utc).isoformat())
                if table == "pyq":
                    row.setdefault("source", "pyq")  # DB column default (mock_exam_columns.sql)
                self.tables.setdefault(table, []).append(row)
                out.append(row)
            return httpx.Response(201, json=copy.deepcopy(out) if rep else None)
        if method == "PATCH":
            rows = self.filtered(table, params)
            for r in rows:
                r.update(body)
            return httpx.Response(200, json=copy.deepcopy(rows) if rep else None)
        raise AssertionError(f"fake PostgREST: unsupported {method}")


def pyq_row(status="approved", *, source="mock", mock_name="Mock Test 1", mock_position=1,
            uploader_name=None, question="Q", **extra):
    now = datetime.now(timezone.utc).isoformat()
    row = {
        "id": str(uuid.uuid4()), "subject": "Physics", "chapter": "1", "class": 12, "year": 2020,
        "question": question, "option_a": "a", "option_b": "b", "option_c": "c", "option_d": "d",
        "correct_answer": "a", "has_diagram": False, "diagram_url": None, "is_active": True,
        "review_status": status, "uploaded_by": "contractor", "uploaded_at": now,
        "uploader_name": uploader_name, "created_at": now,
        "source": source, "mock_name": mock_name, "mock_position": mock_position,
        "source_pdf_filename": None,
    }
    row.update(extra)
    return row


def add_request(**over):
    base = dict(
        mock_name="Mock Test 1", mock_position=50, question="New Q?",
        option_a="A", option_b="B", option_c="C", option_d="D", correct_answer="B",
    )
    base.update(over)
    return server.MockQuestionAddRequest(**base)


async def _noop_embed(*args, **kwargs):
    pass


async def main():
    fake = FakeSupabase()
    server.async_client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handle))
    server._embed_pyq_row_background = _noop_embed
    server.ADMIN_PASSWORD = ADMIN_KEY
    server.CONTRACTOR_PDF_KEY = CONTRACTOR_KEY
    CONTRACTOR = server.CONTRACTOR_UPLOADER_NAME

    # ---- free position: admin ----
    fake.tables["pyq"] = [pyq_row(mock_position=p) for p in range(1, 10)]  # 1-9 taken
    r = await server.admin_mock_question_add(add_request(mock_position=10), _=None)
    check("admin add at a free position succeeds", "created" in r, r)
    created = r["created"]
    check("row saved with source='mock'", created.get("source") == "mock", created)
    check("row starts pending (never auto-approved)", created.get("review_status") == "pending", created)
    check("row attributed to admin", created.get("uploaded_by") == "admin", created)
    check("admin add never sets uploader_name on its own", created.get("uploader_name") is None, created)

    # ---- free position: contractor ----
    r = await server.contractor_mock_question_add(add_request(mock_position=11), _=None)
    check("contractor add at a free position succeeds", "created" in r, r)
    created_c = r["created"]
    check("row attributed to contractor", created_c.get("uploaded_by") == "contractor", created_c)
    check("row defaults uploader_name to the real contractor's name", created_c.get("uploader_name") == CONTRACTOR, created_c)
    check("contractor-added row is also pending, never auto-approved", created_c.get("review_status") == "pending", created_c)

    # ---- occupied position: rejected for both roles, plain-string error ----
    r = await server.admin_mock_question_add(add_request(mock_position=5), _=None)
    check("admin: occupied position rejected", "error" in r and isinstance(r["error"], str) and r["error"], r)
    r = await server.contractor_mock_question_add(add_request(mock_position=5), _=None)
    check("contractor: occupied position rejected too", "error" in r and isinstance(r["error"], str) and r["error"], r)

    # ---- out-of-range position: rejected (0 and 181) ----
    r = await server.admin_mock_question_add(add_request(mock_position=0), _=None)
    check("position 0 rejected", "error" in r and isinstance(r["error"], str), r)
    r = await server.admin_mock_question_add(add_request(mock_position=181), _=None)
    check("position 181 rejected", "error" in r and isinstance(r["error"], str), r)

    # ---- a rejected row's position is free again for a replacement ----
    fake.tables["pyq"] = [pyq_row(mock_position=20, status="rejected")]
    r = await server.admin_mock_question_add(add_request(mock_position=20), _=None)
    check("a rejected row's old position can take a replacement", "created" in r, r)

    # ---- subject always derived from position, server-side ----
    fake.tables["pyq"] = []
    phy = (await server.admin_mock_question_add(add_request(mock_position=1), _=None))["created"]
    che = (await server.admin_mock_question_add(add_request(mock_position=46), _=None))["created"]
    bio = (await server.admin_mock_question_add(add_request(mock_position=91), _=None))["created"]
    check("position 1 -> Physics", phy["subject"] == "Physics", phy)
    check("position 46 -> Chemistry", che["subject"] == "Chemistry", che)
    check("position 91 -> Biology", bio["subject"] == "Biology", bio)
    check("position 45/46 boundary and 90/91 boundary both land correctly",
          (await server.admin_mock_question_add(add_request(mock_position=45), _=None))["created"]["subject"] == "Physics"
          and (await server.admin_mock_question_add(add_request(mock_position=90), _=None))["created"]["subject"] == "Chemistry"
          and (await server.admin_mock_question_add(add_request(mock_position=180), _=None))["created"]["subject"] == "Biology")

    # ---- validation: 4 distinct non-empty options, answer among them ----
    fake.tables["pyq"] = []
    r = await server.admin_mock_question_add(add_request(mock_position=1, option_d=""), _=None)
    check("an empty option is rejected", "error" in r, r)
    r = await server.admin_mock_question_add(add_request(mock_position=1, option_b="A"), _=None)
    check("duplicate option text is rejected (must be 4 distinct options)", "error" in r, r)
    r = await server.admin_mock_question_add(add_request(mock_position=1, correct_answer="E"), _=None)
    check("an answer outside A-D is rejected", "error" in r, r)
    r = await server.admin_mock_question_add(add_request(mock_position=1, question="  "), _=None)
    check("blank question text is rejected", "error" in r, r)

    # ---- GET /admin/mock-question-positions: used/free, rejected rows excluded ----
    fake.tables["pyq"] = [pyq_row(mock_position=p) for p in (1, 2, 3)] + [pyq_row(mock_position=4, status="rejected")]
    pos = await server.admin_mock_question_positions(mock_name="Mock Test 1", _=None)
    check("positions_used holds only live (non-rejected) rows", pos["positions_used"] == [1, 2, 3], pos)
    check("position 4 (rejected) counts as free", 4 in pos["positions_free"], pos)
    check("position 180 (never used) is free", 180 in pos["positions_free"], pos)

    # ---- contractor-added row counts for pay only after admin approval ----
    fake.tables["pyq"] = []
    r = await server.contractor_mock_question_add(add_request(mock_name="Mock Test 9", mock_position=1), _=None)
    new_id = r["created"]["id"]
    tally_before = await server.admin_contractor_tally(_=None)
    row_before = next(x for x in tally_before["rows"] if x["uploader_name"] == CONTRACTOR)
    check("before approval: mock_approved is 0, not counted toward pay", row_before["mock_approved"] == 0, row_before)
    check("before approval: mock_amount_usd is $0.00", row_before["mock_amount_usd"] == 0.0, row_before)

    await server.admin_pyq_review(server.PyqReviewRequest(ids=[new_id], status="approved"), _=None)
    tally_after = await server.admin_contractor_tally(_=None)
    row_after = next(x for x in tally_after["rows"] if x["uploader_name"] == CONTRACTOR)
    check("after admin approval: mock_approved becomes 1", row_after["mock_approved"] == 1, row_after)
    check("after admin approval: mock_amount_usd is one question at PYQ_PRICE_USD_PER_Q",
          row_after["mock_amount_usd"] == round(1 * server.PYQ_PRICE_USD_PER_Q, 2), row_after)

    # ---- Security: CONTRACTOR_PDF_KEY rejected on the admin add endpoint and every other
    # admin-only mock endpoint, but still works on the two contractor-facing routes ----
    client = TestClient(server.app)
    add_body = {"mock_name": "Mock Test 1", "mock_position": 99, "question": "Q", "option_a": "a",
                "option_b": "b", "option_c": "c", "option_d": "d", "correct_answer": "A"}
    resp = client.post("/admin/mock-question-add", json=add_body, headers={"X-Admin-Key": CONTRACTOR_KEY})
    check("contractor key rejected on /admin/mock-question-add", resp.status_code == 401, resp.status_code)
    resp = client.get("/admin/mock-exams-status", headers={"X-Admin-Key": CONTRACTOR_KEY})
    check("contractor key still rejected on /admin/mock-exams-status", resp.status_code == 401, resp.status_code)
    server._admin_login_attempts.clear()  # the two 401s above shouldn't trip the brute-force lockout for what follows

    resp = client.post("/contractor/mock-question-add", json=add_body, headers={"X-Admin-Key": CONTRACTOR_KEY})
    check("contractor key works on its own add endpoint", resp.status_code == 200 and "error" not in resp.json(), resp.text[:200])
    resp = client.post("/admin/mock-question-add", json={**add_body, "mock_position": 98}, headers={"X-Admin-Key": ADMIN_KEY})
    check("admin key still works on the admin add endpoint", resp.status_code == 200 and "error" not in resp.json(), resp.text[:200])
    resp = client.get(f"/admin/mock-question-positions?mock_name=Mock Test 1", headers={"X-Admin-Key": CONTRACTOR_KEY})
    check("contractor key works on the shared read-only positions endpoint", resp.status_code == 200, resp.status_code)

    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    if _failed:
        print("FAILED:")
        for label in _failed:
            print(f"  - {label}")
    return len(_failed) == 0


if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
