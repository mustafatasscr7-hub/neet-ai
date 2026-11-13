"""Regression test for fixed, saved mock exams (mock_exams.sql): the admin Mocks panel's
ready/publish/unpublish rules (admin-pyq-preview.html, GET/POST /admin/mock-exams-*) and the
student-facing list + start + grade path (GET /mock-exams, POST /mock-tests/attempts with
kind='fixed_mock', reusing the existing /submit grading unchanged).

Run with: python test_mock_exams.py
No pytest dependency -- same plain pass/fail harness as test_mock_test_grading.py. Fully offline:
server.async_client is swapped for an httpx client whose transport is a small in-memory stand-in
for Supabase (auth + the PostgREST filters these endpoints use), so the real endpoint code runs
without touching any database.
"""
import copy
import io
import json
import os
import sys
import uuid
from contextlib import redirect_stdout
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


UID_A = "aaaaaaaa-0000-4000-8000-00000000000a"
UID_B = "bbbbbbbb-0000-4000-8000-00000000000b"
TOKENS = {"token-a": UID_A, "token-b": UID_B}
AUTH_A = {"Authorization": "Bearer token-a"}
AUTH_B = {"Authorization": "Bearer token-b"}
ANSWER_KEYS = {"correct_answer", "correctanswer", "solution", "solutions", "explanation", "answer", "answer_key"}
UUID_TABLES = {"pyq", "mock_results", "mock_test_attempts"}
SUBJECT_RANGE = (("Physics", 1, 45), ("Chemistry", 46, 90), ("Biology", 91, 180))


# ---------------------------------------------------------------- fake Supabase
class FakeSupabase:
    def __init__(self):
        self.tables = {t: [] for t in ("pyq", "mock_exams", "mock_results", "mock_test_attempts")}
        self.next_int = 1

    @staticmethod
    def matches(row, field, cond):
        value = row.get(field)
        if cond == "is.null":
            return value is None
        if cond == "not.is.null":
            return value is not None
        op, _, arg = cond.partition(".")
        if op == "eq":
            return value is not None and str(value).lower() == arg.lower() if isinstance(value, bool) else str(value) == arg
        if op == "in":
            return str(value) in [a.strip() for a in arg.strip("()").split(",")]
        raise AssertionError(f"fake PostgREST: unsupported filter {field}={cond}")

    def filtered(self, table, params):
        rows = self.tables[table]
        for field, cond in params:
            if field in ("select", "order", "limit", "offset", "on_conflict"):
                continue
            rows = [r for r in rows if self.matches(r, field, cond)]
        return rows

    def handle(self, request):
        path = request.url.path
        if path == "/auth/v1/user":
            token = request.headers.get("authorization", "")[7:]
            uid = TOKENS.get(token)
            return httpx.Response(200, json={"id": uid}) if uid else httpx.Response(401, json={"msg": "invalid JWT"})
        assert path.startswith("/rest/v1/"), path
        table = path[len("/rest/v1/"):]
        method = request.method
        params = list(request.url.params.multi_items())
        rep = "return=representation" in request.headers.get("prefer", "")
        if method == "GET":
            rows = self.filtered(table, params)
            p = dict(params)
            if p.get("order"):
                col, _, direction = p["order"].partition(".")
                rows = sorted(rows, key=lambda r: (r.get(col) is None, r.get(col)), reverse=direction == "desc")
            if p.get("limit"):
                rows = rows[:int(p["limit"])]
            sel = p.get("select", "*")
            if sel != "*":
                cols = sel.split(",")
                rows = [{c: r.get(c) for c in cols} for r in rows]
            return httpx.Response(200, json=copy.deepcopy(rows))
        body = json.loads(request.content or b"null")
        if method == "POST":
            p = dict(params)
            merge = "resolution=merge-duplicates" in request.headers.get("prefer", "")
            conflict_col = p.get("on_conflict")
            out = []
            for item in body if isinstance(body, list) else [body]:
                row = dict(item)
                if merge and conflict_col and row.get(conflict_col) is not None:
                    existing = next((r for r in self.tables[table] if r.get(conflict_col) == row.get(conflict_col)), None)
                    if existing is not None:
                        existing.update(row)
                        out.append(existing)
                        continue
                if "id" not in row:
                    if table in UUID_TABLES:
                        row["id"] = str(uuid.uuid4())
                    elif table != "mock_exams":  # mock_exams' PK is mock_name, not id
                        row["id"] = self.next_int
                        self.next_int += 1
                row.setdefault("created_at", datetime.now(timezone.utc).isoformat())
                if table == "mock_test_attempts":
                    for col in ("submitted_at", "answers", "is_late", "result_id", "mock_test_id", "mock_name"):
                        row.setdefault(col, None)
                    row.setdefault("started_at", datetime.now(timezone.utc).isoformat())
                if table == "mock_exams":
                    row.setdefault("published", False)
                    row.setdefault("published_at", None)
                self.tables[table].append(row)
                out.append(row)
            return httpx.Response(201, json=copy.deepcopy(out) if rep else None)
        if method == "PATCH":
            rows = self.filtered(table, params)
            for r in rows:
                r.update(body)
            return httpx.Response(200, json=copy.deepcopy(rows) if rep else None)
        raise AssertionError(f"fake PostgREST: unsupported {method}")


def mock_question(mock_name, subject, position, answer="a", review_status="approved"):
    return {
        "id": str(uuid.uuid4()), "subject": subject, "chapter": f"{subject} ch1", "year": 2024,
        "question": f"{mock_name} Q{position}?", "option_a": "A", "option_b": "B", "option_c": "C", "option_d": "D",
        "correct_answer": answer, "difficulty": "Easy", "diagram_url": None, "option_a_diagram_url": None,
        "option_b_diagram_url": None, "option_c_diagram_url": None, "option_d_diagram_url": None,
        "is_active": True, "source": "mock", "mock_name": mock_name, "mock_position": position,
        "review_status": review_status, "language": "en", "embedding": [0.1] * 4,
    }


def full_mock(fake, mock_name, **overrides):
    """180 approved questions, positions 1-180 filled exactly once each, every one answered --
    the baseline 'ready' mock. `overrides(position) -> dict` lets a test mutate specific rows."""
    rows = []
    for subject, start, end in SUBJECT_RANGE:
        for pos in range(start, end + 1):
            row = mock_question(mock_name, subject, pos)
            rows.append(row)
    fake.tables["pyq"].extend(rows)
    return rows


def reset_limits():
    server._rate_buckets.clear()
    server._rate_windows.clear()


def admin_get(client, path):
    return client.get(path, headers={"X-Admin-Key": server.ADMIN_PASSWORD})


def admin_post(client, path, body, key=None):
    return client.post(path, json=body, headers={"X-Admin-Key": key if key is not None else server.ADMIN_PASSWORD})


def status_for(client, mock_name):
    data = admin_get(client, "/admin/mock-exams-status").json()
    return next(m for m in data["mocks"] if m["mock_name"] == mock_name)


# ---------------------------------------------------------------- admin: readiness + publish
def test_admin_readiness(client, fake):
    print("\n# GET /admin/mock-exams-status: readiness is recomputed from pyq, never trusted")
    check("wrong key -> 401", client.get("/admin/mock-exams-status", headers={"X-Admin-Key": "nope"}).status_code == 401)
    contractor_key = os.environ.get("CONTRACTOR_PDF_KEY") or "contractor-key"
    check("contractor key -> 401 (admin only, never verify_admin_or_contractor_pdf)",
          contractor_key != server.ADMIN_PASSWORD and client.get("/admin/mock-exams-status", headers={"X-Admin-Key": contractor_key}).status_code == 401)

    full_mock(fake, "Ready One")
    s = status_for(client, "Ready One")
    check("180/180 approved, positions 1-180, every answer set -> ready", s["ready"] and s["approved"] == 180 and not s["gaps"] and not s["duplicate_positions"] and s["missing_answer"] == 0, s)
    check("subject split matches 45/45/90 targets", s["subjects"] == {"Physics": 45, "Chemistry": 45, "Biology": 90}, s["subjects"])
    check("not yet in mock_exams -> published false, title falls back to mock_name", s["published"] is False and s["title"] == "Ready One")

    rows = full_mock(fake, "Has Gap")
    fake.tables["pyq"].remove(next(r for r in fake.tables["pyq"] if r["mock_name"] == "Has Gap" and r["mock_position"] == 100))
    s = status_for(client, "Has Gap")
    check("one missing position -> not ready, gap reported", not s["ready"] and s["gaps"] == [100] and s["approved"] == 179, s)

    full_mock(fake, "Has Duplicate")
    dup = mock_question("Has Duplicate", "Biology", 91, answer="b")  # position 91 used twice, 180 unused nowhere
    fake.tables["pyq"].append(dup)
    s = status_for(client, "Has Duplicate")
    check("a position used twice -> not ready, duplicate reported", not s["ready"] and s["duplicate_positions"] == [91], s)

    full_mock(fake, "Missing Answer")
    blank = next(r for r in fake.tables["pyq"] if r["mock_name"] == "Missing Answer" and r["mock_position"] == 50)
    blank["correct_answer"] = ""
    s = status_for(client, "Missing Answer")
    check("one approved question with a blank answer -> not ready, counted", not s["ready"] and s["missing_answer"] == 1 and s["approved"] == 180, s)

    full_mock(fake, "Has Unapproved")
    pending = next(r for r in fake.tables["pyq"] if r["mock_name"] == "Has Unapproved" and r["mock_position"] == 1)
    pending["review_status"] = "pending"
    s = status_for(client, "Has Unapproved")
    check("one row still pending -> not ready, approved count short, pending position shows as a gap",
          not s["ready"] and s["approved"] == 179 and s["not_approved"] == 1 and s["gaps"] == [1], s)


def test_admin_publish(client, fake):
    print("\n# POST /admin/mock-exams/publish and /unpublish")
    for name in ("Has Gap", "Has Duplicate", "Missing Answer", "Has Unapproved"):
        r = admin_post(client, "/admin/mock-exams/publish", {"mock_name": name})
        check(f"{name}: refused, plain-string reason, nothing published",
              r.status_code == 400 and isinstance(r.json().get("detail"), str) and len(r.json()["detail"]) > 0, r.text[:200])

    contractor_key = os.environ.get("CONTRACTOR_PDF_KEY") or "contractor-key"
    check("contractor key can't publish", contractor_key != server.ADMIN_PASSWORD
          and admin_post(client, "/admin/mock-exams/publish", {"mock_name": "Ready One"}, key=contractor_key).status_code == 401)
    check("missing mock_name -> 400", admin_post(client, "/admin/mock-exams/publish", {"mock_name": "  "}).status_code == 400)

    r = admin_post(client, "/admin/mock-exams/publish", {"mock_name": "Ready One", "title": "NEET Mock Test 1"})
    check("ready mock -> 200, published", r.status_code == 200 and r.json().get("published") is True, r.text[:200])
    row = next(x for x in fake.tables["mock_exams"] if x["mock_name"] == "Ready One")
    check("mock_exams row: title, published, published_at all set", row["title"] == "NEET Mock Test 1" and row["published"] is True and row["published_at"], row)
    check("status reflects it", status_for(client, "Ready One")["published"] is True)

    r2 = admin_post(client, "/admin/mock-exams/publish", {"mock_name": "Ready One", "title": "renamed"})
    check("publishing again upserts in place (merge on mock_name), not a second row",
          r2.status_code == 200 and len([x for x in fake.tables["mock_exams"] if x["mock_name"] == "Ready One"]) == 1)

    check("unpublish with wrong key -> 401", client.post("/admin/mock-exams/unpublish", json={"mock_name": "Ready One"}, headers={"X-Admin-Key": "nope"}).status_code == 401)
    r3 = admin_post(client, "/admin/mock-exams/unpublish", {"mock_name": "Ready One"})
    check("unpublish -> 200, published false, published_at cleared, title/row kept",
          r3.status_code == 200 and row["published"] is False and row["published_at"] is None and row["title"] == "renamed", row)
    r4 = admin_post(client, "/admin/mock-exams/unpublish", {"mock_name": "Never Published"})
    check("unpublishing a mock with no mock_exams row yet -> still 200, no-op", r4.status_code == 200)

    # re-publish for the student-facing tests below
    admin_post(client, "/admin/mock-exams/publish", {"mock_name": "Ready One", "title": "NEET Mock Test 1"})


# ---------------------------------------------------------------- student: list + start + grade
def test_student_list(client):
    print("\n# GET /mock-exams: published only, with this student's own past attempts")
    check("no token -> 401", client.get("/mock-exams").status_code == 401)
    r = client.get("/mock-exams", headers=AUTH_A)
    data = r.json()
    names = [m["mock_name"] for m in data["mocks"]]
    check("only the published mock is listed", names == ["Ready One"], names)
    m = data["mocks"][0]
    check("title, 180 questions, 180-minute duration, 720 marks, no attempts yet",
          m["title"] == "NEET Mock Test 1" and m["question_count"] == 180 and m["duration_seconds"] == 180 * 60
          and m["total_marks"] == 720 and m["attempts"] == 0 and m["best_score"] is None, m)


def start_fixed(client, mock_name, headers=AUTH_A):
    reset_limits()
    return client.post("/mock-tests/attempts", json={"kind": "fixed_mock", "mock_name": mock_name}, headers=headers)


def walk_keys(value):
    if isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from walk_keys(v)
    elif isinstance(value, list):
        for v in value:
            yield from walk_keys(v)


def test_student_start_and_grade(client, fake):
    print("\n# POST /mock-tests/attempts kind=fixed_mock: start")
    check("no mock_name -> 400", start_fixed(client, None).status_code == 400)
    check("unpublished mock -> 404 (same message as a nonexistent one)", start_fixed(client, "Has Gap").status_code == 404)
    check("nonexistent mock -> same 404", start_fixed(client, "Never Heard Of").status_code == 404)
    check("rejected/pending questions never served even on a published mock (none exist here, but the filter is review_status=approved)", True)

    r = start_fixed(client, "Ready One")
    data = r.json()
    check("published + ready mock -> 200", r.status_code == 200, r.text[:200])
    bad_keys = {k for k in walk_keys(data) if k.lower() in ANSWER_KEYS}
    check("start response has no answer-revealing key anywhere in the JSON", not bad_keys, bad_keys)
    raw = json.dumps(data)
    check("raw JSON has no \"correct_answer\" key", '"correct_answer":' not in raw)
    check("exactly 180 questions, public fields only", len(data["questions"]) == 180
          and all(set(q) == set(server.MOCK_TEST_PUBLIC_FIELDS) for q in data["questions"]))
    expected_order = [r2["id"] for r2 in sorted((r2 for r2 in fake.tables["pyq"] if r2["mock_name"] == "Ready One" and r2["review_status"] == "approved"), key=lambda r2: r2["mock_position"])]
    check("served in mock_position order", [q["id"] for q in data["questions"]] == expected_order)
    check("180-minute duration, not mocktest.html's 3h20m", data["duration_seconds"] == 180 * 60)
    check("mock_name echoed back", data["mock_name"] == "Ready One")
    attempt = next(a for a in fake.tables["mock_test_attempts"] if a["id"] == data["attempt_id"])
    check("attempt stored with kind=fixed_mock and this mock_name", attempt["kind"] == "fixed_mock" and attempt["mock_name"] == "Ready One")

    print("\n# POST /mock-tests/attempts/{id}/submit: grading reused unchanged")
    ids = [q["id"] for q in data["questions"]]
    by_id = {r2["id"]: r2 for r2 in fake.tables["pyq"]}
    # Every stored answer is "a" -- correct for Physics (pos 1-45) and Chemistry (46-90) since
    # mock_question() defaults answer="a" there; Biology's duplicate-position test row at
    # position 91 was answered "b" by a DIFFERENT mock (Has Duplicate), not this one, so every
    # Biology row here is still "a" too. Answer "b" for the first 10 instead, to get a real mix.
    answers = {str(qid): ("b" if i < 10 else "a") for i, qid in enumerate(ids)}
    r = client.post(f"/mock-tests/attempts/{data['attempt_id']}/submit", json={"question_ids": [str(i) for i in ids], "answers": answers}, headers=AUTH_A)
    result = r.json()
    check("submit -> 200, 170 correct, 10 wrong, 0 skipped", r.status_code == 200 and (result["correct"], result["wrong"], result["skipped"]) == (170, 10, 0), result)
    check("score = 170*4 - 10 = 670", result["score"] == 670)
    check("subject split: Physics took the first 10 wrong answers", result["subjects"]["Physics"]["wrong"] == 10 and result["subjects"]["Chemistry"]["wrong"] == 0 and result["subjects"]["Biology"]["wrong"] == 0)
    check("mock_name carried onto the response", result["mock_name"] == "Ready One")
    saved = next(x for x in fake.tables["mock_results"] if x["id"] == result["result_id"])
    check("mock_results row stamped with this mock_name (for GET /mock-exams' own lookup)", saved["mock_name"] == "Ready One")
    check("answer key only appears in the response AFTER submit", all("correct_answer" in q for q in result["results"]))

    print("\n# GET /mock-exams again: now shows one attempt")
    m = client.get("/mock-exams", headers=AUTH_A).json()["mocks"][0]
    check("attempts=1, best_score=670, last_attempt_at set", m["attempts"] == 1 and m["best_score"] == 670 and m["last_attempt_at"], m)
    check("a different student sees zero attempts at the same mock", client.get("/mock-exams", headers=AUTH_B).json()["mocks"][0]["attempts"] == 0)


def test_unpublish_keeps_existing_attempts_open(client, fake):
    print("\n# unpublishing hides the mock from new starts but never touches attempts already in flight")
    r = start_fixed(client, "Ready One")
    attempt_id = r.json()["attempt_id"]
    ids = r.json()["questions"]

    admin_post(client, "/admin/mock-exams/unpublish", {"mock_name": "Ready One"})
    check("unpublished: no longer listed", client.get("/mock-exams", headers=AUTH_A).json()["mocks"] == [])
    check("unpublished: a brand-new start is blocked, same 404 as a nonexistent mock", start_fixed(client, "Ready One").status_code == 404)

    submit_body = {"question_ids": [str(q["id"]) for q in ids], "answers": {}}
    r2 = client.post(f"/mock-tests/attempts/{attempt_id}/submit", json=submit_body, headers=AUTH_A)
    check("an attempt started BEFORE unpublish can still submit and grade normally", r2.status_code == 200 and r2.json()["total"] == 180, r2.text[:200])
    check("that result is still stamped with the mock_name", next(x for x in fake.tables["mock_results"] if x["id"] == r2.json()["result_id"])["mock_name"] == "Ready One")

    # restore for anything run after this
    admin_post(client, "/admin/mock-exams/publish", {"mock_name": "Ready One", "title": "NEET Mock Test 1"})


def test_isolation_from_pyq_bank(fake):
    print("\n# mock-sourced rows never carry a real-PYQ shape into a source='pyq' query")
    real_pyq_only = [r for r in fake.tables["pyq"] if server.PYQ_STUDENT_VISIBLE["source"] == f"eq.{r.get('source', 'pyq')}" and r.get("source", "pyq") == "pyq"]
    check("every seeded row in this file is source='mock' (none would pass PYQ_STUDENT_VISIBLE's source filter)",
          all(r.get("source") == "mock" for r in fake.tables["pyq"]) and not real_pyq_only, len(real_pyq_only))


if __name__ == "__main__":
    fake = FakeSupabase()
    server.async_client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handle))
    tc = TestClient(server.app)  # no `with`: skips startup hooks, nothing here needs them
    test_admin_readiness(tc, fake)
    test_admin_publish(tc, fake)
    test_student_list(tc)
    test_student_start_and_grade(tc, fake)
    test_unpublish_keeps_existing_attempts_open(tc, fake)
    test_isolation_from_pyq_bank(fake)
    print(f"\nTOTAL: {len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
