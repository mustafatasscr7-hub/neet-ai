"""Regression test for server-side mock-test grading (POST /mock-tests/attempts and
POST /mock-tests/attempts/{id}/submit) and for the legacy question endpoints staying unchanged.

Run with: python test_mock_test_grading.py
No pytest dependency -- same plain pass/fail harness as test_student_name_sanitization.py. Fully
offline: server.async_client is swapped for an httpx client whose transport is a small in-memory
stand-in for Supabase (auth + the PostgREST filters these endpoints use), so the real endpoint
code -- including every PostgREST call it makes -- runs without touching any database.
"""
import asyncio
import copy
import io
import json
import os
import sys
import uuid
from contextlib import redirect_stdout
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


UID_A = "aaaaaaaa-0000-4000-8000-00000000000a"
UID_B = "bbbbbbbb-0000-4000-8000-00000000000b"
TOKENS = {"token-a": UID_A, "token-b": UID_B}
AUTH_A = {"Authorization": "Bearer token-a"}
AUTH_B = {"Authorization": "Bearer token-b"}
FORBIDDEN_KEYS = {"correct_answer", "correctanswer", "solution", "solutions", "explanation", "answer", "answer_key", "embedding"}
# Exactly what existing mock_results rows / question_details items look like in production.
RESULT_ROW_KEYS = {"id", "user_id", "score", "correct", "wrong", "skipped", "total", "time_taken", "subject_biology_score",
                   "subject_physics_score", "subject_chemistry_score", "created_at", "source", "mock_test_id", "question_details"}
DETAIL_KEYS = {"chapter", "correct_answer", "diagram_url", "difficulty", "id", "option_a", "option_a_diagram_url", "option_b",
               "option_b_diagram_url", "option_c", "option_c_diagram_url", "option_d", "option_d_diagram_url", "question",
               "status", "subject", "userAnswer", "year"}
UUID_TABLES = {"pyq", "mock_results", "mock_test_attempts"}


# ---------------------------------------------------------------- fake Supabase
class FakeSupabase:
    def __init__(self):
        self.tables = {t: [] for t in ("pyq", "mock_tests", "mock_test_questions", "personalised_test_sets", "mock_results", "mock_test_attempts")}
        self.next_int = 1
        self.reads = []          # (table, key_kind) for every GET
        self.fail_once = {}      # (method, table) -> status code to return once

    def key_kind(self, request):
        key = request.headers.get("apikey")
        return "service" if key == server.SUPABASE_SERVICE_KEY else "anon"

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
        if op in ("gte", "lt"):
            return (str(value) >= arg) if op == "gte" else (str(value) < arg)
        raise AssertionError(f"fake PostgREST: unsupported filter {field}={cond}")

    def filtered(self, table, params):
        rows = self.tables[table]
        for field, cond in params:
            if field in ("select", "order", "limit", "offset"):
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
        status = self.fail_once.pop((method, table), None)
        if status:
            return httpx.Response(status, json={"message": "injected failure"})
        params = list(request.url.params.multi_items())
        rep = "return=representation" in request.headers.get("prefer", "")
        if method == "GET":
            self.reads.append((table, self.key_kind(request)))
            # mock_test_questions' anon policy is dropped by create_mock_test_attempts_table.sql:
            # an anon-key read here would come back empty in production.
            if table == "mock_test_questions" and self.key_kind(request) == "anon":
                return httpx.Response(200, json=[])
            rows = self.filtered(table, params)
            p = dict(params)
            if p.get("order"):
                col, _, direction = p["order"].partition(".")
                rows = sorted(rows, key=lambda r: r.get(col), reverse=direction == "desc")
            if p.get("limit"):
                rows = rows[:int(p["limit"])]
            sel = p.get("select", "*")
            if sel != "*":
                cols = sel.split(",")
                rows = [{c: r.get(c) for c in cols} for r in rows]
            return httpx.Response(200, json=copy.deepcopy(rows))
        body = json.loads(request.content or b"null")
        if method == "POST":
            out = []
            for item in body if isinstance(body, list) else [body]:
                row = dict(item)
                if "id" not in row:
                    if table in UUID_TABLES:
                        row["id"] = str(uuid.uuid4())
                    else:
                        row["id"] = self.next_int
                        self.next_int += 1
                if any(str(r["id"]) == str(row["id"]) for r in self.tables[table]):
                    return httpx.Response(409, json={"code": "23505", "message": "duplicate key value violates unique constraint"})
                row.setdefault("created_at", datetime.now(timezone.utc).isoformat())
                if table == "mock_results":
                    row.setdefault("source", "mock")
                if table == "mock_test_attempts":
                    for col in ("submitted_at", "answers", "is_late", "result_id", "mock_test_id"):
                        row.setdefault(col, None)
                    row.setdefault("started_at", datetime.now(timezone.utc).isoformat())
                self.tables[table].append(row)
                out.append(row)
            return httpx.Response(201, json=copy.deepcopy(out) if rep else None)
        if method == "PATCH":
            rows = self.filtered(table, params)
            for r in rows:
                r.update(body)
            return httpx.Response(200, json=copy.deepcopy(rows) if rep else None)
        raise AssertionError(f"fake PostgREST: unsupported {method}")


def seed(fake):
    letters = ["a", "B", "3", "EMPTY", "", "d", "c ", "2"]
    def q(subject, i, answer):
        return {"id": str(uuid.uuid4()), "subject": subject, "chapter": f"{subject} ch{i % 4}", "year": 2015 + i % 9,
                "question": f"{subject} question {i}?", "option_a": f"A{i}", "option_b": f"B{i}", "option_c": f"C{i}", "option_d": f"D{i}",
                "correct_answer": answer, "difficulty": "Easy", "diagram_url": None, "option_a_diagram_url": None,
                "option_b_diagram_url": None, "option_c_diagram_url": None, "option_d_diagram_url": None,
                "is_active": True, "embedding": [0.1] * 4, "source_pdf_filename": "secret.pdf", "source_tag": "t",
                "reviewed": True, "class": 11, "has_diagram": False, "question_type": "mcq"}
    for subject, n in (("Biology", 120), ("Physics", 60), ("Chemistry", 60)):
        for i in range(n):
            fake.tables["pyq"].append(q(subject, i, letters[i % len(letters)]))
    fake.tables["pyq"].append({**q("Biology", 999, "a"), "is_active": False})
    fake.tables["mock_tests"] += [{"id": 7, "title": "Official 1", "is_published": True}, {"id": 8, "title": "Draft", "is_published": False}]
    for order, (subject, ans) in enumerate([("Physics", "a"), ("Chemistry", "b"), ("Biology", "4"), ("Biology", "c"), ("Physics", "d")], start=1):
        fake.tables["mock_test_questions"].append({**q(subject, 500 + order, ans), "id": 100 + (6 - order), "mock_test_id": 7, "question_order": order})
    fake.tables["mock_test_questions"].append({**q("Physics", 600, "a"), "id": 300, "mock_test_id": 8, "question_order": 1})
    bio_ids = [r["id"] for r in fake.tables["pyq"] if r["subject"] == "Biology" and r["is_active"]][:12]
    fake.tables["personalised_test_sets"].append({"subject": "Biology", "test_number": 1, "title": "Bio set 1", "question_ids": bio_ids})


def walk_keys(value):
    if isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from walk_keys(v)
    elif isinstance(value, list):
        for v in value:
            yield from walk_keys(v)


def reset_limits():
    server._rate_limit_buckets.clear()
    server._mock_test_user_buckets.clear()


def start(client, body, headers=AUTH_A):
    reset_limits()
    return client.post("/mock-tests/attempts", json=body, headers=headers)


def submit(client, attempt_id, question_ids, answers, headers=AUTH_A):
    return client.post(f"/mock-tests/attempts/{attempt_id}/submit", json={"question_ids": question_ids, "answers": answers}, headers=headers)


# ---------------------------------------------------------------- tests
def test_start(client, fake):
    print("\n# POST /mock-tests/attempts: questions without answers, attempt recorded server-side")
    check("no token -> 401", client.post("/mock-tests/attempts", json={"kind": "random"}).status_code == 401)
    check("invalid token -> 401", client.post("/mock-tests/attempts", json={"kind": "random"}, headers={"Authorization": "Bearer nope"}).status_code == 401)
    check("unknown kind -> 400", start(client, {"kind": "exam"}).status_code == 400)
    check("official without mock_test_id -> 400", start(client, {"kind": "official"}).status_code == 400)
    check("personalised with no subjects -> 400", start(client, {"kind": "personalised", "selections": []}).status_code == 400)
    check("personalised with a bad subject -> 400", start(client, {"kind": "personalised", "selections": [{"subject": "Maths", "chapters": []}]}).status_code == 400)
    check("malformed body -> 422 (no stack trace)", start(client, {"kind": ["random"]}).status_code == 422)
    check("unpublished official test -> 404", start(client, {"kind": "official", "mock_test_id": 8}).status_code == 404)
    check("missing catalog test -> 404", start(client, {"kind": "personalised_catalog", "subject": "Biology", "test_number": 99}).status_code == 404)

    bodies = {
        "random": {"kind": "random"},
        "official": {"kind": "official", "mock_test_id": 7},
        "personalised": {"kind": "personalised", "selections": [{"subject": "Physics", "chapters": ["Physics ch1"]}, {"subject": "Chemistry", "chapters": []}], "count": 20},
        "personalised_catalog": {"kind": "personalised_catalog", "subject": "Biology", "test_number": 1},
    }
    started = {}
    for kind, body in bodies.items():
        before = datetime.now(timezone.utc)
        r = start(client, {**body, "started_at": "2000-01-01T00:00:00Z", "duration_seconds": 999999})
        data = r.json() if r.status_code == 200 else {}
        started[kind] = data
        bad_keys = {k for k in walk_keys(data) if k.lower() in FORBIDDEN_KEYS}
        check(f"{kind}: 200 and no answer-revealing key anywhere in the JSON", r.status_code == 200 and not bad_keys, f"{r.status_code} {bad_keys or r.text[:200]}")
        check(f"{kind}: each question carries exactly the public fields",
              data and all(set(q) == set(server.MOCK_TEST_PUBLIC_FIELDS) for q in data["questions"]))
        raw = json.dumps(data)
        check(f"{kind}: raw JSON has no \"correct_answer\" / \"explanation\" / \"solution\" key", all(f'"{s}":' not in raw for s in ("correct_answer", "explanation", "solution")))
        attempt = next((a for a in fake.tables["mock_test_attempts"] if a["id"] == data.get("attempt_id")), None)
        check(f"{kind}: attempt stored for this user with the served question order",
              attempt and attempt["user_id"] == UID_A and attempt["question_ids"] == [str(q["id"]) for q in data["questions"]], str(attempt)[:200])
        if attempt:
            s = server._parse_iso_ts(attempt["started_at"])
            check(f"{kind}: start time is the server's, not the client's", s and abs((s - before).total_seconds()) < 5, attempt["started_at"])
            check(f"{kind}: duration is the server's ({attempt['duration_seconds']}s)", attempt["duration_seconds"] == data["duration_seconds"] != 999999)

    rnd = started["random"]
    subjects = [q["subject"] for q in rnd["questions"]]
    check("random: 90 Biology + 45 Physics + 45 Chemistry, Biology block first",
          subjects.count("Biology") == 90 and subjects.count("Physics") == 45 and subjects.count("Chemistry") == 45 and subjects[:90] == ["Biology"] * 90)
    check("random: 3h20m duration, deadline = start + duration",
          rnd["duration_seconds"] == 12000 and server._parse_iso_ts(rnd["deadline_at"]) - server._parse_iso_ts(rnd["started_at"]) == timedelta(seconds=12000))
    check("random: inactive questions never served", all(q["question"] != "Biology question 999?" for q in rnd["questions"]))
    off = started["official"]
    db_order = [r["id"] for r in sorted((r for r in fake.tables["mock_test_questions"] if r["mock_test_id"] == 7), key=lambda r: r["question_order"])]
    check("official: question order = question_order", [q["id"] for q in off["questions"]] == db_order, str([q["id"] for q in off["questions"]]))
    by_id = {r["id"]: r for r in fake.tables["mock_test_questions"]}
    check("official: option order and text untouched",
          all([q[f"option_{o}"] for o in "abcd"] == [by_id[q["id"]][f"option_{o}"] for o in "abcd"] for q in off["questions"]))
    check("official: mock_test_questions read with the service key only (anon policy is dropped)",
          ("mock_test_questions", "anon") not in fake.reads and ("mock_test_questions", "service") in fake.reads)
    per = started["personalised"]
    check("personalised: requested/available reported, 70s per question (min 300s)",
          per["requested"] == 20 and per["total"] == 20 and per["duration_seconds"] == max(300, 20 * 70))
    check("personalised: only the chosen chapter for Physics",
          all(q["chapter"] == "Physics ch1" for q in per["questions"] if q["subject"] == "Physics"))
    cat = started["personalised_catalog"]
    check("catalog: title, set order, and select=* never used on the secure path",
          cat["title"] == "Bio set 1" and [q["id"] for q in cat["questions"]] == fake.tables["personalised_test_sets"][0]["question_ids"])
    return started


def test_submit(client, fake, started):
    print("\n# POST /mock-tests/attempts/{id}/submit: server grading")
    off = started["official"]
    ids = [str(q["id"]) for q in off["questions"]]
    aid = off["attempt_id"]
    # Keys by question_order: P a, C b, B "4"(=d), B c, P d. Answer: right, wrong, right (digit key), skip, right.
    answers = {ids[0]: "a", ids[1]: "c", ids[2]: "D", ids[4]: "d"}
    check("no token -> 401", client.post(f"/mock-tests/attempts/{aid}/submit", json={"question_ids": ids, "answers": answers}).status_code == 401)
    check("another student's attempt -> 404 (and nothing saved)", submit(client, aid, ids, answers, AUTH_B).status_code == 404 and not fake.tables["mock_results"])
    check("unknown attempt id -> 404", submit(client, str(uuid.uuid4()), ids, answers).status_code == 404)
    check("non-uuid attempt id -> 404", submit(client, "1 or 1=1", ids, answers).status_code == 404)
    check("tampered: question swapped -> 409", submit(client, aid, ids[:-1] + ["999"], answers).status_code == 409)
    check("tampered: reordered -> 409", submit(client, aid, list(reversed(ids)), answers).status_code == 409)
    check("tampered: extra question -> 409", submit(client, aid, ids + ["301"], answers).status_code == 409)
    check("tampered: missing question -> 409", submit(client, aid, ids[1:], answers).status_code == 409)
    check("answer for a question outside the test -> 400", submit(client, aid, ids, {**answers, "300": "a"}).status_code == 400)
    check("answer that isn't a-d -> 400", submit(client, aid, ids, {ids[0]: "e"}).status_code == 400)
    check("answers not an object -> 422", client.post(f"/mock-tests/attempts/{aid}/submit", json={"question_ids": ids, "answers": ["a"]}, headers=AUTH_A).status_code == 422)
    check("empty question_ids -> 400", submit(client, aid, [], {}).status_code == 400)
    check("nothing saved by any rejected submit", not fake.tables["mock_results"] and not next(a for a in fake.tables["mock_test_attempts"] if a["id"] == aid)["submitted_at"])

    r = client.post(f"/mock-tests/attempts/{aid}/submit", json={"question_ids": ids, "answers": answers, "score": 720, "correct": 5, "started_at": "2000-01-01T00:00:00Z"}, headers=AUTH_A)
    data = r.json() if r.status_code == 200 else {}
    check("valid submit -> 200 (client-sent score/start time ignored)", r.status_code == 200 and data.get("score") == 11 and data.get("correct") == 3, f"{r.status_code} {r.text[:300]}")
    check("counts: 3 correct, 1 wrong, 1 skipped, total 5", (data.get("correct"), data.get("wrong"), data.get("skipped"), data.get("total")) == (3, 1, 1, 5))
    check("statuses per question in served order", [q["status"] for q in data.get("results", [])] == ["correct", "wrong", "correct", "skipped", "correct"])
    check("answers + correct keys come back only now", [q["correct_answer"] for q in data.get("results", [])] == ["a", "b", "4", "c", "d"] and data["results"][1]["userAnswer"] == "c")
    check("per-subject breakdown", data.get("subjects", {}).get("Physics") == {"correct": 2, "wrong": 0, "skipped": 0, "total": 2, "score": 8}
          and data["subjects"]["Chemistry"] == {"correct": 0, "wrong": 1, "skipped": 0, "total": 1, "score": 0}
          and data["subjects"]["Biology"]["score"] == 4)
    check("not late, not a replay", data.get("late") is False and data.get("already_submitted") is False)
    rows = fake.tables["mock_results"]
    row = rows[0] if rows else {}
    check("exactly one mock_results row, same columns as existing rows", len(rows) == 1 and set(row) == RESULT_ROW_KEYS, str(set(row) ^ RESULT_ROW_KEYS))
    check("row: user from the token, mock_test_id, source default 'mock', clamped scores",
          row.get("user_id") == UID_A and row.get("mock_test_id") == 7 and row.get("source") == "mock" and row.get("score") == 11
          and row.get("subject_physics_score") == 8 and row.get("subject_chemistry_score") == 0 and row.get("subject_biology_score") == 4)
    check("row: question_details items have exactly the existing keys", all(set(q) == DETAIL_KEYS for q in row.get("question_details", [])))
    check("response result_id = the saved row id", data.get("result_id") == row.get("id"))

    r2 = submit(client, aid, ids, {ids[0]: "b", ids[1]: "b", ids[2]: "b", ids[3]: "b", ids[4]: "b"})
    d2 = r2.json() if r2.status_code == 200 else {}
    check("second submit (different answers) -> same saved result, answers not changed",
          r2.status_code == 200 and d2.get("already_submitted") is True and d2.get("result_id") == data.get("result_id")
          and d2.get("score") == 11 and [q["userAnswer"] for q in d2["results"]] == ["a", "c", "d", None, "d"], r2.text[:200])
    check("still exactly one mock_results row", len(fake.tables["mock_results"]) == 1)
    stored = next(a for a in fake.tables["mock_test_attempts"] if a["id"] == aid)
    check("attempt keeps the first answers", stored["answers"] == {ids[0]: "a", ids[1]: "c", ids[2]: "d", ids[4]: "d"})

    print("\n# grading rules")
    rnd = started["random"]
    rids = [str(q["id"]) for q in rnd["questions"]]
    keys = {str(r["id"]): r["correct_answer"] for r in fake.tables["pyq"]}
    # Answer 'a' everywhere: correct only where the key normalises to a; blank/EMPTY keys skip.
    r = submit(client, rnd["attempt_id"], rids, {qid: "A" for qid in rids})
    d = r.json()
    expect = [("skipped" if server._answer_letter(keys[qid]) is None else "correct" if server._answer_letter(keys[qid]) == "a" else "wrong") for qid in rids]
    check("letter/digit/uppercase/blank/'EMPTY' keys graded like the web and app", [q["status"] for q in d["results"]] == expect)
    check("score = 4 x correct - wrong (unclamped in response)", d["score"] == 4 * expect.count("correct") - expect.count("wrong"))
    row = next(x for x in fake.tables["mock_results"] if x["id"] == d["result_id"])
    check("stored score clamped at 0, like the pages", row["score"] == max(0, d["score"]))
    per = started["personalised"]
    pids = [str(q["id"]) for q in per["questions"]]
    r = submit(client, per["attempt_id"], pids, {})
    prow = next((x for x in fake.tables["mock_results"] if x["id"] == r.json().get("result_id")), {})
    check("personalised: all skipped when nothing answered, saved with source 'personalised'",
          r.json().get("skipped") == len(pids) and prow.get("source") == "personalised" and prow.get("mock_test_id") is None)


def test_late_and_recovery(client, fake):
    print("\n# server-side timer")
    grace = server.MOCK_TEST_SUBMIT_GRACE_SECONDS
    for label, seconds_over, expect_late in (("inside the grace period", grace - 5, False), ("beyond duration + grace", grace + 5, True)):
        d = start(client, {"kind": "personalised_catalog", "subject": "Biology", "test_number": 1}).json()
        attempt = next(a for a in fake.tables["mock_test_attempts"] if a["id"] == d["attempt_id"])
        attempt["started_at"] = (datetime.now(timezone.utc) - timedelta(seconds=attempt["duration_seconds"] + seconds_over)).isoformat()
        ids = [str(q["id"]) for q in d["questions"]]
        r = submit(client, d["attempt_id"], ids, {ids[0]: "a"})
        check(f"{label}: graded and saved, late={expect_late}", r.status_code == 200 and r.json()["late"] is expect_late and attempt["is_late"] is expect_late, r.text[:200])
        check(f"{label}: time_taken from the server clock", r.json()["time_taken"] >= attempt["duration_seconds"] + seconds_over - 2)

    print("\n# crash between claim and result write, then retry")
    d = start(client, {"kind": "personalised_catalog", "subject": "Biology", "test_number": 1}).json()
    ids = [str(q["id"]) for q in d["questions"]]
    before = len(fake.tables["mock_results"])
    fake.fail_once[("POST", "mock_results")] = 500
    buf = io.StringIO()
    with redirect_stdout(buf):
        r1 = submit(client, d["attempt_id"], ids, {ids[0]: "a", ids[1]: "b"})
    check("result write fails -> clean 502", r1.status_code == 502 and "Traceback" not in r1.text and r1.json()["detail"].startswith("Could not submit"))
    check("server log names only the error class, never answers or ids", "SUBMIT ERROR: RuntimeError" in buf.getvalue() and ids[0] not in buf.getvalue() and '"a"' not in buf.getvalue(), buf.getvalue())
    attempt = next(a for a in fake.tables["mock_test_attempts"] if a["id"] == d["attempt_id"])
    attempt["submitted_at"] = (datetime.now(timezone.utc) - timedelta(seconds=45)).isoformat()
    r2 = submit(client, d["attempt_id"], ids, {ids[0]: "d"})
    check("retry grades from the answers stored at the first submit", r2.status_code == 200 and r2.json()["results"][0]["userAnswer"] == "a" and r2.json()["results"][1]["userAnswer"] == "b", r2.text[:200])
    check("exactly one new mock_results row, under the pre-assigned id", len(fake.tables["mock_results"]) == before + 1 and r2.json()["result_id"] == attempt["result_id"])

    print("\n# two submits at the same moment")
    d = start(client, {"kind": "personalised_catalog", "subject": "Biology", "test_number": 1}).json()
    ids = [str(q["id"]) for q in d["questions"]]
    before = len(fake.tables["mock_results"])

    async def race():
        reset_limits()
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            body = {"question_ids": ids, "answers": {ids[0]: "a"}}
            return await asyncio.gather(*[ac.post(f"/mock-tests/attempts/{d['attempt_id']}/submit", json=body, headers=AUTH_A) for _ in range(2)])
    results = asyncio.run(race())
    codes = sorted(r.status_code for r in results)
    check("both succeed with the same result, one row written", codes == [200, 200] and results[0].json()["result_id"] == results[1].json()["result_id"]
          and len(fake.tables["mock_results"]) == before + 1, f"{codes} {[r.text[:80] for r in results]}")
    check("exactly one of them reports already_submitted", sorted(r.json()["already_submitted"] for r in results) == [False, True])


def test_rate_limits(client):
    print("\n# per-user rate limits")
    reset_limits()
    codes = [client.post("/mock-tests/attempts", json={"kind": "personalised_catalog", "subject": "Biology", "test_number": 1}, headers=AUTH_A).status_code for _ in range(11)]
    check("10 starts per 10 minutes, the 11th is 429", codes[:10] == [200] * 10 and codes[10] == 429, str(codes))
    server._rate_limit_buckets.clear()
    check("another student is unaffected", client.post("/mock-tests/attempts", json={"kind": "personalised_catalog", "subject": "Biology", "test_number": 1}, headers=AUTH_B).status_code == 200)
    reset_limits()


def test_legacy_endpoints(client, fake):
    print("\n# legacy endpoints: response shape unchanged for clients that don't opt in")
    reset_limits()
    r = client.get("/mock-test-questions").json()
    check("/mock-test-questions: {questions, total}, 180 questions with correct_answer and the same 16 columns",
          set(r) == {"questions", "total"} and r["total"] == 180 and all(set(q) == set(server.MOCK_QUESTION_COLUMNS.split(",")) for q in r["questions"]))
    r = client.get("/mock-tests/7/questions").json()
    check("/mock-tests/{id}/questions: same shape, correct_answer present, question_order kept",
          set(r) == {"questions", "total"} and r["total"] == 5 and all("correct_answer" in q for q in r["questions"])
          and [q["id"] for q in r["questions"]] == [105, 104, 103, 102, 101])
    check("/mock-tests/{id}/questions: unpublished -> same error as before", client.get("/mock-tests/8/questions").json() == {"error": "Mock test not found"})
    r = client.post("/personalised-test-questions", json={"selections": [{"subject": "Chemistry", "chapters": []}], "count": 7}).json()
    check("/personalised-test-questions: {questions, requested, available} with correct_answer",
          set(r) == {"questions", "requested", "available"} and r["requested"] == 7 and r["available"] == 60 and all("correct_answer" in q for q in r["questions"]))
    check("/personalised-test-questions: errors unchanged",
          client.post("/personalised-test-questions", json={"selections": [], "count": 7}).json() == {"error": "No subjects selected"}
          and client.post("/personalised-test-questions", json={"selections": [{"subject": "Maths", "chapters": []}]}).json() == {"error": "Invalid subject"})
    r = client.post("/personalised-catalog-start", json={"subject": "Biology", "test_number": 1}).json()
    check("/personalised-catalog-start: {questions, title}, still the full select=* rows",
          set(r) == {"questions", "title"} and r["title"] == "Bio set 1" and len(r["questions"]) == 12 and all("embedding" in q and "correct_answer" in q for q in r["questions"]))
    check("/personalised-catalog-start: missing set -> same error", client.post("/personalised-catalog-start", json={"subject": "Biology", "test_number": 99}).json() == {"error": "Test not found"})
    fake.fail_once[("GET", "pyq")] = 500
    with redirect_stdout(io.StringIO()):
        r = client.get("/mock-test-questions").json()
    check("/mock-test-questions: backend failure -> same generic error", r == {"error": "Something went wrong. Please try again."})
    reset_limits()


if __name__ == "__main__":
    fake = FakeSupabase()
    seed(fake)
    server.async_client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handle))
    tc = TestClient(server.app)  # no `with`: skips startup hooks, nothing here needs them
    started = test_start(tc, fake)
    test_submit(tc, fake, started)
    test_late_and_recovery(tc, fake)
    test_rate_limits(tc)
    test_legacy_endpoints(tc, fake)
    print(f"\nTOTAL: {len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
