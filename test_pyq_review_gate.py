"""Regression test for contractor upload review: pending / needs_fix PYQs never reach students.

Run with: python test_pyq_review_gate.py
Same plain pass/fail harness and offline fake-Supabase approach as test_mock_test_grading.py:
server.async_client is swapped for an httpx client backed by an in-memory PostgREST stand-in, so
the real endpoint code runs without touching a database. The fake applies the same column
defaults add_pyq_review_status.sql sets (review_status -> 'pending', uploaded_at -> now()), and
its match_pyq ignores review_status entirely -- the worst case, an RPC that was never updated --
so search results only stay clean if the server's own gate works.

Covers:
  * every student path (PYQ bank chapters/counts/browse, chat PYQ search incl. every tier and a
    failed lookup, legacy + secure mock tests, personalised and catalog tests, ensure-correct-
    answer) returns approved, active rows only;
  * a static guard: any server.py function that reads the pyq table is either an admin/contractor
    route behind its key, or uses PYQ_STUDENT_VISIBLE -- a new student path that forgets the
    filter fails here;
  * uploads: the uploader comes from the key, contractor rows start pending, a contractor can't
    self-approve;
  * review: approve / needs-fix, the contractor's own fix list, a fix goes back to pending,
    warnings and the daily summary.
"""
import ast
import asyncio
import copy
import io
import json
import os
import re
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
UID = "aaaaaaaa-0000-4000-8000-00000000000a"
AUTH = {"Authorization": "Bearer token-a"}
CHAPTER = "Human Reproduction"
UUID_TABLES = {"pyq", "mock_test_attempts", "mock_results"}


def _parse_time(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


# ---------------------------------------------------------------- fake Supabase
class FakeSupabase:
    def __init__(self):
        self.tables = {t: [] for t in ("pyq", "personalised_test_sets", "mock_test_attempts", "mock_results",
                                       "mock_tests", "mock_test_questions")}
        self.pyq_reads = []   # params of every GET on pyq, for the "every read is filtered" check
        self.fail_next = {}   # (method, table) -> status to return once

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
        if op == "ilike":
            return value is not None and str(value).lower() == arg.lower()
        if op == "in":
            return str(value) in [a.strip() for a in arg.strip("()").split(",")]
        if op in ("gte", "lt"):
            a, b = _parse_time(value), _parse_time(arg)
            if a is None or b is None:
                return False
            return a >= b if op == "gte" else a < b
        raise AssertionError(f"fake PostgREST: unsupported filter {field}={cond}")

    def filtered(self, table, params):
        rows = self.tables[table]
        for field, cond in params:
            if field in ("select", "order", "limit", "offset"):
                continue
            if field == "and":
                # Simple, non-nested and=(col.op.val,...) -- all the review filters use.
                for part in cond.strip("()").split(","):
                    col, _, rest = part.partition(".")
                    rows = [r for r in rows if self.matches(r, col, rest)]
                continue
            rows = [r for r in rows if self.matches(r, field, cond)]
        return rows

    def handle(self, request):
        path = request.url.path
        if path == "/auth/v1/user":
            ok = request.headers.get("authorization", "") == "Bearer token-a"
            return httpx.Response(200, json={"id": UID}) if ok else httpx.Response(401, json={"msg": "bad jwt"})
        if path == "/rest/v1/rpc/match_pyq":
            body = json.loads(request.content)
            # Deliberately ignores review_status and is_active (see module docstring).
            rows = [r for r in self.tables["pyq"] if r.get("embedding") is not None]
            rows = sorted(rows, key=lambda r: r.get("_sim", 0), reverse=True)[: body["match_count"]]
            cols = ("id", "year", "subject", "question", "option_a", "option_b", "option_c", "option_d",
                    "correct_answer", "diagram_url", "has_diagram")
            return httpx.Response(200, json=[{**{c: r.get(c) for c in cols}, "similarity": r.get("_sim", 0)} for r in rows])
        assert path.startswith("/rest/v1/"), path
        table = path[len("/rest/v1/"):]
        method = request.method
        status = self.fail_next.pop((method, table), None)
        if status:
            return httpx.Response(status, json={"message": "injected failure"})
        params = list(request.url.params.multi_items())
        p = dict(params)
        headers = {}
        if method == "GET":
            if table == "pyq":
                self.pyq_reads.append(p)
            rows = self.filtered(table, params)
            if p.get("order"):
                col, _, direction = p["order"].partition(".")
                rows = sorted(rows, key=lambda r: (r.get(col) is None, str(r.get(col))), reverse=direction == "desc")
            total = len(rows)
            offset = int(p.get("offset", 0))
            rng = request.headers.get("range")
            if rng:
                start, _, end = rng.partition("-")
                rows = rows[int(start): int(end) + 1]
            else:
                rows = rows[offset:]
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
                row.setdefault("id", str(uuid.uuid4()) if table in UUID_TABLES else len(self.tables[table]) + 1)
                row.setdefault("created_at", datetime.now(timezone.utc).isoformat())
                if table == "pyq":
                    # add_pyq_review_status.sql's defaults: hidden unless the insert says otherwise.
                    row.setdefault("review_status", "pending")
                    row.setdefault("uploaded_at", datetime.now(timezone.utc).isoformat())
                    row.setdefault("embedding", None)
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
            sel = p.get("select")
            result = [{c: r.get(c) for c in sel.split(",")} for r in rows] if sel else rows
            return httpx.Response(200, json=copy.deepcopy(result) if rep else None)
        raise AssertionError(f"fake PostgREST: unsupported {method}")


def pyq_row(status="approved", *, subject="Biology", chapter=CHAPTER, active=True, sim=0.0, uploaded_by=None, **extra):
    row = {
        "id": str(uuid.uuid4()), "subject": subject, "chapter": chapter, "class": 12, "year": 2020,
        "question": f"{status} question {uuid.uuid4().hex[:6]} about gametogenesis?",
        "option_a": "Spermatogonia", "option_b": "Oogonia", "option_c": "Sertoli cells", "option_d": "Leydig cells",
        "correct_answer": "a", "difficulty": "Easy", "has_diagram": False, "diagram_url": None,
        "option_a_diagram_url": None, "option_b_diagram_url": None, "option_c_diagram_url": None,
        "option_d_diagram_url": None, "is_active": active, "review_status": status, "review_note": None,
        "uploaded_by": uploaded_by, "uploaded_at": datetime.now(timezone.utc).isoformat(), "batch_id": None,
        "reviewed_at": None, "reviewed_by": None, "reviewed": status == "approved", "embedding": [0.1] * 4,
        "_sim": sim, "source_tag": "t", "question_type": "mcq", "created_at": datetime.now(timezone.utc).isoformat(),
    }
    row.update(extra)
    return row


def seed(fake):
    """Per subject: 6 approved, plus one each of pending / needs_fix / inactive-approved. In
    Biology the hidden rows are also the closest semantic matches, so search would surface them
    first if nothing filtered them out."""
    hidden = set()
    for subject in ("Biology", "Physics", "Chemistry"):
        for i in range(6):
            fake.tables["pyq"].append(pyq_row(subject=subject, sim=0.5 + i * 0.01 if subject == "Biology" else 0))
        for status, active in (("pending", True), ("needs_fix", True), ("approved", False)):
            r = pyq_row(status, subject=subject, active=active, uploaded_by="contractor", sim=0.95 if subject == "Biology" else 0)
            fake.tables["pyq"].append(r)
            hidden.add(r["id"])
    bio = [r["id"] for r in fake.tables["pyq"] if r["subject"] == "Biology"]
    fake.tables["personalised_test_sets"].append({"subject": "Biology", "test_number": 1, "title": "Bio set", "question_ids": bio})
    return hidden


def reset_limits():
    server._rate_buckets.clear()
    server._rate_windows.clear()


def ids_of(rows):
    return {r["id"] for r in rows}


def visible_ids(fake):
    return {r["id"] for r in fake.tables["pyq"] if r["review_status"] == "approved" and r["is_active"]}


# ---------------------------------------------------------------- student paths
def test_student_paths(client, fake, hidden):
    print("\n# Student paths only ever return approved, active rows")
    fake.pyq_reads.clear()
    leaked = lambda rows: ids_of(rows) & hidden

    reset_limits()
    r = client.get("/pyq-chapters", params={"subject": "Biology"}).json()
    check("pyq-chapters total counts approved+active only", r.get("total") == 6, r)
    check("pyq-chapters per-chapter count excludes hidden rows", r.get("chapters") == [{"name": CHAPTER, "count": 6}], r)

    r = client.get("/pyq-browse", params={"subject": "Biology", "chapter": CHAPTER}).json()
    check("pyq-browse returns the 6 approved rows", len(r.get("questions", [])) == 6, r)
    check("pyq-browse leaks nothing hidden", not leaked(r.get("questions", [])))

    r = client.get("/mock-test-questions").json()
    check("random mock test (legacy) leaks nothing hidden", r.get("questions") and not leaked(r["questions"]), r)

    r = client.post("/personalised-test-questions", json={"selections": [{"subject": "Biology", "chapters": [CHAPTER]}], "count": 50}).json()
    check("personalised test pool is approved rows only", r.get("available") == 6 and not leaked(r.get("questions", [])), r)

    r = client.post("/personalised-catalog-start", json={"subject": "Biology", "test_number": 1}).json()
    check("catalog test drops hidden ids even when the set lists them",
          len(r.get("questions", [])) == 6 and not leaked(r.get("questions", [])), r)

    for kind, extra in (("random", {}), ("personalised", {"selections": [{"subject": "Biology", "chapters": []}], "count": 50}),
                        ("personalised_catalog", {"subject": "Biology", "test_number": 1})):
        reset_limits()
        r = client.post("/mock-tests/attempts", json={"kind": kind, **extra}, headers=AUTH)
        body = r.json()
        check(f"secure mock test ({kind}) leaks nothing hidden",
              r.status_code == 200 and body.get("questions") and not leaked(body["questions"]), (r.status_code, body))

    for hid in hidden:
        r = client.post("/ensure-correct-answer", json={"pyq_id": hid}).json()
        if r.get("error") != "Question not found":
            check("ensure-correct-answer refuses a hidden id", False, r)
            break
    else:
        check("ensure-correct-answer refuses every hidden id", True)

    print("\n# Chat PYQ search (match_pyq ignores review_status here, so only the server gate filters)")
    r = client.post("/pyq", json={"text": "Which cells form sperm during gametogenesis?"}).json()
    check("search: hidden rows (the closest matches) are dropped", r.get("pyqs") and not leaked(r["pyqs"]), r)
    check("search: still returns the approved matches", r.get("match_type") == "exact" and len(r["pyqs"]) == 5, r)
    check("search: chapter backfilled from the gate lookup", all(q.get("chapter") == CHAPTER for q in r["pyqs"]), r)

    # Only hidden rows clear the exact threshold -> must fall through, not return "exact" with nothing.
    saved = {row["id"]: row["_sim"] for row in fake.tables["pyq"]}
    for row in fake.tables["pyq"]:
        row["_sim"] = 0.9 if row["id"] in hidden else 0.1
    r = client.post("/pyq", json={"text": "Which cells form sperm during gametogenesis?", "chapter": CHAPTER}).json()
    check("search: hidden-only exact tier falls through to the chapter tier", r.get("match_type") == "chapter", r)
    check("search: chapter tier leaks nothing hidden", r.get("pyqs") and not leaked(r["pyqs"]), r)
    for row in fake.tables["pyq"]:
        row["_sim"] = saved[row["id"]]

    fake.fail_next[("GET", "pyq")] = 500
    r = client.post("/pyq", json={"text": "Which cells form sperm during gametogenesis?"}).json()
    check("search: a failed gate lookup never lets unchecked rows through", not leaked(r.get("pyqs", [])), r)

    unfiltered = [p for p in fake.pyq_reads if p.get("review_status") != "eq.approved" or p.get("is_active") != "eq.true"]
    check(f"every one of the {len(fake.pyq_reads)} student pyq reads carried the approved+active filter", not unfiltered, unfiltered[:3])


# ---------------------------------------------------------------- static guard
# Functions that read pyq but never serve students: admin/contractor routes (checked below to sit
# behind a key), and helpers only those routes call. Anything else must use PYQ_STUDENT_VISIBLE.
NON_STUDENT_HELPERS = {"_admin_count", "_pyq_dup_corpus", "_pyq_count", "_embed_pyq_row_background", "fetch_page",
                       "report_question"}  # report_question only PATCHes is_active=false
KEY_GUARDS = {"verify_admin", "verify_admin_or_contractor_pdf", "verify_contractor_pdf_review"}


def test_static_guard():
    print("\n# Static guard: no student-facing pyq read without PYQ_STUDENT_VISIBLE")
    src = open(server.__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    problems, seen = [], 0
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        seg = ast.get_source_segment(src, node)
        if not re.search(r'rest/v1/pyq["?]', seg):
            continue
        seen += 1
        if node.name.startswith(("admin_", "contractor_")) and node.name != "contractor_pdf_review_verify":
            guards = set()
            for default in node.args.defaults + node.args.kw_defaults:
                if isinstance(default, ast.Call) and getattr(default.func, "id", "") == "Depends" and default.args:
                    guards.add(getattr(default.args[0], "id", ""))
            if not guards & KEY_GUARDS:
                problems.append(f"{node.name}: admin/contractor-named but not behind a key")
        elif node.name in NON_STUDENT_HELPERS:
            continue
        elif "PYQ_STUDENT_VISIBLE" not in seg:
            problems.append(f"{node.name}: reads pyq without PYQ_STUDENT_VISIBLE")
    check(f"all {seen} functions that touch pyq are classified and filtered", not problems, problems)


# ---------------------------------------------------------------- uploads
def bulk(client, headers, questions, batch_id=None):
    body = {"questions": questions}
    if batch_id is not None:
        body["batch_id"] = batch_id
    return client.post("/admin/pyq-bulk-create", json=body, headers=headers).json()


def new_question(text, **extra):
    q = {"subject": "Biology", "chapter": CHAPTER, "question": text, "option_a": "Mitosis", "option_b": "Meiosis",
         "option_c": "Budding", "option_d": "Fission", "correct_answer": "b", "class": 12, "year": 2021}
    q.update(extra)
    return q


def test_uploads(client, fake):
    print("\n# Uploads: uploader comes from the key, contractor rows start pending")
    r = bulk(client, CONTRACTOR, [new_question("Contractor upload one: what forms the zygote after fertilisation?")], batch_id="20261018-1200-test-pdf-ab12")
    row = r["created"][0]
    check("contractor upload is pending", row["review_status"] == "pending", row)
    check("contractor upload is stamped uploaded_by=contractor", row["uploaded_by"] == "contractor", row)
    check("contractor upload keeps the page's batch id", row["batch_id"] == "20261018-1200-test-pdf-ab12" and r["batch_id"] == row["batch_id"], r)

    r = bulk(client, CONTRACTOR, [new_question("Contractor self-approval attempt: what is spermiogenesis exactly?",
                                               review_status="approved", uploaded_by="admin", batch_id="sneaky")])
    row = r["created"][0]
    check("contractor can't self-approve or pose as admin", row["review_status"] == "pending" and row["uploaded_by"] == "contractor", row)
    check("bad batch id is replaced by a server one", re.fullmatch(r"contractor-[0-9a-f]{12}", row["batch_id"] or "") is not None, row)

    r = bulk(client, ADMIN, [new_question("Admin upload: which hormone triggers ovulation in humans?")])
    row = r["created"][0]
    check("admin upload is approved and stamped admin", row["review_status"] == "approved" and row["uploaded_by"] == "admin", row)

    r = bulk(client, ADMIN, [new_question("Admin undo-restore of a pending contractor row about the corpus luteum?",
                                          review_status="pending", uploaded_by="contractor", batch_id="20261018-0900-restore")])
    row = r["created"][0]
    check("admin restore keeps the original review state", row["review_status"] == "pending" and row["uploaded_by"] == "contractor", row)

    check("wrong key can't upload", client.post("/admin/pyq-bulk-create", json={"questions": [new_question("x?")]},
                                                headers={"X-Admin-Key": "nope"}).status_code == 401)

    reset_limits()
    browse = client.get("/pyq-browse", params={"subject": "Biology", "chapter": CHAPTER}).json()["questions"]
    pending_ids = {r["id"] for r in fake.tables["pyq"] if r["review_status"] == "pending"}
    check("fresh contractor uploads don't show in the PYQ bank", not (ids_of(browse) & pending_ids))
    check("the admin upload does show in the PYQ bank", any("triggers ovulation" in q["question"] for q in browse))


# ---------------------------------------------------------------- review flow
def find_row(fake, fragment):
    return next(r for r in fake.tables["pyq"] if fragment in (r.get("question") or ""))


def test_review_flow(client, fake):
    print("\n# Review: approve / needs fix / contractor fix")
    target = find_row(fake, "Contractor upload one")
    tid = target["id"]

    r = client.post("/admin/pyq-review", json={"ids": [tid], "status": "needs_fix"}, headers=ADMIN).json()
    check("needs fix without a note is refused", "error" in r, r)
    check("contractor key can't review", client.post("/admin/pyq-review", json={"ids": [tid], "status": "approved"},
                                                      headers=CONTRACTOR).status_code == 401)
    r = client.post("/admin/pyq-review", json={"ids": [tid], "status": "needs_fix", "note": "Option C is wrong, check the PDF"}, headers=ADMIN).json()
    check("needs fix with a note is saved", target["review_status"] == "needs_fix" and target["review_note"].startswith("Option C"), r)
    check("needs fix records when and who", target["reviewed_at"] and target["reviewed_by"] == "admin")

    r = client.get("/contractor/pyq-needs-fix", headers=CONTRACTOR).json()
    listed = {x["id"]: x for x in r["rows"]}
    expected = {x["id"] for x in fake.tables["pyq"] if x["uploaded_by"] == "contractor" and x["review_status"] == "needs_fix"}
    check("contractor sees it in their needs-fix list with the note",
          tid in listed and (listed[tid]["review_note"] or "").startswith("Option C"), r)
    check("the list is exactly the contractor's own needs-fix rows", set(listed) == expected, (set(listed), expected))
    check("needs-fix list is contractor-key only", client.get("/contractor/pyq-needs-fix", headers=ADMIN).status_code == 401)

    reset_limits()
    browse = client.get("/pyq-browse", params={"subject": "Biology", "chapter": CHAPTER}).json()["questions"]
    check("a needs-fix row stays hidden from students", tid not in ids_of(browse))

    r = client.patch(f"/contractor/pyq-fix/{tid}", json={"option_c": "Budding (corrected)", "class": 12}, headers=CONTRACTOR).json()
    check("contractor fix saves the change", target["option_c"] == "Budding (corrected)", r)
    check("contractor fix goes back to pending", target["review_status"] == "pending", r)
    check("the admin's note is kept for the re-check", target["review_note"].startswith("Option C"))
    r = client.patch(f"/contractor/pyq-fix/{tid}", json={"option_c": "again"}, headers=CONTRACTOR).json()
    check("a second fix is refused once it's back to pending", "error" in r and target["option_c"] == "Budding (corrected)", r)

    approved = next(r for r in fake.tables["pyq"] if r["review_status"] == "approved" and r["is_active"])
    r = client.patch(f"/contractor/pyq-fix/{approved['id']}", json={"question": "vandalised"}, headers=CONTRACTOR).json()
    check("contractor can't edit an approved row", "error" in r and approved["question"] != "vandalised", r)
    admin_row = find_row(fake, "triggers ovulation")
    client.post("/admin/pyq-review", json={"ids": [admin_row["id"]], "status": "needs_fix", "note": "check"}, headers=ADMIN)
    r = client.patch(f"/contractor/pyq-fix/{admin_row['id']}", json={"question": "vandalised"}, headers=CONTRACTOR).json()
    check("contractor can't edit an admin upload, even one marked needs fix", "error" in r and admin_row["question"] != "vandalised", r)
    reset_limits()
    browse = client.get("/pyq-browse", params={"subject": "Biology", "chapter": CHAPTER}).json()["questions"]
    check("admin sending an approved row to needs fix hides it from students", admin_row["id"] not in ids_of(browse))

    r = client.post("/admin/pyq-review", json={"ids": [tid, admin_row["id"]], "status": "approved"}, headers=ADMIN).json()
    check("approve clears the note and marks reviewed", target["review_status"] == "approved" and target["review_note"] is None
          and target["reviewed"] is True, r)
    reset_limits()
    browse = client.get("/pyq-browse", params={"subject": "Biology", "chapter": CHAPTER}).json()["questions"]
    check("once approved, it reaches students", {tid, admin_row["id"]} <= ids_of(browse))

    r = client.post("/admin/pyq-review", json={"ids": ["not-a-uuid"], "status": "approved"}, headers=ADMIN).json()
    check("review refuses malformed ids", "error" in r, r)


# ---------------------------------------------------------------- warnings + summary
def test_warnings_and_summary(client, fake):
    print("\n# Warnings (rule-based) and the daily summary")
    w = server._pyq_basic_warnings
    codes = lambda row: [x["code"] for x in w(row)]
    base = pyq_row()
    check("a clean row has no warnings", w(base) == [], w(base))
    check("missing answer key", codes({**base, "correct_answer": ""}) == ["answer_key"])
    check("invalid answer key", "is not A-D" in w({**base, "correct_answer": "E"})[0]["text"])
    check("digit answer keys count as valid", w({**base, "correct_answer": "3"}) == [])
    check("empty option", w({**base, "option_c": " "})[0]["text"] == "Option C empty")
    check("an image-only option isn't empty", w({**base, "option_c": "", "option_c_diagram_url": "https://x/y.png"}) == [])
    check("identical options", w({**base, "option_b": "spermatogonia."})[0]["text"] == "Options A and B are identical")
    check("diagram mentioned but missing", codes({**base, "question": "In the figure shown below, identify X"}) == ["diagram"])
    check("'given below are two statements' is not a diagram mention", w({**base, "question": "Given below are two statements about meiosis"}) == [])
    check("has_diagram but nothing uploaded", codes({**base, "has_diagram": True}) == ["diagram"])
    check("diagram present -> no warning", w({**base, "question": "See the diagram", "diagram_url": "https://x/d.png"}) == [])
    check("chapter from another subject", "is a Physics chapter" in w({**base, "chapter": "Current Electricity"})[0]["text"])
    check("chapter not in the syllabus", "not in the syllabus" in w({**base, "chapter": "Cell Biology Misc"})[0]["text"])
    check("dropped old-syllabus chapter is accepted", w({**base, "chapter": "Reproduction in Organisms"}) == [])
    check("class mismatch", "is Class 12" in w({**base, "class": 11})[0]["text"])
    check("missing chapter", codes({**base, "chapter": None}) == ["chapter"])

    original = find_row(fake, "triggers ovulation")
    bulk(client, CONTRACTOR, [new_question(original["question"], correct_answer="")], batch_id="20261018-1300-dups")
    server._pyq_dup_corpus_cache.clear()
    r = client.get("/admin/pyq-search", params={"full": "true", "warnings": "true", "review_status": "pending",
                                                "batch_id": "20261018-1300-dups"}, headers=ADMIN).json()
    check("search filters by status + batch", len(r["rows"]) == 1, r)
    warn = {x["code"]: x for x in r["rows"][0]["warnings"]}
    check("duplicate of an existing question is flagged, with its id", warn.get("duplicate", {}).get("duplicate_of") == original["id"], warn)
    check("missing answer key flagged on the same row", "answer_key" in warn, warn)

    r = client.get("/admin/pyq-search", params={"full": "true", "uploaded_by": "contractor", "review_status": "pending"}, headers=ADMIN).json()
    check("uploaded-by filter", r["rows"] and all(x["uploaded_by"] == "contractor" and x["review_status"] == "pending" for x in r["rows"]), r)
    today = datetime.now(server.IST).date().isoformat()
    yesterday = (datetime.now(server.IST) - timedelta(days=1)).date().isoformat()
    r_today = client.get("/admin/pyq-search", params={"full": "true", "uploaded_date": today}, headers=ADMIN).json()
    r_old = client.get("/admin/pyq-search", params={"full": "true", "uploaded_date": yesterday}, headers=ADMIN).json()
    check("uploaded-date filter (India day)", r_today["total"] > 0 and r_old["total"] == 0, (r_today.get("total"), r_old.get("total")))
    check("bad date is refused", "error" in client.get("/admin/pyq-search", params={"uploaded_date": "18/10"}, headers=ADMIN).json())

    s = client.get("/admin/pyq-review-summary", headers=ADMIN).json()
    pending = sum(1 for r in fake.tables["pyq"] if r["review_status"] == "pending")
    contractor_today = sum(1 for r in fake.tables["pyq"] if r.get("uploaded_by") == "contractor")
    checked = sum(1 for r in fake.tables["pyq"] if r.get("reviewed_at"))
    check("summary: pending count", s["pending"] == pending, s)
    check("summary: today's contractor uploads", s["today"]["uploaded_by_contractor"] == contractor_today, s)
    check("summary: checked today", s["today"]["checked"] == checked, s)
    check("summary: batches listed with counts", any(b["batch_id"] == "20261018-1300-dups" and b["pending"] == 1 for b in s["batches"]), s["batches"])
    check("summary is admin-only", client.get("/admin/pyq-review-summary", headers=CONTRACTOR).status_code == 401)


def main():
    fake = FakeSupabase()
    hidden = seed(fake)
    server.async_client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handle))
    server.ADMIN_PASSWORD = ADMIN_KEY
    server.CONTRACTOR_PDF_KEY = CONTRACTOR_KEY

    async def fake_embedding(text):
        return [0.1] * 4
    server.get_embedding = fake_embedding
    client = TestClient(server.app)  # no `with`: skips startup hooks, nothing here needs them

    test_student_paths(client, fake, hidden)
    test_static_guard()
    test_uploads(client, fake)
    test_review_flow(client, fake)
    test_warnings_and_summary(client, fake)

    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
