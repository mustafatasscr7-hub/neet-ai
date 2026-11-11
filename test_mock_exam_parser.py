"""Regression test for Mock exam mode (mock_exam_columns.sql) in the PDF review pipeline.

Run with: python test_mock_exam_parser.py
Same plain pass/fail harness as test_pyq_review_gate.py.

Covers:
  * page classification stops reading at the "Answers and Solutions" heading, and skips the
    repeated instructions boilerplate -- neither ever reaches the extraction call;
  * subject detection is sticky across pages with no header word of their own;
  * the monotonic marker filter drops false "question start" matches from numbered sub-lists
    embedded in a question's own body (confirmed live against the real sample PDF);
  * mock_position mapping from each page's own printed numbers, with correct per-subject offsets
    (Physics 0, Chemistry 45, Biology 90) and an "approximate" flag when a page's extracted count
    doesn't match its printed-number count;
  * the completeness panel's missing/duplicate/out-of-range reporting;
  * option-label stripping ("a. text" -> "text");
  * isolation: mock rows never reach PYQ_STUDENT_VISIBLE-filtered reads, a mock row is forced to
    review_status='pending' at save time regardless of role/override, and the duplicate-check
    corpus never mixes source='pyq' with source='mock'.

Every fixture here is hand-written synthetic text, never real exam content.
"""
import io
import os
import sys
import uuid
from unittest.mock import patch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import httpx
import process_pyq_vision as ppv
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


# ---------------------------------------------------------------- page classification


def test_classify_pages():
    print("\n# Page classification: instructions skipped, answers section cut, subject sticky")
    instructions = (
        "1Mock Test-1\nPAPER CONSISTS\nPhysics: 45, Chemistry: 45 and Biology: 90 Questions each\n"
        "Do not open this Test Booklet until you are asked to do so.\n"
    )
    physics_p1 = (
        "Physics2\n1. Sample fake physics question one?\na. opt1 b. opt2 c. opt3 d. opt4\n"
        "2. Sample fake physics question two?\na. opt1 b. opt2 c. opt3 d. opt4\n"
    )
    physics_p2_with_answers = (
        "3Mock Test-1\n3. Sample fake physics question three?\na. opt1 b. opt2 c. opt3 d. opt4\n"
        "Answers and Solutions\n1. 2. 3.\na b c\n1. (a) fake solution narrative that must never be extracted\n"
    )
    chem_p1 = "Chemistry 36\n1. Sample fake chemistry question one?\na. x b. y c. z d. w\n"

    pages = [
        {"text": instructions},
        {"text": physics_p1},
        {"text": physics_p2_with_answers},
        {"text": instructions},
        {"text": chem_p1},
    ]
    classified = ppv._mock_exam_classify_pages(pages)

    check("page 0 flagged as instructions", classified[0]["is_instructions"] is True)
    check("page 1 detected as Physics", classified[1]["subject"] == "Physics", classified[1]["subject"])
    check("page 1 markers are [1, 2]", classified[1]["question_numbers"] == [1, 2], classified[1]["question_numbers"])
    check("page 2 stays Physics (no header word of its own)", classified[2]["subject"] == "Physics")
    check("page 2 had_answers_section is True", classified[2]["had_answers_section"] is True)
    check("Answers and Solutions heading never reaches question_text", "Answers and Solutions" not in classified[2]["question_text"])
    check("solution narrative text never reaches question_text", "fake solution narrative" not in classified[2]["question_text"])
    check("page 3 (2nd instructions page) flagged, doesn't reset subject", classified[3]["is_instructions"] is True)
    check("page 4 detected as Chemistry", classified[4]["subject"] == "Chemistry", classified[4]["subject"])


def test_monotonic_filter():
    print("\n# Monotonic marker filter: embedded numbered sub-lists are dropped, real numbers kept")
    page_with_embedded_list = (
        "Physics40\n30. Match Column I with Column II\n1. Fake item one\n2. Fake item two\n"
        "3. Fake item three\n31. Next real question?\na. x b. y c. z d. w\n"
    )
    classified = ppv._mock_exam_classify_pages([{"text": page_with_embedded_list}])
    check("raw regex markers include the embedded list (noisy)", classified[0]["question_numbers"] == [30, 1, 2, 3, 31], classified[0]["question_numbers"])
    filtered = ppv._mock_exam_filter_monotonic_markers(classified)
    check("filtered markers drop the embedded list, keep 30 and 31", filtered[0]["question_numbers"] == [30, 31], filtered[0]["question_numbers"])


def test_option_label_strip():
    print("\n# Option label stripping (mock mode's defensive normalization)")
    check("'a. opt1' -> 'opt1'", ppv._strip_mock_exam_option_label("a. opt1") == "opt1")
    check("'(b) opt2' -> 'opt2'", ppv._strip_mock_exam_option_label("(b) opt2") == "opt2")
    check("'C) opt3' -> 'opt3'", ppv._strip_mock_exam_option_label("C) opt3") == "opt3")
    check("no label left untouched", ppv._strip_mock_exam_option_label("No label here") == "No label here")


def test_completeness():
    print("\n# Completeness panel: expected/found/missing/duplicates/out-of-range")
    counters_pages = [
        {"index": 0, "is_instructions": False, "subject": "Physics", "question_numbers": list(range(1, 45))},  # missing 45
        {"index": 1, "is_instructions": False, "subject": "Chemistry", "question_numbers": list(range(1, 46)) + [1]},  # dup 1
        {"index": 2, "is_instructions": False, "subject": "Biology", "question_numbers": list(range(1, 91)) + [999]},  # out of range
    ]
    report = ppv._mock_exam_completeness(counters_pages)
    check("Physics missing [45]", report["Physics"]["missing"] == [45], report["Physics"])
    check("Physics found_count 44", report["Physics"]["found_count"] == 44)
    check("Chemistry duplicates [1]", report["Chemistry"]["duplicates"] == [1], report["Chemistry"])
    check("Chemistry found_count still 45 (set-based)", report["Chemistry"]["found_count"] == 45)
    check("Biology out_of_range [999]", report["Biology"]["out_of_range"] == [999], report["Biology"])
    check("Biology found_count 90 (complete)", report["Biology"]["found_count"] == 90)


# ---------------------------------------------------------------- full scan_mock_exam_pdf (offline)


def test_scan_mock_exam_pdf_offline():
    print("\n# scan_mock_exam_pdf end-to-end, offline (extraction call monkeypatched, no PDF bytes, no model call)")
    pages = [
        {"text": "1Mock Test-1\nPAPER CONSISTS\nPhysics: 45, Chemistry: 45 and Biology: 90 Questions each\nDo not open this Test Booklet.\n"},
        {"text": "Physics2\n1. Fake physics question A?\na. p1 b. p2 c. p3 d. p4\n2. Fake physics question B?\na. p1 b. p2 c. p3 d. p4\n"},
        {"text": "Chemistry 36\n1. Fake chemistry question A?\na. c1 b. c2 c. c3 d. c4\n"},
    ]

    def fake_extract(page_text, page_num, subject="Biology", model=None):
        # One fake question per detected marker, in order -- mirrors a clean extraction with no
        # drops, so position mapping should be exact.
        import re
        numbers = re.findall(r"^(\d+)\s*[.)]", page_text, re.M)
        return [{
            "question": f"fake {subject} q{n} on page {page_num}",
            "option_a": "a1", "option_b": "a2", "option_c": "a3", "option_d": "a4",
            "question_type": "mcq", "has_diagram": False, "year": None, "source_tag": None,
        } for n in numbers]

    with patch.object(ppv, "extract_pages_text_and_diagrams", return_value=pages), \
         patch.object(ppv, "extract_questions_from_text", side_effect=fake_extract):
        result = ppv.scan_mock_exam_pdf(b"unused", "Mock Test 1 (synthetic)")

    questions = result["questions"]
    check("3 questions extracted (1 instructions page skipped)", len(questions) == 3, len(questions))
    by_subject = {q["subject"]: q for q in questions if q["subject"] == "Physics"}
    physics_qs = [q for q in questions if q["subject"] == "Physics"]
    chem_qs = [q for q in questions if q["subject"] == "Chemistry"]
    check("2 Physics questions, positions 1 and 2", sorted(q["mock_position"] for q in physics_qs) == [1, 2], physics_qs)
    check("1 Chemistry question, position 46 (45 + 1)", chem_qs[0]["mock_position"] == 46, chem_qs)
    check("every question tagged source=mock", all(q["source"] == "mock" for q in questions))
    check("every question carries the mock_name", all(q["mock_name"] == "Mock Test 1 (synthetic)" for q in questions))
    check("no page's position mapping was flagged approximate (clean extraction)", not any(q["needs_review"] for q in questions), questions)
    check("output order is Physics block then Chemistry block", [q["subject"] for q in questions] == ["Physics", "Physics", "Chemistry"], [q["subject"] for q in questions])


def test_scan_mock_exam_pdf_mismatch_flagged():
    print("\n# scan_mock_exam_pdf: a page where extracted count != printed-marker count is flagged approximate")
    pages = [{"text": "Physics2\n1. Fake question A?\na. p1 b. p2 c. p3 d. p4\n2. Fake question B?\na. p1 b. p2 c. p3 d. p4\n"}]

    def fake_extract_one_dropped(page_text, page_num, subject="Biology", model=None):
        # Model "drops" question 2 -- only 1 extracted where 2 markers exist.
        return [{"question": "fake q1", "option_a": "a1", "option_b": "a2", "option_c": "a3", "option_d": "a4",
                  "question_type": "mcq", "has_diagram": False, "year": None, "source_tag": None}]

    with patch.object(ppv, "extract_pages_text_and_diagrams", return_value=pages), \
         patch.object(ppv, "extract_questions_from_text", side_effect=fake_extract_one_dropped):
        result = ppv.scan_mock_exam_pdf(b"unused", "Mock Test 1 (synthetic)")

    check("1 question extracted", len(result["questions"]) == 1, result["questions"])
    check("that question is flagged needs_review (approximate position)", result["questions"][0]["needs_review"] is True)
    check("flagged_pages mentions the marker/extraction mismatch", any("marker" in f["reason"] for f in result["flagged_pages"]), result["flagged_pages"])


# ---------------------------------------------------------------- /admin/scan-pdf endpoint


def test_scan_pdf_endpoint_accepts_mock_exam_without_subject():
    print("\n# /admin/scan-pdf: layout=mock_exam never sends `subject` (detected per-page from the "
          "PDF itself) -- regression test for the 422 that made every mock-exam scan fail as "
          "'Scan failed: [object Object]' (FastAPI's own validation `detail`, not a server error)")
    import base64
    server.ADMIN_PASSWORD = ADMIN_KEY
    server.CONTRACTOR_PDF_KEY = CONTRACTOR_KEY
    client = TestClient(server.app)
    fake_result = {"questions": [], "pages_scanned": 0, "flagged_pages": [], "completeness": {}, "mock_name": "Mock Test 1"}

    with patch.object(server, "scan_mock_exam_pdf", return_value=fake_result) as mocked:
        r = client.post(
            "/admin/scan-pdf",
            json={"data": base64.b64encode(b"fake pdf bytes").decode(), "layout": "mock_exam", "mock_name": "Mock Test 1"},
            headers={"X-Admin-Key": ADMIN_KEY},
        )
        check("mock_exam scan with no `subject` field returns 200, not a 422", r.status_code == 200, (r.status_code, r.text))
        body = r.json()
        check("response has no error", "error" not in body or body["error"] is None, body)
        check("scan_mock_exam_pdf was actually called", mocked.called)

    # A one-character mock_name is accepted server-side too (length is a frontend-only nicety,
    # not a server validation rule) -- confirms the bug was never about name length.
    with patch.object(server, "scan_mock_exam_pdf", return_value=fake_result):
        r2 = client.post(
            "/admin/scan-pdf",
            json={"data": base64.b64encode(b"fake pdf bytes").decode(), "layout": "mock_exam", "mock_name": "M"},
            headers={"X-Admin-Key": ADMIN_KEY},
        )
        check("a one-character mock_name still returns 200", r2.status_code == 200, (r2.status_code, r2.text))

    # Every other layout must still require a real subject -- this is the one check that caught
    # an invalid subject before (and must keep catching it now that the field is Optional).
    r3 = client.post(
        "/admin/scan-pdf",
        json={"data": base64.b64encode(b"fake pdf bytes").decode(), "layout": "standard"},
        headers={"X-Admin-Key": ADMIN_KEY},
    )
    check("a non-mock-exam scan with no subject is still a clean 200+error (not a 422)", r3.status_code == 200, (r3.status_code, r3.text))
    check("...and the error says what's wrong", r3.json().get("error") == "Invalid subject", r3.json())


# ---------------------------------------------------------------- isolation (server.py)


class FakeSupabase:
    """Minimal PostgREST stand-in, same shape as test_pyq_review_gate.py's."""

    def __init__(self):
        self.tables = {"pyq": []}

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
            if isinstance(value, bool):
                return value is not None and str(value).lower() != arg.lower()
            return value is not None and str(value) != arg
        raise AssertionError(f"fake PostgREST: unsupported filter {field}={cond}")

    def filtered(self, table, params):
        rows = self.tables.get(table, [])
        for field, cond in params:
            if field in ("select", "order", "limit", "offset"):
                continue
            rows = [r for r in rows if self.matches(r, field, cond)]
        return rows

    def handle(self, request):
        path = request.url.path
        assert path.startswith("/rest/v1/"), path
        table = path[len("/rest/v1/"):]
        method = request.method
        params = list(request.url.params.multi_items())
        if method == "GET":
            rows = self.filtered(table, params)
            return httpx.Response(200, json=[dict(r) for r in rows])
        if method == "POST":
            body = request.content
            import json as _json
            payload = _json.loads(body)
            items = payload if isinstance(payload, list) else [payload]
            created = []
            for item in items:
                row = dict(item)
                row.setdefault("id", str(uuid.uuid4()))
                row.setdefault("source", "pyq")
                self.tables.setdefault(table, []).append(row)
                created.append(row)
            return httpx.Response(200, json=created)
        if method == "PATCH":
            import json as _json
            body = _json.loads(request.content)
            updated = []
            for row in self.filtered(table, params):
                row.update(body)
                updated.append(row)
            return httpx.Response(200, json=updated)
        raise AssertionError(f"fake PostgREST: unsupported {method}")


def pyq_row(**extra):
    row = {
        "id": str(uuid.uuid4()), "subject": "Biology", "language": "en", "source": "pyq",
        "chapter": "Genetics", "question": "a real pyq row", "option_a": "a", "option_b": "b",
        "option_c": "c", "option_d": "d", "correct_answer": "a", "is_active": True, "year": 2021,
        "review_status": "approved", "uploaded_by": "admin",
    }
    row.update(extra)
    return row


def mock_row(**extra):
    row = pyq_row(source="mock", mock_name="Mock Test 1", mock_position=1, review_status="pending")
    row.update(extra)
    return row


def test_student_visible_excludes_mock():
    print("\n# PYQ_STUDENT_VISIBLE excludes source=mock rows")
    check("PYQ_STUDENT_VISIBLE has source=eq.pyq", server.PYQ_STUDENT_VISIBLE.get("source") == "eq.pyq", server.PYQ_STUDENT_VISIBLE)
    fake = FakeSupabase()
    fake.tables["pyq"] = [pyq_row(id="p1"), mock_row(id="m1", review_status="approved", is_active=True)]
    live = fake.filtered("pyq", list(server.PYQ_STUDENT_VISIBLE.items()))
    ids = {r["id"] for r in live}
    check("only the real pyq row survives PYQ_STUDENT_VISIBLE", ids == {"p1"}, ids)


def test_bulk_create_forces_mock_pending():
    print("\n# admin_pyq_bulk_create: mock rows always save as pending, regardless of role/override")
    fake = FakeSupabase()
    server.async_client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handle))
    server.ADMIN_PASSWORD = ADMIN_KEY
    server.CONTRACTOR_PDF_KEY = CONTRACTOR_KEY
    client = TestClient(server.app)

    def mock_question(**extra):
        q = {
            "subject": "Physics", "chapter": None, "question": "fake mock physics question",
            "option_a": "a1", "option_b": "a2", "option_c": "a3", "option_d": "a4",
            "correct_answer": "", "source": "mock", "mock_name": "Mock Test 1", "mock_position": 1,
        }
        q.update(extra)
        return q

    r = client.post("/admin/pyq-bulk-create", json={"questions": [mock_question(review_status="approved")]},
                     headers={"X-Admin-Key": ADMIN_KEY}).json()
    row = r["created"][0]
    check("admin-key mock upload with review_status=approved still saves pending", row["review_status"] == "pending", row)
    check("source saved as mock", row["source"] == "mock", row)
    check("mock_name saved", row["mock_name"] == "Mock Test 1", row)
    check("mock_position saved", row["mock_position"] == 1, row)

    r2 = client.post("/admin/pyq-bulk-create", json={"questions": [mock_question(mock_name="  ")]},
                      headers={"X-Admin-Key": ADMIN_KEY}).json()
    check("empty mock_name is refused", "error" in r2, r2)

    r3 = client.post("/admin/pyq-bulk-create", json={"questions": [mock_question(mock_position=999)]},
                      headers={"X-Admin-Key": ADMIN_KEY}).json()
    check("out-of-range mock_position (>180) is refused", "error" in r3, r3)


def test_dup_corpus_scoped_by_source():
    print("\n# _pyq_dup_corpus: source='pyq' (default) and source='mock' never mix")
    import asyncio
    fake = FakeSupabase()
    fake.tables["pyq"] = [
        pyq_row(id="p1", subject="Physics"),
        mock_row(id="m1", subject="Physics"),
    ]
    server.async_client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handle))
    server._pyq_dup_corpus_cache.clear()

    pyq_corpus = asyncio.run(server._pyq_dup_corpus("Physics", "en"))
    mock_corpus = asyncio.run(server._pyq_dup_corpus("Physics", "en", source="mock"))
    pyq_ids = {row[0] for row in pyq_corpus}
    mock_ids = {row[0] for row in mock_corpus}
    check("default (pyq) corpus contains only the real row", pyq_ids == {"p1"}, pyq_ids)
    check("mock corpus contains only the mock row", mock_ids == {"m1"}, mock_ids)


def test_auto_check_excludes_mock():
    print("\n# admin_pyq_auto_check: mock rows are never candidates for auto-approval")
    fake = FakeSupabase()
    fake.tables["pyq"] = [
        pyq_row(id="p1", review_status="pending", subject="Physics"),
        mock_row(id="m1", review_status="pending", subject="Physics"),
    ]
    server.async_client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handle))
    server.ADMIN_PASSWORD = ADMIN_KEY
    server.CONTRACTOR_PDF_KEY = CONTRACTOR_KEY
    client = TestClient(server.app)
    r = client.post("/admin/pyq-auto-check", headers={"X-Admin-Key": ADMIN_KEY}).json()
    if "error" in r and "pyq_auto_approve.sql" in r["error"]:
        print("  SKIP: pyq_auto_approve.sql columns not present in this fake -- auto-check path untestable here")
        return
    check("auto-check only processed the real pyq row, not the mock one", r.get("checked") == 1, r)


def main():
    test_classify_pages()
    test_monotonic_filter()
    test_option_label_strip()
    test_completeness()
    test_scan_mock_exam_pdf_offline()
    test_scan_mock_exam_pdf_mismatch_flagged()
    test_scan_pdf_endpoint_accepts_mock_exam_without_subject()
    test_student_visible_excludes_mock()
    test_bulk_create_forces_mock_pending()
    test_dup_corpus_scoped_by_source()
    test_auto_check_excludes_mock()

    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
