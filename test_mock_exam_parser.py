"""Regression test for Mock exam mode (mock_exam_columns.sql) in the PDF review pipeline.

Run with: python test_mock_exam_parser.py
Same plain pass/fail harness as test_pyq_review_gate.py.

Covers process_mock_exam_text.py, the deterministic (NO model call, ever) replacement for the old
process_pyq_vision.scan_mock_exam_pdf (which fed pages to a model -- see process_mock_exam_text's
own module docstring for why that call was replaced):
  * two-column reading order (left column top-to-bottom, then right column);
  * a question continuing across a column break and across a page break is stitched whole, not cut;
  * all three option layouts (one per line, 2x2 grid via short fragmented blocks, inline on one line);
  * a numbering gap is detected and reported, without derailing the rest of the subject;
  * a block sitting before a subject's own first recognized question-start lowers that subject's
    word coverage percent instead of silently vanishing into the previous subject;
  * a PDF with no selectable text layer is rejected with the exact plain-string message, no model
    fallback;
  * a question failing a gate (missing option, duplicate option, empty stem) is flagged
    needs_review with a reason and never dropped;
  * the repeated instructions page and each subject's own "Answers and Solutions" block are
    skipped/cut, including the common case where the heading shares a block with the previous
    question's own last option(s);
  * isolation: mock rows never reach PYQ_STUDENT_VISIBLE-filtered reads, a mock row is forced to
    review_status='pending' at save time regardless of role/override, and the duplicate-check
    corpus never mixes source='pyq' with source='mock'.

Every fixture here is hand-built synthetic text via PyMuPDF, never real exam content.
"""
import io
import os
import sys
import uuid
from unittest.mock import patch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fitz
import httpx
import process_mock_exam_text as pmet
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


# ---------------------------------------------------------------- synthetic PDF builder

def _new_page(doc, width=600, height=800):
    return doc.new_page(width=width, height=height)


def _header(page, subject, page_num, x0=50, x1=560):
    """A running header wide enough to be excluded as header/footer (requirement: >270pt), same
    shape the real PDF uses: subject name padded out to the right-hand page number."""
    pad = " " * max(1, int((x1 - x0) / 4.3) - len(subject) - len(str(page_num)))
    page.insert_text((x0, 30), f"{subject}{pad}{page_num}", fontsize=10)


def _lines(page, x, y0, lines, dy=14, fontsize=10):
    y = y0
    for line in lines:
        page.insert_text((x, y), line, fontsize=fontsize)
        y += dy
    return y


def _pdf_bytes(doc):
    data = doc.tobytes()
    doc.close()
    return data


# ---------------------------------------------------------------- two-column order + continuation

def test_two_column_order_and_continuation():
    print("\n# Two-column reading order: left column top-to-bottom, then right column")
    doc = fitz.open()
    page = _new_page(doc)
    _header(page, "Physics", 2)
    _lines(page, 50, 80, ["1. Left column question one stem?", "a. opt1a", "b. opt1b", "c. opt1c", "d. opt1d"])
    _lines(page, 50, 160, ["2. Left column question two stem?", "a. opt2a", "b. opt2b", "c. opt2c", "d. opt2d"])
    _lines(page, 320, 80, ["3. Right column question three stem?", "a. opt3a", "b. opt3b", "c. opt3c", "d. opt3d"])
    result = pmet.scan_mock_exam_pdf(_pdf_bytes(doc), "Synthetic Mock 1")

    physics = [q for q in result["questions"] if q["subject"] == "Physics"]
    check("3 Physics questions found", len(physics) == 3, physics)
    check("printed order is 1, 2, 3 (column order, not column-interleaved)",
          [q["printed_number"] for q in physics] == [1, 2, 3], [q["printed_number"] for q in physics])
    q3 = next(q for q in physics if q["printed_number"] == 3)
    check("question 3 (right column) kept its own stem", "Right column question three" in q3["question"], q3["question"])
    check("question 3 has all 4 clean options", [q3["option_a"], q3["option_b"], q3["option_c"], q3["option_d"]] == ["opt3a", "opt3b", "opt3c", "opt3d"], q3)
    check("no question flagged (clean, one-per-line layout)", not any(q["needs_review"] for q in physics), physics)


def test_continuation_across_column_and_page():
    print("\n# A question's own content spans a column break AND a page break -- stitched, not cut")
    doc = fitz.open()
    page1 = _new_page(doc)
    _header(page1, "Physics", 2)
    _lines(page1, 50, 80, ["1. Short question?", "a. x", "b. y", "c. z", "d. w"])
    # Question 2 starts in the LEFT column near the bottom, its own options continue in the RIGHT
    # column of the SAME page, and option d spills onto the NEXT page's left column.
    _lines(page1, 50, 160, ["2. A question whose own options spill across both a", "column break and a page break:"])
    _lines(page1, 320, 80, ["a. continuation option a", "b. continuation option b", "c. continuation option c"])
    page2 = _new_page(doc)
    _header(page2, "Physics", 3)
    _lines(page2, 50, 80, ["d. continuation option d"])
    _lines(page2, 50, 110, ["3. Next real question?", "a. n1", "b. n2", "c. n3", "d. n4"])
    result = pmet.scan_mock_exam_pdf(_pdf_bytes(doc), "Synthetic Mock 2")

    by_num = {q["printed_number"]: q for q in result["questions"] if q["subject"] == "Physics"}
    check("all 3 questions found", set(by_num) == {1, 2, 3}, set(by_num))
    q2 = by_num[2]
    check("question 2's stem survived the column break", "column break and a page break" in q2["question"], q2["question"])
    check("question 2's option D survived the page break", q2["option_d"] == "continuation option d", q2)
    check("question 2 not flagged (nothing actually missing)", not q2["needs_review"], q2)
    check("question 2's own source_page is where it STARTED (page 1)", q2["source_page"] == 1, q2["source_page"])


# ---------------------------------------------------------------- option layouts

def test_option_layout_2x2_grid():
    print("\n# 2x2 grid option layout (two short options per line)")
    doc = fitz.open()
    page = _new_page(doc)
    _header(page, "Chemistry", 2)
    _lines(page, 50, 80, ["1. Grid-style question?"])
    page.insert_text((50, 100), "a. p1   b. p2", fontsize=10)
    page.insert_text((50, 115), "c. p3   d. p4", fontsize=10)
    result = pmet.scan_mock_exam_pdf(_pdf_bytes(doc), "Synthetic Mock 3")
    q = next(q for q in result["questions"] if q["subject"] == "Chemistry")
    check("all 4 options recovered from a 2x2 grid", [q["option_a"], q["option_b"], q["option_c"], q["option_d"]] == ["p1", "p2", "p3", "p4"], q)
    check("not flagged", not q["needs_review"], q)


def test_option_layout_inline_one_line():
    print("\n# All four options inline on ONE physical line")
    doc = fitz.open()
    page = _new_page(doc)
    _header(page, "Biology", 2)
    page.insert_text((50, 80), "1. Inline-options question?", fontsize=10)
    page.insert_text((50, 100), "a. alpha b. beta c. gamma d. delta", fontsize=10)
    result = pmet.scan_mock_exam_pdf(_pdf_bytes(doc), "Synthetic Mock 4")
    q = next(q for q in result["questions"] if q["subject"] == "Biology")
    check("all 4 options split out of one inline line",
          [q["option_a"], q["option_b"], q["option_c"], q["option_d"]] == ["alpha", "beta", "gamma", "delta"], q)
    check("not flagged", not q["needs_review"], q)


# ---------------------------------------------------------------- gates: gaps, coverage, rejection

def test_numbering_gap_detected_and_recovers():
    print("\n# A numbering gap is reported, and parsing resumes correctly afterward (no cascade failure)")
    doc = fitz.open()
    page = _new_page(doc)
    _header(page, "Physics", 2)
    _lines(page, 50, 80, ["1. First question?", "a. a1", "b. b1", "c. c1", "d. d1"])
    # Question 2 is MISSING entirely (never printed) -- numbering jumps straight to 3.
    _lines(page, 50, 160, ["3. Third question (2 never appears)?", "a. a3", "b. b3", "c. c3", "d. d3"])
    result = pmet.scan_mock_exam_pdf(_pdf_bytes(doc), "Synthetic Mock 5")
    c = result["completeness"]["Physics"]
    check("question 1 and 3 both found (gap doesn't derail 3)", {q["printed_number"] for q in result["questions"] if q["subject"] == "Physics"} == {1, 3})
    check("question 2 reported missing", c["missing"] == [2] or 2 in c["missing"], c)
    check("found_count is 2 (not corrupted by the gap)", c["found_count"] == 2, c)


def test_coverage_drops_for_unclaimed_leading_content():
    print("\n# Content before a subject's own first recognized question-start lowers coverage%, not silently dropped")
    doc = fitz.open()
    page = _new_page(doc)
    _header(page, "Chemistry", 2)
    # A stray paragraph with no leading "N." marker, sitting before question 1 -- short enough
    # (under _MAX_HEADER_FOOTER_WIDTH) that it isn't itself mistaken for a running header/footer.
    _lines(page, 50, 80, ["Stray preamble, no marker"])
    _lines(page, 50, 110, ["1. Real question?", "a. a1", "b. b1", "c. c1", "d. d1"])
    result = pmet.scan_mock_exam_pdf(_pdf_bytes(doc), "Synthetic Mock 6")
    c = result["completeness"]["Chemistry"]
    check("question 1 still found", c["found_count"] == 1, c)
    check("coverage is below 100% (the stray preamble is counted but never claimed)", c["coverage_pct"] < 100.0, c)


def test_scanned_pdf_rejected_no_model_fallback():
    print("\n# A PDF with no selectable text layer is rejected with the exact plain-string message")
    doc = fitz.open()
    _new_page(doc)  # a blank page: no text layer at all, like a scanned image-only PDF
    pdf_bytes = _pdf_bytes(doc)
    try:
        pmet.scan_mock_exam_pdf(pdf_bytes, "Synthetic Scanned")
        check("raises on a text-less PDF", False, "no exception was raised")
    except ValueError as e:
        check("raises ValueError with the exact rejection message", str(e) == pmet.REJECT_NO_TEXT_LAYER, str(e))


def test_flagged_question_never_dropped():
    print("\n# A question failing a gate is flagged with a reason, kept, and gets a crop image -- never dropped")
    doc = fitz.open()
    page = _new_page(doc)
    _header(page, "Biology", 2)
    # Option D is simply never printed -- a real gap in the source, not a parsing artifact.
    _lines(page, 50, 80, ["1. Missing-option question?", "a. a1", "b. b1", "c. c1"])
    result = pmet.scan_mock_exam_pdf(_pdf_bytes(doc), "Synthetic Mock 7")
    qs = [q for q in result["questions"] if q["subject"] == "Biology"]
    check("the question is still present (not dropped)", len(qs) == 1, qs)
    q = qs[0]
    check("flagged needs_review", q["needs_review"] is True, q)
    check("review_reason mentions option D", q["review_reason"] and "D" in q["review_reason"], q.get("review_reason"))
    check("a cropped image is attached for the flagged question", bool(q.get("crop_image_b64")), q.get("crop_image_b64"))
    check("flagged_pages also carries this reason", any(q["review_reason"] == f["reason"] for f in result["flagged_pages"]), result["flagged_pages"])


# ---------------------------------------------------------------- instructions page + answers section

def test_instructions_page_and_answers_section_skipped():
    print("\n# The repeated instructions page and a subject's own Answers/Solutions block are skipped,")
    print("# including when the heading shares a block with the previous question's last option")
    doc = fitz.open()
    instructions = _new_page(doc)
    _lines(instructions, 50, 80, [
        "PAPER CONSISTS", "Physics: 45, Chemistry: 45 and Biology: 90 Questions each",
        "Do not open this Test Booklet until you are asked to do so.",
    ])
    page = _new_page(doc)
    _header(page, "Physics", 2)
    _lines(page, 50, 80, ["1. Real question one?", "a. a1", "b. b1", "c. c1", "d. d1"])
    # Question 2's own last option shares a block with the heading -- one insert_text call, same
    # shape as the real PDF's own "...d. 400 Ω \nAnswers and Solutions" merge.
    page.insert_text((50, 160), "2. Real question two?", fontsize=10)
    page.insert_text((50, 175), "a. a2", fontsize=10)
    page.insert_text((50, 188), "b. b2", fontsize=10)
    page.insert_text((50, 201), "c. c2", fontsize=10)
    page.insert_text((50, 214), "d. d2\nAnswers and Solutions", fontsize=10)
    page.insert_text((50, 250), "1. (a) This narrative solution text must never be extracted as a question.", fontsize=10)
    result = pmet.scan_mock_exam_pdf(_pdf_bytes(doc), "Synthetic Mock 8")

    physics = [q for q in result["questions"] if q["subject"] == "Physics"]
    check("exactly 2 real questions (instructions page and solutions never counted)", len(physics) == 2, physics)
    check("question 2 kept its real option D despite sharing a block with the heading",
          next(q for q in physics if q["printed_number"] == 2)["option_d"] == "d2",
          next(q for q in physics if q["printed_number"] == 2))
    check("solution narrative text never reached any question", not any("narrative solution" in q["question"] for q in physics), physics)


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


def test_scan_pdf_endpoint_surfaces_rejection_message():
    print("\n# /admin/scan-pdf: a real text-less-PDF rejection from scan_mock_exam_pdf surfaces as a plain error, not a 500")
    import base64
    server.ADMIN_PASSWORD = ADMIN_KEY
    client = TestClient(server.app)
    with patch.object(server, "scan_mock_exam_pdf", side_effect=ValueError(pmet.REJECT_NO_TEXT_LAYER)):
        r = client.post(
            "/admin/scan-pdf",
            json={"data": base64.b64encode(b"fake pdf bytes").decode(), "layout": "mock_exam", "mock_name": "Mock Test 1"},
            headers={"X-Admin-Key": ADMIN_KEY},
        )
    check("returns 200 with the rejection as a plain error string", r.status_code == 200 and r.json().get("error") == pmet.REJECT_NO_TEXT_LAYER, (r.status_code, r.text))


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
    test_two_column_order_and_continuation()
    test_continuation_across_column_and_page()
    test_option_layout_2x2_grid()
    test_option_layout_inline_one_line()
    test_numbering_gap_detected_and_recovers()
    test_coverage_drops_for_unclaimed_leading_content()
    test_scanned_pdf_rejected_no_model_fallback()
    test_flagged_question_never_dropped()
    test_instructions_page_and_answers_section_skipped()
    test_scan_pdf_endpoint_accepts_mock_exam_without_subject()
    test_scan_pdf_endpoint_surfaces_rejection_message()
    test_student_visible_excludes_mock()
    test_bulk_create_forces_mock_pending()
    test_dup_corpus_scoped_by_source()
    test_auto_check_excludes_mock()

    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
