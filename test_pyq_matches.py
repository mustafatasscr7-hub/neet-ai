"""Regression test for the "Duplicate" pill's cross-check panel endpoint (/admin/pyq-matches) on
admin-pyq-preview.html.

Run with: python test_pyq_matches.py
Same offline fake-Supabase approach as test_pyq_contractor_duplicates.py: server.async_client is
swapped for an httpx client backed by an in-memory PostgREST stand-in, and the endpoint coroutine
is awaited directly (not through TestClient) so the module-level corpus cache behaves exactly as
it does live.

Covers:
  * the target's own id is never returned as one of its own matches;
  * only same-subject, same-language rows are considered (a different subject or a different
    language never comes back as a match even with identical text);
  * a rejected row is never returned as a match (kept as reject-reason proof only);
  * an inactive row is never returned as a match;
  * matches are capped at 5, sorted by similarity (highest first), with the true total reported;
  * each match carries id, match_percent, uploader_name, uploaded_at, review_status, subject,
    chapter, year, question, options and figure urls;
  * a question id that doesn't exist comes back as an error, not a crash;
  * the endpoint refuses a request without admin auth, and the contractor key doesn't work on it.
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
    """Minimal PostgREST stand-in -- same shape as test_pyq_contractor_duplicates.py's."""

    def __init__(self):
        self.tables = {"pyq": []}

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
            offset = int(p.get("offset") or 0)
            rows = rows[offset:]
            if p.get("limit"):
                rows = rows[: int(p["limit"])]
            return httpx.Response(200, json=copy.deepcopy(rows))
        raise AssertionError(f"fake PostgREST: unsupported {method}")


_seq = 0


def pyq_row(question, **extra):
    global _seq
    _seq += 1
    row = {
        "id": str(uuid.uuid4()), "subject": "Biology", "language": "en", "chapter": "Genetics",
        "question": question, "option_a": "opt a", "option_b": "opt b",
        "option_c": "opt c", "option_d": "opt d", "correct_answer": "a",
        "is_active": True, "year": 2021, "source_tag": None, "class": 12, "has_diagram": False,
        "diagram_url": None, "option_a_diagram_url": None, "option_b_diagram_url": None,
        "option_c_diagram_url": None, "option_d_diagram_url": None,
        "created_at": f"2024-01-{_seq:02d}T00:00:00Z",
        "review_status": "pending", "review_note": None, "uploaded_by": "contractor",
        "uploaded_at": f"2024-01-{_seq:02d}T00:00:00Z", "uploader_name": "Mr. Md Moniruzzaman",
    }
    row.update(extra)
    return row


TEXT = "Mendels law of independent assortment explains gene segregation across chromosomes"


async def main():
    fake = FakeSupabase()
    server.async_client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handle))
    server.ADMIN_PASSWORD = ADMIN_KEY
    server.CONTRACTOR_PDF_KEY = CONTRACTOR_KEY

    def reset(rows):
        server._pyq_dup_corpus_cache.clear()
        fake.tables["pyq"] = rows

    # ---- excludes the row itself ------------------------------------------------------------
    target = pyq_row(TEXT)
    reset([target])
    data = await server.admin_pyq_matches(target["id"], None)
    check("no error with no other rows", "error" not in data, data)
    check("no matches when it's the only row", data["matches"] == [], data)

    # ---- same subject/language match comes back with full fields ----------------------------
    match1 = pyq_row(TEXT, subject="Biology", language="en")
    reset([target, match1])
    data = await server.admin_pyq_matches(target["id"], None)
    check("no error", "error" not in data, data)
    check("exactly 1 match", len(data["matches"]) == 1, data)
    m = data["matches"][0]
    check("target's own id excluded from its matches", m["id"] != target["id"], m)
    check("match is the other row", m["id"] == match1["id"], m)
    check("match_percent is a high percent for identical text", m["match_percent"] >= 90, m)
    for field in ("id", "match_percent", "uploader_name", "uploaded_at", "review_status",
                  "subject", "chapter", "year", "question", "option_a", "option_b", "option_c",
                  "option_d", "diagram_url"):
        check(f"match carries '{field}'", field in m, m)

    # ---- different subject never matches, even with identical text --------------------------
    diff_subject = pyq_row(TEXT, subject="Physics", language="en")
    reset([target, diff_subject])
    data = await server.admin_pyq_matches(target["id"], None)
    check("different subject never matched", data["matches"] == [], data)

    # ---- different language never matches, even with identical text -------------------------
    diff_language = pyq_row(TEXT, subject="Biology", language="hi")
    reset([target, diff_language])
    data = await server.admin_pyq_matches(target["id"], None)
    check("different language never matched", data["matches"] == [], data)

    # ---- a rejected row is never returned as a match ------------------------------------------
    rejected = pyq_row(TEXT, subject="Biology", language="en", review_status="rejected")
    reset([target, rejected])
    data = await server.admin_pyq_matches(target["id"], None)
    check("rejected row never matched", data["matches"] == [], data)

    # ---- an inactive row is never returned as a match ------------------------------------------
    inactive = pyq_row(TEXT, subject="Biology", language="en", is_active=False)
    reset([target, inactive])
    data = await server.admin_pyq_matches(target["id"], None)
    check("inactive row never matched", data["matches"] == [], data)

    # ---- capped at 5, sorted by similarity, true total reported ------------------------------
    close_text_tpl = TEXT + " extra filler word number {}"
    seven = [pyq_row(close_text_tpl.format(i), subject="Biology", language="en",
                      created_at=f"2023-0{(i % 9) + 1}-01T00:00:00Z") for i in range(7)]
    reset([target] + seven)
    data = await server.admin_pyq_matches(target["id"], None)
    check("capped at 5", len(data["matches"]) == 5, data["matches"])
    check("true total (7) reported uncapped", data["total_matches"] == 7, data)
    pcts = [mm["match_percent"] for mm in data["matches"]]
    check("matches sorted by similarity, highest first", pcts == sorted(pcts, reverse=True), pcts)

    # ---- unknown id -> error, not a crash ------------------------------------------------------
    reset([target])
    data = await server.admin_pyq_matches(str(uuid.uuid4()), None)
    check("unknown id returns an error", "error" in data, data)

    # ---- auth -----------------------------------------------------------------------------------
    reset([target, match1])
    client = TestClient(server.app)  # no `with`: skips startup hooks, nothing here needs them
    resp = client.get(f"/admin/pyq-matches?id={target['id']}")
    check("no key -> 401", resp.status_code == 401, resp.status_code)
    resp = client.get(f"/admin/pyq-matches?id={target['id']}", headers={"X-Admin-Key": "wrong"})
    check("wrong key -> 401", resp.status_code == 401, resp.status_code)
    resp = client.get(f"/admin/pyq-matches?id={target['id']}", headers={"X-Admin-Key": CONTRACTOR_KEY})
    check("contractor key -> 401 (admin-only)", resp.status_code == 401, resp.status_code)
    server._admin_login_attempts.clear()  # the 3 failed attempts above would otherwise trip the cooldown
    resp = client.get(f"/admin/pyq-matches?id={target['id']}", headers={"X-Admin-Key": ADMIN_KEY})
    check("real admin key -> 200", resp.status_code == 200, resp.status_code)

    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
