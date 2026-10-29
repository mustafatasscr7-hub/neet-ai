"""Regression test for the Contractor Queue's "Duplicates" button (/admin/pyq-contractor-duplicates)
and the small cache-invalidation addition to /admin/pyq-delete that backs it.

Run with: python test_pyq_contractor_duplicates.py
Same offline fake-Supabase approach as test_pyq_delete_log.py / test_pyq_review_gate.py:
server.async_client is swapped for an httpx client backed by an in-memory PostgREST stand-in,
and the endpoint coroutines are awaited directly (not through TestClient) so the fire-and-forget
delete-log task and the module-level corpus cache behave exactly as they do live.

Covers:
  * 1, 3 and 8 same-year matches on one question come back as exactly that many matches,
    sorted by similarity (highest first);
  * more than 50 matches on one question are capped at 50, with the true total reported so the
    frontend can say "and N more";
  * a different-year copy comes back with year_relation="different" (an "Also asked in" case,
    never grouped as a same-year duplicate);
  * a missing year on either side comes back as year_relation="unknown";
  * a mix of same-year and different-year matches on one question reports both relations
    correctly in the same match list;
  * three mutual same-year copies: exactly two come back flagged, the third (the one with an
    earlier created_at, since none are approved/in the bank) is left as the group's original
    and never appears in `flagged`, with both flagged entries pointing at it via original_id;
  * an already-approved row anchors its group as the original even if it was uploaded later;
  * deleting a group's original invalidates that subject's corpus cache so the next check picks
    a new original instead of still pointing at the deleted row;
  * the per-subject corpus is read from the DB once and reused (cached) across repeated checks
    within the TTL -- this endpoint never re-downloads the whole bank per press;
  * the endpoint refuses a request without admin auth;
  * CONTRACTOR_DUP_THRESHOLD matches /admin/pyq-duplicates' own default 50% rule.
"""
import asyncio
import copy
import io
import json
import os
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


class FakeSupabase:
    """Minimal PostgREST stand-in -- same shape as test_pyq_delete_log.py's, plus a GET counter
    so the test can confirm the per-subject corpus is actually cached, not re-fetched."""

    def __init__(self):
        self.tables = {"pyq": [], "pyq_delete_log": []}
        self.missing_tables = set()
        self.get_calls = []  # list of (table, params-as-dict) for every GET, for the cache test

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
        if table in self.missing_tables:
            return httpx.Response(404, json={"message": f"relation \"{table}\" does not exist"})
        params = list(request.url.params.multi_items())
        p = dict(params)
        if method == "GET":
            self.get_calls.append((table, p))
            rows = self.filtered(table, params)
            total = len(rows)
            headers = {}
            if "count=exact" in request.headers.get("prefer", ""):
                headers["content-range"] = f"0-{max(len(rows) - 1, 0)}/{total}"
            offset = int(p.get("offset") or 0)
            rows = rows[offset:]
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


_seq = 0


def pyq_row(question, uploader_name, review_status, year, created_at=None, **extra):
    global _seq
    _seq += 1
    row = {
        "id": str(uuid.uuid4()), "subject": "Biology", "chapter": "Genetics",
        "question": question, "option_a": "opt a", "option_b": "opt b",
        "option_c": "opt c", "option_d": "opt d", "correct_answer": "a",
        "is_active": True, "year": year, "source_tag": None, "class": 12, "has_diagram": False,
        "diagram_url": None, "option_a_diagram_url": None, "option_b_diagram_url": None,
        "option_c_diagram_url": None, "option_d_diagram_url": None,
        "created_at": created_at or f"2024-01-{_seq:02d}T00:00:00Z",
        "review_status": review_status, "review_note": None, "uploaded_by": "contractor",
        "uploaded_at": created_at or f"2024-01-{_seq:02d}T00:00:00Z", "uploader_name": uploader_name,
    }
    row.update(extra)
    return row


CONTRACTOR = None  # set in main() after the module is loaded


async def main():
    global CONTRACTOR
    CONTRACTOR = server.CONTRACTOR_UPLOADER_NAME
    fake = FakeSupabase()
    server.async_client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handle))
    server.ADMIN_PASSWORD = ADMIN_KEY

    check("threshold matches /admin/pyq-duplicates' own 50% default",
          server.CONTRACTOR_DUP_THRESHOLD == 0.5, server.CONTRACTOR_DUP_THRESHOLD)

    # ---- 1, 3, 8 same-year matches, sorted by similarity -----------------------------------
    server._pyq_dup_corpus_cache.clear()
    q1 = pyq_row("Mendel's law of independent assortment explains gene segregation across chromosomes question one", CONTRACTOR, "pending", 2021)
    one_match = pyq_row(q1["question"], "Admin", "approved", 2021)  # same text, same year, in the bank
    fake.tables["pyq"] = [q1, one_match]
    data = await server.admin_pyq_contractor_duplicates(None)
    check("no error", "error" not in data, data)
    check("1-match question is flagged", q1["id"] in data["flagged"], data.get("flagged"))
    check("exactly 1 match", len(data["flagged"][q1["id"]]["matches"]) == 1, data["flagged"].get(q1["id"]))
    check("match is same-year (duplicate)", data["flagged"][q1["id"]]["matches"][0]["year_relation"] == "same",
          data["flagged"][q1["id"]])
    check("match location is 'bank' (Admin/approved, not the pending contractor queue)",
          data["flagged"][q1["id"]]["matches"][0]["location"] == "bank", data["flagged"][q1["id"]])

    server._pyq_dup_corpus_cache.clear()
    q3 = pyq_row("Photosynthesis light reaction occurs in the thylakoid membrane of chloroplast stroma question three", CONTRACTOR, "pending", 2020)
    three_matches = [pyq_row(q3["question"], "Admin", "approved", 2020, created_at=f"2023-0{i}-01T00:00:00Z") for i in range(1, 4)]
    fake.tables["pyq"] = [q3] + three_matches
    data = await server.admin_pyq_contractor_duplicates(None)
    check("exactly 3 matches", len(data["flagged"][q3["id"]]["matches"]) == 3, data["flagged"].get(q3["id"]))

    server._pyq_dup_corpus_cache.clear()
    q8 = pyq_row("Krebs cycle takes place in the mitochondrial matrix producing ATP and NADH question eight", CONTRACTOR, "pending", 2019)
    eight_matches = [pyq_row(q8["question"], "Admin", "approved", 2019, created_at=f"2023-0{(i % 9) + 1}-01T00:00:00Z") for i in range(8)]
    fake.tables["pyq"] = [q8] + eight_matches
    data = await server.admin_pyq_contractor_duplicates(None)
    matches8 = data["flagged"][q8["id"]]["matches"]
    check("exactly 8 matches (frontend shows 5 + 'Show all 8')", len(matches8) == 8, matches8)
    overlaps = [m["overlap"] for m in matches8]
    check("matches sorted by similarity, highest first", overlaps == sorted(overlaps, reverse=True), overlaps)
    check("not capped at 8 (well under 50)", data["flagged"][q8["id"]]["capped"] is False, data["flagged"][q8["id"]])

    # ---- hard cap at 50, "and N more" --------------------------------------------------------
    server._pyq_dup_corpus_cache.clear()
    q_many = pyq_row("Ohms law states that current through a conductor is proportional to voltage question many", CONTRACTOR, "pending", 2018)
    sixty_matches = [pyq_row(q_many["question"], "Admin", "approved", 2018, created_at=f"2022-{(i % 12) + 1:02d}-01T00:00:00Z") for i in range(60)]
    fake.tables["pyq"] = [q_many] + sixty_matches
    data = await server.admin_pyq_contractor_duplicates(None)
    info = data["flagged"][q_many["id"]]
    check("matches list capped at 50", len(info["matches"]) == 50, len(info["matches"]))
    check("true total (60) reported uncapped", info["total_matches"] == 60, info["total_matches"])
    check("capped flag is true", info["capped"] is True, info)

    # ---- different-year copy: year_relation, no same-year grouping ---------------------------
    server._pyq_dup_corpus_cache.clear()
    q_diff = pyq_row("Newtons second law relates force to mass and acceleration of a body question diff", CONTRACTOR, "pending", 2022)
    other_year = pyq_row(q_diff["question"], "Admin", "approved", 2019)
    fake.tables["pyq"] = [q_diff, other_year]
    data = await server.admin_pyq_contractor_duplicates(None)
    check("different-year match flagged (shown, not hidden)", q_diff["id"] in data["flagged"], data.get("flagged"))
    rel = data["flagged"][q_diff["id"]]["matches"][0]["year_relation"]
    check("year_relation is 'different' (Also asked in, no duplicate badge)", rel == "different", rel)
    check("never grouped -- no original_id assigned", "original_id" not in data["flagged"][q_diff["id"]],
          data["flagged"][q_diff["id"]])

    # ---- missing year on either side: 'unknown' ----------------------------------------------
    server._pyq_dup_corpus_cache.clear()
    q_noyear = pyq_row("Avogadros number is the number of particles in one mole of a substance question noyear", CONTRACTOR, "pending", None)
    has_year = pyq_row(q_noyear["question"], "Admin", "approved", 2020)
    fake.tables["pyq"] = [q_noyear, has_year]
    data = await server.admin_pyq_contractor_duplicates(None)
    rel = data["flagged"][q_noyear["id"]]["matches"][0]["year_relation"]
    check("missing year -> year_relation 'unknown'", rel == "unknown", rel)

    # ---- mix of same-year and different-year matches on one question -------------------------
    server._pyq_dup_corpus_cache.clear()
    q_mix = pyq_row("DNA replication is semiconservative as shown by the Meselson Stahl experiment question mix", CONTRACTOR, "pending", 2021)
    same = pyq_row(q_mix["question"], "Admin", "approved", 2021)
    diff = pyq_row(q_mix["question"], "Admin", "approved", 2017)
    fake.tables["pyq"] = [q_mix, same, diff]
    data = await server.admin_pyq_contractor_duplicates(None)
    rels = sorted(m["year_relation"] for m in data["flagged"][q_mix["id"]]["matches"])
    check("mix shows both relations in the same match list", rels == ["different", "same"], rels)

    # ---- three mutual same-year copies: two flagged, one original ----------------------------
    server._pyq_dup_corpus_cache.clear()
    copy_text = "The alternation of generations is seen in bryophytes pteridophytes and gymnosperms question group"
    c1 = pyq_row(copy_text, CONTRACTOR, "pending", 2021, created_at="2024-05-01T00:00:00Z")  # earliest
    c2 = pyq_row(copy_text, CONTRACTOR, "pending", 2021, created_at="2024-05-02T00:00:00Z")
    c3 = pyq_row(copy_text, CONTRACTOR, "pending", 2021, created_at="2024-05-03T00:00:00Z")
    fake.tables["pyq"] = [c1, c2, c3]
    data = await server.admin_pyq_contractor_duplicates(None)
    flagged_ids = set(data["flagged"].keys())
    check("exactly two of the three copies are flagged", len(flagged_ids) == 2, flagged_ids)
    check("the earliest-uploaded copy is the one left as the original (not flagged)",
          c1["id"] not in flagged_ids and c2["id"] in flagged_ids and c3["id"] in flagged_ids, flagged_ids)
    check("both flagged copies point at the same original_id (c1)",
          all(data["flagged"][i]["original_id"] == c1["id"] for i in flagged_ids), data["flagged"])
    check("original_in_queue is true (c1 is itself a pending contractor row)",
          all(data["flagged"][i]["original_in_queue"] is True for i in flagged_ids), data["flagged"])

    # ---- an already-approved row anchors the group even if uploaded later --------------------
    server._pyq_dup_corpus_cache.clear()
    copy_text2 = "Ribosomes are the site of protein synthesis in both prokaryotic and eukaryotic cells question group two"
    approved_later = pyq_row(copy_text2, "Admin", "approved", 2021, created_at="2024-09-01T00:00:00Z")  # latest, but approved
    pending_earlier = pyq_row(copy_text2, CONTRACTOR, "pending", 2021, created_at="2024-01-01T00:00:00Z")  # earliest, but pending
    fake.tables["pyq"] = [approved_later, pending_earlier]
    data = await server.admin_pyq_contractor_duplicates(None)
    check("the approved row is the original despite being uploaded later",
          data["flagged"][pending_earlier["id"]]["original_id"] == approved_later["id"],
          data["flagged"].get(pending_earlier["id"]))
    check("that original is correctly marked as not in the queue",
          data["flagged"][pending_earlier["id"]]["original_in_queue"] is False,
          data["flagged"][pending_earlier["id"]])

    # ---- deleting the original invalidates the corpus cache so the next check re-picks -------
    server._pyq_dup_corpus_cache.clear()
    copy_text3 = "Transpiration pull explains the ascent of water in the xylem of tall trees question group three"
    orig = pyq_row(copy_text3, CONTRACTOR, "pending", 2021, created_at="2024-02-01T00:00:00Z")
    survivor = pyq_row(copy_text3, CONTRACTOR, "pending", 2021, created_at="2024-02-02T00:00:00Z")
    fake.tables["pyq"] = [orig, survivor]
    data = await server.admin_pyq_contractor_duplicates(None)
    check("setup: orig is the original, survivor flagged", survivor["id"] in data["flagged"] and orig["id"] not in data["flagged"],
          data["flagged"])
    r = await server.admin_pyq_delete(orig["id"], None)
    await asyncio.sleep(0)
    check("delete of the original succeeds", r.get("success") is True, r)
    check("corpus cache for Biology was invalidated by the delete", "Biology" not in server._pyq_dup_corpus_cache,
          list(server._pyq_dup_corpus_cache.keys()))
    data2 = await server.admin_pyq_contractor_duplicates(None)
    check("after the original is deleted, the survivor is no longer flagged against it",
          survivor["id"] not in data2["flagged"], data2["flagged"])

    # ---- corpus is cached, not re-fetched on a repeat check within the TTL -------------------
    server._pyq_dup_corpus_cache.clear()
    fake.tables["pyq"] = [q1, one_match]
    fake.get_calls.clear()
    await server.admin_pyq_contractor_duplicates(None)
    corpus_fetches_1 = sum(1 for table, p in fake.get_calls if table == "pyq" and "subject" in p)
    await server.admin_pyq_contractor_duplicates(None)
    corpus_fetches_2 = sum(1 for table, p in fake.get_calls if table == "pyq" and "subject" in p)
    check("first check reads the per-subject corpus from the DB once", corpus_fetches_1 == 1, corpus_fetches_1)
    check("second check within the TTL reuses the cache -- no second corpus read",
          corpus_fetches_2 == corpus_fetches_1, (corpus_fetches_1, corpus_fetches_2))

    # ---- auth ---------------------------------------------------------------------------------
    client = TestClient(server.app)  # no `with`: skips startup hooks, nothing here needs them
    resp = client.get("/admin/pyq-contractor-duplicates")
    check("no key -> 401", resp.status_code == 401, resp.status_code)
    resp = client.get("/admin/pyq-contractor-duplicates", headers={"X-Admin-Key": "wrong"})
    check("wrong key -> 401", resp.status_code == 401, resp.status_code)

    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
