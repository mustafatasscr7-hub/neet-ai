"""Mock test for the contractor-upload "uploader name" feature (uploader_name.sql + server.py).

Run with: .venv/Scripts/python.exe test_uploader_name.py
Reuses test_pyq_review_gate.py's offline fake-Supabase harness (server.async_client swapped for
an httpx client backed by an in-memory PostgREST stand-in) -- no real Supabase is touched.

Covers:
  * a valid name is accepted and stored on /admin/pyq-bulk-create (the one save endpoint both the
    PDF-scan flow and the Paste-LaTeX flow use -- Paste LaTeX parses client-side only, it never
    hits its own server endpoint);
  * an invalid name is rejected with an actual HTTP 400 on both /admin/pyq-bulk-create and
    /admin/scan-pdf;
  * a missing name stores null, both as "key omitted" (pre-migration-safe) and on a legacy row
    that never had the column;
  * a contractor "needs fix" save (PATCH /contractor/pyq-fix) never touches uploader_name;
  * /admin/pyq-search's uploader_name filter returns the right rows for all four dropdown values,
    including the "__earlier__" (no name) sentinel;
  * /contractor-pdf-review/verify returns the allowed-names list.
"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import httpx
import server
from fastapi.testclient import TestClient
from test_pyq_review_gate import ADMIN, CONTRACTOR, ADMIN_KEY, CONTRACTOR_KEY, CHAPTER, FakeSupabase, new_question, pyq_row

_passed = []
_failed = []


def check(label, condition, detail=""):
    if condition:
        _passed.append(label)
        print(f"  PASS: {label}")
    else:
        _failed.append(label)
        print(f"  FAIL: {label}  {detail}")


def bulk_raw(client, headers, questions, **extra):
    body = {"questions": questions, **extra}
    return client.post("/admin/pyq-bulk-create", json=body, headers=headers)


def ids_of(rows):
    return {r["id"] for r in rows}


def main():
    fake = FakeSupabase()
    server.async_client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handle))
    server.ADMIN_PASSWORD = ADMIN_KEY
    server.CONTRACTOR_PDF_KEY = CONTRACTOR_KEY
    client = TestClient(server.app)

    print("\n# /contractor-pdf-review/verify returns the allowed names")
    r = client.post("/contractor-pdf-review/verify", headers=CONTRACTOR).json()
    check("verify response carries uploader_names", r.get("uploader_names") == ["Mr. Md Moniruzzaman", "Admin"], r)

    print("\n# A valid name is stored (the single save endpoint both portal flows use)")
    r = bulk_raw(client, CONTRACTOR, [new_question("Uploader test: valid name A", chapter=CHAPTER)],
                 batch_id="20261026-1000-upname-a", uploader_name="Mr. Md Moniruzzaman")
    check("valid name request succeeds", r.status_code == 200, r.text)
    row_a = r.json()["created"][0]
    check("valid name is stored on the created row", row_a.get("uploader_name") == "Mr. Md Moniruzzaman", row_a)

    r = bulk_raw(client, CONTRACTOR, [new_question("Uploader test: valid name B", chapter=CHAPTER)],
                 batch_id="20261026-1000-upname-b", uploader_name="Admin")
    row_b = r.json()["created"][0]
    check("the other valid name is stored too", row_b.get("uploader_name") == "Admin", row_b)

    print("\n# An invalid name is rejected with a real HTTP 400")
    r = bulk_raw(client, CONTRACTOR, [new_question("Uploader test: invalid name", chapter=CHAPTER)],
                 batch_id="20261026-1000-upname-bad", uploader_name="Dr. Someone Else")
    check("pyq-bulk-create: invalid name -> HTTP 400", r.status_code == 400, (r.status_code, r.text))
    check("invalid-name request created nothing",
          not any("invalid name" in (x.get("question") or "").lower() for x in fake.tables["pyq"]), None)

    r = client.post("/admin/scan-pdf", json={"subject": "Biology", "data": "***not-valid-base64***",
                                              "uploader_name": "Dr. Someone Else"}, headers=CONTRACTOR)
    check("scan-pdf: invalid name -> HTTP 400", r.status_code == 400, (r.status_code, r.text))

    r = client.post("/admin/scan-pdf", json={"subject": "Biology", "data": "***not-valid-base64***",
                                              "uploader_name": "Admin"}, headers=CONTRACTOR)
    check("scan-pdf: valid name passes validation (fails later, on the bad PDF bytes, not the name)",
          r.status_code == 200 and r.json().get("error") == "Could not decode PDF data", r.json())

    print("\n# A missing name stores null")
    r = bulk_raw(client, CONTRACTOR, [new_question("Uploader test: missing name", chapter=CHAPTER)],
                 batch_id="20261026-1000-upname-missing")
    row_missing = r.json()["created"][0]
    check("omitted uploader_name stores null", row_missing.get("uploader_name") is None, row_missing)

    r = client.post("/admin/scan-pdf", json={"subject": "Biology", "data": "***not-valid-base64***"}, headers=CONTRACTOR)
    check("scan-pdf: missing name is accepted (no 400)",
          r.status_code == 200 and r.json().get("error") == "Could not decode PDF data", r.json())

    print("\n# A legacy pre-migration row (no uploader_name key at all) behaves as 'earlier upload'")
    legacy = pyq_row(subject="Biology", chapter=CHAPTER, uploaded_by="contractor")
    legacy.pop("uploader_name", None)
    fake.tables["pyq"].append(legacy)

    print("\n# Needs-fix never touches uploader_name")
    r = bulk_raw(client, CONTRACTOR, [new_question("Uploader test: needs fix keeps name", chapter=CHAPTER)],
                 batch_id="20261026-1000-upname-fix", uploader_name="Mr. Md Moniruzzaman")
    fix_row = r.json()["created"][0]
    tid = fix_row["id"]
    client.post("/admin/pyq-review", json={"ids": [tid], "status": "needs_fix", "note": "fix option C"}, headers=ADMIN)
    client.patch(f"/contractor/pyq-fix/{tid}", json={"option_c": "corrected"}, headers=CONTRACTOR)
    target = next(x for x in fake.tables["pyq"] if x["id"] == tid)
    check("needs-fix save keeps the original uploader name",
          target["option_c"] == "corrected" and target["uploader_name"] == "Mr. Md Moniruzzaman", target)

    print("\n# /admin/pyq-search's uploader_name filter, all four dropdown values")
    def search_ids(**params):
        r = client.get("/admin/pyq-search", params={"full": "true", "chapter": CHAPTER, **params}, headers=ADMIN).json()
        return ids_of(r["rows"]), r

    anyone_ids, _ = search_ids()
    check("'Anyone' (no filter) includes every row created above",
          {row_a["id"], row_b["id"], row_missing["id"], legacy["id"], tid} <= anyone_ids, anyone_ids)

    name1_ids, r1 = search_ids(uploader_name="Mr. Md Moniruzzaman")
    check("'Mr. Md Moniruzzaman' filter returns exactly that uploader's rows",
          row_a["id"] in name1_ids and tid in name1_ids and row_b["id"] not in name1_ids
          and row_missing["id"] not in name1_ids and legacy["id"] not in name1_ids, name1_ids)
    check("'Mr. Md Moniruzzaman' filter's reported total matches the row count",
          r1["total"] == len(r1["rows"]) == len(name1_ids), r1.get("total"))

    name2_ids, _ = search_ids(uploader_name="Admin")
    check("'Admin' filter returns exactly that uploader's rows",
          name2_ids == {row_b["id"]}, name2_ids)

    earlier_ids, r_earlier = search_ids(uploader_name="__earlier__")
    check("'Earlier uploads (no name)' filter returns the null/missing-column rows, nothing else",
          earlier_ids == {row_missing["id"], legacy["id"]}, earlier_ids)
    check("'Earlier uploads' total matches its row count", r_earlier["total"] == len(earlier_ids), r_earlier.get("total"))

    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
