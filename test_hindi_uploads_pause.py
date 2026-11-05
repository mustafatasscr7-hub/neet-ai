"""HINDI_UPLOADS_ENABLED kill switch for Hindi PDF uploads (they cost real Gemini-vision money).

Run with: python test_hindi_uploads_pause.py
Same plain pass/fail harness and offline fake-Supabase approach as test_pyq_review_gate.py /
test_uploader_name.py: server.async_client is swapped for an httpx client backed by an in-memory
PostgREST stand-in, and the real Gemini-calling functions (scan_scanned_pdf_bytes,
estimate_scanned_pdf_cost, scan_pdf_bytes) are swapped for call-counting fakes, so the test proves
"never called" rather than just "returned an error eventually".

Covers:
  * language='hi' with the switch off is rejected on /admin/scan-pdf and /admin/scan-pdf/estimate
    BEFORE pdf_bytes is even decoded (never mind a Gemini call) -- both admin and contractor keys;
  * language='hi' with the switch on reaches the real (faked) Gemini call as before;
  * language='en' is unaffected by the switch either way;
  * /admin/pyq-bulk-create rejects a brand-new Hindi row (no id) while the switch is off, but still
    allows a Hindi row RESTORE (id set -- undo-delete, no new cost) and any English row;
  * GET /admin/pdf-review-config reports the live value of the switch to both keys.
"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import base64

import httpx
import server
from fastapi.testclient import TestClient
from test_pyq_review_gate import ADMIN, CONTRACTOR, ADMIN_KEY, CONTRACTOR_KEY, CHAPTER, FakeSupabase, new_question

_passed = []
_failed = []


def check(label, condition, detail=""):
    if condition:
        _passed.append(label)
        print(f"  PASS: {label}")
    else:
        _failed.append(label)
        print(f"  FAIL: {label}  {detail}")


FAKE_DATA = base64.b64encode(b"not a real pdf, just needs to be valid base64").decode()


def setup():
    fake = FakeSupabase()
    server.async_client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handle))
    server.ADMIN_PASSWORD = ADMIN_KEY
    server.CONTRACTOR_PDF_KEY = CONTRACTOR_KEY

    calls = {"scanned": 0, "estimate": 0, "standard": 0}

    def fake_scan_scanned(pdf_bytes, subject, confirm_over_cap=False, language="en", **kw):
        calls["scanned"] += 1
        return {"blocks": [], "flagged_pages": []}

    def fake_estimate(pdf_bytes, subject, language="en", **kw):
        calls["estimate"] += 1
        return {"pages": 1, "cached_pages": 0, "estimated_cost_usd": 0.01}

    def fake_scan_standard(pdf_bytes, subject, **kw):
        calls["standard"] += 1
        return {"blocks": [], "flagged_pages": []}

    server.scan_scanned_pdf_bytes = fake_scan_scanned
    server.estimate_scanned_pdf_cost = fake_estimate
    server.scan_pdf_bytes = fake_scan_standard

    return TestClient(server.app), fake, calls


def bulk_raw(client, headers, questions, **extra):
    return client.post("/admin/pyq-bulk-create", json={"questions": questions, **extra}, headers=headers)


def main():
    for who, headers in (("admin", ADMIN), ("contractor", CONTRACTOR)):
        print(f"\n# /admin/scan-pdf, Hindi, switch OFF ({who})")
        client, fake, calls = setup()
        server.HINDI_UPLOADS_ENABLED = False
        r = client.post("/admin/scan-pdf", json={
            "subject": "Biology", "data": FAKE_DATA, "language": "hi", "layout": "scanned", "confirm_over_cap": True,
        }, headers=headers)
        check(f"{who}: hi scan rejected with the paused message", r.json().get("error") == "Hindi uploads are paused", r.json())
        check(f"{who}: hi scan never reaches the Gemini call", calls["scanned"] == 0, calls)

        print(f"\n# /admin/scan-pdf/estimate, Hindi, switch OFF ({who})")
        r = client.post("/admin/scan-pdf/estimate", json={"subject": "Biology", "data": FAKE_DATA, "language": "hi"}, headers=headers)
        check(f"{who}: hi estimate rejected with the paused message", r.json().get("error") == "Hindi uploads are paused", r.json())
        check(f"{who}: hi estimate never reaches the cost-estimate call", calls["estimate"] == 0, calls)

    print("\n# /admin/scan-pdf, Hindi, switch ON")
    client, fake, calls = setup()
    server.HINDI_UPLOADS_ENABLED = True
    r = client.post("/admin/scan-pdf", json={
        "subject": "Biology", "data": FAKE_DATA, "language": "hi", "layout": "scanned", "confirm_over_cap": True,
    }, headers=ADMIN)
    check("switch on: hi scan succeeds", r.status_code == 200 and "error" not in r.json(), r.json())
    check("switch on: hi scan reaches the (faked) Gemini call exactly once", calls["scanned"] == 1, calls)

    print("\n# /admin/scan-pdf/estimate, Hindi, switch ON")
    r = client.post("/admin/scan-pdf/estimate", json={"subject": "Biology", "data": FAKE_DATA, "language": "hi"}, headers=ADMIN)
    check("switch on: hi estimate succeeds", r.status_code == 200 and "error" not in r.json(), r.json())
    check("switch on: hi estimate reaches the cost-estimate call exactly once", calls["estimate"] == 1, calls)

    for enabled in (False, True):
        print(f"\n# /admin/scan-pdf, English, switch {'ON' if enabled else 'OFF'} -- unaffected")
        client, fake, calls = setup()
        server.HINDI_UPLOADS_ENABLED = enabled
        r = client.post("/admin/scan-pdf", json={"subject": "Biology", "data": FAKE_DATA, "language": "en"}, headers=ADMIN)
        check("english scan still works", r.status_code == 200 and "error" not in r.json(), r.json())
        check("english scan reaches the standard (faked) scan call", calls["standard"] == 1, calls)

        r = client.post("/admin/scan-pdf/estimate", json={"subject": "Biology", "data": FAKE_DATA, "language": "en"}, headers=ADMIN)
        check("english estimate still works", r.status_code == 200 and "error" not in r.json(), r.json())

    print("\n# /admin/pyq-bulk-create: new Hindi row, switch OFF -- rejected, nothing created")
    client, fake, calls = setup()
    server.HINDI_UPLOADS_ENABLED = False
    r = bulk_raw(client, CONTRACTOR, [new_question("Pause test: new hindi row", language="hi")], batch_id="20261105-pause-new-hi")
    check("new hindi row rejected with the paused message", r.json().get("error") == "Hindi uploads are paused", r.json())
    check("nothing was created", not any("Pause test: new hindi row" in (x.get("question") or "") for x in fake.tables["pyq"]), fake.tables["pyq"])

    print("\n# /admin/pyq-bulk-create: Hindi RESTORE (id set), switch OFF -- still allowed, no data changes")
    r = bulk_raw(client, ADMIN, [new_question("Pause test: restored hindi row", language="hi", id="11111111-0000-4000-8000-000000000001")],
                 batch_id="20261105-pause-restore-hi")
    check("hindi restore (id set) is not blocked by the switch", r.json().get("error") is None, r.json())
    created = r.json().get("created") or []
    check("hindi restore created the row with the given id",
          len(created) == 1 and created[0].get("id") == "11111111-0000-4000-8000-000000000001", created)

    print("\n# /admin/pyq-bulk-create: English row, switch OFF -- unaffected")
    r = bulk_raw(client, CONTRACTOR, [new_question("Pause test: english row", language="en")], batch_id="20261105-pause-en")
    check("english row unaffected by the switch", r.json().get("error") is None and len(r.json().get("created") or []) == 1, r.json())

    print("\n# /admin/pyq-bulk-create: new Hindi row, switch ON -- allowed (still force-pended)")
    server.HINDI_UPLOADS_ENABLED = True
    r = bulk_raw(client, CONTRACTOR, [new_question("Pause test: new hindi row, switch on", language="hi")], batch_id="20261105-pause-new-hi-on")
    created = r.json().get("created") or []
    check("new hindi row allowed once the switch is on", len(created) == 1 and created[0].get("review_status") == "pending", r.json())

    print("\n# GET /admin/pdf-review-config reflects the live switch value")
    for who, headers in (("admin", ADMIN), ("contractor", CONTRACTOR)):
        for enabled in (False, True):
            server.HINDI_UPLOADS_ENABLED = enabled
            r = client.get("/admin/pdf-review-config", headers=headers)
            check(f"{who}: config reports hindi_uploads_enabled={enabled}",
                  r.status_code == 200 and r.json().get("hindi_uploads_enabled") is enabled, r.json())

    print(f"\nTOTAL: {len(_passed)} passed, {len(_failed)} failed")
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
