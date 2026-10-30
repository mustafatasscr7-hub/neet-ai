"""GET/PUT /me/settings (cross-device Personalize + Answer Style mirror): same offline-mock shape
as test_todo_tasks.py/test_personalisation_view.py -- Supabase is an httpx.MockTransport, nothing
here ever reaches the real Supabase project or writes anything real.

Checks:
1. Logged out -> 401 on both GET and PUT.
2. Invalid answer_style -> 400, valid partial updates only touch the given field.
3. GET with no saved row -> {personalize: None, answer_style: None}, no other keys.
4. Cross-user isolation: one student's PUT never touches another's row.
5. Missing table -> {"available": False} on both endpoints, never a 500.
6. "user_settings" is in the Delete Account hard-delete list.

Run with: python test_me_settings.py
"""
import io
import json
import os
import sys

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


class FakeSupabase:
    def __init__(self):
        self.rows = {}  # user_id -> {"personalize":..., "answer_style":..., "updated_at":...}

    def _param(self, request, name):
        raw = request.url.params.get(name)
        return raw[3:] if raw and raw.startswith("eq.") else raw

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/auth/v1/user":
            uid = TOKENS.get(request.headers.get("authorization", "")[7:])
            return httpx.Response(200, json={"id": uid}) if uid else httpx.Response(401, json={})
        if path == "/rest/v1/user_settings":
            user_id = self._param(request, "user_id")
            if request.method == "GET":
                row = self.rows.get(user_id)
                return httpx.Response(200, json=[row] if row else [])
            if request.method == "POST":  # upsert (on_conflict=user_id, merge-duplicates)
                body = json.loads(request.content)
                uid = body["user_id"]
                existing = self.rows.get(uid, {})
                existing.update(body)
                self.rows[uid] = existing
                return httpx.Response(201, json=[existing])
        return httpx.Response(200, json=[])


class MissingTableSupabase:
    """Every /rest/v1/user_settings call answers like PostgREST does when the table has never
    been created (PGRST205-style 404) -- auth still works normally."""
    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/auth/v1/user":
            uid = TOKENS.get(request.headers.get("authorization", "")[7:])
            return httpx.Response(200, json={"id": uid}) if uid else httpx.Response(401, json={})
        if path == "/rest/v1/user_settings":
            return httpx.Response(404, json={"code": "PGRST205", "message": "Could not find the table 'public.user_settings'"})
        return httpx.Response(200, json=[])


def auth(token):
    return {"Authorization": f"Bearer {token}", "X-Real-IP": "203.0.113.40"}


def test_logged_out(client):
    print("\n# logged out")
    r = client.get("/me/settings")
    check("GET no token -> 401", r.status_code == 401, r.text)
    r = client.put("/me/settings", json={"personalize": True})
    check("PUT no token -> 401", r.status_code == 401, r.text)


def test_get_no_saved_row(client):
    print("\n# GET before anything was ever saved")
    r = client.get("/me/settings", headers=auth("token-a"))
    check("200 OK", r.status_code == 200, r.text)
    data = r.json()
    check("personalize is null, answer_style is null, no other keys",
          data == {"personalize": None, "answer_style": None}, data)


def test_invalid_answer_style(client):
    print("\n# invalid answer_style rejected")
    r = client.put("/me/settings", json={"answer_style": "verbose"}, headers=auth("token-a"))
    check("bad answer_style -> 400", r.status_code == 400, r.text)
    r = client.get("/me/settings", headers=auth("token-a"))
    check("row still untouched after the rejected PUT", r.json() == {"personalize": None, "answer_style": None}, r.text)


def test_partial_put_only_touches_given_field(client):
    print("\n# partial PUT changes only the given field")
    r = client.put("/me/settings", json={"personalize": False}, headers=auth("token-a"))
    check("PUT personalize only -> 200", r.status_code == 200, r.text)
    check("response reflects personalize=false, answer_style still null",
          r.json() == {"personalize": False, "answer_style": None}, r.text)

    r = client.put("/me/settings", json={"answer_style": "concise"}, headers=auth("token-a"))
    check("PUT answer_style only -> 200", r.status_code == 200, r.text)
    check("answer_style now concise, personalize still false (untouched by this PUT)",
          r.json() == {"personalize": False, "answer_style": "concise"}, r.text)

    r = client.get("/me/settings", headers=auth("token-a"))
    check("GET reflects both saved fields", r.json() == {"personalize": False, "answer_style": "concise"}, r.text)

    r = client.put("/me/settings", json={"answer_style": "medium"}, headers=auth("token-a"))
    check("'medium' is a valid answer_style -> 200", r.status_code == 200 and r.json()["answer_style"] == "medium", r.text)


def test_empty_put_rejected(client):
    print("\n# PUT with no recognised fields -> 400")
    r = client.put("/me/settings", json={}, headers=auth("token-a"))
    check("empty body -> 400", r.status_code == 400, r.text)


def test_cross_user_isolation(client):
    print("\n# one student cannot read or change another student's row")
    client.put("/me/settings", json={"personalize": True, "answer_style": "detailed"}, headers=auth("token-a"))
    r = client.put("/me/settings", json={"personalize": False, "answer_style": "concise"}, headers=auth("token-b"))
    check("B's PUT -> 200", r.status_code == 200, r.text)

    r = client.get("/me/settings", headers=auth("token-a"))
    check("A's row unaffected by B's PUT", r.json() == {"personalize": True, "answer_style": "detailed"}, r.text)

    r = client.get("/me/settings", headers=auth("token-b"))
    check("B sees only B's own values", r.json() == {"personalize": False, "answer_style": "concise"}, r.text)


def test_missing_table(client):
    print("\n# table not created yet -> available:false, never a 500")
    r = client.get("/me/settings", headers=auth("token-a"))
    check("GET -> 200 with available:false (not 500)", r.status_code == 200 and r.json().get("available") is False, r.text)
    r = client.put("/me/settings", json={"personalize": True}, headers=auth("token-a"))
    check("PUT -> 200 with available:false (not 500)", r.status_code == 200 and r.json().get("available") is False, r.text)
    # Validation still happens before ever touching Supabase, table or not.
    r = client.put("/me/settings", json={"answer_style": "nonsense"}, headers=auth("token-a"))
    check("invalid answer_style still 400 even with a missing table", r.status_code == 400, r.text)


def test_delete_account_table_list():
    print("\n# Delete Account hard-delete list includes user_settings")
    check("'user_settings' is in ACCOUNT_DELETE_TABLES", "user_settings" in server.ACCOUNT_DELETE_TABLES, server.ACCOUNT_DELETE_TABLES)


if __name__ == "__main__":
    fake = FakeSupabase()
    server.async_client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handle))
    tc = TestClient(server.app)
    test_logged_out(tc)
    test_get_no_saved_row(tc)
    test_invalid_answer_style(tc)
    test_partial_put_only_touches_given_field(tc)
    test_empty_put_rejected(tc)
    test_cross_user_isolation(tc)
    test_delete_account_table_list()

    missing = MissingTableSupabase()
    server.async_client = httpx.AsyncClient(transport=httpx.MockTransport(missing.handle))
    tc2 = TestClient(server.app)
    test_missing_table(tc2)

    print(f"\nTOTAL: {len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
