"""POST /referral/register takes the referred account from the login token, never the body.

Run with: python test_referral_register.py
Plain pass/fail harness like test_client_ip.py. Fully offline: Supabase is an httpx.MockTransport
holding user_referral_codes and a referrals table with referred_id UNIQUE (409 on a second row).
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


UID_A = "aaaaaaaa-0000-4000-8000-00000000000a"   # new student
UID_B = "bbbbbbbb-0000-4000-8000-00000000000b"   # someone else, never logged in here
UID_R = "cccccccc-0000-4000-8000-00000000000c"   # referrer
TOKENS = {"token-a": UID_A, "token-r": UID_R}
CODES = {"7K4XPQR": UID_R, "AAAA222": UID_A}


class FakeSupabase:
    def __init__(self):
        self.referrals = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/auth/v1/user":
            uid = TOKENS.get(request.headers.get("authorization", "")[7:])
            return httpx.Response(200, json={"id": uid}) if uid else httpx.Response(401, json={})
        if path == "/rest/v1/user_referral_codes":
            uid = CODES.get(request.url.params["code"][3:])
            return httpx.Response(200, json=[{"user_id": uid}] if uid else [])
        if path == "/rest/v1/referrals" and request.method == "POST":
            row = json.loads(request.content)
            if any(r["referred_id"] == row["referred_id"] for r in self.referrals):
                return httpx.Response(409, json={"code": "23505"})
            self.referrals.append(row)
            return httpx.Response(201, json=[row])
        return httpx.Response(200, json=[])


def register(client, body, token=None):
    headers = {"X-Real-IP": "203.0.113.10"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return client.post("/referral/register", json=body, headers=headers)


def test_register(client, fake):
    print("\n# /referral/register")
    r = register(client, {"referred_id": UID_A, "referral_code": "7K4XPQR"})
    check("no login token -> 401, nothing stored", r.status_code == 401 and not fake.referrals, f"{r.status_code} {fake.referrals}")
    r = register(client, {"referred_id": UID_A, "referral_code": "7K4XPQR"}, token="forged")
    check("invalid token -> 401", r.status_code == 401 and not fake.referrals)
    r = register(client, {"referred_id": UID_B, "referral_code": " 7k4xpqr "}, token="token-a")
    check("old-client body (with someone else's referred_id) -> success", r.json() == {"success": True}, r.text)
    check("stored against the token's account, not the body's id",
          fake.referrals == [{"referrer_id": UID_R, "referred_id": UID_A, "referral_code": "7K4XPQR"}], str(fake.referrals))
    r = register(client, {"referral_code": "7K4XPQR"}, token="token-a")
    check("already-referred account -> rejected", r.json() == {"error": "This account has already used a referral code"}, r.text)
    r = register(client, {"referral_code": "7K4XPQR"}, token="token-r")
    check("own code -> rejected as self-referral", r.json() == {"error": "You can't refer yourself"}, r.text)
    r = register(client, {"referral_code": "NOPE999"}, token="token-r")
    check("unknown code -> rejected", r.json() == {"error": "Invalid referral code"}, r.text)
    check("only the one valid registration was stored", len(fake.referrals) == 1)


if __name__ == "__main__":
    fake = FakeSupabase()
    server.async_client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handle))
    tc = TestClient(server.app)  # no `with`: skips startup hooks, nothing here needs them
    test_register(tc, fake)
    print(f"\nTOTAL: {len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
