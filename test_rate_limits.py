"""Rate limits keyed on account / device / IP (server.py's rate_limit, _check_otp_limits, the guest
daily budget). Carrier NAT puts many real students on one public IP, so nothing here may block one
student because of another on the same IP -- while a script inventing device ids still stops at the
per-IP ceilings.

Run with: python test_rate_limits.py
Plain pass/fail harness like test_client_ip.py. Fully offline: Supabase (auth + REST) is an
httpx.MockTransport, the OTP send is stubbed, and the clock is a fake so windows can be crossed.
"""
import asyncio
import io
import json
import os
import sys
import time as real_time
import uuid

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import httpx
import server
from fastapi import Depends
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
AUTH_A = [("Authorization", "Bearer token-a")]
AUTH_B = [("Authorization", "Bearer token-b")]
IP1, IP2 = "203.0.113.10", "198.51.100.20"
DEV1, DEV2 = "11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222"


def at(ip, *extra, device=None):
    """Headers as Railway's edge delivers them for a client at `ip`."""
    headers = [("X-Real-IP", ip), ("X-Forwarded-For", f"{ip}, 79.127.178.82")]
    if device:
        headers.append(("X-Device-Id", device))
    return headers + [h for group in extra for h in group]


def new_device():
    return str(uuid.uuid4())


class FakeClock:
    """Stands in for the `time` module inside server.py so tests can step past windows."""
    def __init__(self):
        self.now = real_time.time()

    def time(self):
        return self.now

    def __getattr__(self, name):
        return getattr(real_time, name)


USAGE_KEYS = {"guest_usage_log": "ip", "guest_device_usage_log": "device_id", "usage_log": "user_id"}
USAGE_RPCS = {"increment_guest_usage": ("guest_usage_log", "p_ip"), "increment_guest_device_usage": ("guest_device_usage_log", "p_device_id"),
              "increment_daily_usage": ("usage_log", "p_user_id")}


class FakeSupabase:
    """Auth (token -> user id) plus the three daily-usage tables and their increment RPCs."""
    def __init__(self):
        self.auth_calls = 0
        self.requests = []
        self.rows = {}  # (table, key value, usage_date) -> row

    def row(self, table, key, day):
        return self.rows.setdefault((table, key, day), {USAGE_KEYS[table]: key, "usage_date": day, "tokens_used": 0, "limit_reached_at": None})

    def tokens(self, table, key, day=None):
        r = self.rows.get((table, key, day or server._ist_today()))
        return r["tokens_used"] if r else 0

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/auth/v1/user":
            self.auth_calls += 1
            token = request.headers.get("authorization", "")[7:]
            uid = TOKENS.get(token)
            return httpx.Response(200, json={"id": uid}) if uid else httpx.Response(401, json={"msg": "bad jwt"})
        params = request.url.params
        self.requests.append((request.method, path, dict(params)))
        if path.startswith("/rest/v1/rpc/"):
            fn = path.rsplit("/", 1)[1]
            if fn in USAGE_RPCS:
                body = json.loads(request.content)
                table, key_param = USAGE_RPCS[fn]
                self.row(table, body[key_param], body["p_date"])["tokens_used"] += body["p_tokens"]
            return httpx.Response(200, json=None)
        table = path.rsplit("/", 1)[1]
        if table in USAGE_KEYS:
            key = params[USAGE_KEYS[table]][3:]  # "eq.<value>"
            dates = params["usage_date"]
            days = dates[4:-1].split(",") if dates.startswith("in.(") else [dates[3:]]
            matches = [r for (t, k, d), r in self.rows.items() if t == table and k == key and d in days]
            if request.method == "PATCH":
                for r in matches:
                    r.update(json.loads(request.content))
                return httpx.Response(204)
            return httpx.Response(200, json=[dict(r) for r in matches])
        return httpx.Response(200, json=[])


def reset():
    server._rate_buckets.clear()
    server._rate_windows.clear()
    server._rate_last_sweep = 0.0
    server._admin_login_attempts.clear()


# Test-only routes, so the limiter can be exercised without a real handler's side effects.
@server.app.post("/__t/account")
async def _t_account(_: None = Depends(server.rate_limit(3, 60, by="account")), uid: str = Depends(server._get_verified_user_id)):
    return {"uid": uid}


@server.app.post("/__t/device")
async def _t_device(_: None = Depends(server.rate_limit(3, 60))):
    return {}


@server.app.get("/__t/item/{item_id}")
async def _t_item(item_id: int, _: None = Depends(server.rate_limit(3, 60))):
    return {}


def codes(client, n, method, path, headers_fn):
    return [client.request(method, path, headers=headers_fn(i)).status_code for i in range(n)]


# ---------------------------------------------------------------- tests
def test_route_table():
    print("\n# every limited route has the approved key and limit")
    expected = {
        ("POST", "/chat"): (15, 60, "account"), ("POST", "/solve"): (15, 60, "account"),
        ("POST", "/title"): (15, 60, "account"), ("POST", "/summarize-answer"): (15, 60, "account"),
        ("POST", "/solve-alternate-method"): (20, 60, "account"),
        ("POST", "/report-question"): (20, 60, "account"), ("POST", "/report-diagram"): (20, 60, "account"),
        ("POST", "/mock-tests/attempts"): (20, 60, "account"), ("POST", "/mock-tests/attempts/{attempt_id}/submit"): (30, 60, "account"),
        ("POST", "/chat-image-upload"): (10, 60, "account"), ("POST", "/chat-pdf-upload"): (10, 60, "account"),
        ("POST", "/session-feedback"): (10, 60, "account"), ("POST", "/account/delete"): (3, 3600, "account"),
        ("POST", "/pyq"): (15, 60, "device"), ("POST", "/diagram-match"): (15, 60, "device"),
        ("POST", "/classify-difficulty"): (30, 60, "device"), ("POST", "/ensure-correct-answer"): (30, 60, "device"),
        ("GET", "/pyq-browse"): (30, 60, "device"), ("GET", "/pyq-chapters"): (30, 60, "device"),
        ("GET", "/diagrams"): (30, 60, "device"), ("GET", "/neet-exam-dates"): (30, 60, "device"),
        ("GET", "/mock-test-questions"): (20, 60, "device"), ("GET", "/mock-tests/{mock_test_id}/questions"): (20, 60, "device"),
        ("POST", "/personalised-test-questions"): (20, 60, "device"), ("POST", "/personalised-catalog-start"): (20, 60, "device"),
        ("POST", "/admin/scan-pdf"): (10, 300, "ip"), ("POST", "/admin/scan-pdf/ocr-job"): (10, 300, "ip"),
        ("POST", "/admin/scan-pdf/estimate"): (20, 300, "ip"), ("POST", "/admin/mock-test-scan"): (5, 300, "ip"),
        ("POST", "/chat-alternate-method"): (20, 60, "device"),
        ("GET", "/guest-usage-status"): (30, 60, "device"),
    }
    found = {}
    for route in server.app.routes:
        if not getattr(route, "dependant", None) or route.path.startswith("/__t/"):
            continue
        for dep in route.dependant.dependencies:
            spec = getattr(dep.call, "rate_limit_spec", None)
            if spec:
                for method in route.methods:
                    found[(method, route.path)] = spec
    missing = {k: v for k, v in expected.items() if found.get(k) != v}
    extra = {k: v for k, v in found.items() if k not in expected}
    check(f"{len(expected)} routes match the approved table", not missing, str({k: (v, found.get(k)) for k, v in missing.items()}))
    check("no limited route outside the table", not extra, str(extra))


def test_accounts_share_an_ip(client, fake):
    print("\n# logged in: limited per account, never by who else is on the IP")
    reset()
    a = codes(client, 4, "POST", "/__t/account", lambda i: at(IP1, AUTH_A))
    check("account A: 3 allowed, 4th is 429", a == [200, 200, 200, 429], str(a))
    b = codes(client, 3, "POST", "/__t/account", lambda i: at(IP1, AUTH_B))
    check("account B on the same IP still gets its full 3", b == [200, 200, 200], str(b))
    r = client.post("/__t/account", headers=at(IP2, AUTH_A))
    check("account A from a second IP shares A's bucket (429)", r.status_code == 429, str(r.status_code))
    check("logged-in requests create no per-IP guest ceiling keys", not [k for k in server._rate_buckets if k.startswith("g:")], str(list(server._rate_buckets)))
    reset()
    before = fake.auth_calls
    client.post("/__t/account", headers=at(IP1, AUTH_A))
    check("one Supabase token check per request (limiter + handler share it)", fake.auth_calls - before == 1, str(fake.auth_calls - before))


def test_devices_share_an_ip(client):
    print("\n# guests: limited per device")
    reset()
    d1 = codes(client, 4, "POST", "/__t/device", lambda i: at(IP1, device=DEV1))
    check("device 1: 3 allowed, 4th is 429", d1 == [200, 200, 200, 429], str(d1))
    d2 = codes(client, 3, "POST", "/__t/device", lambda i: at(IP1, device=DEV2))
    check("device 2 on the same IP still gets its full 3", d2 == [200, 200, 200], str(d2))
    r = client.post("/__t/device", headers=at(IP2, device=DEV1))
    check("device 1 from another IP is still limited (same device bucket)", r.status_code == 429, str(r.status_code))
    reset()
    a = codes(client, 4, "POST", "/__t/device", lambda i: at(IP1, AUTH_A, device=DEV1))
    check("route that doesn't check login: a logged-in student is limited per device", a == [200, 200, 200, 429], str(a))


def test_rotating_device_ids(client):
    print("\n# a script inventing a device id per request stops at the per-IP ceiling (10 x 3 = 30)")
    reset()
    got = codes(client, 31, "POST", "/__t/account", lambda i: at(IP1, device=new_device()))
    check("30 allowed, the 31st is 429", got[:30] == [200] * 30 and got[30] == 429, str(got))
    r = client.post("/__t/account", headers=at(IP2, device=new_device()))
    check("a guest on another IP is unaffected", r.status_code == 200, str(r.status_code))
    r = client.post("/__t/account", headers=at(IP1, AUTH_A, device=new_device()))
    check("a logged-in student on the saturated IP is unaffected", r.status_code == 200, str(r.status_code))


def test_device_id_validation(client):
    print("\n# device id shape")
    request = lambda value: server.Request({"type": "http", "headers": [(b"x-device-id", value.encode())]})
    good = [DEV1, DEV1.upper(), "dev-lq3k2abc-k3j4h5g6f7", "dev-1697000000000-k3j4h5g6f7"]
    bad = ["", "not-a-uuid", "dev-", "dev-ab", "x" * 100, "dev-" + "a" * 61, DEV1 + "0", "abc,ip.eq.1.2.3.4",
           "dev-a)or(1=1", "11111111-1111-4111-8111-11111111111g", " ", "dev-ABC_def"]
    check("website/app shapes accepted (UUIDs lowercased)", [server._device_id(request(v)) for v in good] == [DEV1, DEV1, good[2], good[3]],
          str([server._device_id(request(v)) for v in good]))
    check("everything else treated as no device id", all(server._device_id(request(v)) == "" for v in bad),
          str([v for v in bad if server._device_id(request(v))]))
    reset()
    got = codes(client, 4, "POST", "/__t/device", lambda i: at(IP1, device=f"bogus,{i}"))
    check("malformed ids fall back to the IP bucket (3, then 429)", got == [200, 200, 200, 429], str(got))
    check("...keyed on the IP, not on the junk", any(k == f"ip:{IP1}:/__t/device" for k in server._rate_buckets)
          and not any("bogus" in k for k in server._rate_buckets), str(list(server._rate_buckets)))


def test_no_device_header(client):
    print("\n# old clients (no device id): per IP at the route's own limit, as before")
    reset()
    got = codes(client, 4, "POST", "/__t/device", lambda i: at(IP1))
    check("3 allowed, the 4th is 429", got == [200, 200, 200, 429], str(got))
    check("another IP unaffected", client.post("/__t/device", headers=at(IP2)).status_code == 200)


def test_route_template(client):
    print("\n# buckets are per route template, not per URL")
    reset()
    got = [client.get(f"/__t/item/{i}", headers=at(IP1, device=DEV1)).status_code for i in range(1, 5)]
    check("/__t/item/1..4 share one bucket (4th is 429)", got == [200, 200, 200, 429], str(got))
    reset()
    client.get("/mock-tests/7/questions", headers=at(IP1, device=DEV1))
    check("real route keyed on '/mock-tests/{mock_test_id}/questions'",
          f"d:{DEV1}:/mock-tests/{{mock_test_id}}/questions" in server._rate_buckets, str(list(server._rate_buckets)))


def test_ipv6(client):
    print("\n# IPv6: one /64 is one IP")
    k = server._ip_key
    check("same /64 -> same key", k("2001:db8:1:2::1") == k("2001:db8:1:2:ffff:ee:dd:9") == "2001:db8:1:2::/64")
    check("different /64 -> different key", k("2001:db8:1:3::1") != k("2001:db8:1:2::1"))
    check("IPv4 and IPv4-mapped IPv6 keyed as the IPv4", k("203.0.113.10") == k("::ffff:203.0.113.10") == "203.0.113.10")
    check("non-address peer passes through", k("testclient") == "testclient")
    reset()
    got = [client.post("/__t/device", headers=at(f"2001:db8:1:2::{i + 1}")).status_code for i in range(4)]
    check("4 addresses inside one /64, no device id: 4th is 429", got == [200, 200, 200, 429], str(got))
    check("a different /64 is unaffected", client.post("/__t/device", headers=at("2001:db8:1:3::1")).status_code == 200)


def test_otp(client):
    print("\n# OTP: per phone (3 / 10 min, 10 / day) and loosely per IP (60 / 10 min, 300 / day)")
    sent = []

    class Ok:
        status_code = 200

        def json(self):
            return {}

    original_post, original_time = server.http_requests.post, server.time
    server.http_requests.post = lambda url, **kw: (sent.append(kw["json"]["phone"]), Ok())[1]
    server.time = clock = FakeClock()
    try:
        reset()
        send = lambda phone, ip: client.post("/auth/send-otp", json={"phone": phone}, headers=at(ip))
        phone = "+919876500001"
        got = [send(phone, f"203.0.113.{i + 1}").status_code for i in range(4)]
        r = send(phone, "203.0.113.99")
        check("same phone from 4 different IPs: 3 sent, then 429", got == [200, 200, 200, 429], str(got))
        check("...with the per-number message", "this number" in r.json().get("detail", ""), r.text)
        total = 3
        for _ in range(3):  # three more 10-minute windows: 3 + 3 + 1 more before the daily 10
            clock.now += 601
            total += sum(send(phone, IP2).status_code == 200 for _ in range(3))
        clock.now += 601
        r = send(phone, IP2)
        check("10 sends per number per day, the 11th is 429 even in a fresh 10-minute window",
              total == 10 and r.status_code == 429, f"total={total} {r.status_code}")
        reset()
        got = [send(f"+91987650{i:04d}", IP1).status_code for i in range(30)]
        check("30 different students on one IP all get codes (was: 4th blocked)", got == [200] * 30, str(got))
        got = [send(f"+91987651{i:04d}", IP1).status_code for i in range(31)]
        check("61st send from one IP within 10 minutes is 429", got[:30] == [200] * 30 and got[30] == 429, str(got[-3:]))
        check("...with the generic message (not blaming the number)", "this number" not in send("+919876529999", IP1).json().get("detail", ""))
        check("another IP still sends", send("+919876530000", IP2).status_code == 200)
        reset()
        day = []
        for window in range(6):
            day += [send(f"+9198766{window}{i:04d}", IP1).status_code for i in range(60)]
            clock.now += 601
        check("300 sends per IP per day, then 429", day[:300] == [200] * 300 and day[300] == 429, str(day.count(200)))
        before = len(sent)
        send("+919876600000", IP1)
        check("a blocked request never reaches Supabase", len(sent) == before)
        r = client.post("/auth/send-otp", json={"phone": "98765"}, headers=at(IP2))
        check("invalid number is still a 400, not counted", r.status_code == 400 and not [k for k in server._rate_buckets if IP2 in k and k.startswith("otp-")])
    finally:
        server.http_requests.post, server.time = original_post, original_time
        reset()


def test_admin(client):
    print("\n# admin login: strict per IP, unchanged")
    reset()
    got = [client.post("/admin/verify", headers=at(IP1, [("X-Admin-Key", "wrong")])).status_code for _ in range(4)]
    check("3 wrong keys -> 401 x3, 4th is 429", got == [401, 401, 401, 429], str(got))
    r = client.post("/admin/verify", headers=at(IP1, [("X-Admin-Key", server.ADMIN_PASSWORD)]))
    check("right key refused from the locked IP during the cooldown", r.status_code == 429, str(r.status_code))
    r = client.post("/admin/verify", headers=at(IP2, [("X-Admin-Key", server.ADMIN_PASSWORD)]))
    check("another IP unaffected", r.status_code == 200, str(r.status_code))
    scope = {"type": "http", "path": "/", "query_string": b"", "headers": [(b"x-real-ip", IP1.encode()), (b"x-device-id", DEV1.encode())]}
    keys = server._rate_limit_keys(server.Request(scope),
                                   10, 300, UID_A, "ip")
    check("admin scan routes key on the IP even with an account and a device id", keys == [(f"ip:{IP1}:/", 10, 300)], str(keys))
    reset()


def test_backstop(client, fake):
    print("\n# per-IP backstop runs before the Supabase token check")
    original = server.RATE_BACKSTOP_PER_MINUTE
    server.RATE_BACKSTOP_PER_MINUTE = 20
    try:
        reset()
        before = fake.auth_calls
        got = [client.post("/__t/account", headers=at(IP1, [("Authorization", f"Bearer junk-{i}")], device=new_device())).status_code for i in range(25)]
        check("junk-token flood: 20 through, then 429", got.count(429) >= 5 and got[20:] == [429] * 5, str(got))
        check("Supabase asked only for the 20 that passed the backstop", fake.auth_calls - before == 20, str(fake.auth_calls - before))
        check("another IP unaffected", client.post("/__t/account", headers=at(IP2, AUTH_A)).status_code == 200)
    finally:
        server.RATE_BACKSTOP_PER_MINUTE = original
        reset()


def test_sweep_and_cap(client):
    print("\n# memory: idle buckets swept, bucket count capped")
    original_time, original_cap = server.time, server.RATE_BUCKET_CAP
    server.time = clock = FakeClock()
    try:
        reset()
        for _ in range(5):
            client.post("/__t/device", headers=at(IP1, device=new_device()))
        check("buckets created", len(server._rate_buckets) >= 5)
        clock.now += 121
        client.post("/__t/device", headers=at(IP2, device=DEV1))
        check("after the window + sweep interval only the fresh request's buckets remain",
              set(server._rate_buckets) == {f"all:{IP2}", f"g:{IP2}:/__t/device", f"d:{DEV1}:/__t/device"}, str(list(server._rate_buckets)))
        server.RATE_BUCKET_CAP = len(server._rate_buckets) + 1
        client.post("/__t/device", headers=at(IP1, device=DEV2))
        client.post("/__t/device", headers=at(IP1, device=new_device()))
        check("at the cap, a new device id falls back to the IP bucket",
              f"ip:{IP1}:/__t/device" in server._rate_buckets and sum(k.startswith("d:") for k in server._rate_buckets) == 1,
              str(list(server._rate_buckets)))
        r = client.post("/__t/device", headers=at(IP2, device=DEV1))
        check("a device already tracked keeps its own bucket", r.status_code == 200 and len(server._rate_buckets[f"d:{DEV1}:/__t/device"]) == 2)
    finally:
        server.time, server.RATE_BUCKET_CAP = original_time, original_cap
        reset()


TOKENS_PER_ANSWER = 6000


async def fake_stream_response(*args):
    """Stands in for server.stream_response: bills one answer the way the real streaming helpers
    do -- from a task spawned mid-stream -- so the guest's device must reach it via the context."""
    user_id, ip = args[7], args[10]
    task = asyncio.create_task(server.log_token_usage(user_id, TOKENS_PER_ANSWER, ip))
    yield "ok"
    await task


def test_guest_budget(client, fake):
    print("\n# guest daily budget: per device, plus 280,000 tokens per IP per IST day")
    original_stream = server.stream_response
    server.stream_response = fake_stream_response
    today = server._ist_today()
    ask = lambda *headers: client.post("/chat", json={"text": "what is osmosis"}, headers=at(*headers[1:], device=headers[0]) if headers[0] else at(*headers[1:]))
    try:
        reset()
        dev_a, dev_b, dev_c = new_device(), new_device(), new_device()
        got = [ask(dev_a, IP1).status_code for _ in range(4)]
        check("device A: 3 answers (18,000 tokens > 14,000), the 4th is 402", got == [200, 200, 200, 402], str(got))
        r = ask(dev_a, IP1)
        check("...with the log-in message and a 24h retry", r.json()["detail"]["message"].startswith("Guest limit reached")
              and 86000 < r.json()["detail"]["retry_after_seconds"] <= 86400, r.text)
        check("tokens billed to device A's row from inside the stream", fake.tokens("guest_device_usage_log", dev_a) == 18000,
              str(fake.tokens("guest_device_usage_log", dev_a)))
        check("device B on the same IP still gets answers", ask(dev_b, IP1).status_code == 200)
        check("the IP row adds up every guest on it (A + B)", fake.tokens("guest_usage_log", IP1) == 24000, str(fake.tokens("guest_usage_log", IP1)))

        fake.row("guest_usage_log", IP1, today)["tokens_used"] = server.GUEST_IP_DAILY_CEILING
        r = ask(dev_c, IP1)
        secs = server._seconds_until_ist_midnight()
        check("IP at 280,000 today: a brand-new device on it is 402 until IST midnight",
              r.status_code == 402 and abs(r.json()["detail"]["retry_after_seconds"] - secs) <= 2, r.text)
        check("a new device on another IP is unaffected", ask(new_device(), IP2).status_code == 200)
        r = client.post("/chat", json={"text": "what is osmosis"}, headers=at(IP1, AUTH_A, device=dev_c))
        check("a logged-in student on that IP is unaffected (own account budget)", r.status_code == 200, r.text)
        check("...and billed to the account, not the guest rows", fake.tokens("usage_log", UID_A) == TOKENS_PER_ANSWER
              and fake.tokens("guest_device_usage_log", dev_c) == 0)

        ip3 = "192.0.2.33"
        got = [ask(None, ip3).status_code for _ in range(4)]
        check("no device id (old client): the IP's own 14,000 budget, as before", got == [200, 200, 200, 402], str(got))
        check("...billed to the IP row only", fake.tokens("guest_usage_log", ip3) == 18000
              and not [k for k in fake.rows if k[0] == "guest_device_usage_log" and k[1] == ip3])

        print("\n# /guest-usage-status")
        s = client.get("/guest-usage-status", headers=at(IP2, device=dev_a)).json()
        check("device A: its own usage and cooldown (from any IP)", s["tokens_used"] == 18000 and s["retry_after_seconds"] > 86000, str(s))
        s = client.get("/guest-usage-status", headers=at(IP1, device=new_device())).json()
        check("fresh device on an IP past the ceiling: shown as used up until midnight",
              s["tokens_used"] == server.DAILY_TOKEN_BUDGET_GUEST and abs(s["retry_after_seconds"] - secs) <= 2, str(s))
        s = client.get("/guest-usage-status", headers=at(IP2, device=new_device())).json()
        check("fresh device on a normal IP: nothing used", s == {"tokens_used": 0, "budget": 14000, "retry_after_seconds": None}, str(s))
        s = client.get("/guest-usage-status", headers=at(ip3)).json()
        check("no device id: the IP row", s["tokens_used"] == 18000 and s["retry_after_seconds"] > 86000, str(s))

        print("\n# /merge-guest-usage")
        before_b = fake.tokens("guest_device_usage_log", dev_b)
        r = client.post("/merge-guest-usage", json={"user_id": "ignored"}, headers=at(IP1, AUTH_B, device=dev_a))
        check("merges only this device's usage into the account", r.json() == {"merged": 18000}
              and fake.tokens("usage_log", UID_B) == 18000, f"{r.text} {fake.tokens('usage_log', UID_B)}")
        check("device row zeroed, other devices on the IP untouched, IP total kept",
              fake.tokens("guest_device_usage_log", dev_a) == 0 and fake.tokens("guest_device_usage_log", dev_b) == before_b
              and fake.tokens("guest_usage_log", IP1) == server.GUEST_IP_DAILY_CEILING)
        r = client.post("/merge-guest-usage", json={"user_id": "x"}, headers=at(IP1, device=dev_b))
        check("no login token -> 401", r.status_code == 401)
    finally:
        server.stream_response = original_stream
        reset()


def test_cors(client):
    print("\n# CORS preflight allows X-Device-Id from the website")
    r = client.options("/chat", headers={"Origin": "https://neet-ai-jet.vercel.app", "Access-Control-Request-Method": "POST",
                                         "Access-Control-Request-Headers": "content-type,authorization,x-device-id"})
    allowed = r.headers.get("access-control-allow-headers", "").lower()
    check("preflight 200 and x-device-id allowed", r.status_code == 200 and "x-device-id" in allowed, f"{r.status_code} {allowed}")


if __name__ == "__main__":
    fake = FakeSupabase()
    server.async_client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handle))
    tc = TestClient(server.app)  # no `with`: skips startup hooks, nothing here needs them
    test_route_table()
    test_accounts_share_an_ip(tc, fake)
    test_devices_share_an_ip(tc)
    test_rotating_device_ids(tc)
    test_device_id_validation(tc)
    test_no_device_header(tc)
    test_route_template(tc)
    test_ipv6(tc)
    test_otp(tc)
    test_admin(tc)
    test_backstop(tc, fake)
    test_sweep_and_cap(tc)
    test_guest_budget(tc, fake)
    test_cors(tc)
    print(f"\nTOTAL: {len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
