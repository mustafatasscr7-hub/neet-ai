"""Regression test for server.py's _client_ip (client IP behind Railway's edge).

Run with: python test_client_ip.py
No pytest dependency -- same plain pass/fail harness as test_student_name_sanitization.py. Fully
offline: requests go through FastAPI's in-process TestClient, and the one handler that would read
Supabase (/guest-usage-status) has its lookup replaced.

What the headers look like in production (measured through Railway's edge 2026-10-15): the edge
strips whatever X-Forwarded-For / X-Real-IP the client sent and delivers
  X-Real-IP: <client>        X-Forwarded-For: <client>, <edge>
In Aug 2024 the edge APPENDED to the client's X-Forwarded-For instead ("<spoofed>, <client>"),
which the old "leftmost entry" rule would have trusted. These tests feed spoofed values in both
shapes and check that the admin lockout, the per-IP rate limits and the guest budget key all stay
on the real client.
"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server
from fastapi.testclient import TestClient
from starlette.requests import Request

_passed = []
_failed = []


def check(label, condition, detail=""):
    if condition:
        _passed.append(label)
        print(f"  PASS: {label}")
    else:
        _failed.append(label)
        print(f"  FAIL: {label}  {detail}")


REAL = "203.0.113.50"       # the student's real address, as Railway's edge reports it
OTHER = "198.51.100.20"     # a second, different real client
EDGE = "79.127.178.82"      # Railway edge hop, appended after the client
PEER = "100.64.0.7"         # socket peer inside Railway's network
SPOOFS = ["1.1.1.1", "8.8.8.8", "9.9.9.9", "4.4.4.4", "5.5.5.5", "6.6.6.6", "7.7.7.7",
          "11.11.11.11", "12.12.12.12", "13.13.13.13", "14.14.14.14", "15.15.15.15"]


def make_request(headers, client=(PEER, 51234)):
    scope = {"type": "http", "method": "GET", "path": "/", "query_string": b"",
             "headers": [(k.lower().encode(), v.encode()) for k, v in headers], "client": client}
    return Request(scope)


def ip_for(headers, count=2, client=(PEER, 51234)):
    saved = server.TRUSTED_PROXY_COUNT
    server.TRUSTED_PROXY_COUNT = count
    try:
        return server._client_ip(make_request(headers, client))
    finally:
        server.TRUSTED_PROXY_COUNT = saved


def old_leftmost_rule(headers):
    """The helper as it was before this fix, for comparison."""
    xff = dict((k.lower(), v) for k, v in headers).get("x-forwarded-for")
    return xff.split(",")[0].strip() if xff else PEER


def railway_today(spoof=None):
    # Edge strips the client's copies; the spoof never arrives.
    return [("X-Real-IP", REAL), ("X-Forwarded-For", f"{REAL}, {EDGE}")]


def railway_append(spoof, real=REAL):
    # 2024-style edge: client's own X-Forwarded-For kept in front of what the edge appends.
    return [("X-Real-IP", real), ("X-Forwarded-For", f"{spoof}, {real}, {EDGE}")]


def test_helper_positions():
    print("\n# _client_ip: which value is trusted")
    check("Railway today: X-Real-IP + '<client>, <edge>' -> client", ip_for(railway_today()) == REAL)
    check("old rule was also right for today's shape (not exploitable now)", old_leftmost_rule(railway_today()) == REAL)
    check("append shape, one spoof -> client", ip_for(railway_append("1.1.1.1")) == REAL)
    check("append shape, several spoofs -> client",
          ip_for([("X-Real-IP", REAL), ("X-Forwarded-For", f"1.1.1.1, 2.2.2.2, 3.3.3.3, {REAL}, {EDGE}")]) == REAL)
    check("old rule WOULD trust the spoof in the append shape", old_leftmost_rule(railway_append("1.1.1.1")) == "1.1.1.1")
    check("no X-Real-IP: X-Forwarded-For counted from the right, spoofs ignored",
          ip_for([("X-Forwarded-For", f"1.1.1.1, 2.2.2.2, {REAL}, {EDGE}")]) == REAL)
    check("X-Forwarded-For split across several header lines is joined before counting",
          ip_for([("X-Forwarded-For", "1.1.1.1"), ("X-Forwarded-For", f"{REAL}, {EDGE}")]) == REAL)
    check("two X-Real-IP lines (the 2024 duplication bug): the proxy's later one wins",
          ip_for([("X-Real-IP", "1.1.1.1"), ("X-Real-IP", REAL)]) == REAL)
    check("garbage X-Real-IP falls back to X-Forwarded-For",
          ip_for([("X-Real-IP", "unknown"), ("X-Forwarded-For", f"1.1.1.1, {REAL}, {EDGE}")]) == REAL)
    check("too few X-Forwarded-For entries -> socket peer, never the client-made entry",
          ip_for([("X-Forwarded-For", "1.1.1.1")]) == PEER)
    check("no forwarding headers -> socket peer", ip_for([]) == PEER)
    check("Forwarded / True-Client-IP / CF-Connecting-IP are never read",
          ip_for([("Forwarded", "for=1.1.1.1"), ("True-Client-IP", "2.2.2.2"), ("CF-Connecting-IP", "3.3.3.3")]) == PEER)
    check("TRUSTED_PROXY_COUNT=0 ignores every header (local dev)",
          ip_for(railway_append("1.1.1.1"), count=0, client=("127.0.0.1", 5000)) == "127.0.0.1")
    check("TRUSTED_PROXY_COUNT=1 takes the last X-Forwarded-For entry",
          ip_for([("X-Forwarded-For", f"1.1.1.1, {REAL}")], count=1) == REAL)
    check("TRUSTED_PROXY_COUNT=3 takes the third from the right",
          ip_for([("X-Forwarded-For", f"1.1.1.1, {REAL}, 10.0.0.1, {EDGE}")], count=3) == REAL)
    check("no client on the scope -> 'unknown'", ip_for([], client=None) == "unknown")

    print("\n# _parse_ip normalisation")
    for raw, want in [(" 203.0.113.50 ", REAL), ("203.0.113.50:4431", REAL), ("[2001:DB8:0::1]:443", "2001:db8::1"),
                      ("2001:db8:0:0::1", "2001:db8::1"), ('"203.0.113.50"', REAL), ("unknown", None), ("", None),
                      ("1.2.3", None), ("<script>", None)]:
        check(f"_parse_ip({raw!r}) -> {want!r}", server._parse_ip(raw) == want, repr(server._parse_ip(raw)))

    print("\n# TRUSTED_PROXY_COUNT env parsing")
    saved = os.environ.get("TRUSTED_PROXY_COUNT")
    try:
        for raw, want in [(None, 2), ("0", 0), ("1", 1), ("3", 3), ("-4", 0), ("two", 2), ("", 2)]:
            if raw is None:
                os.environ.pop("TRUSTED_PROXY_COUNT", None)
            else:
                os.environ["TRUSTED_PROXY_COUNT"] = raw
            check(f"TRUSTED_PROXY_COUNT={raw!r} -> {want}", server._trusted_proxy_count() == want)
    finally:
        if saved is None:
            os.environ.pop("TRUSTED_PROXY_COUNT", None)
        else:
            os.environ["TRUSTED_PROXY_COUNT"] = saved


def test_admin_lockout(client):
    print("\n# admin 3-strike lockout keys on the real client, not the spoofed entry")
    for shape_name, shape in [("append shape (X-Real-IP + XFF)", railway_append),
                              ("XFF only, no X-Real-IP", lambda s, real=REAL: [("X-Forwarded-For", f"{s}, {real}, {EDGE}")])]:
        server._admin_login_attempts.clear()
        codes = [client.post("/admin/verify", headers=shape(SPOOFS[i]) + [("X-Admin-Key", "wrong")]).status_code for i in range(3)]
        check(f"{shape_name}: 3 wrong keys, a new spoof each time -> 401 x3", codes == [401, 401, 401], str(codes))
        r = client.post("/admin/verify", headers=shape(SPOOFS[3]) + [("X-Admin-Key", "wrong")])
        check(f"{shape_name}: 4th attempt with yet another spoof is locked out (429)", r.status_code == 429, f"{r.status_code} {r.text}")
        r = client.post("/admin/verify", headers=shape(SPOOFS[4]) + [("X-Admin-Key", server.ADMIN_PASSWORD)])
        check(f"{shape_name}: even the right key is refused for that client while locked", r.status_code == 429, str(r.status_code))
        check(f"{shape_name}: only the real IP was recorded", set(server._admin_login_attempts) == {REAL}, str(server._admin_login_attempts))
        r = client.post("/admin/verify", headers=shape(SPOOFS[5], real=OTHER) + [("X-Admin-Key", server.ADMIN_PASSWORD)])
        check(f"{shape_name}: a different real client is unaffected", r.status_code == 200, f"{r.status_code} {r.text}")
    server._admin_login_attempts.clear()


def test_rate_limit(client):
    print("\n# no-device fallback rate limit (POST /session-feedback: 10 per 60s) keys on the real client")
    server._rate_buckets.clear()
    body = {"rating": "good", "ai_reply_count": 5}
    codes = [client.post("/session-feedback", json=body, headers=railway_append(SPOOFS[i])).status_code for i in range(10)]
    check("10 requests, a new spoofed entry each time -> all reach the handler (401, no token)", codes == [401] * 10, str(codes))
    r = client.post("/session-feedback", json=body, headers=railway_append(SPOOFS[10]))
    check("11th with a fresh spoof is rate limited (429)", r.status_code == 429, f"{r.status_code} {r.text}")
    r = client.post("/session-feedback", json=body, headers=[("X-Real-IP", REAL), ("X-Forwarded-For", "1.1.1.1, 2.2.2.2, 3.3.3.3")])
    check("several spoofed entries and a short chain: still the same bucket (429)", r.status_code == 429, str(r.status_code))
    expected = {f"all:{REAL}", f"g:{REAL}:/session-feedback", f"ip:{REAL}:/session-feedback"}
    check("buckets keyed on the real IP only", set(server._rate_buckets) == expected, str(list(server._rate_buckets)))
    r = client.post("/session-feedback", json=body, headers=railway_append(SPOOFS[11], real=OTHER))
    check("a different real client still gets through", r.status_code == 401, str(r.status_code))
    server._rate_buckets.clear()


def test_guest_budget_key(client):
    print("\n# guest daily budget (guest_usage_log) keys on the real client")
    seen = []
    original_fetch, original_cooldown = server._fetch_usage_rows, server._active_cooldown_seconds

    async def fake_fetch(table, column, value):
        seen.append((table, column, value))
        return "2026-10-15", "2026-10-14", {}

    server._fetch_usage_rows = fake_fetch
    server._active_cooldown_seconds = lambda *a, **k: None
    try:
        for i in range(4):
            client.get("/guest-usage-status", headers=railway_append(SPOOFS[i]))
    finally:
        server._fetch_usage_rows, server._active_cooldown_seconds = original_fetch, original_cooldown
    check("4 guest lookups with 4 different spoofs all use the real IP",
          [v for _, _, v in seen] == [REAL] * 4 and all(t == "guest_usage_log" for t, _, _ in seen), str(seen))


if __name__ == "__main__":
    test_helper_positions()
    tc = TestClient(server.app)  # no `with`: skips startup hooks, nothing here needs them
    test_admin_lockout(tc)
    test_rate_limit(tc)
    test_guest_budget_key(tc)
    print(f"\nTOTAL: {len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
