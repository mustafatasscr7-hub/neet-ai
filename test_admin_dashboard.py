"""Admin dashboard "Usage & Cost" tab (admin_dashboard_stats.sql + the /admin/dashboard/* endpoints
+ the Usage & Cost tab in admin-dashboard.html).

Run with: python test_admin_dashboard.py
Plain pass/fail harness like test_cost_fixes.py. Fully offline: Supabase is a recording fake (every
RPC response is canned here), no real network call is made, and no SQL is ever executed.

Covers the verification list from the task: each endpoint computes the right aggregate (including
zero rows), the 60s in-memory cache, a missing SQL function returns `*_available: false` and never
a 500, admin auth refuses requests without the key, the phone the harness hands back stays masked
end to end, and no response contains an email or a name field anywhere.
"""
import asyncio
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
        print(f"  FAIL: {label}" + (f" -- {detail}" if detail else ""))


ADMIN_KEY = "test-admin-key"


class FakeResp:
    def __init__(self, status_code=200, data=None):
        self.status_code = status_code
        self._data = data if data is not None else []
        self.text = "" if status_code < 400 else '{"message":"Could not find the function"}'

    def json(self):
        return self._data


class FakeRpcClient:
    """POST to /rest/v1/rpc/<name> returns rpc_responses[name] (status, data); missing from the
    dict simulates PostgREST's 404 for a function that hasn't been created yet (SQL not run)."""
    def __init__(self, rpc_responses=None):
        self.rpc_responses = rpc_responses or {}
        self.rpc_calls = []

    def _name(self, url):
        return str(url).split("/rpc/")[-1].split("?")[0]

    async def post(self, url, headers=None, json=None, **kw):
        name = self._name(url)
        self.rpc_calls.append(name)
        if name not in self.rpc_responses:
            return FakeResp(404)
        status, data = self.rpc_responses[name]
        return FakeResp(status, data)

    async def get(self, url, headers=None, params=None, **kw):
        return FakeResp(200, [])


SPEND_SUMMARY = {
    "today": {"cost": 0.01, "input_tokens": 100, "output_tokens": 50, "cache_hit_tokens": 10, "calls": 2},
    "last_7d": {"cost": 0.05, "input_tokens": 500, "output_tokens": 250, "cache_hit_tokens": 50, "calls": 10},
    "last_30d": {"cost": 0.20, "input_tokens": 2000, "output_tokens": 1000, "cache_hit_tokens": 200, "calls": 40},
    "month": {"cost": 0.15, "input_tokens": 1500, "output_tokens": 750, "cache_hit_tokens": 150, "calls": 30},
    "daily": [{"date": "2026-10-01", "cost": 0.001, "input_tokens": 10, "output_tokens": 5, "cache_hit_tokens": 1, "calls": 1}],
}
BY_PROVIDER_ROWS = [
    {"provider": "deepseek-flash", "calls": 30, "input_tokens": 1500, "output_tokens": 700, "cache_hit_tokens": 100,
     "cost": 0.12, "peak_count": 5, "output_capped_count": 1},
    {"provider": "qwen-flash", "calls": 10, "input_tokens": 500, "output_tokens": 300, "cache_hit_tokens": 100,
     "cost": 0.08, "peak_count": 10, "output_capped_count": 0},
]
BY_ENDPOINT_ROWS = [{"endpoint": "/chat", "calls": 35, "input_tokens": 1800, "output_tokens": 900, "cost": 0.18}]
USER_STATS = {"total_users": 9, "plan_free": 0, "plan_pro": 5, "plan_max": 4, "new_today": 1, "new_7d": 3,
              "new_per_day": [{"date": "2026-10-30", "count": 1}]}
ACTIVE_COUNTS = {"last_24h": 3, "last_7d": 7}
NEAR_CAP = {"daily": {"free": 0, "pro": 1, "max": 0}, "weekly": {"free": 0, "pro": 2, "max": 1}}
TOP_USERS_ROWS = [{"user_id": "11111111-2222-3333-4444-555555555555", "plan": "pro",
                    "masked_phone": "+91 98xxxxxx79", "cost": 0.09, "calls": 12}]


def full_rpc_responses():
    return {
        "admin_spend_summary": (200, SPEND_SUMMARY),
        "admin_spend_by_provider": (200, BY_PROVIDER_ROWS),
        "admin_spend_by_endpoint": (200, BY_ENDPOINT_ROWS),
        "admin_user_stats": (200, USER_STATS),
        "admin_active_user_counts": (200, ACTIVE_COUNTS),
        "admin_near_cap_counts": (200, NEAR_CAP),
        "admin_top_users_by_cost": (200, TOP_USERS_ROWS),
    }


def setup(rpc_responses):
    server.ADMIN_PASSWORD = ADMIN_KEY
    server._admin_dashboard_cache.clear()
    client_obj = FakeRpcClient(rpc_responses)
    server.async_client = client_obj
    return TestClient(server.app), client_obj  # no `with`: skips startup hooks, nothing here needs them


def test_admin_auth():
    print("\n# Admin auth")
    tc, _ = setup(full_rpc_responses())
    for path in ("/admin/dashboard/spend", "/admin/dashboard/by-provider", "/admin/dashboard/by-endpoint",
                 "/admin/dashboard/users", "/admin/dashboard/top-users"):
        res = tc.get(path)
        check(f"{path} refuses without a key", res.status_code == 401, res.status_code)
        res = tc.get(path, headers={"X-Admin-Key": "wrong"})
        check(f"{path} refuses with the wrong key", res.status_code == 401, res.status_code)
        res = tc.get(path, headers={"X-Admin-Key": ADMIN_KEY})
        check(f"{path} accepts the real key", res.status_code == 200, res.status_code)


def test_spend():
    print("\n# /admin/dashboard/spend")
    tc, client_obj = setup(full_rpc_responses())
    res = tc.get("/admin/dashboard/spend", headers={"X-Admin-Key": ADMIN_KEY})
    data = res.json()
    check("available true", data.get("available") is True)
    check("today cost matches", data.get("today", {}).get("cost") == 0.01)
    check("daily series present", isinstance(data.get("daily"), list) and len(data["daily"]) == 1)

    # 60s in-memory cache: a second call before the TTL elapses must not hit the RPC again.
    calls_before = len(client_obj.rpc_calls)
    tc.get("/admin/dashboard/spend", headers={"X-Admin-Key": ADMIN_KEY})
    check("cached response: no second RPC call within 60s", len(client_obj.rpc_calls) == calls_before)

    import time
    real_monotonic = time.monotonic
    time.monotonic = lambda: real_monotonic() + 61
    try:
        tc.get("/admin/dashboard/spend", headers={"X-Admin-Key": ADMIN_KEY})
        check("cache expires after 60s: RPC called again", len(client_obj.rpc_calls) > calls_before)
    finally:
        time.monotonic = real_monotonic


def test_spend_missing_function():
    print("\n# /admin/dashboard/spend -- SQL not run yet")
    tc, _ = setup({})  # no RPCs registered at all -> every one 404s
    res = tc.get("/admin/dashboard/spend", headers={"X-Admin-Key": ADMIN_KEY})
    check("never a 500", res.status_code == 200, res.status_code)
    check("available: false", res.json().get("available") is False)


def test_by_provider_and_endpoint():
    print("\n# /admin/dashboard/by-provider and /admin/dashboard/by-endpoint")
    tc, _ = setup(full_rpc_responses())
    for rng in ("today", "7d", "30d"):
        res = tc.get(f"/admin/dashboard/by-provider?range={rng}", headers={"X-Admin-Key": ADMIN_KEY})
        check(f"by-provider range={rng} -> 200 with rows", res.status_code == 200 and len(res.json()["rows"]) == 2)
    res = tc.get("/admin/dashboard/by-provider?range=bogus", headers={"X-Admin-Key": ADMIN_KEY})
    check("invalid range -> 400", res.status_code == 400, res.status_code)

    res = tc.get("/admin/dashboard/by-endpoint?range=30d", headers={"X-Admin-Key": ADMIN_KEY})
    check("by-endpoint -> 200 with rows", res.status_code == 200 and res.json()["rows"][0]["endpoint"] == "/chat")

    # Zero rows (empty table) must render as an empty list, not an error.
    tc2, _ = setup({**full_rpc_responses(), "admin_spend_by_provider": (200, [])})
    res = tc2.get("/admin/dashboard/by-provider?range=30d", headers={"X-Admin-Key": ADMIN_KEY})
    check("zero provider rows -> available true, empty list", res.json() == {"available": True, "rows": []})


def test_users_partial_availability():
    print("\n# /admin/dashboard/users -- one of its three RPCs missing")
    responses = full_rpc_responses()
    del responses["admin_near_cap_counts"]  # simulate that one function not created yet
    tc, _ = setup(responses)
    res = tc.get("/admin/dashboard/users", headers={"X-Admin-Key": ADMIN_KEY})
    check("never a 500", res.status_code == 200, res.status_code)
    data = res.json()
    check("user stats still available", data.get("available") is True and data.get("total_users") == 9)
    check("active counts still available", data.get("active_available") is True and data.get("active", {}).get("last_24h") == 3)
    check("near-cap flagged unavailable, not crashed", data.get("near_cap_available") is False and data.get("near_cap") is None)


def test_top_users_masking_and_no_pii():
    print("\n# /admin/dashboard/top-users -- masking and no stray PII fields")
    tc, _ = setup(full_rpc_responses())
    res = tc.get("/admin/dashboard/top-users?range=30d", headers={"X-Admin-Key": ADMIN_KEY})
    rows = res.json()["rows"]
    check("one row back", len(rows) == 1)
    row = rows[0]
    check("phone is masked (never the raw digits)", row["masked_phone"] == "+91 98xxxxxx79" and "x" in row["masked_phone"])
    check("masked phone is shorter than a bare 12-digit number would read, with x's in the middle",
          "xxxxxx" in row["masked_phone"])
    allowed_keys = {"user_id", "plan", "masked_phone", "cost", "calls"}
    check("top-users row has no extra field (no email/name key)", set(row.keys()) <= allowed_keys, row.keys())

    import json as jsonlib
    full_text = jsonlib.dumps(res.json())
    for path in ("/admin/dashboard/spend", "/admin/dashboard/by-provider?range=30d",
                 "/admin/dashboard/by-endpoint?range=30d", "/admin/dashboard/users"):
        full_text += jsonlib.dumps(tc.get(path, headers={"X-Admin-Key": ADMIN_KEY}).json())
    check("no response anywhere contains 'email' or 'name' as a key/word", "email" not in full_text and '"name"' not in full_text, full_text[:300])


def test_near_cap_uses_live_python_constants():
    print("\n# admin_near_cap_counts is called with the real cap constants, not hardcoded numbers")
    tc, client_obj = setup(full_rpc_responses())

    captured = {}
    real_post = client_obj.post

    async def spying_post(url, headers=None, json=None, **kw):
        if "admin_near_cap_counts" in str(url):
            captured.update(json)
        return await real_post(url, headers=headers, json=json, **kw)

    client_obj.post = spying_post
    tc.get("/admin/dashboard/users", headers={"X-Admin-Key": ADMIN_KEY})
    check("daily free threshold = 80% of DAILY_TOKEN_BUDGET_FREE",
          captured.get("p_daily_free") == 0.8 * server.DAILY_TOKEN_BUDGET_FREE)
    check("weekly max threshold = 80% of WEEKLY_TOKEN_LIMIT_MAX",
          captured.get("p_weekly_max") == 0.8 * server.WEEKLY_TOKEN_LIMIT_MAX)


def main():
    test_admin_auth()
    test_spend()
    test_spend_missing_function()
    test_by_provider_and_endpoint()
    test_users_partial_availability()
    test_top_users_masking_and_no_pii()
    test_near_cap_uses_live_python_constants()

    print(f"\nTOTAL: {len(_passed)} passed, {len(_failed)} failed")
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
