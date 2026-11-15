"""Tests for NCERT_V2_ENABLED's tolerant parsing (_parse_bool_env in server.py) and the
/admin/retrieval-status diagnostic endpoint. Fully offline: no network, no real Supabase/
DeepSeek client touched -- the endpoint itself does no I/O at all.

Run with: python test_ncert_v2_tolerant_flag_and_status.py
Plain pass/fail harness like test_admin_ncert_content.py.
"""
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
CONTRACTOR_KEY = "test-contractor-key"

# (raw env value, expected result) -- covers unset, case, whitespace, one layer of wrapping
# quotes (the Railway env-UI misconfig this was built to tolerate), and plain garbage.
PARSE_TABLE = [
    (None, False),
    ("", False),
    ("true", True),
    ("True", True),
    ("TRUE", True),
    ("  true  ", True),
    ("1", True),
    ("yes", True),
    ("YES", True),
    ("on", True),
    ("ON", True),
    ('"true"', True),
    ("'true'", True),
    ('"TRUE"', True),
    (' "true" ', True),
    ("false", False),
    ("0", False),
    ("no", False),
    ("off", False),
    ("garbage", False),
    ('"false"', False),
    ('""', False),
    ("'", False),
    ("'true", False),  # mismatched/unclosed quote -- deliberately not stripped
]


def test_parse_bool_env_table():
    print("\n# _parse_bool_env parse table")
    for raw, expected in PARSE_TABLE:
        result = server._parse_bool_env(raw)
        check(f"_parse_bool_env({raw!r}) == {expected!r}", result == expected, result)


def setup():
    server.ADMIN_PASSWORD = ADMIN_KEY
    server.CONTRACTOR_PDF_KEY = CONTRACTOR_KEY
    server._admin_login_attempts.clear()
    return TestClient(server.app)  # no `with`: skips startup hooks, nothing here needs them


def test_retrieval_status_auth():
    print("\n# /admin/retrieval-status auth -- admin ok, contractor key rejected, no key rejected")
    tc = setup()

    res = tc.get("/admin/retrieval-status")
    check("refuses without a key", res.status_code == 401, res.status_code)

    res = tc.get("/admin/retrieval-status", headers={"X-Admin-Key": CONTRACTOR_KEY})
    check("refuses the contractor key", res.status_code == 401, res.status_code)

    res = tc.get("/admin/retrieval-status", headers={"X-Admin-Key": ADMIN_KEY})
    check("accepts the real admin key", res.status_code == 200, (res.status_code, res.text[:200]))


def test_retrieval_status_shape():
    print("\n# /admin/retrieval-status response shape")
    tc = setup()
    saved_enabled = server.NCERT_V2_ENABLED
    saved_counts = dict(server._ncert_retrieval_counts)
    saved_raw = os.environ.get("NCERT_V2_ENABLED")
    saved_commit = os.environ.get("RAILWAY_GIT_COMMIT_SHA")
    try:
        server.NCERT_V2_ENABLED = True
        os.environ["NCERT_V2_ENABLED"] = ' "true" '
        os.environ["RAILWAY_GIT_COMMIT_SHA"] = "fake0000commit0000sha0000for0000this0000test"  # pragma: allowlist secret
        server._ncert_retrieval_counts["v1"] = 3
        server._ncert_retrieval_counts["v2"] = 7

        res = tc.get("/admin/retrieval-status", headers={"X-Admin-Key": ADMIN_KEY})
        data = res.json()
        check("v2_enabled reflects server.NCERT_V2_ENABLED", data.get("v2_enabled") is True, data)
        check("v2_raw is a quoted repr of the live raw env value",
              data.get("v2_raw") == repr(' "true" '), data.get("v2_raw"))
        check("deployed_commit is shortened", data.get("deployed_commit") == "fake000", data.get("deployed_commit"))
        check("retrieval_counts carries the in-memory tally",
              data.get("retrieval_counts") == {"v1": 3, "v2": 7}, data.get("retrieval_counts"))

        del os.environ["NCERT_V2_ENABLED"]
        del os.environ["RAILWAY_GIT_COMMIT_SHA"]
        server.NCERT_V2_ENABLED = False
        res = tc.get("/admin/retrieval-status", headers={"X-Admin-Key": ADMIN_KEY})
        data = res.json()
        check("v2_raw is \"unset\" when the env var isn't set", data.get("v2_raw") == "unset", data.get("v2_raw"))
        check("deployed_commit is None when RAILWAY_GIT_COMMIT_SHA isn't set",
              data.get("deployed_commit") is None, data.get("deployed_commit"))
    finally:
        server.NCERT_V2_ENABLED = saved_enabled
        server._ncert_retrieval_counts.clear()
        server._ncert_retrieval_counts.update(saved_counts)
        if saved_raw is None:
            os.environ.pop("NCERT_V2_ENABLED", None)
        else:
            os.environ["NCERT_V2_ENABLED"] = saved_raw
        if saved_commit is None:
            os.environ.pop("RAILWAY_GIT_COMMIT_SHA", None)
        else:
            os.environ["RAILWAY_GIT_COMMIT_SHA"] = saved_commit


if __name__ == "__main__":
    test_parse_bool_env_table()
    test_retrieval_status_auth()
    test_retrieval_status_shape()
    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
