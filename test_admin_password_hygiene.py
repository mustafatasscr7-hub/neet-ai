"""Regression tests for the 2026-11-24 hygiene pass: ADMIN_PASSWORD has no hardcoded default
(fails closed when unset, works once set), admin/contractor error responses no longer leak raw
exception text, and no PDF the site itself serves got swept into .gitignore's blanket *.pdf rule.

Run with: python test_admin_password_hygiene.py
Plain pass/fail harness, same shape as test_security_hardening.py. Fully offline.
"""
import io
import os
import re
import subprocess
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


def test_admin_refuses_all_logins_when_unset():
    print("\n# ADMIN_PASSWORD unset -- every admin login is refused")
    saved = server.ADMIN_PASSWORD
    server.ADMIN_PASSWORD = None
    try:
        with TestClient(server.app) as client:
            for key in [None, "", "neetai-admin-2027", "anything-at-all"]:
                server._admin_login_attempts.clear()  # isolate from the 3-strike lockout, tested elsewhere
                headers = {"X-Admin-Key": key} if key is not None else {}
                r = client.post("/admin/verify", headers=headers)
                check(f"X-Admin-Key={key!r} -> 401 when ADMIN_PASSWORD is unset", r.status_code == 401, str(r.status_code))
    finally:
        server.ADMIN_PASSWORD = saved
        server._admin_login_attempts.clear()


def test_admin_works_when_set():
    print("\n# ADMIN_PASSWORD set -- the real key passes, a wrong one still 401s")
    saved = server.ADMIN_PASSWORD
    server.ADMIN_PASSWORD = "unit-test-admin-key"  # pragma: allowlist secret
    server._admin_login_attempts.clear()
    try:
        with TestClient(server.app) as client:
            r = client.post("/admin/verify", headers={"X-Admin-Key": "unit-test-admin-key"})
            check("correct key -> 200", r.status_code == 200, str(r.status_code))
            server._admin_login_attempts.clear()
            r = client.post("/admin/verify", headers={"X-Admin-Key": "wrong"})
            check("wrong key -> 401", r.status_code == 401, str(r.status_code))
    finally:
        server.ADMIN_PASSWORD = saved
        server._admin_login_attempts.clear()


def test_forced_exception_returns_generic_message():
    print("\n# a handler that raises returns the generic message, never the exception text")
    saved_admin = server.ADMIN_PASSWORD
    saved_count = server._admin_count
    server.ADMIN_PASSWORD = "unit-test-admin-key"  # pragma: allowlist secret
    secret_detail = "SECRET_DB_DETAIL_f3a9c1 column pyq.correct_answer does not exist"  # pragma: allowlist secret

    def boom(params):
        raise RuntimeError(secret_detail)

    server._admin_count = boom
    server._admin_login_attempts.clear()
    try:
        with TestClient(server.app) as client:
            r = client.get("/admin/pyq-stats", headers={"X-Admin-Key": "unit-test-admin-key"})
            check("forced exception still returns 200 (error carried in the body)", r.status_code == 200, str(r.status_code))
            body_text = r.text
            check("response body contains the generic message", "Something went wrong" in body_text, body_text)
            check("response body does NOT contain the raw exception text", secret_detail not in body_text, body_text)
    finally:
        server.ADMIN_PASSWORD = saved_admin
        server._admin_count = saved_count
        server._admin_login_attempts.clear()


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PDF_REF_RE = re.compile(r'["\'(]([^"\'()]*\.pdf)["\')]', re.IGNORECASE)


def _tracked_pdf_paths():
    out = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    return {line for line in out.splitlines() if line.lower().endswith(".pdf")}


def _pdf_paths_referenced_in_source():
    referenced = set()
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in (".git", "node_modules")]
        for name in files:
            if not name.endswith((".html", ".js")):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except OSError:
                continue
            for m in PDF_REF_RE.finditer(text):
                candidate = m.group(1)
                # Only a real relative/bare filename reference, not a URL or a UI label/comment.
                if "/" in candidate or re.match(r'^[\w .\-]+\.pdf$', candidate, re.IGNORECASE):
                    if " " not in candidate or "/" in candidate:
                        referenced.add(candidate.lstrip("/"))
    return referenced


def test_site_served_pdfs_still_tracked():
    print("\n# any PDF path the site's own html/js references is still a tracked file")
    tracked = _tracked_pdf_paths()
    referenced = _pdf_paths_referenced_in_source()
    missing = sorted(p for p in referenced if p not in tracked and os.path.isfile(os.path.join(REPO_ROOT, p)))
    check("every site-referenced PDF that exists on disk is still git-tracked", not missing, str(missing))


if __name__ == "__main__":
    test_admin_refuses_all_logins_when_unset()
    test_admin_works_when_set()
    test_forced_exception_returns_generic_message()
    test_site_served_pdfs_still_tracked()
    print(f"\nTOTAL: {len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
