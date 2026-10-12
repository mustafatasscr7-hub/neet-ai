"""Regression test for server.py's _usable_student_name guard.

Run with: python test_student_name_sanitization.py
No pytest dependency -- this repo has no existing test runner, so this uses plain asserts and a
small pass/fail harness, matching test_pyq_extraction.py's own pattern.

Covers the fix where /chat's name_context would address a student by whatever raw student_name
the caller sent, even when that was actually their phone number or email (the profile-display
fallback chat.html/chatStore.tsx use when no real display name is set) -- that's a data-leak-
shaped mistake, not a personalization touch, so the server must refuse to use it regardless of
what any given caller sends.
"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server

_passed = []
_failed = []


def check(label, condition, detail=""):
    if condition:
        _passed.append(label)
        print(f"  PASS: {label}")
    else:
        _failed.append(label)
        print(f"  FAIL: {label}" + (f" -- {detail}" if detail else ""))


def test_usable_student_name():
    print("\n=== _usable_student_name ===")
    cases = [
        ("9876543210", "", "bare phone number"),
        ("+919876543210", "", "phone number with country code"),
        ("+91 98765 43210", "", "phone number with spaces"),
        ("(987) 654-3210", "", "phone number with punctuation"),
        ("student@example.com", "", "email address"),
        ("Aarav Sharma", "Aarav Sharma", "real display name"),
        ("Priya", "Priya", "single-word real name"),
        ("", "", "empty string"),
        ("   ", "", "whitespace only"),
        (None, "", "None"),
        ("  Rohan Verma  ", "Rohan Verma", "real name with surrounding whitespace is trimmed"),
    ]
    for raw, expected, label in cases:
        result = server._usable_student_name(raw)
        check(f"{label}: {raw!r} -> {expected!r}", result == expected, detail=f"got {result!r}")


if __name__ == "__main__":
    test_usable_student_name()

    print(f"\nTOTAL: {len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
