"""NEET_SYLLABUS must number chapters like the rationalised NCERT books (2026-27).

Run with: python test_syllabus_numbering.py
Same small pass/fail harness as test_answer_quality.py. Fully offline.

The pre-2023 chapter order used to live here, so citations showed old numbers -- e.g.
"Excretory Products and their Elimination" cited as Chapter 19 instead of 16.
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


KNOWN = [
    ("Biology", 11, "Breathing and Exchange of Gases", 14),
    ("Biology", 11, "Excretory Products and their Elimination", 16),
    ("Biology", 11, "Chemical Coordination and Integration", 19),
    ("Biology", 12, "Sexual Reproduction in Flowering Plants", 1),
    ("Biology", 12, "Principles of Inheritance and Variation", 4),
    ("Biology", 12, "Ecosystem", 12),
    ("Chemistry", 11, "Thermodynamics", 5),
    ("Chemistry", 11, "Hydrocarbons", 9),
    ("Chemistry", 12, "Solutions", 1),
    ("Chemistry", 12, "Amines", 9),
    ("Physics", 12, "Current Electricity", 3),
]


def test_known_chapters():
    for subject, cls, name, num in KNOWN:
        got = server._ncert_chapter_citation(subject, cls, name)
        check(f"{subject} {cls} {name} -> Ch {num}", got == f"NCERT Class {cls}, Chapter {num} — {name}", got)


def test_hindi_row_uses_english_lookup():
    got = server._ncert_chapter_citation("Biology", "11", "उत्सर्जी उत्पाद एवं उनका निष्कासन",
                                         lookup_name="Excretory Products and their Elimination")
    check("Hindi excretion row -> Ch 16", got.startswith("NCERT Class 11, Chapter 16 — "), got)


def test_dropped_chapter_gets_no_number():
    # ncert_content still holds Transport in Plants; token overlap alone would cite it as Ch 12.
    got = server._ncert_chapter_citation("Biology", 11, "Transport in Plants")
    check("Transport in Plants -> numberless", got == "NCERT Class 11, Biology — Transport in Plants", got)
    got = server._ncert_chapter_citation("Biology", 11, "पौधों में परिवहन", lookup_name="Transport in Plants")
    check("Hindi Transport in Plants -> numberless", "Chapter" not in got, got)


def test_chapter_counts():
    counts = {s: {c: len(v) for c, v in classes.items()} for s, classes in server.NEET_SYLLABUS.items()}
    check("Biology 19 + 13 chapters", counts["Biology"] == {11: 19, 12: 13}, counts["Biology"])
    check("Chemistry 9 + 10 chapters", counts["Chemistry"] == {11: 9, 12: 10}, counts["Chemistry"])


if __name__ == "__main__":
    test_known_chapters()
    test_hindi_row_uses_english_lookup()
    test_dropped_chapter_gets_no_number()
    test_chapter_counts()
    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
