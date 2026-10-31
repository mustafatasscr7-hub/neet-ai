"""Unit test for rank_suggested_chapters (GET /me/suggested-chapters' ranking function) --
pure-function test, no HTTP/Supabase involved.

Checks:
1. Mistakes only: a compute_weak_chapters entry with no matching doubts ranks by its own total,
   reason follows whichever count (mock_wrong vs saved) is larger.
2. Doubts only: a chapter with no mistake/saved qualification but >=2 recent doubts appears with
   reason "doubts".
3. Both: a chapter that already qualifies via mistakes also has doubts on it -- reason stays
   "mistakes"/"saved", but its score is nudged above an equal-total chapter with no doubts.
4. A single doubt (count 1) is ignored -- never surfaces a chapter on its own.
5. Doubts older than DOUBT_SUGGESTION_LOOKBACK_DAYS score 0 and don't nudge or qualify anything.

Run with: python test_suggested_chapters_ranking.py
"""
import io
import os
import sys
from datetime import datetime, timedelta, timezone

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
        print(f"  FAIL: {label}  {detail}")


def doubt_row(subject, chapter, days_ago):
    created = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return {"subject": subject, "chapter": chapter, "created_at": created.isoformat()}


print("1. Mistakes only (no doubts at all)")
weak = server.compute_weak_chapters(
    mistake_rows=[],
    mock_wrong_rows=[
        {"subject": "Physics", "chapter": "Laws of Motion"},
        {"subject": "Physics", "chapter": "Laws of Motion"},
    ],
)
doubt_scores = server.compute_doubt_scores([])
ranked = server.rank_suggested_chapters(weak, doubt_scores)
check("one chapter ranked", len(ranked) == 1, ranked)
check("reason is mistakes (mock_wrong_count >= saved_count)", ranked and ranked[0]["reason"] == "mistakes", ranked)
check("score equals the mistake total (2)", ranked and ranked[0]["score"] == 2, ranked)

print("\n2. Doubts only (no mistakes/saved qualification)")
doubt_rows = [doubt_row("Biology", "Ecosystem", 0), doubt_row("Biology", "Ecosystem", 1)]
doubt_scores = server.compute_doubt_scores(doubt_rows)
ranked = server.rank_suggested_chapters([], doubt_scores)
check("chapter surfaces from doubts alone", len(ranked) == 1, ranked)
check("reason is doubts", ranked and ranked[0]["reason"] == "doubts", ranked)
check(
    "doubt-only score stays below one mistake/saved event's weight (< 1.0)",
    ranked and ranked[0]["score"] < 1.0,
    ranked,
)

print("\n3. Both: a mistake-qualified chapter also has doubts on it")
weak = server.compute_weak_chapters(
    mistake_rows=[],
    mock_wrong_rows=[
        {"subject": "Chemistry", "chapter": "Thermodynamics"},
        {"subject": "Chemistry", "chapter": "Thermodynamics"},
        {"subject": "Chemistry", "chapter": "Equilibrium"},
        {"subject": "Chemistry", "chapter": "Equilibrium"},
    ],
)
doubt_rows = [doubt_row("Chemistry", "Thermodynamics", 0), doubt_row("Chemistry", "Thermodynamics", 1)]
doubt_scores = server.compute_doubt_scores(doubt_rows)
ranked = server.rank_suggested_chapters(weak, doubt_scores)
by_chapter = {c["chapter"]: c for c in ranked}
check("both chapters present", set(by_chapter) == {"Thermodynamics", "Equilibrium"}, ranked)
check("Thermodynamics keeps reason 'mistakes', not 'doubts'", by_chapter["Thermodynamics"]["reason"] == "mistakes", ranked)
check(
    "Thermodynamics (has doubts) outranks the equal-total Equilibrium (no doubts)",
    by_chapter["Thermodynamics"]["score"] > by_chapter["Equilibrium"]["score"],
    ranked,
)

print("\n4. A single doubt (count 1) is ignored")
doubt_scores = server.compute_doubt_scores([doubt_row("Physics", "Gravitation", 0)])
ranked = server.rank_suggested_chapters([], doubt_scores)
check("no chapter surfaces from one doubt alone", len(ranked) == 0, ranked)

print("\n5. Old doubts (outside the lookback window) are ignored")
old_rows = [
    doubt_row("Biology", "Evolution", server.DOUBT_SUGGESTION_LOOKBACK_DAYS + 5),
    doubt_row("Biology", "Evolution", server.DOUBT_SUGGESTION_LOOKBACK_DAYS + 10),
]
doubt_scores = server.compute_doubt_scores(old_rows)
check("old doubts are dropped entirely, not just down-weighted", doubt_scores == {}, doubt_scores)
ranked = server.rank_suggested_chapters([], doubt_scores)
check("old doubts don't surface a chapter despite count>=2", len(ranked) == 0, ranked)

print(f"\n{len(_passed)} passed, {len(_failed)} failed")
if _failed:
    sys.exit(1)
