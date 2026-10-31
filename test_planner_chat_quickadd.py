"""Planner quick-add from chat (Pro/Max only): the widened _has_planner_trigger regex (English/
Hindi/Hinglish phrasings) and _resolve_planner_due_date's date-word resolution. Both are pure
functions -- no network, no Supabase, nothing to mock. The real precision (is this actually a
task request, e.g. "add 3 and 5" vs "add bio 30 Qs tomorrow") lives in _extract_planner_task's
Qwen-Flash call, not the regex -- see its own docstring -- so that's out of scope for this file.

Run with: python test_planner_chat_quickadd.py
"""
import asyncio
import io
import os
import sys
from datetime import datetime, timedelta

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


# 12 phrasings that must trigger the cheap regex pre-filter (English, Hindi, Hinglish).
MUST_TRIGGER = [
    "add a task biology for tomorrow 30 Qs",
    "planner mein biology 30 Qs kal daal do",
    "remind me to revise optics on Friday",
    "schedule chemistry revision for Monday",
    "put bio revision for tomorrow",
    "jod do chemistry 20 questions kal",
    "jodo physics ke 10 numericals parso",
    "daal dena biology revision kal ke liye",
    "जोड़ दो कल के लिए रसायन विज्ञान 20 प्रश्न",
    "प्लानर में कल जीव विज्ञान जोड़ो",
    "टास्क add karo physics revision kal",
    "parso tak yeh complete karna hai, add kar do planner mein",
]

# 8 phrasings with no trigger word at all -- genuine subject questions/smalltalk that must stay
# on the normal chat path. (Borderline over-triggers like "add 3 and 5"/"what is a task force"
# deliberately DO fire this regex today, by design -- see _extract_planner_task's own docstring
# for how those are resolved to is_task:false one step later, not here.)
MUST_NOT_TRIGGER = [
    "What is the Krebs cycle?",
    "Explain Newton's second law of motion",
    "Why is the sky blue",
    "2 + 2 ka answer kya hai",
    "mitochondria ki structure batao",
    "क्रेब्स चक्र क्या है",
    "ओम का नियम समझाओ",
    "hi how are you",
]


def test_trigger_regex():
    print("\n# _has_planner_trigger: 12 phrasings that must fire")
    for text in MUST_TRIGGER:
        check(f"triggers: {text!r}", server._has_planner_trigger(text) is True)

    print("\n# _has_planner_trigger: 8 phrasings that must NOT fire")
    for text in MUST_NOT_TRIGGER:
        check(f"does not trigger: {text!r}", server._has_planner_trigger(text) is False)


def test_date_resolution():
    print("\n# _resolve_planner_due_date: 5 date parses")
    # Fixed reference instant so the weekday case is deterministic regardless of when this runs.
    now_ist = datetime(2026, 10, 28, 12, 0, tzinfo=server.IST)  # a Wednesday
    check("fixed reference is a Wednesday", now_ist.strftime("%A") == "Wednesday", now_ist.strftime("%A"))

    today_str = now_ist.strftime("%Y-%m-%d")
    tomorrow_str = (now_ist + timedelta(days=1)).strftime("%Y-%m-%d")
    friday_str = (now_ist + timedelta(days=2)).strftime("%Y-%m-%d")  # next Friday from a Wednesday

    check("'today' -> today's date",
          server._resolve_planner_due_date("today", now_ist) == today_str)
    check("'tomorrow' -> tomorrow's date",
          server._resolve_planner_due_date("tomorrow", now_ist) == tomorrow_str)
    check("'Friday' -> this week's Friday",
          server._resolve_planner_due_date("Friday", now_ist) == friday_str)
    check("'2026-11-05' -> passed through unchanged",
          server._resolve_planner_due_date("2026-11-05", now_ist) == "2026-11-05")
    check("unrecognized text -> None",
          server._resolve_planner_due_date("whenever I feel like it", now_ist) is None)


async def test_stream_response_planner_branch():
    # Extra wiring check beyond the 25 table-driven cases above: stream_response's own branching
    # (auto-add vs confirm-card vs multi-card vs upgrade) with _extract_planner_task mocked out,
    # so this is testing the NEW branch logic itself, not Qwen's own extraction quality.
    print("\n# stream_response planner branch: 4 scenario checks (_extract_planner_task mocked)")
    orig_extract = server._extract_planner_task
    orig_plan = server.get_user_plan
    canned = {}

    async def fake_extract(text, user_id=""):
        return canned.get(text)

    async def fake_plan(user_id):
        return "free" if user_id == "u-free" else "pro"

    server._extract_planner_task = fake_extract
    server.get_user_plan = fake_plan

    async def run(text, user_id):
        chunks = []
        async for c in server.stream_response(text, user_id=user_id, language="en"):
            chunks.append(c)
        return "".join(chunks)

    try:
        canned["add bio 30 Qs tomorrow"] = {
            "tasks": [{"title": "Biology - 30 MCQs", "due_date": "2026-11-01"}],
            "confident": True, "truncated": False, "today_str": "2026-10-31",
        }
        out = await run("add bio 30 Qs tomorrow", "u-pro")
        check("confident single task -> PLANNER_AUTO_ADD marker (no confirm card)",
              out.startswith("PLANNER_AUTO_ADD: yes") and "TITLE: Biology - 30 MCQs" in out and "DUE: 2026-11-01" in out, out)

        canned["remind me to study"] = {
            "tasks": [{"title": "study", "due_date": None}],
            "confident": False, "truncated": False, "today_str": "2026-10-31",
        }
        out = await run("remind me to study", "u-pro")
        check("vague single task -> PLANNER_TASK confirm card (unchanged behavior)",
              out.startswith("PLANNER_TASK: yes") and "TITLE: study" in out, out)

        canned["add bio 30 Qs and chem revision tomorrow"] = {
            "tasks": [
                {"title": "Biology - 30 MCQs", "due_date": "2026-11-01"},
                {"title": "Chemistry revision", "due_date": "2026-11-01"},
            ],
            "confident": False, "truncated": False, "today_str": "2026-10-31",
        }
        out = await run("add bio 30 Qs and chem revision tomorrow", "u-pro")
        check("two tasks -> one PLANNER_MULTI card listing both",
              out.startswith("PLANNER_MULTI: yes") and out.count("TASK: ") == 2
              and "Biology - 30 MCQs" in out and "Chemistry revision" in out, out)

        canned["add bio 30 Qs tomorrow free"] = {
            "tasks": [{"title": "Biology - 30 MCQs", "due_date": "2026-11-01"}],
            "confident": True, "truncated": False, "today_str": "2026-10-31",
        }
        out = await run("add bio 30 Qs tomorrow free", "u-free")
        check("Free plan -> PLANNER_UPGRADE marker, nothing added",
              out.startswith("PLANNER_UPGRADE: yes") and "PLANNER_AUTO_ADD" not in out, out)
    finally:
        server._extract_planner_task = orig_extract
        server.get_user_plan = orig_plan


if __name__ == "__main__":
    test_trigger_regex()
    test_date_resolution()
    asyncio.run(test_stream_response_planner_branch())
    print(f"\nTOTAL: {len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
