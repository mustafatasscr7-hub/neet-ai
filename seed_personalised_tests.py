import random
import requests
from dotenv import load_dotenv
import os

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json"
}

TESTS_PER_SUBJECT = 6
QUESTIONS_PER_TEST = {
    "Biology": 90,
    "Physics": 45,
    "Chemistry": 45
}

for subject, count in QUESTIONS_PER_TEST.items():
    res = requests.get(
        f"{SUPABASE_URL}/rest/v1/pyq",
        headers=headers,
        params={
            "subject": f"eq.{subject}",
            "is_active": "eq.true",
            "select": "id",
            "limit": 5000
        }
    )
    ids = [row["id"] for row in res.json()]
    print(f"{subject}: {len(ids)} active questions available")

    if len(ids) < count:
        print(f"  Skipping {subject} — not enough active questions for a {count}-question test yet.")
        continue

    rows = []
    for test_number in range(1, TESTS_PER_SUBJECT + 1):
        question_ids = random.sample(ids, count)
        rows.append({
            "subject": subject,
            "test_number": test_number,
            "title": f"{subject} Practice Test {test_number}",
            "description": f"NEET-standard {subject} practice test drawing {count} questions from across the {subject} NEET syllabus.",
            "question_ids": question_ids,
            "question_count": count
        })

    upsert_res = requests.post(
        f"{SUPABASE_URL}/rest/v1/personalised_test_sets?on_conflict=subject,test_number",
        headers={**headers, "Prefer": "resolution=merge-duplicates"},
        json=rows
    )
    if upsert_res.status_code in (200, 201):
        print(f"  Seeded {TESTS_PER_SUBJECT} tests for {subject}")
    else:
        print(f"  Failed to seed {subject}: {upsert_res.status_code} {upsert_res.text}")

print("\nDone.")
