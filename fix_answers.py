import requests
import re

SUPABASE_URL = "https://hvhnfttrfouajlyhvunq.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imh2aG5mdHRyZm91YWpseWh2dW5xIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzg5NDI1NTIsImV4cCI6MjA5NDUxODU1Mn0.8RaSu5yiTpQCEUGKrFO2y6oiRXSqdBehIz933j9Z7WA"

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json"
}

num_to_letter = {"1": "a", "2": "b", "3": "c", "4": "d"}

def extract_answer(text):
    if not text:
        return None, text
    match = re.search(r'Ans\.\s*\[(\d)\]', text, re.IGNORECASE)
    if match:
        num = match.group(1)
        letter = num_to_letter.get(num)
        clean_text = re.sub(r'\s*Ans\.\s*\[\d\]', '', text).strip()
        return letter, clean_text
    return None, text

def fix_all_questions():
    print("Fetching all questions...")
    offset = 0
    limit = 1000
    all_questions = []

    while True:
        res = requests.get(
            f"{SUPABASE_URL}/rest/v1/pyq?select=*&limit={limit}&offset={offset}",
            headers=headers
        )
        batch = res.json()
        if not batch:
            break
        all_questions.extend(batch)
        if len(batch) < limit:
            break
        offset += limit

    print(f"Total questions: {len(all_questions)}")
    fixed = 0

    for q in all_questions:
        qid = q['id']
        correct = q.get('correct_answer', '')
        
        # Check all options for Ans. [X]
        found_answer = None
        updates = {}

        for opt in ['option_a', 'option_b', 'option_c', 'option_d']:
            letter, clean = extract_answer(q.get(opt, ''))
            if clean != q.get(opt, ''):
                updates[opt] = clean
            if letter and not found_answer:
                found_answer = letter

        if found_answer and not correct:
            updates['correct_answer'] = found_answer

        if updates:
            patch = requests.patch(
                f"{SUPABASE_URL}/rest/v1/pyq?id=eq.{qid}",
                headers=headers,
                json=updates
            )
            if patch.status_code in [200, 204]:
                fixed += 1
                if fixed % 50 == 0:
                    print(f"Fixed {fixed} questions...")
            else:
                print(f"Error on {qid}: {patch.text}")

    print(f"\nDone! Fixed {fixed} out of {len(all_questions)} questions.")

fix_all_questions()