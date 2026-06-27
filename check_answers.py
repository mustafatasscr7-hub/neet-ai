import requests
import re

SUPABASE_URL = "https://hvhnfttrfouajlyhvunq.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imh2aG5mdHRyZm91YWpseWh2dW5xIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzg5NDI1NTIsImV4cCI6MjA5NDUxODU1Mn0.8RaSu5yiTpQCEUGKrFO2y6oiRXSqdBehIz933j9Z7WA"

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}"
}

res = requests.get(f"{SUPABASE_URL}/rest/v1/pyq?select=correct_answer&limit=1000", headers=headers)
questions = res.json()

empty = 0
has_value = {}
for q in questions:
    ans = q.get('correct_answer', '')
    if not ans:
        empty += 1
    else:
        val = ans.strip()
        has_value[val] = has_value.get(val, 0) + 1

print(f"Empty correct_answer: {empty}")
print(f"Values found: {has_value}")