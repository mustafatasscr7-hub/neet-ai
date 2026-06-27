import requests

SUPABASE_URL = "https://hvhnfttrfouajlyhvunq.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imh2aG5mdHRyZm91YWpseWh2dW5xIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzg5NDI1NTIsImV4cCI6MjA5NDUxODU1Mn0.8RaSu5yiTpQCEUGKrFO2y6oiRXSqdBehIz933j9Z7WA"

headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}

# Questions with chapter tagged
res1 = requests.get(f"{SUPABASE_URL}/rest/v1/pyq?chapter=not.is.null&select=id&limit=1000", headers=headers)
print(f"Questions with chapter: {len(res1.json())}")

# Questions with correct answer
res2 = requests.get(f"{SUPABASE_URL}/rest/v1/pyq?correct_answer=not.is.null&correct_answer=neq.&select=id&limit=2000", headers=headers)
print(f"Questions with correct answer: {len(res2.json())}")python check_data.py