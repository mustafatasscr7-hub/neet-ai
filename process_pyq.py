import os
import re
import time
import requests
import openai

SUPABASE_URL = "https://hvhnfttrfouajlyhvunq.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imh2aG5mdHRyZm91YWpseWh2dW5xIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzg5NDI1NTIsImV4cCI6MjA5NDUxODU1Mn0.8RaSu5yiTpQCEUGKrFO2y6oiRXSqdBehIz933j9Z7WA"
OPENAI_KEY = "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"

openai_client = openai.OpenAI(api_key=OPENAI_KEY)

def get_embedding(text):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text[:500])
    return response.data[0].embedding

def update_embedding(record_id, embedding):
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.patch(
        f"{SUPABASE_URL}/rest/v1/pyq?id=eq.{record_id}",
        headers=headers,
        json={"embedding": embedding},
        timeout=30)
    return response.status_code

def get_all_questions():
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }
    response = requests.get(
        f"{SUPABASE_URL}/rest/v1/pyq?embedding=is.null&select=id,question",
        headers=headers,
        timeout=30)
    return response.json()

def add_embeddings():
    print("Fetching questions without embeddings...")
    questions = get_all_questions()
    print(f"Found {len(questions)} questions to embed")

    for i, q in enumerate(questions):
        try:
            embedding = get_embedding(q['question'])
            status = update_embedding(q['id'], embedding)
            if status in [200, 204]:
                print(f"  {i+1}/{len(questions)} done")
            else:
                print(f"  {i+1}/{len(questions)} failed — status {status}")
        except Exception as e:
            print(f"  {i+1} error: {e}")

        time.sleep(0.3)

    print("\nAll embeddings added!")

if __name__ == "__main__":
    add_embeddings()