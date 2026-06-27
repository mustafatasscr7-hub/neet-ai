import os
import time
import PyPDF2
import requests
import openai

SUPABASE_URL = "https://hvhnfttrfouajlyhvunq.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imh2aG5mdHRyZm91YWpseWh2dW5xIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzg5NDI1NTIsImV4cCI6MjA5NDUxODU1Mn0.8RaSu5yiTpQCEUGKrFO2y6oiRXSqdBehIz933j9Z7WA"
OPENAI_KEY = "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"

openai_client = openai.OpenAI(api_key=OPENAI_KEY)

PDF_FOLDER = "ncert-pdfs"
CHUNK_SIZE = 500

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        print(f"  Error reading {pdf_path}: {e}")
    return text

def split_into_chunks(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def get_embedding(text):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def upload_chunk(content, subject, class_num, filename):
    embedding = get_embedding(content)
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "subject": subject,
        "class": class_num,
        "chapter_name": filename,
        "content": content,
        "embedding": embedding
    }
    for attempt in range(3):
        try:
            response = requests.post(
                f"{SUPABASE_URL}/rest/v1/ncert_content",
                headers=headers,
                json=data,
                timeout=30
            )
            return response.status_code
        except Exception as e:
            print(f"  Retry {attempt+1}/3: {e}")
            time.sleep(3)
    return 0

def get_subject(filename):
    if "bo" in filename:
        return "Biology"
    elif "ch" in filename:
        return "Chemistry"
    elif "ph" in filename:
        return "Physics"
    return "Biology"

def get_class(filename):
    if filename.startswith("le"):
        return 12
    elif filename.startswith("ke"):
        return 11
    return 11

def process_all_pdfs():
    pdf_files = []
    for root, dirs, files in os.walk(PDF_FOLDER):
        for file in files:
            if file.endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))

    print(f"Found {len(pdf_files)} PDF files")

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"\nProcessing: {filename}")

        subject = get_subject(filename)
        class_num = get_class(filename)

        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            print(f"  No text found — skipping")
            continue

        chunks = split_into_chunks(text)
        print(f"  Subject: {subject} | Class: {class_num} | Chunks: {len(chunks)}")

        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) > 100:
                status = upload_chunk(chunk, subject, class_num, filename)
                print(f"  Chunk {i+1}/{len(chunks)} → Status: {status}")
                time.sleep(0.5)

    print("\nDone! All NCERT content uploaded.")

if __name__ == "__main__":
    process_all_pdfs()