import os
import time
import PyPDF2
import requests
import openai
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
OPENAI_KEY = os.environ["OPENAI_KEY"]

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