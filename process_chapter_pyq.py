import os
import re
import json
import requests
import anthropic

SUPABASE_URL = "https://hvhnfttrfouajlyhvunq.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imh2aG5mdHRyZm91YWpseWh2dW5xIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzg5NDI1NTIsImV4cCI6MjA5NDUxODU1Mn0.8RaSu5yiTpQCEUGKrFO2y6oiRXSqdBehIz933j9Z7WA"
ANTHROPIC_KEY = "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"

client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json"
}

# Chapter name mapping from filename
CHAPTER_MAP = {
    "PLANT_KINGDOM": ("Plant Kingdom", "Biology", 11),
    "PLANT KINGDOM": ("Plant Kingdom", "Biology", 11),
    "LIVING_WORLD": ("The Living World", "Biology", 11),
    "THE LIVING WORLD": ("The Living World", "Biology", 11),
    "BIOLOGICAL_CLASSIFICATION": ("Biological Classification", "Biology", 11),
    "BIOLOGICAL CLASSIFICATION": ("Biological Classification", "Biology", 11),
    "ANIMAL_TISSUE": ("Structural Organisation in Animals", "Biology", 11),
    "ANIMAL TISSUE": ("Structural Organisation in Animals", "Biology", 11),
    "ANATOMY_OF_FLOWERING_PLANTS": ("Anatomy of Flowering Plants", "Biology", 11),
    "ANATOMY Of Flowering Plants": ("Anatomy of Flowering Plants", "Biology", 11),
    "MORPHOLOGY_OF_FLOWERING_PLANTS": ("Morphology of Flowering Plants", "Biology", 11),
    "Morphology_Of_Flowering_Plants": ("Morphology of Flowering Plants", "Biology", 11),
    "CELL_THE_UNIT_OF_LIFE": ("Cell: The Unit of Life", "Biology", 11),
    "CELL THE UNIT OF LIFE": ("Cell: The Unit of Life", "Biology", 11),
    "PHOTOSYNTHESIS_IN_HIGHER_PLANTS": ("Photosynthesis in Higher Plants", "Biology", 11),
    "PHOTOSYNTHESIS IN HIGHER PLANTS": ("Photosynthesis in Higher Plants", "Biology", 11),
    "RESPIRATION_IN_PLANTS": ("Respiration in Plants", "Biology", 11),
    "Respiration_In_Plants": ("Respiration in Plants", "Biology", 11),
    "PLANT_GROWTH_AND_DEVELOPMENT": ("Plant Growth and Development", "Biology", 11),
    "Plant_Growth_And_Development": ("Plant Growth and Development", "Biology", 11),
    "BREATHING_AND_EXCHANGE_OF_GASES": ("Breathing and Exchange of Gases", "Biology", 11),
    "HUMAN_REPRODUCTION": ("Human Reproduction", "Biology", 12),
    "HUMAN REPRODUCTION": ("Human Reproduction", "Biology", 12),
    "MOLECULAR_BASIS_OF_INHERITANCE": ("Molecular Basis of Inheritance", "Biology", 12),
    "Molecular_Basis_Of_Inheritance": ("Molecular Basis of Inheritance", "Biology", 12),
}

def get_chapter_info(filename):
    filename_upper = filename.upper().replace("-", "_")
    for key, value in CHAPTER_MAP.items():
        if key.upper() in filename_upper:
            return value
    return None

def read_pdf(filepath):
    import subprocess
    result = subprocess.run(
        ['python', '-c', f'''
import pdfplumber
with pdfplumber.open(r"{filepath}") as pdf:
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    print(text)
'''],
        capture_output=True, text=True
    )
    return result.stdout

def extract_questions_with_claude(text, chapter, subject, class_num):
    # Split text into chunks of 6000 chars with overlap
    chunk_size = 6000
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+chunk_size])
        i += chunk_size - 500  # 500 char overlap to catch split questions

    # Get answer key from last part of text
    answer_key_text = text[-3000:]
    
    all_questions = []
    
    for idx, chunk in enumerate(chunks):
        print(f"  Processing chunk {idx+1}/{len(chunks)}...")
        prompt = f"""Extract NEET PYQ questions from this text chunk for chapter: {chapter} ({subject} Class {class_num}).

Return ONLY a valid JSON array. No other text, no markdown, no backticks.
If no complete questions found in this chunk, return empty array: []

Each question object must have:
- question: string (question text only, no Q1. prefix)
- option_a: string
- option_b: string
- option_c: string
- option_d: string
- correct_answer: string ("a", "b", "c", or "d" — use answer key below to match: 1=a, 2=b, 3=c, 4=d)
- year: integer (from NEET YYYY or AIPMT YYYY)
- subject: "{subject}"
- chapter: "{chapter}"
- class: {class_num}
- has_diagram: false

ANSWER KEY (from end of document):
{answer_key_text}

TEXT CHUNK:
{chunk}
"""
        try:
            message = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=6000,
                messages=[{"role": "user", "content": prompt}]
            )
            result = message.content[0].text.strip()
            if result.startswith("```"):
                result = re.sub(r'^```[a-z]*\n?', '', result)
                result = re.sub(r'\n?```$', '', result)
            questions = json.loads(result)
            all_questions.extend(questions)
        except Exception as e:
            print(f"  Chunk {idx+1} error: {e}")
            continue
    
    # Remove duplicates based on question text
    seen = set()
    unique = []
    for q in all_questions:
        key = q.get('question', '')[:50]
        if key not in seen:
            seen.add(key)
            unique.append(q)
    
    return json.dumps(unique)

def upload_to_supabase(questions):
    success = 0
    for q in questions:
        res = requests.post(
            f"{SUPABASE_URL}/rest/v1/pyq",
            headers=headers,
            json=q
        )
        if res.status_code in [200, 201]:
            success += 1
        else:
            print(f"  Error uploading: {res.text[:100]}")
    return success

def process_pdf(filepath):
    filename = os.path.basename(filepath)
    print(f"\nProcessing: {filename}")
    
    chapter_info = get_chapter_info(filename)
    if not chapter_info:
        print(f"  Could not identify chapter from filename. Skipping.")
        return 0
    
    chapter, subject, class_num = chapter_info
    print(f"  Chapter: {chapter} | Subject: {subject} | Class: {class_num}")
    
    # Read PDF text
    print(f"  Reading PDF...")
    text = read_pdf(filepath)
    if not text or len(text) < 100:
        print(f"  Could not read PDF text. Skipping.")
        return 0
    
    print(f"  Extracted {len(text)} characters")
    
    # Extract questions with Claude
    print(f"  Extracting questions with Claude...")
    try:
        result = extract_questions_with_claude(text, chapter, subject, class_num)
        questions = json.loads(result)
        print(f"  Found {len(questions)} questions")
        
        # Upload to Supabase
        print(f"  Uploading to Supabase...")
        success = upload_to_supabase(questions)
        print(f"  Uploaded {success}/{len(questions)} questions")
        return success
        
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        print(f"  Raw result: {result[:200]}")
        return 0
    except Exception as e:
        print(f"  Error: {e}")
        return 0

def main():
    pdf_folder = r"C:\Users\hakim\Documents\neet-ai\chapter-Qs"
    
    if not os.path.exists(pdf_folder):
        print(f"Folder not found: {pdf_folder}")
        return
    
    pdfs = []
    for root, dirs, files in os.walk(pdf_folder):
        for f in files:
            if f.endswith('.pdf'):
                pdfs.append(os.path.join(root, f))
    print(f"Found {len(pdfs)} PDFs")
    
    total = 0
    for filepath in pdfs:
        count = process_pdf(filepath)
        total += count
    
    print(f"\n✅ Done! Total questions uploaded: {total}")

main()