import os
import json
import base64
import fitz
from anthropic import Anthropic
from dotenv import load_dotenv
import requests as http_requests

load_dotenv()

ANTHROPIC_KEY = os.getenv("ANTHROPIC_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

client = Anthropic(api_key=ANTHROPIC_KEY)

VISION_MODEL = "claude-haiku-4-5-20251001"

def build_extraction_prompt(subject):
    return f"""You are extracting NEET {subject} questions from one page of a scanned PDF.

For EACH complete question visible on this page, extract:
- question (full question text, including any sub-statements A/B/C/D or i/ii/iii if part of the question)
- option_a, option_b, option_c, option_d (exact text of each option)
- question_type: one of "mcq", "match_column", "assertion_reason", "statement_based"
- has_diagram: true if the question references a figure, diagram, image, or shows a chemical structure/graph/apparatus, otherwise false
- year (if a NEET year tag like [NEET-2024] appears directly next to this question, else null)

Do NOT guess or infer a chapter name - that is handled separately. Do NOT attempt to determine the correct answer, even if you can solve the question - leave that to a human reviewer.

If a question is cut off at the top or bottom of this page (incomplete), SKIP it entirely - do not guess missing parts.
If there are no complete questions on this page (e.g. this page is a cover page, instructions, or an answer key), return an empty array.

Return ONLY a JSON array, no other text. Example:
[
  {{
    "question": "...",
    "option_a": "...",
    "option_b": "...",
    "option_c": "...",
    "option_d": "...",
    "question_type": "mcq",
    "has_diagram": false,
    "year": 2024
  }}
]
"""

# Kept for backward compatibility with the standalone Biology CLI batch job below.
EXTRACTION_PROMPT = build_extraction_prompt("Biology")

def pdf_to_page_images(pdf_source):
    """pdf_source may be a file path (str) or raw PDF bytes."""
    doc = fitz.open(pdf_source) if isinstance(pdf_source, str) else fitz.open(stream=pdf_source, filetype="pdf")
    images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        images.append(base64.standard_b64encode(img_bytes).decode("utf-8"))
    doc.close()
    return images

def extract_questions_from_page(image_b64, page_num, subject="Biology", model=VISION_MODEL):
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64
                        }
                    },
                    {
                        "type": "text",
                        "text": build_extraction_prompt(subject)
                    }
                ]
            }
        ]
    )

    text = message.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    try:
        return json.loads(text)
    except Exception as e:
        print(f"    JSON parse error on page {page_num}: {e}")
        return []

def scan_pdf_bytes(pdf_bytes, subject):
    """Callable entry point for the admin review pipeline (server.py's /admin/scan-pdf).
    Returns raw extracted questions - no chapter/class/correct_answer guessing, that's
    left to the TF-IDF classifier and manual review on the frontend."""
    page_images = pdf_to_page_images(pdf_bytes)
    questions = []
    for page_num, img in enumerate(page_images, 1):
        for q in extract_questions_from_page(img, page_num, subject=subject):
            questions.append({
                "question": q.get("question", ""),
                "option_a": q.get("option_a", ""),
                "option_b": q.get("option_b", ""),
                "option_c": q.get("option_c", ""),
                "option_d": q.get("option_d", ""),
                "correct_answer": "",
                "question_type": q.get("question_type", "mcq"),
                "has_diagram": bool(q.get("has_diagram", False)),
                "year": q.get("year"),
                "source_page": page_num
            })
    return {"questions": questions, "pages_scanned": len(page_images)}

def insert_question(q, subject="Biology"):
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "question": q.get("question"),
        "option_a": q.get("option_a"),
        "option_b": q.get("option_b"),
        "option_c": q.get("option_c"),
        "option_d": q.get("option_d"),
        "correct_answer": "EMPTY",
        "question_type": q.get("question_type", "mcq"),
        "chapter": q.get("chapter"),
        "year": q.get("year"),
        "subject": subject,
        "has_diagram": q.get("has_diagram", False),
        "is_active": True
    }
    res = http_requests.post(
        f"{SUPABASE_URL}/rest/v1/pyq",
        headers=headers,
        json=payload
    )
    return res.status_code

if __name__ == "__main__":
    folder = "bio_pdfs_to_process"
    pdf_files = [f for f in os.listdir(folder) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDFs")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder, pdf_file)
        print(f"\nProcessing {pdf_file}...")
        try:
            page_images = pdf_to_page_images(pdf_path)
            print(f"  {len(page_images)} pages found")
            total_inserted = 0
            for page_num, img in enumerate(page_images, 1):
                questions = extract_questions_from_page(img, page_num)
                print(f"  Page {page_num}: {len(questions)} questions")
                for q in questions:
                    status = insert_question(q)
                    if status == 201:
                        total_inserted += 1
            print(f"  TOTAL inserted for {pdf_file}: {total_inserted}")
        except Exception as e:
            print(f"  ERROR on {pdf_file}: {e}")

    print("\nAll done.")