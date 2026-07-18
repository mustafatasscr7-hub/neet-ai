import os
import json
import re
import base64
import concurrent.futures
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

TEXT_MODEL = "claude-haiku-4-5-20251001"
MIN_TEXT_LENGTH = 40
MIN_ALNUM_RATIO = 0.3
DIAGRAM_MARKER = "<<<DIAGRAM_HERE>>>"

def build_text_extraction_prompt(subject):
    return f"""You are extracting NEET {subject} questions from raw text extracted programmatically
from one page of a PDF. The text may have minor formatting artifacts (missing line breaks,
irregular spacing) from the extraction process - use your judgement to reconstruct the
original question structure.

The literal marker "{DIAGRAM_MARKER}" has been inserted into the text at the exact position
where a real image/figure/diagram appears in the original PDF page (based on its actual
position on the page, not a guess). Use this to determine has_diagram PER QUESTION: if the
marker falls within a question's own text, or immediately after its stem/options and before
the next question starts, set has_diagram=true for THAT question only - not for other
questions on the same page that have no marker near them.

For EACH complete question in this text, extract:
- question (full question text, including any sub-statements A/B/C/D or i/ii/iii if part of the question) - do not include the marker itself in the question text
- option_a, option_b, option_c, option_d (exact text of each option)
- question_type: one of "mcq", "match_column", "assertion_reason", "statement_based"
- has_diagram: true only if the {DIAGRAM_MARKER} marker is positioned within or immediately next to THIS question, otherwise false
- year (if a NEET year tag like [NEET-2024] appears directly next to this question, else null)

Do NOT guess or infer a chapter name - that is handled separately. Do NOT attempt to determine
the correct answer, even if an answer key or answer text appears elsewhere in this text or you
can solve the question yourself - leave that to a human reviewer.

If a question is cut off at the top or bottom of this page (incomplete), SKIP it entirely - do not guess missing parts.
If there are no complete questions in this text (e.g. this is a cover page, instructions, or an answer key), return an empty array.

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

TEXT FROM THE PDF PAGE:
---
{{page_text}}
---
"""

def looks_garbled_or_empty(text, min_length=MIN_TEXT_LENGTH, min_alnum_ratio=MIN_ALNUM_RATIO):
    """Heuristic for 'this page probably has no real extractable text' - e.g. a scanned/image
    page slipped into a batch that's supposed to be text-layer PDFs only. Tuned against real
    text-layer PYQ pages, which measured 0.42-0.76 alnum ratio with 1000+ characters."""
    stripped = (text or "").strip()
    if len(stripped) < min_length:
        return True
    alnum_count = sum(c.isalnum() for c in stripped)
    return (alnum_count / len(stripped)) < min_alnum_ratio

def extract_pages_text_and_diagrams(pdf_bytes):
    """Free/local, no API call: raw text per page (PyMuPDF), with a marker inserted at the
    actual on-page position of each real (non-watermark) image, so Claude can attribute
    has_diagram to the correct individual question rather than the whole page.

    Images that repeat identically (same xref) across multiple pages are treated as a
    watermark/logo/letterhead, not a real per-question diagram - confirmed empirically on
    real PYQ PDFs, where the branding image on every page would otherwise flag every question.
    get_image_rects() (not the richer get_text("dict") image blocks, whose position/identity
    fields turned out unreliable on real files) gives the real display bbox for a confirmed xref."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_xrefs = [set(img[0] for img in page.get_images(full=True)) for page in doc]

    xref_page_count = {}
    for xrefs in page_xrefs:
        for xref in xrefs:
            xref_page_count[xref] = xref_page_count.get(xref, 0) + 1
    template_xrefs = {xref for xref, count in xref_page_count.items() if count > 1}

    pages = []
    for i, page in enumerate(doc):
        real_diagram_xrefs = page_xrefs[i] - template_xrefs
        text_entries = []  # (y0, text)
        for x0, y0, x1, y1, block_text, block_no, block_type in page.get_text("blocks"):
            if block_type == 0 and block_text.strip():
                text_entries.append((y0, block_text.strip()))

        # A one-time (non-repeating) image sitting ABOVE the first real question on the page -
        # e.g. a chapter-title banner - is a decorative header, not a per-question diagram.
        # Repeating images are already excluded above; this catches the one-time-only case.
        question_start_re = re.compile(r"^\d+\s*[.)]")
        question_start_ys = [y0 for y0, text in text_entries if question_start_re.match(text)]
        content_start_y = min(question_start_ys) if question_start_ys else 0

        entries = list(text_entries)  # (y0, text) - text blocks and diagram markers, sorted into reading order
        for xref in real_diagram_xrefs:
            for rect in page.get_image_rects(xref):
                if rect.y0 >= content_start_y:
                    entries.append((rect.y0, DIAGRAM_MARKER))
        entries.sort(key=lambda e: e[0])
        pages.append({"text": "\n".join(content for _, content in entries)})
    doc.close()
    return pages

def extract_questions_from_text(page_text, page_num, subject="Biology", model=TEXT_MODEL):
    prompt = build_text_extraction_prompt(subject).replace("{page_text}", page_text)
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
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

def scan_pdf_bytes(pdf_bytes, subject, max_workers=5):
    """Callable entry point for the admin review pipeline (server.py's /admin/scan-pdf).
    Text-layer PDFs only (not scanned/image PDFs) - extracts real text + programmatic diagram
    detection for free, then a TEXT-ONLY (no Vision, cheaper) Claude call per page to structure
    it into questions. Returns raw extracted questions - no chapter/class/correct_answer
    guessing, that's left to the TF-IDF classifier and manual review on the frontend.

    Pages are sent to Claude concurrently - doing them one at a time made scanning a
    multi-page PDF take minutes."""
    pages = extract_pages_text_and_diagrams(pdf_bytes)
    flagged_pages = []
    to_extract = {}  # page index -> page text, only for pages that pass the safety check
    for i, page in enumerate(pages):
        if looks_garbled_or_empty(page["text"]):
            flagged_pages.append({"page": i + 1, "reason": "Extraction found little or no usable text - possible scanned/image page, or a cover/blank page."})
        else:
            to_extract[i] = page["text"]

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(extract_questions_from_text, text, idx + 1, subject=subject): idx
            for idx, text in to_extract.items()
        }
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"    Error extracting page {idx + 1}: {e}")
                flagged_pages.append({"page": idx + 1, "reason": f"Extraction call failed: {e}"})
                results[idx] = []

    questions = []
    for i, page in enumerate(pages):
        page_num = i + 1
        for q in results.get(i, []):
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
    flagged_pages.sort(key=lambda f: f["page"])
    return {"questions": questions, "pages_scanned": len(pages), "flagged_pages": flagged_pages}

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