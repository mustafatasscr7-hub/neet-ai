import os
import json
import re
import base64
import concurrent.futures
import fitz
from anthropic import Anthropic
from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv
import requests as http_requests

load_dotenv()

ANTHROPIC_KEY = os.getenv("ANTHROPIC_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

client = Anthropic(api_key=ANTHROPIC_KEY)
# Text-only extraction (extract_questions_from_text, below) runs on Gemini 3.5 Flash-Lite --
# moved off DeepSeek V4 Flash, which wasn't extracting accurately enough for this task (real
# comparison on 3 PDF pages, see the migration commit). The vision path
# (extract_questions_from_page, image bytes sent to Claude) stays on `client` above -- it's a
# genuine vision task and the actual scanned-PDF path, untouched by this swap.
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

VISION_MODEL = "claude-haiku-4-5-20251001"

def build_extraction_prompt(subject):
    return f"""You are extracting NEET {subject} questions from one page of a scanned PDF.

For EACH complete question visible on this page, extract:
- question (full question text, including any sub-statements A/B/C/D or i/ii/iii if part of the question)
- option_a, option_b, option_c, option_d (exact text of each option)
- question_type: one of "mcq", "match_column", "assertion_reason", "statement_based"
- has_diagram: true if the question references a figure, diagram, image, or shows a chemical structure/graph/apparatus, otherwise false
- year (the 4-digit exam year, extracted from any bracketed exam-source tag near this question
  regardless of exact wording or exam name -- e.g. [NEET-2024], [NEET 2018], [AIPMT 1990],
  [AIPMT Screening 2008], [Re-NEET 2024] all count. Extract just the year as an integer. If no
  such tag appears, use null -- do not guess a year from context.)
- source_tag (the FULL exam-source tag text near this question, verbatim, without the brackets
  -- e.g. "AIPMT 1990", "NEET 2018", "AIPMT Screening 2008", "Re-NEET 2024". This is the same
  tag `year` was extracted from, just kept in full instead of reduced to a number. If no such
  tag appears, use null.)

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
    "year": 2024,
    "source_tag": "AIPMT Screening 2008"
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

    return _parse_extraction_json(text, page_num)

TEXT_MODEL = "gemini-3.5-flash-lite"
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

A number or unit immediately followed by "^{{...}}" (e.g. "10^{{24}}", "m^{{3}}") marks a
genuine superscript from the original PDF, detected from the source's actual font size and
vertical position, not a guess. Preserve these as LaTeX exponent notation wrapped in single
dollar signs in your output -- e.g. source text "6.023 × 10^{{24}}" should become "$6.023 \\times
10^{{24}}$" in the extracted question/option. Never strip the ^{{...}} markup or flatten it back
into plain concatenated digits (e.g. "1024"). This font-size detection occasionally
mis-highlights an unrelated glyph as superscript (e.g. a degree symbol ° rendered in a broken
font shows up as a raised "^{{1}}" mid-word) -- if a ^{{...}} span clearly isn't a real
mathematical exponent or unit power given the surrounding context, use your own scientific
judgement to write what the notation most likely actually is instead of outputting it literally.

More generally, whenever you write ANY mathematical notation in the extracted question/option
text -- subscripts (e.g. mu-naught as $\\mu_{{0}}$), superscripts/exponents, Greek letters used
as symbols (e.g. $\\varepsilon$, $\\theta$), fractions, or any other LaTeX command -- always wrap
the whole math expression in single dollar signs ($...$), covering just the mathematical part,
not the surrounding plain-English words. Never output raw LaTeX (backslash commands, ^, _, {{}})
outside of $...$ delimiters -- unwrapped LaTeX displays as literal text with visible braces and
backslashes to the student, not as rendered math. Example: source text "dimensions of
(mu0 epsilon0)^-1/2 are" should become "dimensions of $(\\mu_{{0}}\\varepsilon_{{0}})^{{-1/2}}$
are", not left as plain "(mu0 epsilon0)^-1/2" or only partially wrapped.

For EACH complete question in this text, extract:
- question (full question text, including any sub-statements A/B/C/D or i/ii/iii if part of the question) - do not include the marker itself in the question text
- option_a, option_b, option_c, option_d (exact text of each option)
- question_type: one of "mcq", "match_column", "assertion_reason", "statement_based"
- has_diagram: true only if the {DIAGRAM_MARKER} marker is positioned within or immediately next to THIS question, otherwise false
- year (the 4-digit exam year, extracted from any bracketed exam-source tag near this question
  regardless of exact wording or exam name -- e.g. [NEET-2024], [NEET 2018], [AIPMT 1990],
  [AIPMT Screening 2008], [Re-NEET 2024] all count. Extract just the year as an integer. If no
  such tag appears, use null -- do not guess a year from context.)
- source_tag (the FULL exam-source tag text near this question, verbatim, without the brackets
  -- e.g. "AIPMT 1990", "NEET 2018", "AIPMT Screening 2008", "Re-NEET 2024". This is the same
  tag `year` was extracted from, just kept in full instead of reduced to a number. If no such
  tag appears, use null.)

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
    "year": 2024,
    "source_tag": "AIPMT Screening 2008"
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

def _reconstruct_line_text(line):
    """Joins a line's spans into plain text, wrapping any span that's genuinely superscript-
    formatted (meaningfully smaller font AND raised above the line's own baseline, vs. the
    line's own dominant-size spans) in ^{...}. get_text("blocks")/plain get_text("text") only
    return character content with no font-size/position info, so a real exponent like "10"
    followed by a raised, smaller "24" span gets silently concatenated into the wrong number
    "1024" with no trace two separate spans were ever superscripted -- confirmed via
    get_text("dict") on a real NEET PDF (QP_2019.pdf p1: normal-text spans share one origin y
    at font size 10.08, the exponent digit span sits ~4pt higher at size 7.44, ~74% as tall)."""
    spans = [s for s in line.get("spans", []) if s.get("text")]
    if not spans:
        return ""
    sizes = [s["size"] for s in spans if s["text"].strip()]
    if not sizes:
        return "".join(s["text"] for s in spans)
    base_size = max(sizes)
    baseline_y = min(s["origin"][1] for s in spans if s["size"] >= base_size - 0.1)

    out = ""
    in_super = False
    for s in spans:
        text = s["text"]
        if not text:
            continue
        is_super = bool(text.strip()) and s["size"] < base_size * 0.85 and s["origin"][1] < baseline_y - 1
        if is_super and not in_super:
            out += "^{"
            in_super = True
        elif not is_super and in_super:
            out += "}"
            in_super = False
        out += text
    if in_super:
        out += "}"
    return out

def _block_text_from_dict(block):
    line_texts = [_reconstruct_line_text(line) for line in block.get("lines", [])]
    return "\n".join(t for t in line_texts if t)

def extract_pages_text_and_diagrams(pdf_bytes):
    """Free/local, no API call: raw text per page (PyMuPDF), with a marker inserted at the
    actual on-page position of each real (non-watermark) image, so the extraction model can attribute
    has_diagram to the correct individual question rather than the whole page. Also preserves
    real superscript exponents (see _reconstruct_line_text) that plain block-mode text would
    silently flatten into wrong plain-digit numbers.

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
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            block_text = _block_text_from_dict(block)
            if block_text.strip():
                text_entries.append((block["bbox"][1], block_text.strip()))

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

def _fix_unescaped_json_backslashes(text):
    """Gemini/Claude are inconsistent about JSON-escaping the literal backslashes in LaTeX
    commands they write (e.g. \\mu, \\varepsilon, \\times, \\text, \\frac) inside the extracted
    question/option strings -- confirmed live: the exact same prompt, same content, sometimes
    correctly emits \\\\mu and sometimes invalid raw \\mu, non-deterministically. Prompting
    harder didn't fix it (tried, including response_mime_type="application/json").

    IMPORTANT: this used to only double a backslash NOT already followed by a "valid JSON escape
    character" (one of \\"/bfnrtu), on the theory that those were probably intentional. That was
    wrong in a dangerous way -- \\t, \\b, \\f, \\n, \\r are ALSO the first two characters of common
    LaTeX commands (\\text, \\theta, \\times, \\tan, \\to, \\frac, \\beta, \\bar, \\nabla, \\rho,
    \\rightarrow...), and json.loads() treats \\t etc as VALID escapes -- it doesn't raise, so the
    old fallback-on-exception logic never even ran for these. The result was silent corruption:
    "\\text{AlF}_3" parsed as an actual TAB CHARACTER followed by the literal text "ext{AlF}_3",
    with no error anywhere, visible only much later as garbled "ext{...}" in the admin UI (real
    bug, reported live 2026-08-10).

    Simply doubling every lone backslash unconditionally isn't right either -- confirmed live on
    the same real extraction batch that reproduced the bug above: a genuinely-intended "\\n\\n"
    paragraph break between "Statement I:" and "Statement II:" in a real question also showed up
    as a lone backslash, and blanket-doubling it would turn a real line break into literal visible
    "\\n" text. The two cases are only distinguishable by context: the extraction prompt requires
    ALL math to be wrapped in $...$, so a lone backslash INSIDE an active $...$ span is essentially
    always an under-escaped LaTeX command, while one OUTSIDE any $ span is essentially always
    genuine prose (a real newline, etc). This walks the string tracking $-parity and only doubles
    backslashes while "inside" an odd number of $ signs; an already-correct \\\\ or \\" pair is
    consumed atomically (2 chars at once) wherever it appears, in or out of math mode, so its
    second character is never independently re-examined and re-doubled."""
    out = []
    in_math = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == '$':
            in_math = not in_math
            out.append(ch)
            i += 1
            continue
        if ch == '\\' and i + 1 < n:
            nxt = text[i + 1]
            if nxt in ('\\', '"'):
                out.append(ch); out.append(nxt); i += 2; continue
            if in_math:
                out.append('\\\\'); i += 1; continue
            out.append(ch); i += 1; continue
        out.append(ch)
        i += 1
    return ''.join(out)

def _parse_extraction_json(text, page_num):
    """Shared by both extraction paths (Gemini text-only and Claude Vision). Always runs the
    backslash fix BEFORE parsing rather than only as an exception fallback -- see
    _fix_unescaped_json_backslashes's docstring for why "did the first parse succeed" can't be
    trusted as a signal here. The fix is a verified no-op on already-correct JSON, so this is
    strictly safer than the old try-original-then-fallback order, never worse."""
    try:
        return json.loads(_fix_unescaped_json_backslashes(text))
    except Exception:
        pass
    try:
        return json.loads(text)  # last resort, in case the fix itself broke something unexpected
    except Exception as e:
        print(f"    JSON parse error on page {page_num}: {e}")
        return []

def extract_questions_from_text(page_text, page_num, subject="Biology", model=TEXT_MODEL):
    prompt = build_text_extraction_prompt(subject).replace("{page_text}", page_text)
    response = gemini_client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(max_output_tokens=4096)
    )

    text = (response.text or "").strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    return _parse_extraction_json(text, page_num)

def scan_pdf_bytes(pdf_bytes, subject, max_workers=5):
    """Callable entry point for the admin review pipeline (server.py's /admin/scan-pdf).
    Text-layer PDFs only (not scanned/image PDFs) - extracts real text + programmatic diagram
    detection for free, then a TEXT-ONLY (no Vision, cheaper) Gemini call per page to structure
    it into questions. Returns raw extracted questions - no chapter/class/correct_answer
    guessing, that's left to the TF-IDF classifier and manual review on the frontend.

    Pages are sent to Gemini concurrently - doing them one at a time made scanning a
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
                "source_tag": q.get("source_tag"),
                "source_page": page_num
            })
    flagged_pages.sort(key=lambda f: f["page"])
    return {"questions": questions, "pages_scanned": len(pages), "flagged_pages": flagged_pages}

def slice_pdf_pages(pdf_bytes, start_page, end_page):
    """start_page/end_page are 1-indexed, inclusive, matching how an admin reads printed
    page numbers. Builds a standalone in-memory PDF for just that range via PyMuPDF, so
    scan_pdf_bytes() runs completely unmodified against it -- no changes to the proven
    single-subject PYQ scan path."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total = len(doc)
    if start_page < 1 or end_page < start_page or end_page > total:
        doc.close()
        raise ValueError(f"Invalid page range {start_page}-{end_page} for a {total}-page PDF")
    doc.select(list(range(start_page - 1, end_page)))
    out_bytes = doc.tobytes()
    doc.close()
    return out_bytes

MOCK_TEST_SUBJECT_ORDER = ["Physics", "Chemistry", "Biology"]

def scan_mock_test_pdf(pdf_bytes, ranges, max_workers=3):
    """ranges: {"Physics": (start,end), "Chemistry": (start,end), "Biology": (start,end)}.
    Slices once per subject, then calls the existing scan_pdf_bytes() 3x concurrently (each
    already parallelizes its own pages internally). question_order is assigned in fixed
    Physics->Chemistry->Biology order regardless of dict/completion order, so the served
    exam's section order can't get scrambled by an out-of-order scan completion."""
    sliced = {s: slice_pdf_pages(pdf_bytes, *ranges[s]) for s in MOCK_TEST_SUBJECT_ORDER}
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_subject = {executor.submit(scan_pdf_bytes, sliced[s], s): s for s in MOCK_TEST_SUBJECT_ORDER}
        for future in concurrent.futures.as_completed(future_to_subject):
            s = future_to_subject[future]
            try:
                results[s] = future.result()
            except Exception as e:
                results[s] = {"error": str(e), "questions": [], "pages_scanned": 0, "flagged_pages": []}

    questions = []
    flagged_pages = []
    per_subject_counts = {}
    errors = {}
    order = 1
    for s in MOCK_TEST_SUBJECT_ORDER:
        r = results[s]
        if r.get("error"):
            errors[s] = r["error"]
        for f in r.get("flagged_pages", []):
            flagged_pages.append({**f, "subject": s})
        subj_questions = r.get("questions", [])
        per_subject_counts[s] = len(subj_questions)
        for q in subj_questions:
            questions.append({**q, "subject": s, "question_order": order})
            order += 1

    return {
        "questions": questions,
        "flagged_pages": flagged_pages,
        "per_subject_counts": per_subject_counts,
        "errors": errors
    }

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
        "source_tag": q.get("source_tag"),
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