"""Extraction mode for SCANNED, image-only PYQ PDFs (no text layer at all) -- e.g. a phone/scanner
capture of a coaching book saved through macOS Preview, where every page is one full-page JPEG.
process_pyq_vision.scan_pdf_bytes ("Standard") flags every such page as "little or no usable text"
and never sends it anywhere; process_pyq_book_3col needs a real text layer too. This mode renders
each page to an image and sends it to Gemini (the same client and model the Standard pipeline
already uses for its text-only extraction) to read.

Deliberately a SEPARATE module, never auto-selected -- same reasoning as process_pyq_book_3col's
own docstring: neither of the other two modes is modified or routed through here.

Split of responsibilities (this is what makes the tricky parts unit-testable without a model):
  - The model does the part that genuinely needs vision: transcribe each question/continuation/
    answer-key item on ONE page and report where it sits ("box_2d", normalized 0-1000).
  - Plain Python does everything structural, deterministically: column/band reading order
    (order_page_items), joining a question that continues across a column or page break
    (merge_document_items), numbering gaps/duplicates, and attaching answer-key letters ONLY by
    exact question-number match (never guessed).

Built against DISHA_36_YEARS_BIOLOGY_11thTEST.pdf (4 scanned pages, 2 columns, "NCERT | Page-x |
year" tags, a full-width answer-key table on the last page between a 2-column question region
and a 2-column "Hints & Solutions" region -- the band-splitting in order_page_items exists for
exactly that page shape).
"""
import concurrent.futures
import hashlib
import json
import os
import re

import fitz
from google.genai import types as genai_types

from process_pyq_vision import (
    gemini_client,
    TEXT_MODEL,
    _fix_unescaped_json_backslashes,
    wrap_bare_latex_notation,
    format_numbered_substatements,
    _flag_suspicious_mcq_options,
)
from process_pyq_book_3col import _YEAR_DIGITS_RE, _normalize_ws

# Same Gemini model as the Standard pipeline's extraction call (process_pyq_vision.TEXT_MODEL),
# here given an image instead of extracted text.
SCANNED_MODEL = TEXT_MODEL
# The DISHA source scans are ~1600px across a 595pt-wide page (~195 DPI native), so 200 DPI keeps
# every pixel the scan actually has without upsampling noise into a much bigger upload.
SCANNED_DPI = 200
SCANNED_JPEG_QUALITY = 90
# Bump whenever the prompt or the shape of the cached per-page response changes, so old cache
# entries are never reused for a different prompt.
SCANNED_PROMPT_VERSION = "scanned-v1"
DEFAULT_MAX_PAGES = 50

# Mirrors server.py's GEMINI_FLASH_LITE_RATE (USD per 1M tokens) -- duplicated rather than
# imported because server.py imports this module, not the other way round.
# test_pyq_scanned.py asserts the two stay equal.
GEMINI_RATE_PER_M = {"input": 0.30, "output": 2.50}
# Per-page token estimate for the pre-run cost estimate, calibrated against the real 4-page DISHA
# run: input was 2260 on every page (fixed-size image + prompt), output 856-2850 (thinking
# included) depending on how many questions the page holds. Rounded UP past the worst page on
# purpose -- an estimate that comes in under the real bill is the worse failure.
EST_INPUT_TOKENS_PER_PAGE = 2300
EST_OUTPUT_TOKENS_PER_PAGE = 3500

SCANNED_PAGE_CACHE_DIR = os.getenv(
    "SCANNED_PAGE_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".scan_cache", "scanned_pages"),
)

_LETTERS = ("a", "b", "c", "d")


def build_scanned_page_prompt(subject):
    return f"""You are reading ONE scanned page of a NEET {subject} coaching book of previous-year
questions. The scan may be slightly tilted, unevenly lit, or cropped at an edge.

Question text is normally laid out in TWO columns. Some blocks span the full page width (chapter
title banners, trend-analysis tables, answer-key tables, section headings).

Return a JSON array. Every item MUST have "kind" and "box_2d": [ymin, xmin, ymax, xmax], the
item's own region on the page, normalized to 0-1000. Item kinds:

1. "question" -- a question whose own printed number (e.g. "12.") is visible on this page:
   {{"kind": "question", "number": 12, "question": "stem text", "options": {{"a": "...", "b": "...",
     "c": "...", "d": "..."}}, "year_tag": "2019", "ncert_tag": "Page-7 / N-8",
     "question_type": "mcq", "has_diagram": false, "box_2d": [...]}}
   - Include ONLY options whose (a)/(b)/(c)/(d) marker is printed inside this item's own region.
     If the question runs off the bottom of the column and its remaining options continue at the
     top of the next column or next page, leave those letters OUT -- never invent them.
   - box_2d covers the question's region within ONE column only.
   - question_type: one of "mcq", "match_column", "assertion_reason", "statement_based".
   - has_diagram: true only if the question shows or refers to a figure/diagram/structure.

2. "continuation" -- text at the TOP of a column that has NO question number of its own because it
   continues a question from the previous column or the previous page:
   {{"kind": "continuation", "text": "unmarked leading text, if any (the rest of a stem or of the
     previous option)", "options": {{"b": "...", "c": "..."}}, "box_2d": [...]}}

3. "answer_key" -- an answer-key table or list mapping question numbers to option letters:
   {{"kind": "answer_key", "entries": {{"1": "b", "2": "c"}}, "box_2d": [...]}}
   Read every cell; letters only (a/b/c/d). Omit a number whose letter you cannot read.

4. "chapter_start" -- a large chapter-OPENING title banner (e.g. a big "1  The Living World"
   banner at the start of a chapter). NOT a running header/footer such as
   "Class-XI/Chapter-1/The Living World".
   {{"kind": "chapter_start", "title": "The Living World", "box_2d": [...]}}

5. "solutions_start" -- the heading where a hints / solutions / explanations section begins
   (e.g. "Hints & Solutions"). Do NOT transcribe anything inside that section: its numbered
   entries are explanations, not questions.
   {{"kind": "solutions_start", "box_2d": [...]}}

Skip entirely: trend-analysis tables, topic headings (e.g. "Topic-1: What is Living?"), running
headers and footers, page numbers, watermarks / channel handles, legends.

Tags: a year tag sits at the right end of a question's first line(s), often shaded -- e.g. "2016",
"2003, 04", "1990,91,92,96", "2010 M", "Ph-II 2022". Copy it verbatim into year_tag (null if
none). An NCERT tag looks like "NCERT | Page-11 / N-8 | 2021": put "Page-11 / N-8" in ncert_tag
and "2021" in year_tag. Remove both tags from the question text.

Match-the-column and statement questions: keep the columns/statements in the question text, one
row per line separated by "\\n". Copy options exactly as printed, e.g. "(i), (iii) and (iv) only".
When the options are printed as a grid under an "(A) (B) (C) (D)" header row, write each option
as "(A)-(i), (B)-(iv), (C)-(iii), (D)-(ii)".

Math and chemistry notation: wrap ONLY the mathematical part in single dollar signs using LaTeX,
e.g. "a lower $O_{{2}}$ requirement", "$CO_{{2}}$", "$10^{{-3}}$". Plain words stay outside $...$.

Transcribe exactly what is printed: do not correct wording, do not solve anything, and never
supply an answer that is not printed in an answer key. Use "" for an option you cannot read. If
text is illegible or cut off at the page edge, transcribe what you can and add
"legibility_issue": "<short description>" to that item.

Return ONLY the JSON array, no other text."""


def build_scanned_page_prompt_hindi(subject):
    """Hindi-language variant of build_scanned_page_prompt. Deliberately drops the "answer_key"/
    "solutions_start" item kinds entirely -- the task this was built for wants the answer key
    ignored outright (never transcribed, never used to fill correct_answer), so the model is
    simply never told that kind exists. merge_document_items/finalize_segments handle that fine
    unchanged: with no answer_key items ever produced, every question's own `key.get(number)`
    lookup in finalize_segments is just always empty, which is exactly "correct_answer always
    comes back ''" with no extra code needed -- see scan_scanned_pdf_bytes's language=="hi"
    branch for the one place this does need an explicit guard (suppressing the "no answer key
    found" flag, which only makes sense when a key was ever expected)."""
    return f"""You are reading ONE scanned page of a Hindi-language NEET {subject} previous-year-
questions book or paper. The scan may be slightly tilted, unevenly lit, or cropped at an edge. The
question text is in Hindi (Devanagari script), though option values may legitimately include
English/Latin letters, numerals, or chemical/mathematical notation.

Question text is normally laid out in TWO columns. Some blocks span the full page width (chapter
title banners, weightage/trend-analysis tables, tracker tables, section banners, footnotes,
running headers/footers, or an answer-key block). SKIP all of those entirely -- do not transcribe
them, including any answer-key block. Never report or guess an answer for any question; that is
left entirely to a human reviewer.

Extract EVERY complete, numbered MCQ visible on this page. Read the LEFT column fully, top to
bottom, before the RIGHT column.

Return a JSON array. Every item MUST have "kind" and "box_2d": [ymin, xmin, ymax, xmax], the
item's own region on the page, normalized to 0-1000. Item kinds:

1. "question" -- a question whose own printed number (e.g. "12.") is visible on this page:
   {{"kind": "question", "number": 12, "question": "Hindi stem text, Devanagari preserved exactly
     as printed", "options": {{"a": "...", "b": "...", "c": "...", "d": "..."}}, "year_tag":
     "2019", "question_type": "mcq", "has_diagram": false, "box_2d": [...]}}
   - Include ONLY options whose marker is printed inside this item's own region. If the question
     runs off the bottom of the column and its remaining options continue at the top of the next
     column or next page, leave those letters OUT -- never invent them.
   - box_2d covers the question's region within ONE column only.
   - question_type: one of "mcq", "match_column", "assertion_reason", "statement_based".
   - has_diagram: true only if the question shows or refers to a figure/diagram/structure.

2. "continuation" -- text at the TOP of a column that has NO question number of its own because it
   continues a question from the previous column or the previous page (a question may genuinely
   start in one column and finish in the next -- this is how that gets stitched back together):
   {{"kind": "continuation", "text": "unmarked leading text, if any", "options": {{"b": "...", "c":
     "..."}}, "box_2d": [...]}}

3. "chapter_start" -- a large chapter-OPENING title banner. NOT a running header/footer.
   {{"kind": "chapter_start", "title": "...", "box_2d": [...]}}

Keep all Devanagari text exactly as printed -- do not transliterate, translate, or normalize it.

A year tag sits in brackets at the end of a question (e.g. "[2019]", "[Mains 2010]",
"[RE-EXAM 2026]") -- copy the bracket's contents verbatim into year_tag (null if none) and REMOVE
it from the question text itself; it does not belong in the question. Keep the full tag text
(e.g. "Mains 2010"), not just the digits.

Math and chemistry notation: wrap ONLY the mathematical part in single dollar signs using LaTeX,
e.g. "$CO_{{2}}$", "$10^{{-3}}$". Plain Hindi/English words stay outside $...$.

Transcribe exactly what is printed: do not correct wording, do not solve anything, and never
supply an answer. Use "" for an option you cannot read. If text is illegible or cut off at the
page edge, transcribe what you can and add "legibility_issue": "<short description>" to that item.

Return ONLY the JSON array, no other text."""


# ---------------------------------------------------------------- page rendering + cache

def render_page_jpeg(page, dpi=SCANNED_DPI):
    """Renders through PyMuPDF rather than pulling the embedded JPEG out directly -- the DISHA
    scans are stored landscape and rotated into place by the page's own transform, which only a
    real render applies."""
    pix = page.get_pixmap(dpi=dpi)
    return pix.tobytes("jpg", jpg_quality=SCANNED_JPEG_QUALITY)


def page_cache_key(image_bytes, subject, language="en", tag=""):
    """tag distinguishes cache entries that share the same page image, subject and language but
    aren't interchangeable -- e.g. Hindi mode's two independent extraction passes over the same
    page (see extract_page_items_two_pass), which must never read back each other's cached result."""
    h = hashlib.sha256()
    for part in (SCANNED_PROMPT_VERSION, SCANNED_MODEL, subject, language, tag, str(SCANNED_DPI)):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    h.update(image_bytes)
    return h.hexdigest()


def _cache_path(key, cache_dir):
    return os.path.join(cache_dir, f"{key}.json")


def cache_get(key, cache_dir=None):
    path = _cache_path(key, cache_dir or SCANNED_PAGE_CACHE_DIR)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def cache_put(key, value, cache_dir=None):
    """Never fatal -- a read-only or full disk just means this page gets paid for again next
    time, not a failed scan."""
    cache_dir = cache_dir or SCANNED_PAGE_CACHE_DIR
    try:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = _cache_path(key, cache_dir) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)
        os.replace(tmp, _cache_path(key, cache_dir))
    except OSError as e:
        print(f"    Scanned mode: could not write page cache ({e}) -- non-fatal")


def render_pdf_pages(pdf_bytes, subject, language="en"):
    """Returns [(image_bytes, cache_key), ...], one per page. For language="hi", cache_key is
    just pass 1's key (a representative, not the whole story -- Hindi mode caches its two passes
    independently, see extract_page_items_two_pass/page_cache_key's own docstring); good enough
    for estimate_scanned_pdf_cost's "how many pages are already cached" approximation."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        out = []
        for page in doc:
            img = render_page_jpeg(page)
            tag = "two_pass" if language == "hi" else ""
            out.append((img, page_cache_key(img, subject, language, tag)))
        return out
    finally:
        doc.close()


# How many separate dark runs along one edge strip of the scan count as "text cut off by the
# edge". Measured live on DISHA: the two genuinely cropped right edges (pages 2 and 4) had 33 and
# 25 runs; every other edge had at most 4 (a thumb in the margin, the scan border, a banner).
_CROPPED_EDGE_MIN_RUNS = 10


def cropped_edge_sides(page):
    """Which sides ('left'/'right') of a page's scanned image cut through lines of text. The
    model can't be relied on to report this itself -- seen live on DISHA page 4, it silently
    completed "photograp" / "pressed ar" into whole words without setting legibility_issue -- so
    it's measured from the pixels instead. Only for a page whose single largest image covers most
    of the page (i.e. a real scan); never fatal."""
    try:
        import numpy as np
        rects = [r for xref in {img[0] for img in page.get_images(full=True)} for r in page.get_image_rects(xref)]
        if not rects:
            return []
        rect = max(rects, key=lambda r: r.width * r.height)
        if rect.width * rect.height < 0.6 * page.rect.width * page.rect.height:
            return []
        pix = page.get_pixmap(dpi=100, colorspace=fitz.csGRAY, clip=rect)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        h = arr.shape[0]
        body = arr[int(h * 0.1):int(h * 0.9)]  # skip header/footer decoration bands
        sides = []
        for side, strip in (("left", body[:, :6]), ("right", body[:, -6:])):
            dark_rows = (strip < 90).any(axis=1).astype(int)
            if int((np.diff(dark_rows) == 1).sum()) >= _CROPPED_EDGE_MIN_RUNS:
                sides.append(side)
        return sides
    except Exception as e:
        print(f"    Scanned mode: edge-crop check skipped for a page ({e}) -- non-fatal")
        return []


def pdf_cropped_edges(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return [cropped_edge_sides(page) for page in doc]
    finally:
        doc.close()


# ---------------------------------------------------------------- cost

def tokens_to_usd(input_tokens, output_tokens):
    return (input_tokens * GEMINI_RATE_PER_M["input"] + output_tokens * GEMINI_RATE_PER_M["output"]) / 1_000_000


def estimate_scanned_pdf_cost(pdf_bytes, subject, cache_dir=None, language="en"):
    """Pre-run estimate for the admin UI. Pages already in the cache cost nothing, so they're
    counted separately rather than folded into the estimate. Hindi mode runs TWO independent
    passes per page (see extract_page_items_two_pass), so its per-page token estimate is doubled."""
    rendered = render_pdf_pages(pdf_bytes, subject, language)
    cached = sum(1 for _, key in rendered if cache_get(key, cache_dir) is not None)
    uncached = len(rendered) - cached
    multiplier = 2 if language == "hi" else 1
    return {
        "pages": len(rendered),
        "cached_pages": cached,
        "uncached_pages": uncached,
        "estimated_usd": round(tokens_to_usd(uncached * EST_INPUT_TOKENS_PER_PAGE * multiplier,
                                              uncached * EST_OUTPUT_TOKENS_PER_PAGE * multiplier), 4),
        "max_pages": DEFAULT_MAX_PAGES,
        "model": SCANNED_MODEL,
    }


# ---------------------------------------------------------------- model call

def parse_page_response(text):
    """Model text -> list of item dicts. Reuses process_pyq_vision's backslash repair (the same
    unescaped-LaTeX-in-JSON problem applies to any Gemini JSON here). Raises on a genuine parse
    failure, same contract as process_pyq_vision._parse_extraction_json, so the caller flags the
    page instead of it silently contributing nothing."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        items = json.loads(_fix_unescaped_json_backslashes(text))
    except Exception:
        try:
            items = json.loads(text)
        except Exception as e:
            raise ValueError(f"Could not parse model response as JSON: {e}") from e
    if isinstance(items, dict):
        items = items.get("items", [items])
    if not isinstance(items, list):
        raise ValueError("Model response was not a JSON array")
    return [it for it in items if isinstance(it, dict)]


def extract_page_items(image_bytes, subject, model=SCANNED_MODEL, language="en"):
    """One Gemini call for one page. Returns (items, usage) where usage has input/output tokens
    (thinking tokens count as output, which is how Gemini bills them)."""
    prompt = build_scanned_page_prompt_hindi(subject) if language == "hi" else build_scanned_page_prompt(subject)
    response = gemini_client.models.generate_content(
        model=model,
        contents=[
            genai_types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt,
        ],
        config=genai_types.GenerateContentConfig(max_output_tokens=16384, response_mime_type="application/json"),
    )
    um = response.usage_metadata
    usage = {
        "input_tokens": (um.prompt_token_count or 0) if um else 0,
        "output_tokens": ((um.candidates_token_count or 0) + (getattr(um, "thoughts_token_count", 0) or 0)) if um else 0,
    }
    return parse_page_response(response.text), usage


def _compare_hindi_passes(items1, items2):
    """Hindi-only, per page: question numbers whose transcribed text disagrees between two
    independent extraction passes over the SAME page image -- a two-pass check can't silently
    average two different transcriptions away, so a genuine disagreement is left for a human to
    settle instead (see scan_scanned_pdf_bytes's language=="hi" branch for where this feeds into
    needs_review/review_reason)."""
    by_num_1 = {it.get("number"): it for it in items1 if it.get("kind") == "question" and it.get("number") is not None}
    by_num_2 = {it.get("number"): it for it in items2 if it.get("kind") == "question" and it.get("number") is not None}
    disagreements = []
    for num in sorted(by_num_1.keys() & by_num_2.keys()):
        a, b = by_num_1[num], by_num_2[num]
        if _normalize_ws(_clean(a.get("question"))) != _normalize_ws(_clean(b.get("question"))):
            disagreements.append(num)
            continue
        if _norm_options(a.get("options")) != _norm_options(b.get("options")):
            disagreements.append(num)
    return disagreements


def extract_page_items_two_pass(image_bytes, subject, model=SCANNED_MODEL):
    """Hindi-only entry point: runs extract_page_items TWICE over the same page image
    (independent model calls) and reports any question number whose text disagrees between the
    two. Returns (items, usage, disagreements, pass_failed): items is the usable pass's result
    (pass 1, when both succeed), usage is every call actually made combined, disagreements is a
    sorted list of question numbers to flag, pass_failed is True when exactly one of the two
    calls raised (a transient model/parse error on just one of them -- confirmed live: Gemini
    occasionally returns truncated/invalid JSON on one call of a pair with no retry) -- the page
    still gets the OTHER pass's result instead of losing the whole page, but with no partner to
    cross-check against, so the comparison is skipped for this page.

    Raises only if BOTH calls fail -- same contract as extract_page_items, so the caller (scan_
    scanned_pdf_bytes) flags the whole page exactly as it already does for the English path.

    No caching of its own -- scan_scanned_pdf_bytes caches this function's whole combined result
    under one page-level key, the same way it already caches a single English pass."""
    results = []
    for _ in range(2):
        try:
            results.append(extract_page_items(image_bytes, subject, model=model, language="hi"))
        except Exception as e:
            results.append(e)
    successes = [r for r in results if not isinstance(r, Exception)]
    if not successes:
        raise results[0]
    usage = {
        "input_tokens": sum(r[1].get("input_tokens", 0) for r in successes),
        "output_tokens": sum(r[1].get("output_tokens", 0) for r in successes),
    }
    if len(successes) == 1:
        return successes[0][0], usage, [], True
    items1, items2 = successes[0][0], successes[1][0]
    return items1, usage, _compare_hindi_passes(items1, items2), False


# ---------------------------------------------------------------- reading order (deterministic)

_GUTTER = 500           # page midline in box_2d's 0-1000 space
# A column-only box may poke this far past the midline -- model boxes are approximate (seen live:
# DISHA page 4's left-column Q38 came back with xmax=573), and tilted scans make it worse. A real
# full-width block (answer-key table, banner, section heading) reaches well into BOTH columns.
_GUTTER_SLACK = 100


def _normalize_box(box):
    """Gemini sometimes wraps a single box in an extra list ([[y0, x0, y1, x1]] -- seen live on
    DISHA page 1's chapter banner)."""
    if isinstance(box, (list, tuple)) and len(box) == 1 and isinstance(box[0], (list, tuple)):
        return list(box[0])
    return box


def _valid_box(box):
    return (isinstance(box, (list, tuple)) and len(box) == 4
            and all(isinstance(v, (int, float)) for v in box)
            and box[0] <= box[2] and box[1] <= box[3])


def item_column(box):
    """'full' if the box clearly spans the gutter, else 'left'/'right' by horizontal center."""
    _, xmin, _, xmax = box
    if xmin < _GUTTER - _GUTTER_SLACK and xmax > _GUTTER + _GUTTER_SLACK:
        return "full"
    return "left" if (xmin + xmax) / 2 < _GUTTER else "right"


def order_page_items(items):
    """True reading order for one page: the page is cut into horizontal bands by its full-width
    items (answer-key table, banner); within each band the left column is read top to bottom,
    then the right column, and each full-width item sits between the band above and below it.
    This is what DISHA's last page needs -- Q38's own options (b)-(d) at the top of the RIGHT
    column must come before the full-width answer key, and the 2-column solutions region below
    the key must come after it, which a plain "all left, then all right" sort gets wrong.

    Returns (ordered_items, used_geometry). If any item lacks a usable box_2d, the model's own
    order is kept unchanged (used_geometry=False) so the caller can flag the page -- a partial
    geometric sort mixed with unknown positions would just be a different guess.

    A "continuation" always sorts FIRST within its own band+column, whatever its y says: by
    definition it's the unnumbered text at the top of a column, and the model's boxes are only
    approximate -- seen live on DISHA page 4, where Q38's box was reported starting ABOVE Q37's
    continuation grid, which a pure y-sort then glued onto Q38 instead of Q37."""
    for it in items:
        if "box_2d" in it:
            it["box_2d"] = _normalize_box(it["box_2d"])
    if not items or not all(_valid_box(it.get("box_2d")) for it in items):
        return list(items), False
    cols = [item_column(it["box_2d"]) for it in items]
    full_tops = sorted(it["box_2d"][0] for it, c in zip(items, cols) if c == "full")

    def key(idx):
        it, col = items[idx], cols[idx]
        ymin = it["box_2d"][0]
        if col == "full":
            band = full_tops.index(ymin)
            return (band * 2 + 1, 0, 0, ymin, idx)
        band = sum(1 for t in full_tops if t < ymin)
        return (band * 2, 0 if col == "left" else 1, 0 if it.get("kind") == "continuation" else 1, ymin, idx)

    return [items[i] for i in sorted(range(len(items)), key=key)], True


# ---------------------------------------------------------------- joining + flagging (deterministic)

def _clean(text):
    return text.strip() if isinstance(text, str) else ""


def _int_or_none(v):
    try:
        return int(str(v).strip().rstrip("."))
    except (TypeError, ValueError):
        return None


def _norm_options(raw):
    out = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            letter = str(k).strip().strip("()").lower()
            if letter in _LETTERS and _clean(v):
                out[letter] = _clean(v)
    return out


def _append(a, b):
    return f"{a} {b}".strip() if a else b


# Lowercase only, and only at the start of the text or after whitespace: "(A) Herbarium" in a
# match-the-column stem is a column label, not an option marker. That's also why this doesn't
# reuse process_pyq_book_3col._parse_options, whose marker regex is case-insensitive.
_OPTION_MARKER_RE = re.compile(r'(?:^|(?<=\s))\(([a-d])\)\s*')


def split_option_markers(text):
    """'lead (c) X (d) Y' -> ('lead', {'c': 'X', 'd': 'Y'}). Whatever sits before the first
    marker is returned as lead; text between one marker and the next belongs to the first."""
    matches = list(_OPTION_MARKER_RE.finditer(text or ""))
    if not matches:
        return _normalize_ws(text or ""), {}
    lead = _normalize_ws(text[:matches[0].start()])
    opts = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = _normalize_ws(text[m.end():end])
        if content and m.group(1) not in opts:
            opts[m.group(1)] = content
    return lead, opts


def _join_continuation(q, item):
    """Attach a continuation fragment to the open question q (mutates q). Unmarked leading text
    continues whatever was being written when the column/page ended: the last option if any
    option has started, otherwise the stem. Lettered options fill their own slots; a letter that
    already has text gets the new text appended and a flag, rather than one silently overwriting
    the other.

    Option markers inside "text" are split out here rather than trusted to the model's own
    text/options split -- seen live on DISHA page 2: Q15's "(c) Felis tigris ... (d) E.coli ..."
    came back entirely inside "text", which would otherwise have been glued onto option (b)."""
    lead, text_opts = split_option_markers(_clean(item.get("text")))
    if lead:
        last = next((l for l in reversed(_LETTERS) if q["options"].get(l)), None)
        if last:
            q["options"][last] = _append(q["options"][last], lead)
        else:
            q["question"] = _append(q["question"], lead)
    incoming = dict(text_opts)
    incoming.update(_norm_options(item.get("options")))
    for letter, val in sorted(incoming.items()):
        if q["options"].get(letter):
            q["options"][letter] = _append(q["options"][letter], val)
            q["issues"].append(f"option ({letter}) text was found twice (before and after a column/page break) and was joined -- check it")
        else:
            q["options"][letter] = val
    if item.get("legibility_issue"):
        q["issues"].append(f"model reported: {_clean(item['legibility_issue'])}")
    q["continued"] = True


def _new_question(item, page):
    number = _int_or_none(item.get("number"))
    year_tag = _clean(item.get("year_tag")) or None
    q = {
        "number": number,
        "question": _clean(item.get("question")),
        "options": _norm_options(item.get("options")),
        "year_tag": year_tag,
        "ncert_tag": _clean(item.get("ncert_tag")) or None,
        "question_type": item.get("question_type") if item.get("question_type") in ("mcq", "match_column", "assertion_reason", "statement_based") else "mcq",
        "has_diagram": bool(item.get("has_diagram")),
        "page": page,
        "issues": [],
        "continued": False,
    }
    if number is None:
        q["issues"].append("no question number could be read")
    if item.get("legibility_issue"):
        q["issues"].append(f"model reported: {_clean(item['legibility_issue'])}")
    return q


def merge_document_items(pages_items):
    """pages_items: [(page_num, ordered_items), ...] in page order (each page already run through
    order_page_items). Walks the whole document as ONE reading-order sequence, so a question that
    runs off the bottom of a column or page is completed by the very next continuation item,
    wherever it lands.

    Returns (segments, page_flags). A segment is one chapter's worth of questions plus the answer
    key found inside it ({"questions": [...], "answer_key": {num: letter}, "key_conflicts": [...],
    "starts_at_chapter": bool}) -- numbering restarts at each chapter_start, so gaps, duplicates
    and key matching are all scoped per segment. Anything that can't be attached anywhere is
    reported in page_flags, never dropped silently."""
    segments = []
    page_flags = []

    def new_segment(at_chapter):
        seg = {"questions": [], "answer_key": {}, "key_conflicts": [], "starts_at_chapter": at_chapter}
        segments.append(seg)
        return seg

    seg = new_segment(False)
    open_q = None
    in_solutions = False

    for page, items in pages_items:
        for item in items:
            kind = item.get("kind")
            if kind == "chapter_start":
                seg = new_segment(True) if (seg["questions"] or seg["answer_key"]) else seg
                seg["starts_at_chapter"] = True
                open_q = None
                in_solutions = False
            elif kind == "solutions_start":
                open_q = None
                in_solutions = True
            elif kind == "answer_key":
                # A key never continues a question -- whatever follows it can't be that
                # question's tail either.
                open_q = None
                entries = item.get("entries") if isinstance(item.get("entries"), dict) else {}
                for k, v in entries.items():
                    num = _int_or_none(k)
                    letter = str(v).strip().strip("()").lower()
                    if num is None or letter not in _LETTERS:
                        continue
                    prev = seg["answer_key"].get(num)
                    if prev and prev != letter:
                        seg["key_conflicts"].append((num, prev, letter, page))
                    else:
                        seg["answer_key"][num] = letter
            elif in_solutions:
                continue  # explanations, not questions
            elif kind == "question":
                open_q = _new_question(item, page)
                seg["questions"].append(open_q)
                item["_question_number"] = open_q["number"]
            elif kind == "continuation":
                if open_q is not None:
                    _join_continuation(open_q, item)
                    item["_question_number"] = open_q["number"]
                else:
                    snippet = _clean(item.get("text")) or " / ".join(f"({l}) {v}" for l, v in _norm_options(item.get("options")).items())
                    page_flags.append({
                        "page": page,
                        "reason": f"Text at the top of a column continues a question that isn't in this scan (it probably started on an earlier page): “{snippet[:80]}”. Use \"+ Add Missed Question\" if it's needed.",
                    })
    return [s for s in segments if s["questions"] or s["answer_key"]], page_flags


def find_numbering_issues(numbers, starts_at_chapter):
    """numbers: question numbers in reading order (None for unreadable ones). Returns
    (missing, duplicates). A gap is any integer missing between the lowest and highest number
    found; the run from 1 up to the first number only counts when this segment begins at a
    chapter banner -- a scan that starts mid-chapter legitimately begins at e.g. Q10."""
    found = [n for n in numbers if n is not None]
    if not found:
        return [], []
    low = 1 if starts_at_chapter else min(found)
    seen, dupes = set(), []
    for n in found:
        if n in seen and n not in dupes:
            dupes.append(n)
        seen.add(n)
    missing = sorted(set(range(low, max(found) + 1)) - seen)
    return missing, dupes


def _maybe_resplit_options(q):
    """Fallback: if (a)-(d) markers ended up inside the stem (seen live: Q9's options arrived as a
    page-2 continuation with no options of its own, so they were joined onto the stem), split
    them back out deterministically. Never overwrites an option that's already filled."""
    if len(q["options"]) == 4:
        return
    lead, recovered = split_option_markers(q["question"])
    if recovered and "a" in recovered:
        q["question"] = lead
        for l, v in recovered.items():
            q["options"].setdefault(l, v)


_ROMAN_GRID_RE = re.compile(r'^\(?([ivx]+)\)?\s+\(?([ivx]+)\)?\s+\(?([ivx]+)\)?\s+\(?([ivx]+)\)?$')


def _expand_match_grid_options(q):
    """A match-the-column answer grid printed under an "(A) (B) (C) (D)" header reads back as
    bare "(i) (iv) (iii) (ii)" per option, which means nothing once the header row is gone.
    Rewritten as "(A)-(i), (B)-(iv), ..." only when every option is exactly four roman numerals
    and the stem really has (A)-(D) column labels -- anything else is left as transcribed."""
    if not all(q["options"].get(l) for l in _LETTERS):
        return
    if not all(f"({c})" in q["question"] for c in "ABCD"):
        return
    grids = {l: _ROMAN_GRID_RE.match(q["options"][l].strip()) for l in _LETTERS}
    if all(grids.values()):
        for l, m in grids.items():
            q["options"][l] = ", ".join(f"({c})-({r})" for c, r in zip("ABCD", m.groups()))


def finalize_segments(segments, track_answer_key=True):
    """Segments -> (questions in the existing scan output shape, flags). Applies the same text
    post-processing the Standard pipeline does (bare-LaTeX wrapping, numbered sub-statements,
    the identical/empty-option check), then every review flag this mode knows about.

    track_answer_key=False (Hindi mode): the model is never asked for an answer key at all (see
    build_scanned_page_prompt_hindi), so `key` is always empty and every question would otherwise
    get flagged as "no answer key found" -- a false positive, since correct_answer="" is the
    INTENDED result here, not a defect. key_conflicts/unmatched_keys never fire either way (both
    require an answer_key item to exist), so only the no_key flag below needs the guard."""
    questions, flags = [], []
    no_key = []
    for seg in segments:
        numbers = [q["number"] for q in seg["questions"]]
        missing, dupes = find_numbering_issues(numbers, seg["starts_at_chapter"])
        key = seg["answer_key"]

        for q in seg["questions"]:
            _maybe_resplit_options(q)
            _expand_match_grid_options(q)
            if q["number"] in dupes:
                q["issues"].append(f"question number {q['number']} appears more than once")
            missing_opts = [l for l in _LETTERS if not q["options"].get(l)]
            if missing_opts:
                q["issues"].append(f"only {4 - len(missing_opts)} of 4 options found (missing {', '.join(l.upper() for l in missing_opts)})")
            if not q["question"]:
                q["issues"].append("question text came back empty")

            answer = key.get(q["number"]) if q["number"] is not None and q["number"] not in dupes else None
            if answer is None:
                no_key.append(q)

            out = {
                "question": format_numbered_substatements(wrap_bare_latex_notation(q["question"])),
                "option_a": wrap_bare_latex_notation(q["options"].get("a", "")),
                "option_b": wrap_bare_latex_notation(q["options"].get("b", "")),
                "option_c": wrap_bare_latex_notation(q["options"].get("c", "")),
                "option_d": wrap_bare_latex_notation(q["options"].get("d", "")),
                "correct_answer": answer or "",
                "question_type": q["question_type"],
                "has_diagram": q["has_diagram"],
                "year": int(m.group(0)) if (m := _YEAR_DIGITS_RE.search(q["year_tag"] or "")) else None,
                "source_tag": q["year_tag"],
                "ncert_tag": q["ncert_tag"],
                "question_number": q["number"],
                "source_page": q["page"],
                "needs_review": False,
                "review_reason": None,
            }
            # Same identical/empty-option check the Standard pipeline runs on every question --
            # only consulted when this mode hasn't already flagged a missing option itself.
            if not missing_opts:
                _flag_suspicious_mcq_options(out)
                if out["needs_review"]:
                    q["issues"].append(out["review_reason"])
            if q["issues"]:
                out["needs_review"] = True
                out["review_reason"] = "; ".join(q["issues"])
                label = f"Q{q['number']}" if q["number"] is not None else "Unnumbered question"
                flags.append({"page": q["page"], "reason": f"{label}: {out['review_reason'].rstrip('.')}."})
            questions.append(out)

        if missing:
            by_num = {q["number"]: q for q in seg["questions"] if q["number"] is not None}
            for n in missing:
                nxt = next((by_num[k] for k in sorted(by_num) if k > n), None)
                page = nxt["page"] if nxt else seg["questions"][-1]["page"]
                flags.append({"page": page, "reason": f"Q{n} is missing from the numbering sequence -- it was never found on any scanned page. Check the page and use \"+ Add Missed Question\" if needed."})
        for num, prev, new, page in seg["key_conflicts"]:
            flags.append({"page": page, "reason": f"Answer key lists two different answers for Q{num} ({prev} and {new}) -- kept {prev}, please check."})
        unmatched_keys = sorted(set(key) - {q["number"] for q in seg["questions"]})
        if unmatched_keys and seg["questions"]:
            # Normal when the key covers more of the chapter than this scan range did -- worth a
            # note, not a per-number flag, and never attached to anything.
            flags.append({"page": seg["questions"][-1]["page"], "reason": f"Answer key has entries for {len(unmatched_keys)} question number(s) not in this scan (e.g. Q{unmatched_keys[0]}) -- not attached to anything."})

    if track_answer_key and no_key:
        nums = ", ".join(f"Q{q['number']}" if q["number"] is not None else "?" for q in no_key[:15])
        more = f" and {len(no_key) - 15} more" if len(no_key) > 15 else ""
        flags.append({"page": no_key[0]["page"], "reason": f"No answer key found for {len(no_key)} question(s) ({nums}{more}) -- left blank, set them manually. Keys are only ever taken from a printed answer key, matched by question number."})
    return questions, flags


def cropped_edge_flags(cropped_by_page, pages_items):
    """One flag per page whose scan cuts through text at an edge, naming the questions sitting
    on that side (every question touching the page, if the model's boxes put none there)."""
    flags = []
    for (page, items), sides in zip(pages_items, cropped_by_page):
        for side in sides:
            on_page = [it for it in items if "_question_number" in it]
            near = [it for it in on_page if _valid_box(it.get("box_2d")) and item_column(it["box_2d"]) in (side, "full")]
            nums = sorted({it["_question_number"] for it in (near or on_page) if it["_question_number"] is not None})
            which = f" -- check {', '.join(f'Q{n}' for n in nums)} against the printed page" if nums else ""
            flags.append({"page": page, "reason": f"The scan's {side} edge cuts through lines of text on this page. The model may have filled in cut-off words itself{which}."})
    return flags


# ---------------------------------------------------------------- Hindi-only review flags

# Devanagari block (includes the ।/॥ dandas and Devanagari digits), Latin letters/digits (English
# terms, chemical formulae, numerals legitimately mixed into Hindi text), whitespace, and the
# punctuation/LaTeX syntax characters this pipeline's own text already uses elsewhere (wrap_bare_
# latex_notation, _fix_unescaped_json_backslashes). Anything outside this -- e.g. Arabic script --
# is almost certainly an OCR/transcription slip, not real content.
_HINDI_ALLOWED_CHAR_RE = re.compile(
    r'[ऀ-ॿ]|[A-Za-z0-9]|[\s.,;:!?()\[\]{}/\\$^_%+=<>|\'"-]|[°πμΩ×÷≤≥≈]'
)
_HINDI_STEM_TERMINATORS = set('?:.।॥')


def hindi_review_flags(q):
    """Review-flag reasons specific to Hindi scanned-PDF extraction (task requirement: flag non-
    Devanagari/non-Latin characters, a missing year, and a stem that doesn't end on a real
    sentence boundary). Duplicate options are already covered by _flag_suspicious_mcq_options,
    which finalize_segments runs on every question regardless of language -- not duplicated here.
    Returns a list of human-readable reason strings, empty when nothing looks off."""
    issues = []
    all_text = " ".join(filter(None, [
        q.get("question"), q.get("option_a"), q.get("option_b"), q.get("option_c"), q.get("option_d"),
    ]))
    bad_chars = sorted({ch for ch in all_text if ch.strip() and not _HINDI_ALLOWED_CHAR_RE.match(ch)})
    if bad_chars:
        issues.append(f"unexpected script/characters found ({' '.join(bad_chars[:10])}) -- check against the scan")
    if q.get("year") is None:
        issues.append("no exam year tag found")
    stem = (q.get("question") or "").rstrip()
    stem = stem.rstrip('"\')’” ')  # trailing quote/paren wrappers don't count as the real ending
    if stem and stem[-1] not in _HINDI_STEM_TERMINATORS:
        issues.append("question stem doesn't end in ?, :, . or । -- may be cut off mid-sentence")
    return issues


# ---------------------------------------------------------------- entry point

def scan_scanned_pdf_bytes(pdf_bytes, subject, max_pages=DEFAULT_MAX_PAGES, confirm_over_cap=False,
                           max_workers=4, cache_dir=None, extract_fn=None, language="en"):
    """Entry point for /admin/scan-pdf with layout="scanned". Same return shape as
    process_pyq_vision.scan_pdf_bytes ({"questions", "pages_scanned", "flagged_pages"}) plus a
    "cost" summary, so it drops into the existing review UI unchanged.

    language="hi" switches to the two-pass Hindi pipeline (build_scanned_page_prompt_hindi,
    extract_page_items_two_pass) and layers on hindi_review_flags + two-pass-disagreement flags
    per question, on top of everything the English path already checks.

    extract_fn exists for tests (a fake model); production always uses extract_page_items (English)
    or extract_page_items_two_pass (Hindi)."""
    extract_fn = extract_fn or (extract_page_items_two_pass if language == "hi" else extract_page_items)
    rendered = render_pdf_pages(pdf_bytes, subject, language)
    if len(rendered) > max_pages and not confirm_over_cap:
        return {"error": f"This PDF has {len(rendered)} pages, over the {max_pages}-page cap for one scanned-mode run. Confirm to run it anyway.",
                "needs_confirmation": True, "pages": len(rendered)}

    flagged = []
    page_results = {}
    page_disagreements = {}  # Hindi only: idx -> [question numbers two passes disagreed on]
    to_call = {}
    for idx, (img, key) in enumerate(rendered):
        hit = cache_get(key, cache_dir)
        if hit is not None:
            page_results[idx] = hit["items"]
            page_disagreements[idx] = hit.get("disagreements", [])
        else:
            to_call[idx] = (img, key)

    in_tok = out_tok = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(extract_fn, img, subject): (idx, key) for idx, (img, key) in to_call.items()}
        for fut in concurrent.futures.as_completed(futures):
            idx, key = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                print(f"    Scanned mode: error on page {idx + 1}: {e}")
                flagged.append({"page": idx + 1, "reason": f"Reading this scanned page failed: {e}. Nothing from this page is below -- re-run to retry it (other pages are cached and won't be charged again)."})
                page_results[idx] = []
                page_disagreements[idx] = []
                continue
            # extract_page_items returns (items, usage); extract_page_items_two_pass (Hindi) adds
            # a disagreements list and a pass_failed flag (one of its two calls raised, so this
            # page's result is only a single, uncross-checked pass).
            items, usage = result[0], result[1]
            disagreements = result[2] if len(result) > 2 else []
            pass_failed = len(result) > 3 and result[3]
            if pass_failed:
                flagged.append({"page": idx + 1, "reason": "One of the two extraction passes over this page failed (a transient model error) -- only a single, uncross-checked pass is below. Re-run to retry both passes for this page."})
            in_tok += usage.get("input_tokens", 0)
            out_tok += usage.get("output_tokens", 0)
            page_results[idx] = items
            page_disagreements[idx] = disagreements
            # Not cached when a pass failed -- caching it would make "re-run to retry" a lie (the
            # retry would just load this same single-pass result back out of the cache instead of
            # actually calling the model again for the missing pass).
            if not pass_failed:
                cache_put(key, {"items": items, "usage": usage, "disagreements": disagreements}, cache_dir)

    pages_items = []
    for idx in range(len(rendered)):
        ordered, used_geometry = order_page_items(page_results.get(idx, []))
        if not used_geometry and ordered:
            flagged.append({"page": idx + 1, "reason": "The model didn't give a usable position for every item on this page, so its own reading order was kept -- check that no question was joined to the wrong continuation."})
        pages_items.append((idx + 1, ordered))
        kinds = {it.get("kind") for it in ordered}
        if idx in page_results and not kinds & {"question", "continuation", "answer_key"} and not any(f["page"] == idx + 1 for f in flagged):
            flagged.append({"page": idx + 1, "reason": "No questions, continuations or answer key were found on this scanned page. If it does have questions, check it manually."})

    segments, page_flags = merge_document_items(pages_items)
    questions, q_flags = finalize_segments(segments, track_answer_key=(language != "hi"))
    flagged.extend(page_flags)
    flagged.extend(cropped_edge_flags(pdf_cropped_edges(pdf_bytes), pages_items))
    flagged.extend(q_flags)

    if language == "hi":
        disagree_at = {(idx + 1, num) for idx, nums in page_disagreements.items() for num in nums}
        for q in questions:
            issues = hindi_review_flags(q)
            if (q.get("source_page"), q.get("question_number")) in disagree_at:
                issues.append("the two independent extraction passes disagree on this question's text -- please verify against the scan")
            if not issues:
                continue
            combined = "; ".join(issues)
            q["review_reason"] = f"{q['review_reason']}; {combined}" if q.get("needs_review") else combined
            q["needs_review"] = True
            label = f"Q{q['question_number']}" if q.get("question_number") is not None else "Unnumbered question"
            flagged.append({"page": q.get("source_page", 1), "reason": f"{label}: {combined}"})

    flagged.sort(key=lambda f: f["page"])

    cost = {
        "model": SCANNED_MODEL,
        "pages_called": len(to_call),
        "pages_cached": len(rendered) - len(to_call),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "usd": round(tokens_to_usd(in_tok, out_tok), 6),
    }
    print(f"SCANNED MODE COST: pages_called={cost['pages_called']} pages_cached={cost['pages_cached']} input={in_tok} output={out_tok} cost=${cost['usd']:.6f}", flush=True)
    return {"questions": questions, "pages_scanned": len(rendered), "flagged_pages": flagged, "cost": cost}
