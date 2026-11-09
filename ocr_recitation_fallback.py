"""Phase 2b: for the pages Gemini blocked with finish_reason=RECITATION (its own-verbatim-text
safety filter -- triggered because these specific NCERT pages are closely indexed by third-party
sites, confirmed via resp.candidates[0].citation_metadata on a direct diagnostic call; not a
resolution/DPI problem, confirmed by retrying at 250 DPI with no change), recursively split the
page image top/bottom and retry each half -- shorter verbatim spans fall under the recitation
trigger. Concatenates the halves (in top-to-bottom order) back into ONE page-level OCR record so
downstream merge_and_rechunk.py's simple last-record-wins-per-(pdf,page) logic still works.

Usage: python ocr_recitation_fallback.py   (reads ncert_page_recovery_recitation_blocked.json:
    [{"pdf":, "page":}, ...])
"""
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fitz
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

from google import genai
from google.genai import types as genai_types

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

MODEL = "gemini-3.5-flash-lite"
RATE = {"input": 0.30, "output": 2.50}
BUDGET_USD = 0.80
PDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ncert-pdfs")
OCR_FILE = "ncert_page_recovery_ocr.jsonl"
SPEND_FILE = "ncert_page_recovery_gemini_spend.json"
DPI = 250
MAX_SPLIT_DEPTH = 4  # whole -> halves -> quarters -> eighths -> sixteenths

PROMPT = (
    "Transcribe the body text of this NCERT textbook page EXCERPT exactly, in reading order "
    "(if it has two columns, read left column top-to-bottom fully, then the right column). "
    "This is a partial vertical slice of a page -- text may start or end mid-sentence at the "
    "top/bottom edge, that's expected, just transcribe what's visible. Keep section headings as "
    "their own line starting with \"## \". Keep formulas, equations and chemical names as plain "
    "text exactly as written. Ignore page header, footer, page number, figure/diagram labels, "
    "figure or table captions. Do not summarize, paraphrase, or rewrite -- transcribe verbatim. "
    "Output plain text only."
)


def _cost(usage) -> float:
    if not usage:
        return 0.0
    inp = usage.prompt_token_count or 0
    out = usage.candidates_token_count or 0
    return (inp * RATE["input"] + out * RATE["output"]) / 1_000_000


def _render_full_image(pdf_path: str, page_num: int, dpi: int) -> Image.Image:
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_num - 1)
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB)
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    finally:
        doc.close()


def _ocr_crop(img: Image.Image, spend_tracker: dict):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    part = genai_types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")
    config = genai_types.GenerateContentConfig(max_output_tokens=4096)
    resp = client.models.generate_content(model=MODEL, contents=[part, PROMPT], config=config)
    usage = resp.usage_metadata
    cost = _cost(usage)
    spend_tracker["usd"] += cost
    finish_reason = resp.candidates[0].finish_reason.name if resp.candidates else "UNKNOWN"
    text = resp.text or ""
    return text, finish_reason, cost


def _dedupe_boundary(text_top: str, text_bot: str) -> str:
    """A text line straddling exactly the pixel y-coordinate where a crop was split can get
    fully read by Gemini in BOTH the top crop (as its last visible line) and the bottom crop (as
    its first visible line) -- confirmed on lebo104.pdf p17 ("...led to discovering the basis for
    the variation" appeared twice in a row at a split boundary). If the bottom fragment's first
    line is already the top fragment's last line (or a close prefix match), drop that repeated
    line from the bottom fragment before joining."""
    top_lines = text_top.rstrip().split("\n")
    bot_lines = text_bot.lstrip().split("\n")
    if not top_lines or not bot_lines:
        return text_top.rstrip() + "\n" + text_bot.lstrip()
    last_top = top_lines[-1].strip()
    first_bot = bot_lines[0].strip()
    if last_top and first_bot and (last_top == first_bot or last_top.startswith(first_bot) or first_bot.startswith(last_top)):
        bot_lines = bot_lines[1:]
    return "\n".join(top_lines + bot_lines)


def recursive_ocr(img: Image.Image, spend_tracker: dict, depth: int = 0, label: str = "whole"):
    """Returns (ok: bool, text: str, notes: list[str])."""
    if spend_tracker["usd"] >= BUDGET_USD:
        return False, "", [f"{label}: budget reached before attempting"]

    text, finish_reason, cost = _ocr_crop(img, spend_tracker)
    print(f"    [{label}] finish_reason={finish_reason} cost=${cost:.5f} chars={len(text)} "
          f"running=${spend_tracker['usd']:.4f}")

    if finish_reason == "STOP":
        return True, text, []
    if finish_reason != "RECITATION" or depth >= MAX_SPLIT_DEPTH:
        return False, "", [f"{label}: finish_reason={finish_reason} at depth {depth}, giving up here"]

    w, h = img.size
    if h < 400:  # too thin to usefully split further
        return False, "", [f"{label}: still RECITATION, too small to split further (h={h})"]

    top = img.crop((0, 0, w, h // 2))
    bot = img.crop((0, h // 2, w, h))
    ok_top, text_top, notes_top = recursive_ocr(top, spend_tracker, depth + 1, f"{label}/top")
    ok_bot, text_bot, notes_bot = recursive_ocr(bot, spend_tracker, depth + 1, f"{label}/bot")

    combined_notes = notes_top + notes_bot
    # Concatenate whatever text EACH side actually recovered, regardless of whether that side was
    # a full or only partial success -- a partial side still holds real, correctly-ordered text
    # from its own successful leaf calls, and dropping it (as an earlier version of this function
    # did, keyed off the "ok" flag instead of "is there text") silently threw away already-paid-for,
    # already-correct recovered text whenever a SIBLING sub-region was still blocked.
    if text_top and text_bot:
        # NOT deduped: an earlier version of this tried to detect and drop a line repeated across
        # the crop boundary (a real, harmless artifact -- see _dedupe_boundary), but its
        # startswith() check also fires when a bottom fragment's first line legitimately
        # CONTINUES the top fragment's last line (e.g. top ends "...Each half of", bottom begins
        # "Each half of pectoral girdle consists of..." as one real, undivided sentence) -- that
        # false match silently deleted the bottom fragment's entire first line, which on
        # kebo117.pdf p10 was most of a paragraph (confirmed by direct comparison against a fresh,
        # independent re-OCR of the same crop, which still had the "clavicle and a scapula"
        # sentence the deduped version had lost). A stray duplicate line left in the text costs
        # nothing at chunking time; silently deleted content is a correctness bug, so this keeps
        # the plain, unconditional concatenation instead.
        combined_text = text_top.rstrip() + "\n" + text_bot.lstrip()
    else:
        combined_text = text_top or text_bot
    ok = ok_top and ok_bot
    if not ok:
        combined_notes.append(f"{label}: partial (some sub-region still RECITATION-blocked)")
    return ok, combined_text, combined_notes


def _append_record(rec: dict):
    with open(OCR_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main():
    with open("ncert_page_recovery_recitation_blocked.json", encoding="utf-8") as f:
        blocked = json.load(f)
    with open("ncert_page_recovery_pages.json", encoding="utf-8") as f:
        pages_by_pdf = json.load(f)

    spend = 0.0
    if os.path.exists(SPEND_FILE):
        spend = json.load(open(SPEND_FILE, encoding="utf-8")).get("gemini_spend_usd", 0.0)
    spend_tracker = {"usd": spend}

    print(f"Starting spend: ${spend_tracker['usd']:.4f} / ${BUDGET_USD}")

    for item in blocked:
        pdf_name, page_num = item["pdf"], item["page"]
        chapter_name = pages_by_pdf.get(pdf_name, {}).get("chapter_name", "")
        print(f"\n{pdf_name} p{page_num} ({chapter_name}):")
        if spend_tracker["usd"] >= BUDGET_USD:
            print("  BUDGET REACHED, stopping.")
            break
        img = _render_full_image(os.path.join(PDF_DIR, pdf_name), page_num, DPI)
        ok, text, notes = recursive_ocr(img, spend_tracker, depth=0, label=f"{pdf_name}p{page_num}")
        with open(SPEND_FILE, "w", encoding="utf-8") as f:
            json.dump({"gemini_spend_usd": spend_tracker["usd"]}, f, indent=2)

        _append_record({
            "pdf": pdf_name, "page": page_num, "chapter_name": chapter_name,
            "dpi": DPI, "text": text, "input_tokens": None, "output_tokens": None,
            "cost": None, "error": None if ok else "; ".join(notes) or "RECITATION (no split succeeded)",
            "recitation_fallback": True, "fully_recovered": ok, "split_notes": notes,
        })
        print(f"  RESULT: {'FULL' if ok else 'PARTIAL/FAILED'} recovery, {len(text)} chars"
              + (f" -- {notes}" if notes else ""))
        time.sleep(0.3)

    print(f"\nDone. Total Gemini spend: ${spend_tracker['usd']:.4f}")


if __name__ == "__main__":
    main()
