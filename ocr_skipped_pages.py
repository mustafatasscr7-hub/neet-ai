"""Phase 2: transcribe each rendered skipped-page image with Gemini 3.5 Flash-Lite, single pass.
Checkpointed to ncert_page_recovery_ocr.jsonl (resumable). Tracks real spend against a hard
$0.80 budget and stops (does not loop) on a quota/balance error or once the budget is reached.

Usage:
    python ocr_skipped_pages.py              # 200 DPI pass over every page not yet checkpointed
    python ocr_skipped_pages.py --retry-250  # re-OCR specific pages at 250 DPI (reads
                                              # ncert_page_recovery_retry.json: [{"pdf":, "page":}, ...])
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fitz
from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types as genai_types

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Missing GEMINI_API_KEY in .env", file=sys.stderr)
    sys.exit(1)
client = genai.Client(api_key=GEMINI_API_KEY)

MODEL = "gemini-3.5-flash-lite"
# ai.google.dev/gemini-api/docs/pricing, per 1M tokens -- same constant as server.py's
# GEMINI_FLASH_LITE_RATE.
RATE = {"input": 0.30, "output": 2.50}
BUDGET_USD = 0.80

PDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ncert-pdfs")
IMG_DIR = os.path.join("ncert_text_clean", "page_images")
OCR_FILE = "ncert_page_recovery_ocr.jsonl"
SPEND_FILE = "ncert_page_recovery_gemini_spend.json"

PROMPT = (
    "Transcribe the body text of this NCERT textbook page exactly, in reading order "
    "(if the page has two columns, read the left column top-to-bottom fully, then the right "
    "column). Keep section headings as their own line starting with \"## \". Keep formulas, "
    "equations and chemical names as plain text exactly as written. Ignore the page header, "
    "footer, page number, figure/diagram labels, and figure or table captions -- do not "
    "transcribe those. Do not summarize, paraphrase, or rewrite anything -- transcribe the "
    "actual body text verbatim. Output plain text only, no markdown other than the \"## \" "
    "heading lines, no commentary."
)


def _cost(usage) -> float:
    if not usage:
        return 0.0
    inp = usage.prompt_token_count or 0
    out = usage.candidates_token_count or 0
    return (inp * RATE["input"] + out * RATE["output"]) / 1_000_000


def _is_quota_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(s in msg for s in ("quota", "resource_exhausted", "429", "billing", "insufficient", "balance"))


def _load_spend() -> float:
    if os.path.exists(SPEND_FILE):
        with open(SPEND_FILE, encoding="utf-8") as f:
            return json.load(f).get("gemini_spend_usd", 0.0)
    return 0.0


def _save_spend(total: float):
    with open(SPEND_FILE, "w", encoding="utf-8") as f:
        json.dump({"gemini_spend_usd": total}, f, indent=2)


def _load_done() -> set:
    done = set()
    if os.path.exists(OCR_FILE):
        with open(OCR_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((rec["pdf"], rec["page"]))
    return done


def _append_record(rec: dict):
    with open(OCR_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def render_page_bytes(pdf_path: str, page_num: int, dpi: int) -> bytes:
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_num - 1)
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB)
        return pix.tobytes("jpg", jpg_quality=90)
    finally:
        doc.close()


def ocr_one(image_bytes: bytes):
    part = genai_types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
    config = genai_types.GenerateContentConfig(max_output_tokens=4096)
    resp = client.models.generate_content(model=MODEL, contents=[part, PROMPT], config=config)
    usage = resp.usage_metadata
    cost = _cost(usage)
    return resp.text or "", (usage.prompt_token_count or 0) if usage else 0, \
        (usage.candidates_token_count or 0) if usage else 0, cost


def run_main_pass():
    with open("ncert_page_recovery_pages.json", encoding="utf-8") as f:
        pages_by_pdf = json.load(f)

    done = _load_done()
    spend = _load_spend()
    print(f"Starting spend so far: ${spend:.4f} / ${BUDGET_USD:.2f} budget. {len(done)} pages already OCR'd.")

    for pdf_name, info in sorted(pages_by_pdf.items()):
        for page_num in info["pages"]:
            if (pdf_name, page_num) in done:
                continue
            if spend >= BUDGET_USD:
                print(f"BUDGET REACHED (${spend:.4f} >= ${BUDGET_USD:.2f}). Stopping. "
                      f"{pdf_name} page {page_num} and remaining pages NOT processed.")
                return spend
            img_path = os.path.join(IMG_DIR, f"{os.path.splitext(pdf_name)[0]}_p{page_num}.jpg")
            with open(img_path, "rb") as f:
                image_bytes = f.read()
            try:
                text, in_tok, out_tok, cost = ocr_one(image_bytes)
            except Exception as e:
                if _is_quota_error(e):
                    print(f"QUOTA/BALANCE ERROR on {pdf_name} page {page_num}: {e}. Stopping (not looping).")
                    _save_spend(spend)
                    return spend
                print(f"OCR FAILED on {pdf_name} page {page_num}: {e}. Marking as failed, continuing.")
                _append_record({
                    "pdf": pdf_name, "page": page_num, "chapter_name": info["chapter_name"],
                    "dpi": 200, "text": "", "input_tokens": 0, "output_tokens": 0, "cost": 0.0,
                    "error": str(e),
                })
                continue
            spend += cost
            _save_spend(spend)
            _append_record({
                "pdf": pdf_name, "page": page_num, "chapter_name": info["chapter_name"],
                "dpi": 200, "text": text, "input_tokens": in_tok, "output_tokens": out_tok,
                "cost": cost, "error": None,
            })
            print(f"OCR'd {pdf_name} p{page_num}  ({in_tok}in/{out_tok}out, ${cost:.5f}, "
                  f"running ${spend:.4f})  chars={len(text)}")
            time.sleep(0.3)
    print(f"\nMain pass done. Total Gemini spend: ${spend:.4f}")
    return spend


def run_retry_pass():
    retry_list_path = "ncert_page_recovery_retry.json"
    if not os.path.exists(retry_list_path):
        print(f"{retry_list_path} not found -- nothing to retry.")
        return
    with open(retry_list_path, encoding="utf-8") as f:
        retry_list = json.load(f)
    with open("ncert_page_recovery_pages.json", encoding="utf-8") as f:
        pages_by_pdf = json.load(f)

    spend = _load_spend()
    for item in retry_list:
        pdf_name, page_num = item["pdf"], item["page"]
        if spend >= BUDGET_USD:
            print(f"BUDGET REACHED. Stopping retry pass. {pdf_name} p{page_num} not retried.")
            break
        pdf_path = os.path.join(PDF_DIR, pdf_name)
        image_bytes = render_page_bytes(pdf_path, page_num, 250)
        try:
            text, in_tok, out_tok, cost = ocr_one(image_bytes)
        except Exception as e:
            if _is_quota_error(e):
                print(f"QUOTA/BALANCE ERROR during retry on {pdf_name} p{page_num}: {e}. Stopping.")
                _save_spend(spend)
                return
            print(f"RETRY FAILED on {pdf_name} p{page_num}: {e}")
            continue
        spend += cost
        _save_spend(spend)
        chapter_name = pages_by_pdf.get(pdf_name, {}).get("chapter_name", "")
        _append_record({
            "pdf": pdf_name, "page": page_num, "chapter_name": chapter_name,
            "dpi": 250, "text": text, "input_tokens": in_tok, "output_tokens": out_tok,
            "cost": cost, "error": None, "retry": True,
        })
        print(f"RETRY OCR'd {pdf_name} p{page_num} at 250 DPI ({in_tok}in/{out_tok}out, "
              f"${cost:.5f}, running ${spend:.4f}) chars={len(text)}")
        time.sleep(0.3)
    print(f"\nRetry pass done. Total Gemini spend: ${spend:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retry-250", action="store_true")
    args = parser.parse_args()
    if args.retry_250:
        run_retry_pass()
    else:
        run_main_pass()
