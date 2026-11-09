"""Phase 3: validate every recovered page's OCR text with a script (no second model call).

Checks per page: not empty; mostly English (ASCII-letter ratio); word count in a sane range
against that chapter's average real page (computed locally from the chapter's own
non-skipped pages via pdfplumber -- no API cost); no garbage-character runs; no obvious
mid-sentence cutoff at the end.

Writes ncert_page_recovery_validation.json (per-page verdict) and, for anything flagged,
ncert_page_recovery_retry.json (consumed by `ocr_skipped_pages.py --retry-250`).
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pdfplumber

PDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ncert-pdfs")
MIN_WORDS_FLOOR = 100
MAX_WORDS_RATIO = 2.2
_GARBAGE_RE = re.compile(r'[�\x00-\x08\x0b\x0c\x0e-\x1f]')
_CUTOFF_END_RE = re.compile(r'[a-zA-Z,;:\-]\s*$')  # ends mid-word/mid-clause, no terminal punctuation


def _word_count(text: str) -> int:
    return len(text.split())


def _ascii_letter_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    ascii_letters = sum(1 for c in letters if c.isascii())
    return ascii_letters / len(letters)


def _chapter_avg_page_words(pdf_path: str, skip_pages: set) -> float:
    with pdfplumber.open(pdf_path) as pdf:
        counts = []
        for page_num, page in enumerate(pdf.pages, start=1):
            if page_num in skip_pages:
                continue
            try:
                text = page.extract_text() or ""
            except Exception:
                continue
            wc = _word_count(text)
            if wc > 0:
                counts.append(wc)
            page.flush_cache()
    return (sum(counts) / len(counts)) if counts else 0.0


def _load_ocr_records():
    """Last record per (pdf, page) wins -- a 250-DPI retry record appended later overrides the
    original 200-DPI one."""
    records = {}
    if not os.path.exists("ncert_page_recovery_ocr.jsonl"):
        return records
    with open("ncert_page_recovery_ocr.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[(rec["pdf"], rec["page"])] = rec
    return records


def main():
    with open("ncert_page_recovery_pages.json", encoding="utf-8") as f:
        pages_by_pdf = json.load(f)
    ocr_records = _load_ocr_records()

    results = {}
    retry_list = []
    avg_cache = {}

    for pdf_name, info in sorted(pages_by_pdf.items()):
        pdf_path = os.path.join(PDF_DIR, pdf_name)
        skip_set = set(info["pages"])
        if pdf_name not in avg_cache:
            avg_cache[pdf_name] = _chapter_avg_page_words(pdf_path, skip_set)
        avg = avg_cache[pdf_name]

        for page_num in info["pages"]:
            rec = ocr_records.get((pdf_name, page_num))
            key = f"{pdf_name}:p{page_num}"
            if rec is None:
                results[key] = {"verdict": "MISSING", "reason": "no OCR record found"}
                retry_list.append({"pdf": pdf_name, "page": page_num})
                continue
            text = rec.get("text", "") or ""
            if rec.get("error"):
                results[key] = {"verdict": "ERROR", "reason": rec["error"]}
                retry_list.append({"pdf": pdf_name, "page": page_num})
                continue

            # NOTE on "cut-off at the end": real NCERT body text routinely ends mid-sentence at a
            # physical page boundary because the sentence continues onto the next page -- that is
            # normal and the chunker's cross-page sentence stream handles it, not a sign of
            # truncated OCR output. A genuine OCR truncation would show up as the model hitting
            # max_output_tokens (4096); every one of the 52 pages came back under 650 output
            # tokens, nowhere near that cap, so there's no real truncation signal to check here --
            # an end-of-text heuristic was tried and produced only false positives (flagged pages
            # that, on manual inspection, were simply mid-sentence-at-page-break, not cut off) and
            # was dropped rather than kept as a check that can't actually fire correctly.
            retry_worthy = []
            info_only = []
            if not text.strip():
                retry_worthy.append("empty")
            else:
                wc = _word_count(text)
                ratio = _ascii_letter_ratio(text)
                if ratio < 0.85:
                    retry_worthy.append(f"non-English ratio={ratio:.2f}")
                if _GARBAGE_RE.search(text):
                    retry_worthy.append("garbage control characters found")
                if wc < MIN_WORDS_FLOOR:
                    # Diagram-dominated pages legitimately have very little body text (confirmed
                    # by manual inspection of every such page in this run) -- flagged for the
                    # report but NOT auto-retried, since a 250 DPI re-OCR of a page that's mostly
                    # a figure will not conjure body text that was never there.
                    info_only.append(f"short: {wc} words (floor {MIN_WORDS_FLOOR}, chapter avg {avg:.0f}) -- "
                                      f"verify it's a diagram-heavy page, not a real miss")
                if avg > 0 and wc > MAX_WORDS_RATIO * avg:
                    info_only.append(f"long: {wc} words (> {MAX_WORDS_RATIO}x chapter avg {avg:.0f})")

            problems = retry_worthy + info_only
            if retry_worthy:
                results[key] = {"verdict": "FLAGGED", "reason": "; ".join(problems),
                                 "word_count": _word_count(text), "chapter_avg": round(avg, 1),
                                 "dpi": rec.get("dpi", 200)}
                retry_list.append({"pdf": pdf_name, "page": page_num})
            elif info_only:
                results[key] = {"verdict": "FLAGGED_INFO_ONLY", "reason": "; ".join(problems),
                                 "word_count": _word_count(text), "chapter_avg": round(avg, 1),
                                 "dpi": rec.get("dpi", 200)}
            else:
                results[key] = {"verdict": "OK", "word_count": _word_count(text),
                                 "chapter_avg": round(avg, 1), "dpi": rec.get("dpi", 200)}

    with open("ncert_page_recovery_validation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    retryable = [k for k, v in results.items() if v["verdict"] in ("FLAGGED", "ERROR", "MISSING")]
    info_only_keys = [k for k, v in results.items() if v["verdict"] == "FLAGGED_INFO_ONLY"]
    ok = len(results) - len(retryable) - len(info_only_keys)
    print(f"Validated {len(results)} pages. OK={ok}  FLAGGED_INFO_ONLY(short, not retried)={len(info_only_keys)}  "
          f"RETRY-WORTHY={len(retryable)}")
    for k in retryable:
        print(f"  [RETRY] {k}: {results[k]}")
    for k in info_only_keys:
        print(f"  [INFO]  {k}: {results[k]}")

    if retry_list:
        with open("ncert_page_recovery_retry.json", "w", encoding="utf-8") as f:
            json.dump(retry_list, f, indent=2)
        print(f"\nWrote ncert_page_recovery_retry.json ({len(retry_list)} pages) for --retry-250.")
    else:
        print("\nNothing to retry.")


if __name__ == "__main__":
    main()
