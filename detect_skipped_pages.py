"""Phase 1: find the EXACT page numbers skipped by ncert_rechunk_v2.py's oversized-content-stream
guard, for the 18 chapters flagged in ncert_rechunk_v2_load_progress.jsonl (52 pages total).

Re-implements only the size check (no pdfplumber text extraction) so this is cheap and safe to
re-run. Writes ncert_page_recovery_pages.json: {pdf: {"chapter_name":, "subject":, "class":,
"pages": [n, ...]}}.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import PyPDF2
from ncert_rechunk_v2 import (
    PDF_DIR, MAX_PAGE_CONTENT_STREAM_BYTES, _page_content_stream_size, _chapter_groups,
)

AFFECTED_PDFS = {
    "kebo102.pdf", "kebo103.pdf", "kebo104.pdf", "kebo108.pdf", "kebo110.pdf", "kebo117.pdf",
    "kebo118.pdf", "lebo101.pdf", "lebo102.pdf", "lebo104.pdf", "lebo105.pdf", "lebo106.pdf",
    "lebo107.pdf", "lebo108.pdf", "lebo109.pdf", "lebo112.pdf", "lebo113.pdf", "lech101.pdf",
}

EXPECTED_COUNTS = {
    "kebo102.pdf": 1, "kebo103.pdf": 1, "kebo104.pdf": 3, "kebo108.pdf": 3, "kebo110.pdf": 2,
    "kebo117.pdf": 2, "kebo118.pdf": 1, "lebo101.pdf": 8, "lebo102.pdf": 7, "lebo104.pdf": 10,
    "lebo105.pdf": 1, "lebo106.pdf": 5, "lebo107.pdf": 2, "lebo108.pdf": 1, "lebo109.pdf": 1,
    "lebo112.pdf": 2, "lebo113.pdf": 1, "lech101.pdf": 1,
}


def main():
    groups = _chapter_groups()
    out = {}
    total_pages = 0
    for subject, class_num, chapter_name, path in groups:
        base = os.path.basename(path)
        if base not in AFFECTED_PDFS:
            continue
        reader = PyPDF2.PdfReader(path)
        skipped = []
        for page_num, page in enumerate(reader.pages, start=1):
            size = _page_content_stream_size(page)
            if size > MAX_PAGE_CONTENT_STREAM_BYTES:
                skipped.append(page_num)
        del reader
        out[base] = {
            "chapter_name": chapter_name, "subject": subject, "class": class_num,
            "pages": skipped, "num_pages_in_pdf": None,
        }
        expected = EXPECTED_COUNTS.get(base)
        flag = "" if len(skipped) == expected else f"  <<< MISMATCH expected {expected}"
        print(f"{base:14s} {chapter_name:45s} skipped={skipped}{flag}")
        total_pages += len(skipped)

    missing = AFFECTED_PDFS - set(out.keys())
    if missing:
        print(f"\nWARNING: never matched against _chapter_groups(): {sorted(missing)}")

    print(f"\nTotal chapters: {len(out)}  Total pages: {total_pages} (expected 18 / 52)")

    with open("ncert_page_recovery_pages.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Wrote ncert_page_recovery_pages.json")


if __name__ == "__main__":
    main()
