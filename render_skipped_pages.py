"""Phase 1b: render each skipped page (from ncert_page_recovery_pages.json) to a 200 DPI JPEG,
one page at a time, never loading the whole PDF's content streams (fitz opens lazily; only the
single requested page is rasterised and then dropped). Checkpointed -- safe to re-run.

Output: ncert_text_clean/page_images/<pdf-stem>_p<page>.jpg
"""
import json
import os

import fitz  # PyMuPDF

OUT_DIR = os.path.join("ncert_text_clean", "page_images")
PDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ncert-pdfs")
DPI = 200


def render_page(pdf_path: str, page_num: int, dpi: int, out_path: str):
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_num - 1)
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB)
        pix.save(out_path, jpg_quality=90)
        del pix
        page.clean_contents()
    finally:
        doc.close()


def main():
    with open("ncert_page_recovery_pages.json", encoding="utf-8") as f:
        pages_by_pdf = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)
    rendered = 0
    skipped_existing = 0
    for pdf_name, info in sorted(pages_by_pdf.items()):
        pdf_path = os.path.join(PDF_DIR, pdf_name)
        stem = os.path.splitext(pdf_name)[0]
        for page_num in info["pages"]:
            out_path = os.path.join(OUT_DIR, f"{stem}_p{page_num}.jpg")
            if os.path.exists(out_path):
                skipped_existing += 1
                continue
            render_page(pdf_path, page_num, DPI, out_path)
            rendered += 1
            print(f"rendered {out_path}")
    print(f"\nDone. {rendered} rendered, {skipped_existing} already existed.")


if __name__ == "__main__":
    main()
