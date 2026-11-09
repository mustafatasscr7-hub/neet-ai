"""Phase 4+5: merge recovered page text back into each affected chapter at the correct page
position, re-chunk the WHOLE chapter with the same rules as ncert_rechunk_v2.py, save the full
cleaned chapter text locally, then embed + upsert the new chunks into ncert_content_v2 (same
on_conflict upsert path as ncert_rechunk_v2.py), finally deleting any stale leftover rows whose
chunk_index is now out of range for that chapter. Chapter by chapter, checkpointed.

Only touches the 18 affected chapters. Never touches ncert_content, Hindi rows, or any other
chapter in ncert_content_v2.

Usage:
    python merge_and_rechunk.py --dry-run   # chunk + report only, no embedding, no DB writes
    python merge_and_rechunk.py --load      # embed + upsert + delete-stale, checkpointed
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv

load_dotenv()

from ncert_rechunk_v2 import (
    PDF_DIR, EMBEDDING_MODEL, _extract_tagged_lines, _build_sentence_stream, chunk_sentences,
    _env_or_die, _NCERT_V2_ON_CONFLICT,
)

TEXT_CLEAN_DIR = "ncert_text_clean"
DRY_RUN_PROGRESS_FILE = "ncert_page_recovery_merge_progress.jsonl"
LOAD_PROGRESS_FILE = "ncert_page_recovery_merge_load_progress.jsonl"
EMBED_SPEND_FILE = "ncert_page_recovery_embed_spend.json"


def _load_ocr_final():
    """Last record per (pdf, page) wins -- picks up 250 DPI retries over the original 200 DPI
    pass automatically since retries are appended later in the file."""
    records = {}
    with open("ncert_page_recovery_ocr.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[(rec["pdf"], rec["page"])] = rec
    return records


def _ocr_text_to_tagged_lines(page_num: int, text: str):
    """Mirrors _classify_line's output shape ((page_num, kind, text)), reading the "## " heading
    convention the Gemini OCR prompt used instead of NCERT's numbered-heading regex (the OCR'd
    text has no 5x-stroke artifact or boilerplate footer to decode/strip -- the prompt already
    told Gemini to skip headers/footers/captions)."""
    tagged = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("## "):
            tagged.append((page_num, "heading", line[3:].strip()))
        else:
            tagged.append((page_num, "text", line))
    return tagged


def build_merged_chapter(pdf_name: str, info: dict, ocr_records: dict, warnings: dict):
    """Returns (chunks, full_text) for one chapter, using the ORIGINAL extraction for every page
    pdfplumber could read, plus the Gemini-recovered text spliced in at the correct page
    positions for the pages that were skipped."""
    pdf_path = os.path.join(PDF_DIR, pdf_name)
    original_tagged = _extract_tagged_lines(pdf_path, warnings)  # still skips the oversized pages

    recovered_tagged = []
    missing_recovery = []
    for page_num in info["pages"]:
        rec = ocr_records.get((pdf_name, page_num))
        if rec is None or rec.get("error") or not (rec.get("text") or "").strip():
            missing_recovery.append(page_num)
            continue
        recovered_tagged.extend(_ocr_text_to_tagged_lines(page_num, rec["text"]))

    merged_tagged = sorted(original_tagged + recovered_tagged, key=lambda t: t[0])
    stream = _build_sentence_stream(merged_tagged, warnings)
    chunks = chunk_sentences(stream)
    for idx, c in enumerate(chunks):
        c.update(subject=info["subject"], class_=info["class"], chapter_name=info["chapter_name"], chunk_index=idx)

    full_text_lines = []
    for page_num, kind, text in merged_tagged:
        full_text_lines.append(f"## {text}" if kind == "heading" else text)
    full_text = "\n".join(full_text_lines)

    return chunks, full_text, missing_recovery


def _load_checkpoint(path: str) -> dict:
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                done[rec["pdf"]] = rec
    return done


def _append_checkpoint(rec: dict, path: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _embed_cost(tokens: int) -> float:
    # text-embedding-3-small: $0.02 / 1M tokens (ai.google.dev isn't this -- platform.openai.com
    # pricing, same model ncert_rechunk_v2.py already uses for every other v2 chapter).
    return tokens * 0.02 / 1_000_000


def _load_embed_spend() -> float:
    if os.path.exists(EMBED_SPEND_FILE):
        with open(EMBED_SPEND_FILE, encoding="utf-8") as f:
            return json.load(f).get("embed_spend_usd", 0.0)
    return 0.0


def _save_embed_spend(v: float):
    with open(EMBED_SPEND_FILE, "w", encoding="utf-8") as f:
        json.dump({"embed_spend_usd": v}, f, indent=2)


def _upsert_chunks(chunks, client, headers, supabase_url, spend_tracker) -> int:
    import requests
    BATCH_SIZE = 20
    inserted = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        rows = []
        for c in batch:
            embedding = None
            for attempt in range(3):
                try:
                    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=c["content"])
                    embedding = resp.data[0].embedding
                    spend_tracker["usd"] += _embed_cost(resp.usage.total_tokens)
                    break
                except Exception as e:
                    print(f"  embed retry {attempt + 1}/3: {e}")
                    time.sleep(3)
            if embedding is None:
                print(f"  SKIPPED (embedding failed 3x): {c['chapter_name']} chunk {c['chunk_index']}")
                continue
            rows.append({
                "subject": c["subject"], "class": c["class_"], "chapter_name": c["chapter_name"],
                "chapter_name_en": None, "language": "en", "heading": c["heading"], "page": c["page"],
                "chunk_index": c["chunk_index"], "content": c["content"], "embedding": embedding,
            })
        if not rows:
            continue
        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{supabase_url}/rest/v1/ncert_content_v2", headers=headers,
                    params={"on_conflict": _NCERT_V2_ON_CONFLICT}, json=rows, timeout=30)
                if resp.status_code < 300:
                    inserted += len(rows)
                    break
                print(f"  insert failed (status {resp.status_code}): {resp.text[:300]}")
                time.sleep(3)
            except Exception as e:
                print(f"  insert retry {attempt + 1}/3: {e}")
                time.sleep(3)
    return inserted


def _delete_stale_rows(subject, class_num, chapter_name, new_chunk_count, headers, supabase_url) -> int:
    """Deletes any leftover row for this chapter whose chunk_index >= the new chunk count --
    e.g. the old chapter had 21 chunks (0..20), the recovered one has 19 (0..18); rows 19 and 20
    are now stale and must go. Uses chunk_index=gte.N via PostgREST's filter syntax. Returns the
    count header's reported deleted-row count (0 if none)."""
    import requests
    resp = requests.delete(
        f"{supabase_url}/rest/v1/ncert_content_v2",
        headers={**headers, "Prefer": "return=representation"},
        params={
            "subject": f"eq.{subject}", "class": f"eq.{class_num}",
            "chapter_name": f"eq.{chapter_name}", "chunk_index": f"gte.{new_chunk_count}",
        },
        timeout=30,
    )
    if resp.status_code >= 300:
        print(f"  WARNING: stale-row delete failed (status {resp.status_code}): {resp.text[:300]}")
        return -1
    try:
        return len(resp.json())
    except Exception:
        return 0


def _count_rows(subject, class_num, chapter_name, headers, supabase_url) -> int:
    # Same count=exact + limit=1 pattern server.py already uses (e.g. the /pyq count query) --
    # gets the real total via content-range without transferring every row's data.
    import requests
    resp = requests.get(
        f"{supabase_url}/rest/v1/ncert_content_v2",
        headers={**headers, "Prefer": "count=exact"},
        params={"subject": f"eq.{subject}", "class": f"eq.{class_num}", "chapter_name": f"eq.{chapter_name}",
                "select": "id", "limit": 1},
        timeout=30,
    )
    cr = resp.headers.get("content-range", "")
    tail = cr.split("/")[-1] if "/" in cr else ""
    return int(tail) if tail.isdigit() else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true")
    args = parser.parse_args()

    with open("ncert_page_recovery_pages.json", encoding="utf-8") as f:
        pages_by_pdf = json.load(f)
    ocr_records = _load_ocr_final()
    checkpoint_path = LOAD_PROGRESS_FILE if args.load else DRY_RUN_PROGRESS_FILE
    done = _load_checkpoint(checkpoint_path)

    os.makedirs(TEXT_CLEAN_DIR, exist_ok=True)

    client = headers = supabase_url = None
    spend_tracker = {"usd": _load_embed_spend() if args.load else 0.0}
    if args.load:
        import openai
        openai_key = _env_or_die("OPENAI_KEY")
        supabase_url = _env_or_die("SUPABASE_URL")
        supabase_service_key = _env_or_die("SUPABASE_SERVICE_KEY")
        client = openai.OpenAI(api_key=openai_key)
        headers = {
            "apikey": supabase_service_key, "Authorization": f"Bearer {supabase_service_key}",
            "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates",
        }
        print(f"Embed spend so far: ${spend_tracker['usd']:.4f} / $0.05 budget")

    EMBED_BUDGET = 0.05
    before_total_chunks = 0
    after_total_chunks = 0

    for pdf_name, info in sorted(pages_by_pdf.items()):
        if pdf_name in done:
            rec = done[pdf_name]
            print(f"SKIP (checkpointed) {pdf_name}: {rec}")
            before_total_chunks += rec.get("before_chunk_count", 0)
            after_total_chunks += rec.get("after_chunk_count", 0)
            continue

        warnings = {}
        chunks, full_text, missing = build_merged_chapter(pdf_name, info, ocr_records, warnings)
        text_path = os.path.join(TEXT_CLEAN_DIR, f"{os.path.splitext(pdf_name)[0]}_clean.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        chapter_name = info["chapter_name"]
        subject, class_num = info["subject"], info["class"]

        if missing:
            print(f"{pdf_name} ({chapter_name}): {len(chunks)} chunks, WARNING missing recovery for pages {missing}")
        else:
            print(f"{pdf_name} ({chapter_name}): {len(chunks)} chunks, all {len(info['pages'])} pages recovered")

        rec = {
            "pdf": pdf_name, "chapter_name": chapter_name, "after_chunk_count": len(chunks),
            "missing_recovery_pages": missing, "warnings": warnings,
        }

        if args.load:
            if spend_tracker["usd"] >= EMBED_BUDGET:
                print(f"EMBEDDING BUDGET REACHED (${spend_tracker['usd']:.4f} >= ${EMBED_BUDGET}). "
                      f"Stopping before {pdf_name}.")
                break
            before_count = _count_rows(subject, class_num, chapter_name, headers, supabase_url)
            inserted = _upsert_chunks(chunks, client, headers, supabase_url, spend_tracker)
            deleted = _delete_stale_rows(subject, class_num, chapter_name, len(chunks), headers, supabase_url)
            after_count = _count_rows(subject, class_num, chapter_name, headers, supabase_url)
            rec.update(before_row_count=before_count, inserted=inserted, deleted_stale=deleted,
                       after_row_count=after_count)
            _save_embed_spend(spend_tracker["usd"])
            print(f"  DB: before={before_count} rows, upserted={inserted}, deleted_stale={deleted}, "
                  f"after={after_count} rows (expect {len(chunks)}), embed spend running ${spend_tracker['usd']:.4f}")
            if after_count != len(chunks):
                print(f"  WARNING: row count mismatch after update! expected {len(chunks)}, got {after_count}")

        _append_checkpoint(rec, checkpoint_path)
        after_total_chunks += len(chunks)

    print(f"\nDone. Chapters processed this run checkpointed to {checkpoint_path}.")
    if args.load:
        print(f"Total embedding spend: ${spend_tracker['usd']:.4f} / ${EMBED_BUDGET}")


if __name__ == "__main__":
    main()
