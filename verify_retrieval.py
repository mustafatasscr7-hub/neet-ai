"""Phase 6: retrieval-only before/after verification (no chat model calls).

"After" = the real live ncert_content_v2 table, via the production match_ncert_v2_hybrid RPC.
"Before" = reconstructed locally: the OLD (pre-recovery) chunk set for that one chapter, built by
running the UNMODIFIED build_chapter_chunks() (the same extraction ncert_rechunk_v2.py --load
originally ran, still skipping the oversized pages) -- i.e. exactly what used to be live before
this run's upsert overwrote it. Embedded once per chapter and ranked locally by cosine similarity
against the same query embedding used for the live RPC call, so "before" and "after" are directly
comparable. (The real pre-recovery rows could not be queried after the fact since the upsert in
merge_and_rechunk.py --load already replaced them in place -- this reconstruction is exact because
build_chapter_chunks()'s extraction logic is untouched; only merge_and_rechunk.py's recovered-page
splice is new.)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv

load_dotenv()

import numpy as np
import openai
import requests

from ncert_rechunk_v2 import PDF_DIR, EMBEDDING_MODEL, build_chapter_chunks, _chapter_groups

OPENAI_KEY = os.environ["OPENAI_KEY"]
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
client = openai.OpenAI(api_key=OPENAI_KEY)

QUERIES = [
    ("Plant Kingdom", "Biology", 11, "Give examples of pteridophyte plants."),
    ("Biological Classification", "Biology", 11, "Who discovered viruses using the tobacco mosaic disease and how?"),
    ("Cell: The Unit of Life", "Biology", 11, "What organelles make up the endomembrane system?"),
    ("Human Reproduction", "Biology", 12, "Why is the scrotum important for spermatogenesis?"),
    ("Principles of Inheritance and Variation", "Biology", 12, "What is a monohybrid cross?"),
    ("Evolution", "Biology", 12, "What are homologous structures and how do forelimbs show it?"),
    ("Locomotion and Movement", "Biology", 11, "What are the bones of the pectoral girdle?"),
    ("Ecosystem", "Biology", 12, "What factors affect the rate of decomposition?"),
    ("Solutions", "Chemistry", 12, "How does Raoult's law relate to Henry's law?"),
]

EMBED_SPEND_FILE = "ncert_page_recovery_embed_spend.json"


def _embed(text: str):
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding, resp.usage.total_tokens


def _cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _chapter_pdf_path(chapter_name: str):
    for subject, class_num, name, path in _chapter_groups():
        if name == chapter_name:
            return path
    raise ValueError(f"no PDF mapped for chapter {chapter_name!r}")


def main():
    spend = 0.0
    if os.path.exists(EMBED_SPEND_FILE):
        spend = json.load(open(EMBED_SPEND_FILE, encoding="utf-8")).get("embed_spend_usd", 0.0)

    old_chunk_cache = {}  # chapter_name -> [(heading, page, embedding), ...]

    headers = {
        "apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
    }

    report_lines = []
    for chapter_name, subject, class_num, query in QUERIES:
        print(f"\n=== {chapter_name} | {query!r} ===")
        query_embedding, qtok = _embed(query)
        spend += qtok * 0.02 / 1_000_000

        # BEFORE: reconstruct old chunk set for this chapter (unmodified extraction, still
        # skipping the oversized pages -- exactly what used to be live), embed once, cache.
        if chapter_name not in old_chunk_cache:
            path = _chapter_pdf_path(chapter_name)
            old_chunks = build_chapter_chunks(subject, class_num, chapter_name, path, warnings={})
            embedded = []
            for c in old_chunks:
                emb, tok = _embed(c["content"])
                spend += tok * 0.02 / 1_000_000
                embedded.append((c["heading"], c["page"], emb))
            old_chunk_cache[chapter_name] = embedded
            print(f"  (reconstructed {len(embedded)} old chunks for {chapter_name}, spend now ${spend:.4f})")
        scored = [(h, p, _cosine(query_embedding, e)) for h, p, e in old_chunk_cache[chapter_name]]
        scored.sort(key=lambda t: -t[2])
        before_top3 = scored[:3]

        # AFTER: real live RPC against the now-updated ncert_content_v2.
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/rpc/match_ncert_v2_hybrid", headers=headers,
            json={"query_embedding": query_embedding, "query_text": query, "match_count": 3,
                  "filter_subject": subject, "filter_class": class_num}, timeout=30)
        after_rows = resp.json() if resp.status_code == 200 else []

        print("  BEFORE (reconstructed old table):")
        for h, p, sim in before_top3:
            print(f"    sim={sim:.3f}  page={p}  heading={h!r}")
        print("  AFTER (live ncert_content_v2):")
        for r in after_rows:
            print(f"    sim={r.get('similarity', 0):.3f}  page={r.get('page')}  heading={r.get('heading')!r}")

        report_lines.append({
            "chapter": chapter_name, "query": query,
            "before": [{"page": p, "heading": h, "sim": round(sim, 3)} for h, p, sim in before_top3],
            "after": [{"page": r.get("page"), "heading": r.get("heading"), "sim": round(r.get("similarity", 0), 3)}
                      for r in after_rows],
        })

    with open(EMBED_SPEND_FILE, "w", encoding="utf-8") as f:
        json.dump({"embed_spend_usd": spend}, f, indent=2)
    with open("ncert_page_recovery_retrieval_verification.json", "w", encoding="utf-8") as f:
        json.dump(report_lines, f, indent=2)

    print(f"\n\nTotal embed spend after verification: ${spend:.4f} / $0.05")


if __name__ == "__main__":
    main()
