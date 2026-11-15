#!/usr/bin/env python
"""Dry-run only: reports how many already-loaded ncert_sections rows contain figure/table
caption or multi-panel letter-label noise wedged mid-sentence (see
ncert_ingest_core.find_caption_panel_runs / strip_figure_caption_noise), and how many words would
change per chapter if that noise were stripped. Read-only -- never writes to Supabase, and never
prints or logs any section's actual text, only aggregate counts.

Usage:
    py ncert_figure_caption_noise_scan.py
"""
import os
import sys
from collections import defaultdict

import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ncert_ingest_core as core

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")

PAGE_SIZE = 1000


def fetch_all_sections():
    headers = {"apikey": SERVICE_KEY, "Authorization": f"Bearer {SERVICE_KEY}"}
    rows = []
    offset = 0
    while True:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/ncert_sections",
            headers=headers,
            params={
                "select": "id,class,subject,chapter,heading,text,word_count",
                "order": "class,subject,chapter,ordinal",
                "limit": PAGE_SIZE,
                "offset": offset,
            },
        )
        resp.raise_for_status()
        page = resp.json()
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return rows


def main():
    if not SUPABASE_URL or not SERVICE_KEY:
        print("Missing SUPABASE_URL / SUPABASE_SERVICE_KEY in environment.", file=sys.stderr)
        sys.exit(1)

    rows = fetch_all_sections()
    print(f"Fetched {len(rows)} ncert_sections rows (read-only, no text printed below).\n")

    # key = (class, subject, chapter)
    per_chapter_total = defaultdict(int)
    per_chapter_affected = defaultdict(int)
    per_chapter_words_removed = defaultdict(int)

    total_affected = 0
    total_words_removed = 0

    for row in rows:
        key = (row.get("class"), row.get("subject"), row.get("chapter"))
        per_chapter_total[key] += 1
        text = row.get("text") or ""
        _cleaned, removed = core.strip_figure_caption_noise(text)
        if removed:
            per_chapter_affected[key] += 1
            per_chapter_words_removed[key] += removed
            total_affected += 1
            total_words_removed += removed

    print(f"{'Class':<6}{'Subject':<12}{'Chapter':<45}{'Sections':<12}{'Affected':<10}{'Words would change'}")
    for key in sorted(per_chapter_total):
        class_num, subject, chapter = key
        total = per_chapter_total[key]
        affected = per_chapter_affected.get(key, 0)
        if not affected:
            continue
        words = per_chapter_words_removed.get(key, 0)
        chapter_label = (chapter or "")[:44]
        print(f"{str(class_num):<6}{str(subject):<12}{chapter_label:<45}{total:<12}{affected:<10}{words}")

    chapters_with_noise = sum(1 for k in per_chapter_total if per_chapter_affected.get(k))
    print(f"\nChapters with at least one affected section: {chapters_with_noise} / {len(per_chapter_total)}")
    print(f"Sections affected overall: {total_affected} / {len(rows)}")
    print(f"Total words that would change overall: {total_words_removed}")
    print("\nDry-run only -- no writes made. Nothing applied to the DB.")


if __name__ == "__main__":
    main()
