#!/usr/bin/env python
"""Load a cleaned NCERT chapter file (from ncert_clean_manual.py) into the
ncert_sections table. Dry-run by default; pass --apply to actually write.

Usage:
    py load_ncert_sections.py ncert_manual/clean/Class11_Biology_The_Living_World.txt [--apply]

Expects the input filename as "Class<N>_<Subject>_<Chapter_Name>.txt" and
uses the server's own Supabase REST pattern (SUPABASE_URL +
SUPABASE_SERVICE_KEY from .env), matching server.py's ADMIN_HEADERS /
on_conflict upsert style.
"""
import argparse
import os
import re
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

HEADING_RE = re.compile(r"^##\s+(.+)$")
FILENAME_RE = re.compile(r"^Class(\d+)_([A-Za-z]+)_(.+)$")


def parse_filename(path: Path):
    m = FILENAME_RE.match(path.stem)
    if not m:
        raise ValueError(
            f"Filename {path.name!r} doesn't match 'Class<N>_<Subject>_<Chapter_Name>.txt'"
        )
    class_num, subject, chapter_slug = m.groups()
    chapter = chapter_slug.replace("_", " ").strip()
    return class_num, subject, chapter


def parse_sections(text: str):
    lines = text.split("\n")
    sections = []
    heading = None
    body = []

    def flush(ordinal):
        if heading is None:
            return None
        body_text = "\n".join(body).strip()
        return {
            "heading": heading,
            "text": body_text,
            "word_count": len(body_text.split()),
            "ordinal": ordinal,
        }

    ordinal = 0
    for line in lines:
        m = HEADING_RE.match(line.strip())
        if m:
            section = flush(ordinal)
            if section:
                sections.append(section)
                ordinal += 1
            heading = m.group(1)
            body = []
        else:
            if line.strip():
                body.append(line)
    section = flush(ordinal)
    if section:
        sections.append(section)
    return sections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_path")
    ap.add_argument("--apply", action="store_true", help="Actually write to Supabase (default: dry-run)")
    args = ap.parse_args()

    in_path = Path(args.input_path)
    class_num, subject, chapter = parse_filename(in_path)
    sections = parse_sections(in_path.read_text(encoding="utf-8"))

    print(f"Class: {class_num}  Subject: {subject}  Chapter: {chapter}")
    print(f"Parsed {len(sections)} sections:")
    for s in sections:
        print(f"  [{s['ordinal']}] {s['heading']!r} ({s['word_count']} words)")

    if not args.apply:
        print("\nDry-run only (no writes). Pass --apply to write to Supabase.")
        return

    load_dotenv()
    supabase_url = os.getenv("SUPABASE_URL")
    service_key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
    if not supabase_url or not service_key:
        print("Missing SUPABASE_URL / SUPABASE_SERVICE_KEY in environment.", file=sys.stderr)
        sys.exit(1)

    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=representation",
    }
    rows = [
        {
            "class": class_num,
            "subject": subject,
            "chapter": chapter,
            "heading": s["heading"],
            "ordinal": s["ordinal"],
            "text": s["text"],
            "word_count": s["word_count"],
        }
        for s in sections
    ]
    resp = requests.post(
        f"{supabase_url}/rest/v1/ncert_sections",
        headers=headers,
        params={"on_conflict": "class,subject,chapter,ordinal"},
        json=rows,
    )
    print(f"\nSupabase response: {resp.status_code}")
    if resp.status_code >= 400:
        print(resp.text, file=sys.stderr)
        sys.exit(1)
    print(f"Wrote {len(rows)} rows.")


if __name__ == "__main__":
    main()
