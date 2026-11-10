#!/usr/bin/env python
"""One-time script: generates a one-line keyword summary for every section already loaded in
ncert_sections (see load_ncert_sections.py / ncert_sections.sql) and stores it in that table's
`keywords` column (see ncert_sections_keywords.sql -- run that migration first).

For each section, asks DeepSeek (thinking disabled) for 10 to 15 key terms and the kinds of
questions the section answers, in plain words, no sentences copied from the book. This exists
because some sections (e.g. "1.1 Diversity in the Living World", 1,100+ words) cover many topics
under one heading, so the picker prompt (_pick_ncert_sections in server.py) needs more than the
heading alone to tell them apart -- see _get_ncert_sections_cache for where this column is read.

Prints only each section's heading and the generated keyword line -- never the book text itself.

Usage:
    py load_ncert_section_keywords.py           # dry-run: calls the model, prints results, writes nothing
    py load_ncert_section_keywords.py --apply    # also PATCHes each row's keywords column
"""
import argparse
import os
import sys

import anthropic
import requests
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_KEY = os.getenv("DEEPSEEK_KEY")
DEEPSEEK_MODEL = "deepseek-flash"
SUPABASE_URL = os.getenv("SUPABASE_URL")
SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")

KEYWORDS_SYSTEM = (
    "You are tagging one section of an NCERT textbook chapter for a search index. You will be "
    "given the section's heading and its full text. Reply with ONLY ONE line: 10 to 15 key terms "
    "and short phrases (comma-separated) covering (a) the specific topics, named concepts, and "
    "terms this section defines or explains, and (b) the kinds of questions a student could ask "
    "that this section -- and only this section -- would answer. Plain words only, no copied "
    "sentences from the text, no numbering, no heading repeated verbatim as a whole, no "
    "explanation, nothing before or after the single line."
)


def fetch_sections():
    headers = {"apikey": SERVICE_KEY, "Authorization": f"Bearer {SERVICE_KEY}"}
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/ncert_sections", headers=headers,
        params={"select": "id,chapter,heading,text,keywords", "order": "chapter,ordinal"},
    )
    resp.raise_for_status()
    return resp.json()


def generate_keywords(client, heading: str, text: str) -> str:
    resp = client.messages.create(
        model=DEEPSEEK_MODEL, max_tokens=120, thinking={"type": "disabled"},
        system=KEYWORDS_SYSTEM,
        messages=[{"role": "user", "content": f"Heading: {heading}\n\nText:\n{text}"}],
    )
    return "".join(b.text for b in resp.content if hasattr(b, "text")).strip()


def write_keywords(section_id: int, keywords: str):
    headers = {
        "apikey": SERVICE_KEY, "Authorization": f"Bearer {SERVICE_KEY}",
        "Content-Type": "application/json", "Prefer": "return=minimal",
    }
    resp = requests.patch(
        f"{SUPABASE_URL}/rest/v1/ncert_sections", headers=headers,
        params={"id": f"eq.{section_id}"}, json={"keywords": keywords},
    )
    resp.raise_for_status()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Write generated keywords to Supabase (default: dry-run)")
    args = ap.parse_args()

    if not DEEPSEEK_KEY or not SUPABASE_URL or not SERVICE_KEY:
        print("Missing DEEPSEEK_KEY / SUPABASE_URL / SUPABASE_SERVICE_KEY in environment.", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com/anthropic")
    sections = fetch_sections()
    print(f"Loaded {len(sections)} sections.\n")

    for s in sections:
        keywords = generate_keywords(client, s["heading"], s["text"])
        print(f"[{s['id']}] {s['heading']!r}")
        print(f"    keywords: {keywords}\n")
        if args.apply:
            write_keywords(s["id"], keywords)

    if not args.apply:
        print("Dry-run only (no writes). Pass --apply to write to Supabase.")
    else:
        print(f"Wrote keywords for {len(sections)} sections.")


if __name__ == "__main__":
    main()
