#!/usr/bin/env python
"""One-command local ingest for official NCERT chapter PDFs into ncert_sections.

Walks a folder tree of PDFs (default ncert_pdfs/, with subfolders named like
Class11_Biology - class and subject come from the folder name, or pass
--class/--subject directly). For each PDF: extracts the text layer in page
order (PyMuPDF), then reuses ncert_clean_manual.py's cleaning rules (dropped
headers/page numbers/reprint lines, rejoined broken lines, "## " headings,
reconstructed tables) and load_ncert_sections.py's section parser, so the
output matches what the existing manual-paste pipeline produces byte-for-byte
on the same source text.

Every chapter is validated by deterministic checks only (no model calls) -
see validate_chapter(). A chapter that fails any check is NOT loaded; it is
reported with the reason(s) and the run moves on to the next one.

Dry-run by default: prints the summary table, writes nothing. --apply writes
sections into ncert_sections (load_ncert_sections.py's own upsert pattern),
then generates each section's one-line keyword summary with DeepSeek exactly
as load_ncert_section_keywords.py does, and stores it. Chapters already
present in ncert_sections (by class, subject, chapter) are skipped unless
--force. Progress is checkpointed per chapter in
ncert_pdfs/.ncert_ingest_checkpoint.json so a re-run resumes instead of
reprocessing (and re-billing DeepSeek for) chapters already loaded.

DeepSeek spend for keyword generation is capped at RUN_COST_CAP per run: once
the running total reaches the cap, the run stops starting new chapters and
reports which chapters it did not reach.

Usage:
    py ncert_ingest.py ncert_pdfs/Class11_Biology
    py ncert_ingest.py ncert_pdfs/Class11_Biology --apply
    py ncert_ingest.py ncert_pdfs --class 11 --subject Biology --force --apply
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

import load_ncert_section_keywords as kw_mod
import ncert_ingest_core as core
from ncert_ingest_core import (
    DEEPSEEK_RATES,
    CST,
    merge_wrapped_headings,
    recover_chapter_title,
    validate_chapter,
    generate_keywords_with_cost,
)

FOLDER_RE = re.compile(r"^Class(\d+)_([A-Za-z]+)$", re.IGNORECASE)

CHECKPOINT_PATH = Path("ncert_pdfs") / ".ncert_ingest_checkpoint.json"
REPORT_PATH = Path("ncert_pdfs") / "ncert_ingest_report.txt"

RUN_COST_CAP = 0.50


def _is_deepseek_peak_hour() -> bool:
    return core.is_deepseek_peak_hour()


def _deepseek_cost(cache_miss_tokens: int, cache_hit_tokens: int, output_tokens: int, is_peak: bool) -> float:
    return core.deepseek_cost(cache_miss_tokens, cache_hit_tokens, output_tokens, is_peak)


def discover_pdfs(root: Path):
    return sorted(p for p in root.rglob("*.pdf"))


def resolve_class_subject(pdf_path: Path, root: Path, class_override, subject_override):
    """Class/subject come from the nearest ancestor folder named
    Class<N>_<Subject>, falling back to --class/--subject for whichever of
    the two that folder name doesn't supply."""
    cls, subj = class_override, subject_override
    if cls and subj:
        return cls, subj
    root_resolved = root.resolve()
    cur = pdf_path.resolve().parent
    while True:
        m = FOLDER_RE.match(cur.name)
        if m:
            cls = cls or m.group(1)
            subj = subj or m.group(2)
            break
        if cur == root_resolved or cur.parent == cur:
            break
        cur = cur.parent
    return cls, subj


def should_skip(key, existing, checkpoint, force):
    if force:
        return False
    if key in existing:
        return True
    ckey = "|".join(key)
    return checkpoint.get(ckey, {}).get("status") == "loaded"


def fetch_existing_chapters(supabase_url, headers):
    resp = requests.get(
        f"{supabase_url}/rest/v1/ncert_sections", headers=headers,
        params={"select": "class,subject,chapter"},
    )
    resp.raise_for_status()
    return {(r["class"], r["subject"], r["chapter"]) for r in resp.json()}


def write_sections(supabase_url, service_key, cls, subj, chapter_title, sections):
    headers = {
        "apikey": service_key, "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }
    rows = [
        {
            "class": cls, "subject": subj, "chapter": chapter_title,
            "heading": s["heading"], "ordinal": s["ordinal"],
            "text": s["text"], "word_count": s["word_count"],
        }
        for s in sections
    ]
    resp = requests.post(
        f"{supabase_url}/rest/v1/ncert_sections", headers=headers,
        params={"on_conflict": "class,subject,chapter,ordinal"}, json=rows,
    )
    resp.raise_for_status()

    get_headers = {"apikey": service_key, "Authorization": f"Bearer {service_key}"}
    resp2 = requests.get(
        f"{supabase_url}/rest/v1/ncert_sections", headers=get_headers,
        params={
            "select": "id,heading,text,ordinal",
            "class": f"eq.{cls}", "subject": f"eq.{subj}", "chapter": f"eq.{chapter_title}",
            "order": "ordinal",
        },
    )
    resp2.raise_for_status()
    return resp2.json()


def format_summary(results):
    header = ["Chapter", "Status", "Sections", "Words", "Flags", "DeepSeek cost"]
    rows = []
    for r in results:
        flags = "; ".join(r.get("flags", [])) or "-"
        rows.append([
            r.get("chapter_title") or "?",
            r["status"],
            str(r.get("sections", 0)),
            str(r.get("words", 0)),
            flags,
            f"${r.get('cost', 0.0):.4f}",
        ])
    widths = [
        max([len(header[i])] + [len(row[i]) for row in rows])
        for i in range(len(header))
    ]

    def fmt_row(cols):
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols))

    lines = [fmt_row(header), " | ".join("-" * w for w in widths)]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default="ncert_pdfs", help="Root folder to scan for PDFs (default: ncert_pdfs)")
    ap.add_argument("--class", dest="class_override", default=None, help="Override the class derived from the folder name")
    ap.add_argument("--subject", dest="subject_override", default=None, help="Override the subject derived from the folder name")
    ap.add_argument("--apply", action="store_true", help="Write to Supabase and generate DeepSeek keywords (default: dry-run)")
    ap.add_argument("--force", action="store_true", help="Reprocess chapters already in ncert_sections / the checkpoint")
    args = ap.parse_args()

    root = Path(args.folder)
    if not root.exists():
        print(f"No such folder: {root}", file=sys.stderr)
        sys.exit(1)

    pdf_paths = discover_pdfs(root)
    if not pdf_paths:
        print(f"No PDFs found under {root}")
        return

    load_dotenv()
    supabase_url = os.getenv("SUPABASE_URL")
    service_key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
    deepseek_key = os.getenv("DEEPSEEK_KEY")

    existing = set()
    if supabase_url and service_key:
        try:
            existing = fetch_existing_chapters(supabase_url, {"apikey": service_key, "Authorization": f"Bearer {service_key}"})
        except Exception as e:
            print(f"Warning: could not fetch existing ncert_sections for the skip-check: {e}", file=sys.stderr)
    else:
        print("Warning: SUPABASE_URL/SUPABASE_SERVICE_KEY not set - skipping the already-loaded DB check (the checkpoint file still applies).", file=sys.stderr)

    checkpoint = {}
    if CHECKPOINT_PATH.exists():
        checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))

    deepseek_client = None
    results = []
    run_cost = 0.0
    cap_hit = False

    for pdf_path in pdf_paths:
        if cap_hit:
            results.append({
                "chapter_title": pdf_path.name, "status": "NOT PROCESSED (cost cap reached)",
                "sections": 0, "words": 0, "flags": [], "cost": 0.0,
            })
            continue

        cls, subj = resolve_class_subject(pdf_path, root, args.class_override, args.subject_override)
        if not cls or not subj:
            results.append({
                "chapter_title": pdf_path.name, "status": "REJECTED", "sections": 0, "words": 0,
                "flags": ["cannot determine class/subject from the folder name; pass --class/--subject"],
                "cost": 0.0,
            })
            continue

        try:
            result = core.process_chapter_pdf_bytes(pdf_path.read_bytes())
        except Exception as e:
            results.append({
                "chapter_title": pdf_path.name, "status": "REJECTED", "sections": 0, "words": 0,
                "flags": [f"could not open/extract PDF: {e}"], "cost": 0.0,
            })
            continue

        chapter_title = result["chapter_title"]
        sections = result["sections"]
        flags = result["flags"]
        words = result["words"]
        row = {"chapter_title": chapter_title or pdf_path.name, "sections": len(sections), "words": words}

        if flags:
            row.update(status="REJECTED", flags=flags, cost=0.0)
            results.append(row)
            continue

        key = (cls, subj, chapter_title)
        if should_skip(key, existing, checkpoint, args.force):
            row.update(status="SKIPPED (already in ncert_sections)", flags=[], cost=0.0)
            results.append(row)
            continue

        if not args.apply:
            row.update(status="OK (dry-run)", flags=[], cost=0.0)
            results.append(row)
            continue

        if not supabase_url or not service_key:
            row.update(status="REJECTED", flags=["missing SUPABASE_URL/SUPABASE_SERVICE_KEY"], cost=0.0)
            results.append(row)
            continue

        try:
            db_rows = write_sections(supabase_url, service_key, cls, subj, chapter_title, sections)
        except Exception as e:
            row.update(status="REJECTED", flags=[f"Supabase write failed: {e}"], cost=0.0)
            results.append(row)
            continue

        chapter_cost = 0.0
        keywords_incomplete = False
        if deepseek_key:
            if deepseek_client is None:
                import anthropic
                deepseek_client = anthropic.Anthropic(api_key=deepseek_key, base_url="https://api.deepseek.com/anthropic")
            is_peak = _is_deepseek_peak_hour()
            for s in db_rows:
                if run_cost >= RUN_COST_CAP:
                    cap_hit = True
                    keywords_incomplete = True
                    break
                try:
                    keywords, cost = generate_keywords_with_cost(deepseek_client, s["heading"], s["text"], is_peak)
                    run_cost += cost
                    chapter_cost += cost
                    kw_mod.write_keywords(s["id"], keywords)
                except Exception as e:
                    print(f"Warning: keyword generation failed for section {s['id']}: {e}", file=sys.stderr)
        else:
            keywords_incomplete = True

        ckey = "|".join(key)
        checkpoint[ckey] = {
            "status": "loaded", "cost": chapter_cost, "pdf": str(pdf_path),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_PATH.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

        status = "LOADED (keywords incomplete)" if keywords_incomplete else "LOADED"
        row.update(status=status, flags=[], cost=chapter_cost)
        results.append(row)

        if cap_hit:
            print(f"DeepSeek run cost cap (${RUN_COST_CAP:.2f}) reached - stopping new chapters.", file=sys.stderr)

    summary = format_summary(results)
    total_cost = sum(r.get("cost", 0.0) for r in results)
    print()
    print(summary)
    print(f"\nTotal DeepSeek cost this run: ${total_cost:.4f}")

    report_lines = [summary, "", f"Total DeepSeek cost this run: ${total_cost:.4f}"]
    if cap_hit:
        note = f"Run cost cap (${RUN_COST_CAP:.2f}) reached - some chapters were not processed. Re-run the same command to resume."
        print(note)
        report_lines.append(note)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\nWrote {REPORT_PATH}")

    if not args.apply:
        print("\nDry-run only (no writes). Pass --apply to write to Supabase and generate keywords.")


if __name__ == "__main__":
    main()
