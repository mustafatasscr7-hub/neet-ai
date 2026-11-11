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

import fitz
import requests
from dotenv import load_dotenv

import ncert_clean_manual as clean_mod
import load_ncert_sections as load_mod
import load_ncert_section_keywords as kw_mod

FOLDER_RE = re.compile(r"^Class(\d+)_([A-Za-z]+)$", re.IGNORECASE)
BARE_HEADING_NUM_RE = re.compile(r"^(\d+(?:\.\d+)+)\s*$")
STANDALONE_CAPS_RE = re.compile(r"^[A-Z][A-Z\s']{2,59}$")

# Private-use-area and dingbat ranges: real NCERT body text should never
# contain these. A high share usually means a broken font/encoding in the
# PDF's text layer (common in scanned-then-OCR'd or poorly re-saved PDFs).
_GARBLED_RANGES = [(0xE000, 0xF8FF), (0x2700, 0x27BF)]  # private-use-area, dingbats
GARBLED_CHAR_RE = re.compile("[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in _GARBLED_RANGES) + "]")
GARBLED_CHAR_PCT_MAX = 0.5
COMMON_WORDS = {"the", "of", "and", "is", "are"}
COMMON_WORD_PCT_MIN = 8.0
MIN_HEADINGS = 3
PAGE_WORD_MIN = 15
PAGE_WORD_MAX = 1500

CHECKPOINT_PATH = Path("ncert_pdfs") / ".ncert_ingest_checkpoint.json"
REPORT_PATH = Path("ncert_pdfs") / "ncert_ingest_report.txt"

# Copied from server.py's DEEPSEEK_RATES / _is_deepseek_peak_hour (checked
# 2026-10-17) so this script can track its own spend without importing the
# whole server module. $ per 1M tokens throughout.
DEEPSEEK_RATES = {
    "peak": {"cache_hit": 0.006, "cache_miss": 0.30, "output": 1.20},
    "off_peak": {"cache_hit": 0.003, "cache_miss": 0.15, "output": 0.60},
}
CST = timezone(timedelta(hours=8))  # China Standard Time, fixed offset (no DST)
RUN_COST_CAP = 0.50


def _is_deepseek_peak_hour() -> bool:
    now_utc = datetime.now(timezone.utc)
    if now_utc.astimezone(CST).weekday() >= 5:
        return False
    hour = now_utc.hour
    return (1 <= hour < 4) or (6 <= hour < 10)


def _deepseek_cost(cache_miss_tokens: int, cache_hit_tokens: int, output_tokens: int, is_peak: bool) -> float:
    r = DEEPSEEK_RATES["peak" if is_peak else "off_peak"]
    return (cache_miss_tokens * r["cache_miss"] + cache_hit_tokens * r["cache_hit"] + output_tokens * r["output"]) / 1_000_000


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


def extract_pages_text(pdf_path: Path):
    doc = fitz.open(str(pdf_path))
    pages = [page.get_text() for page in doc]
    return doc, pages


def merge_wrapped_headings(lines):
    """A PDF's raw text layer often splits a numbered heading across lines
    (the section number alone, then its title on the very next non-blank
    line) where ncert_clean_manual.py's HEADING_RE expects both on one line
    - that module was built for already-typed/pasted text, where a human
    naturally keeps a heading on a single line. Rejoins that one case
    (confirmed against a real NCERT chapter PDF: "1.2.2\\nGenus\\nGenus
    comprises...") so real body headings get recognised the same way.
    Leaves anything that doesn't look like a short heading title alone."""
    out = []
    i, n = 0, len(lines)
    while i < n:
        m = BARE_HEADING_NUM_RE.match(lines[i].strip())
        if m:
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            if j < n:
                candidate = lines[j].strip()
                words = candidate.split()
                if candidate and candidate[0].isupper() and 1 <= len(words) <= 8 and not candidate.endswith((".", "?", "!")):
                    out.append(f"{m.group(1)} {candidate}")
                    i = j + 1
                    continue
        out.append(lines[i])
        i += 1
    return out


def recover_chapter_title(lines, chapter_num):
    """Some PDFs' chapter-opening sidebar puts the ALL-CAPS chapter title
    line immediately BEFORE the "CHAPTER n" line rather than after it
    (observed live: PyMuPDF's block order for the two-column opening-page
    layout put "THE LIVING WORLD" ahead of "CHAPTER  1"), so
    ncert_clean_manual.py's strip_front_matter - which only looks after
    that line - comes back with a number but no title. Falls back to the
    nearest standalone ALL-CAPS line directly above it."""
    if not chapter_num:
        return None
    for i, ln in enumerate(lines):
        m = clean_mod.CHAPTER_TITLE_RE.match(ln.strip())
        if m and m.group(1) == chapter_num:
            for back in lines[max(0, i - 3):i][::-1]:
                stripped = back.strip()
                if not stripped:
                    continue
                if STANDALONE_CAPS_RE.match(stripped):
                    return clean_mod.title_case(stripped)
                break
            return None
    return None


def validate_chapter(chapter_num, chapter_title, blocks, sections, pages):
    """All deterministic, no model calls. Returns a list of flag strings;
    any flag at all means the chapter is rejected (not loaded)."""
    flags = []

    if not chapter_num or not chapter_title:
        flags.append("chapter title not found (no 'CHAPTER n' block)")

    headings = [text for btype, text in blocks if btype == "heading"]
    if len(headings) < MIN_HEADINGS:
        flags.append(f"only {len(headings)} heading(s) found (minimum {MIN_HEADINGS})")

    for i, page_text in enumerate(pages, start=1):
        wc = len(page_text.split())
        if wc < PAGE_WORD_MIN:
            flags.append(f"page {i}: only {wc} word(s) (possibly blank/scanned-image page)")
        elif wc > PAGE_WORD_MAX:
            flags.append(f"page {i}: {wc} words (unusually dense - possible extraction glitch)")

    full_text = "".join(pages)
    total_chars = len(full_text) or 1
    garbled_pct = len(GARBLED_CHAR_RE.findall(full_text)) / total_chars * 100
    if garbled_pct > GARBLED_CHAR_PCT_MAX:
        flags.append(f"garbled font: {garbled_pct:.2f}% private-use/dingbat characters")

    tokens = re.findall(r"[A-Za-z']+", full_text.lower())
    if tokens:
        common_pct = sum(1 for t in tokens if t in COMMON_WORDS) / len(tokens) * 100
        if common_pct < COMMON_WORD_PCT_MIN:
            flags.append(f"garbled font: common English words only {common_pct:.2f}% of tokens")

    empty_sections = [s["heading"] for s in sections if not s["text"].strip()]
    if empty_sections:
        flags.append("empty section(s): " + ", ".join(empty_sections))

    numbered = []
    for btype, text in blocks:
        if btype != "heading":
            continue
        m = clean_mod.HEADING_RE.match(text)
        if m:
            numbered.append(tuple(int(x) for x in m.group(1).split(".")))
    for i in range(1, len(numbered)):
        if numbered[i] < numbered[i - 1]:
            flags.append(
                "section order: heading "
                + ".".join(map(str, numbered[i]))
                + " appears after "
                + ".".join(map(str, numbered[i - 1]))
            )
            break

    return flags


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


def generate_keywords_with_cost(client, heading, text, is_peak):
    resp = client.messages.create(
        model=kw_mod.DEEPSEEK_MODEL, max_tokens=120, thinking={"type": "disabled"},
        system=kw_mod.KEYWORDS_SYSTEM,
        messages=[{"role": "user", "content": f"Heading: {heading}\n\nText:\n{text}"}],
    )
    keywords = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
    usage = resp.usage
    cache_miss = usage.input_tokens + (usage.cache_creation_input_tokens or 0)
    cache_hit = usage.cache_read_input_tokens or 0
    cost = _deepseek_cost(cache_miss, cache_hit, usage.output_tokens, is_peak)
    return keywords, cost


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
            doc, pages = extract_pages_text(pdf_path)
        except Exception as e:
            results.append({
                "chapter_title": pdf_path.name, "status": "REJECTED", "sections": 0, "words": 0,
                "flags": [f"could not open/extract PDF: {e}"], "cost": 0.0,
            })
            continue

        raw_text = "\n\n".join(pages)
        raw_lines = merge_wrapped_headings(raw_text.replace("#", "").split("\n"))
        chapter_num, chapter_title, _ = clean_mod.strip_front_matter(raw_lines)
        if chapter_num and not chapter_title:
            recovered = recover_chapter_title(raw_lines, chapter_num)
            if recovered:
                chapter_title = recovered
                for idx, ln in enumerate(raw_lines):
                    m = clean_mod.CHAPTER_TITLE_RE.match(ln.strip())
                    if m and m.group(1) == chapter_num:
                        # Splice the recovered title in right after the "CHAPTER n"
                        # line, so clean_text's OWN internal strip_front_matter call
                        # (it re-derives this from scratch) finds it too, rather than
                        # this script and clean_mod silently disagreeing.
                        raw_lines = raw_lines[: idx + 1] + [recovered] + raw_lines[idx + 1 :]
                        break
        merged_text = "\n".join(raw_lines)
        cleaned_text, blocks = clean_mod.clean_text(merged_text, doc)
        sections = load_mod.parse_sections(cleaned_text)

        flags = validate_chapter(chapter_num, chapter_title, blocks, sections, pages)
        words = sum(s["word_count"] for s in sections)
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
