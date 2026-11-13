#!/usr/bin/env python
"""Shared NCERT chapter-PDF extraction/clean/validate pipeline, used by both ncert_ingest.py (the
local batch CLI) and server.py's admin NCERT Content endpoints. Pulled out of ncert_ingest.py so
the admin API can reuse the exact same extraction, heading-recovery and validation logic without
importing that script's CLI-only argparse/checkpoint/folder-walking code. Deliberately has no
Supabase or dotenv dependency of its own (unlike ncert_ingest.py) -- callers own their own writes.
"""
import re
import time
from datetime import datetime, timedelta, timezone

import fitz

import ncert_clean_manual as clean_mod
import load_ncert_sections as load_mod
import load_ncert_section_keywords as kw_mod

BARE_HEADING_NUM_RE = re.compile(r"^(\d+(?:\.\d+)+)\s*$")
STANDALONE_CAPS_RE = re.compile(r"^[A-Z][A-Z\s']{2,59}$")

# Private-use-area and dingbat ranges: real NCERT body text should never contain these. A high
# share usually means a broken font/encoding in the PDF's text layer (common in scanned-then-OCR'd
# or poorly re-saved PDFs).
_GARBLED_RANGES = [(0xE000, 0xF8FF), (0x2700, 0x27BF)]  # private-use-area, dingbats
GARBLED_CHAR_RE = re.compile("[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in _GARBLED_RANGES) + "]")
GARBLED_CHAR_PCT_MAX = 0.5
COMMON_WORDS = {"the", "of", "and", "is", "are"}
COMMON_WORD_PCT_MIN = 8.0
MIN_HEADINGS = 3
PAGE_WORD_MIN = 15
PAGE_WORD_MAX = 1500

# Copied from server.py's DEEPSEEK_RATES / _is_deepseek_peak_hour (checked 2026-10-17) so this
# module can track its own spend without importing the whole server module.$ per 1M tokens.
DEEPSEEK_RATES = {
    "peak": {"cache_hit": 0.006, "cache_miss": 0.30, "output": 1.20},
    "off_peak": {"cache_hit": 0.003, "cache_miss": 0.15, "output": 0.60},
}
CST = timezone(timedelta(hours=8))  # China Standard Time, fixed offset (no DST)


def is_deepseek_peak_hour() -> bool:
    now_utc = datetime.now(timezone.utc)
    if now_utc.astimezone(CST).weekday() >= 5:
        return False
    hour = now_utc.hour
    return (1 <= hour < 4) or (6 <= hour < 10)


def deepseek_cost(cache_miss_tokens: int, cache_hit_tokens: int, output_tokens: int, is_peak: bool) -> float:
    r = DEEPSEEK_RATES["peak" if is_peak else "off_peak"]
    return (cache_miss_tokens * r["cache_miss"] + cache_hit_tokens * r["cache_hit"] + output_tokens * r["output"]) / 1_000_000


def extract_pages_text_from_bytes(pdf_bytes: bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [page.get_text() for page in doc]
    return doc, pages


def merge_wrapped_headings(lines):
    """A PDF's raw text layer often splits a numbered heading across lines (the section number
    alone, then its title on the very next non-blank line) where ncert_clean_manual.py's
    HEADING_RE expects both on one line - that module was built for already-typed/pasted text,
    where a human naturally keeps a heading on a single line. Rejoins that one case (confirmed
    against a real NCERT chapter PDF: "1.2.2\\nGenus\\nGenus comprises...") so real body headings
    get recognised the same way. Leaves anything that doesn't look like a short heading title
    alone."""
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
    """Some PDFs' chapter-opening sidebar puts the ALL-CAPS chapter title line immediately BEFORE
    the "CHAPTER n" line rather than after it (observed live: PyMuPDF's block order for the
    two-column opening-page layout put "THE LIVING WORLD" ahead of "CHAPTER  1"), so
    ncert_clean_manual.py's strip_front_matter - which only looks after that line - comes back
    with a number but no title. Falls back to the nearest standalone ALL-CAPS line directly above
    it."""
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
    """All deterministic, no model calls. Returns a list of flag strings; any flag at all means
    the chapter is rejected (not loaded/not applyable from the PDF route)."""
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


def process_chapter_pdf_bytes(pdf_bytes: bytes):
    """Runs the full extract -> clean -> parse-sections -> validate pipeline on one chapter PDF's
    raw bytes, matching ncert_ingest.py's own per-chapter logic byte-for-byte. No Supabase/model
    calls -- deterministic only. Returns a dict: chapter_num, chapter_title, sections, flags,
    words, section_count, pages (PDF page count), extract_seconds (wall-clock time this function
    took -- both logged to ncert_chapter_log by callers, see ncert_chapter_log.sql)."""
    start = time.perf_counter()
    doc, pages = extract_pages_text_from_bytes(pdf_bytes)
    try:
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
                        # Splice the recovered title in right after the "CHAPTER n" line, so
                        # clean_text's OWN internal strip_front_matter call (it re-derives this
                        # from scratch) finds it too, rather than this module and clean_mod
                        # silently disagreeing.
                        raw_lines = raw_lines[: idx + 1] + [recovered] + raw_lines[idx + 1 :]
                        break
        merged_text = "\n".join(raw_lines)
        cleaned_text, blocks = clean_mod.clean_text(merged_text, doc)
        sections = load_mod.parse_sections(cleaned_text)
        flags = validate_chapter(chapter_num, chapter_title, blocks, sections, pages)
        words = sum(s["word_count"] for s in sections)
    finally:
        doc.close()

    return {
        "chapter_num": chapter_num,
        "chapter_title": chapter_title,
        "sections": sections,
        "flags": flags,
        "words": words,
        "section_count": len(sections),
        "pages": len(pages),
        "extract_seconds": round(time.perf_counter() - start, 3),
    }


def generate_keywords_with_cost(client, heading, text, is_peak):
    """One DeepSeek call for one section's keyword line, using the same model/prompt as
    load_ncert_section_keywords.py (kw_mod). Returns (keywords, cost)."""
    resp = client.messages.create(
        model=kw_mod.DEEPSEEK_MODEL, max_tokens=120, thinking={"type": "disabled"},
        system=kw_mod.KEYWORDS_SYSTEM,
        messages=[{"role": "user", "content": f"Heading: {heading}\n\nText:\n{text}"}],
    )
    keywords = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
    usage = resp.usage
    cache_miss = usage.input_tokens + (usage.cache_creation_input_tokens or 0)
    cache_hit = usage.cache_read_input_tokens or 0
    cost = deepseek_cost(cache_miss, cache_hit, usage.output_tokens, is_peak)
    return keywords, cost
