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
FULL_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)+)\s+(.+)$")
STANDALONE_CAPS_RE = re.compile(r"^[A-Z][A-Z\s']{2,59}$")

# A heading's title must look like one: real words, not a figure/table caption's "a)." / "b)."
# letter-in-parentheses continuation, and not a caption line itself ("Figure 6.3 b). It has...").
CAPTION_PREFIX_RE = re.compile(r"^(?:Figure|Fig\.|Table)\b", re.IGNORECASE)
LETTER_PAREN_RE = re.compile(r"^\(?[a-zA-Z]\)\.?(?:\s|$)")
CONTINUATION_END_WORDS = {"of", "and", "the"}
HEADING_CONT_MAX_LINES = 5
HEADING_CONT_SHORT_WORDS = 3

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


def _is_heading_fragment(candidate):
    """True if a line plausibly continues a heading title: short, starts with a capital letter,
    not a sentence (no terminal punctuation), and not a figure/table caption, its "a)."/"b)."
    letter-in-parentheses continuation, or a page-footer reprint line -- none of those are ever
    headings."""
    if not candidate:
        return False
    if CAPTION_PREFIX_RE.match(candidate):
        return False
    if LETTER_PAREN_RE.match(candidate):
        return False
    if clean_mod.REPRINT_RE.match(candidate):
        return False
    words = candidate.split()
    if not (1 <= len(words) <= 8):
        return False
    if not candidate[0].isupper():
        return False
    if candidate.endswith((".", "?", "!")):
        return False
    return True


def merge_wrapped_headings(lines):
    """A PDF's raw text layer often splits a numbered heading across lines - either the section
    number alone with its title on the next non-blank line ("1.2.2\\nGenus\\nGenus comprises..."),
    or the number and the start of its title together with the rest wrapping onto one or more
    further lines ("6.2\\nANATOMY OF DICOTYLEDONOUS AND\\nMONOCOTYLEDONOUS PLANTS",
    "6.2.5\\nDorsiventral\\n(Dicotyledonous) Leaf") - where ncert_clean_manual.py's HEADING_RE
    expects it all on one line (that module was built for already-typed/pasted text, where a
    human naturally keeps a heading on a single line). Rejoins both cases so real body headings
    get recognised the same way, while rejecting figure/table captions (never headings) so a
    caption fragment like "Figure\\n6.3 b). It has..." doesn't get mistaken for one."""
    out = []
    i, n = 0, len(lines)
    while i < n:
        stripped = lines[i].strip()
        bare_m = BARE_HEADING_NUM_RE.match(stripped)
        full_m = None if bare_m else FULL_HEADING_RE.match(stripped)

        if bare_m:
            heading_num = bare_m.group(1)
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            if j >= n or not _is_heading_fragment(lines[j].strip()):
                out.append(lines[i])
                i += 1
                continue
            parts = [lines[j].strip()]
            j += 1
        elif full_m:
            title = full_m.group(2).strip()
            if CAPTION_PREFIX_RE.match(title) or LETTER_PAREN_RE.match(title):
                out.append(lines[i])
                i += 1
                continue
            heading_num = full_m.group(1)
            parts = [title]
            j = i + 1
        else:
            out.append(lines[i])
            i += 1
            continue

        extra = 0
        while extra < HEADING_CONT_MAX_LINES and j < n:
            joined = " ".join(parts)
            joined_words = joined.split()
            last_word = re.sub(r"[^A-Za-z]", "", joined_words[-1]).lower() if joined_words else ""
            ends_open = last_word in CONTINUATION_END_WORDS or joined.rstrip().endswith(",")
            all_caps = bool(joined_words) and joined == joined.upper() and joined != joined.lower()
            nxt = lines[j].strip()
            if not nxt:
                break
            if ends_open and _is_heading_fragment(nxt):
                pass
            elif all_caps and nxt == nxt.upper() and _is_heading_fragment(nxt):
                pass
            elif (
                len(joined_words) <= HEADING_CONT_SHORT_WORDS
                and nxt.startswith("(")
                and len(nxt.split()) <= 4
                and not nxt.rstrip().endswith((".", "?", "!"))
                and not CAPTION_PREFIX_RE.match(nxt)
            ):
                # A short parenthetical qualifier right after a short heading start, e.g.
                # "Dorsiventral" + "(Dicotyledonous) Leaf". Deliberately narrower than the
                # ends_open/all_caps rules above (must literally start with "(") -- a looser
                # "any short capitalised line" rule here previously absorbed unrelated repeated
                # words (a sidebar echo of the same short heading, a stray page-footer line) into
                # the heading text on other chapters.
                pass
            else:
                break
            parts.append(nxt)
            j += 1
            extra += 1

        out.append(f"{heading_num} {' '.join(parts)}")
        i = j
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


MIN_SECTION_WORDS = 3


def validate_sections(sections):
    """Deterministic, PDF-independent revalidation for a hand-edited list of sections (the
    editable-preview flow in server.py, both for a not-yet-applied preview and for a chapter
    already loaded into ncert_sections). Mirrors validate_chapter's own rules (too few headings,
    empty sections, a minimum word count, duplicate headings, out-of-order numbered headings,
    garbled characters) but works directly off {"heading", "text"} dicts instead of the PDF's own
    blocks/pages -- an edit has no PDF page structure to re-check. Any flag means the edited
    sections can't be applied/saved yet. Never raises on malformed input; a missing field is just
    treated as empty."""
    flags = []
    if len(sections) < MIN_HEADINGS:
        flags.append(f"only {len(sections)} heading(s) found (minimum {MIN_HEADINGS})")

    empty = [s.get("heading") or "(no heading)" for s in sections if not (s.get("text") or "").strip()]
    if empty:
        flags.append("empty section(s): " + ", ".join(empty))

    too_short = [
        s.get("heading") or "(no heading)" for s in sections
        if (s.get("text") or "").strip() and len((s.get("text") or "").split()) < MIN_SECTION_WORDS
    ]
    if too_short:
        flags.append(f"section(s) under {MIN_SECTION_WORDS} words: " + ", ".join(too_short))

    seen = set()
    dupes = []
    for s in sections:
        h = (s.get("heading") or "").strip()
        if h in seen and h not in dupes:
            dupes.append(h)
        seen.add(h)
    if dupes:
        flags.append("duplicate heading(s): " + ", ".join(dupes))

    full_text = "\n".join((s.get("text") or "") for s in sections)
    total_chars = len(full_text) or 1
    garbled_pct = len(GARBLED_CHAR_RE.findall(full_text)) / total_chars * 100
    if garbled_pct > GARBLED_CHAR_PCT_MAX:
        flags.append(f"garbled font: {garbled_pct:.2f}% private-use/dingbat characters")

    tokens = re.findall(r"[A-Za-z']+", full_text.lower())
    if tokens:
        common_pct = sum(1 for t in tokens if t in COMMON_WORDS) / len(tokens) * 100
        if common_pct < COMMON_WORD_PCT_MIN:
            flags.append(f"garbled font: common English words only {common_pct:.2f}% of tokens")

    numbered = []
    for s in sections:
        m = clean_mod.HEADING_RE.match((s.get("heading") or "").strip())
        if m:
            try:
                numbered.append(tuple(int(x) for x in m.group(1).split(".")))
            except ValueError:
                continue
    for i in range(1, len(numbered)):
        if numbered[i] < numbered[i - 1]:
            flags.append(
                "section order: heading " + ".".join(map(str, numbered[i]))
                + " appears after " + ".".join(map(str, numbered[i - 1]))
            )
            break

    return flags


# A figure/table caption and its multi-panel letter labels ("Figure 9.9 (a) Prophase (b)
# Metaphase") sometimes land, in the PDF's own text-layer reading order, between two halves of the
# SAME body sentence instead of on their own line -- the page's visual column layout interleaves
# the figure with the paragraph wrapping around it. By the time build_blocks has already joined a
# block's lines with spaces this noise is indistinguishable from real body text except for its own
# shape, so it's detected here, on the final joined block text, instead. A lone "Figure N.N" is
# NEVER removed by itself -- that's indistinguishable from a normal inline cross-reference a real
# NCERT sentence makes all the time ("...as shown in Figure 9.9."). Only a run of 2+ ascending
# single-letter panel markers close together ("(a) ... (b) ...", optionally preceded by the
# "Figure N.N" caption that introduces them) is dropped, since that shape never occurs in real
# NCERT body prose.
FIGURE_CAPTION_PREFIX_RE = re.compile(r"\b(?:Figure|Fig\.|Table)\s+\d+(?:\.\d+)*\b", re.IGNORECASE)
PANEL_LETTER_RE = re.compile(r"\(\s*([a-h])\s*\)", re.IGNORECASE)
CAPTION_RUN_MAX_GAP_WORDS = 6
CAPTION_LABEL_MAX_WORDS = 6
_WORD_RE = re.compile(r"\S+")
_SENTENCE_END_RE = re.compile(r"[.!?]")


def _words_between(text, start, end):
    return len(text[start:end].split())


def _absorb_trailing_label(text, end, max_words):
    """A short panel label right after the run's last marker ("(b) Metaphase") is itself part of
    the noise, but the real sentence resuming right after it ("...before being pulled apart...")
    must NOT be absorbed too -- the two are told apart by case, not by the nearest full stop,
    since the resuming text is a lower-case continuation of the clause the caption interrupted
    while an NCERT figure/panel label is always capitalised. Absorbs consecutive whitespace-
    separated words starting at `end` for as long as each one starts with an uppercase letter
    (stopping early if a word itself contains a sentence-ending mark), up to max_words words.
    Returns the new end offset (unchanged if the very next word isn't capitalised, e.g. there is
    no label at all)."""
    pos = end
    n = len(text)
    for _ in range(max_words):
        ws_end = pos
        while ws_end < n and text[ws_end].isspace():
            ws_end += 1
        if ws_end >= n:
            break
        word = _WORD_RE.match(text, ws_end).group()
        if not word[0].isupper():
            break
        pos = ws_end + len(word)
        if _SENTENCE_END_RE.search(word):
            break
    return pos


def find_caption_panel_runs(text):
    """Returns non-overlapping (start, end) character spans in `text` covering each run of 2+
    ascending single-letter panel markers (see module comment above), extended backward to absorb
    an immediately-preceding 'Figure N.N' / 'Fig. N.N' / 'Table N.N' caption token if one is close
    enough, and forward to absorb a short trailing label after the last marker. Returns [] when
    fewer than 2 qualifying markers exist anywhere in `text` -- the common case for an ordinary
    section, so this is a cheap no-op for almost every call. Deterministic, no model involved."""
    panel_tokens = list(PANEL_LETTER_RE.finditer(text))
    if len(panel_tokens) < 2:
        return []
    figure_tokens = list(FIGURE_CAPTION_PREFIX_RE.finditer(text))

    runs = []
    i = 0
    while i < len(panel_tokens):
        run = [panel_tokens[i]]
        last_letter = panel_tokens[i].group(1).lower()
        j = i + 1
        while j < len(panel_tokens):
            gap_words = _words_between(text, run[-1].end(), panel_tokens[j].start())
            letter = panel_tokens[j].group(1).lower()
            if gap_words > CAPTION_RUN_MAX_GAP_WORDS or letter <= last_letter:
                break
            run.append(panel_tokens[j])
            last_letter = letter
            j += 1
        if len(run) >= 2:
            runs.append(run)
            i = j
        else:
            i += 1

    spans = []
    for run in runs:
        start, end = run[0].start(), run[-1].end()
        for ft in reversed(figure_tokens):
            if ft.end() <= start:
                if _words_between(text, ft.end(), start) <= CAPTION_RUN_MAX_GAP_WORDS:
                    start = ft.start()
                break
        end = _absorb_trailing_label(text, end, CAPTION_LABEL_MAX_WORDS)
        spans.append((start, end))

    spans.sort()
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def strip_figure_caption_noise(text):
    """Drops every detected caption/panel-label run (see find_caption_panel_runs) from `text`,
    collapsing the surrounding whitespace back to a single space so the two sentence halves that
    used to sandwich the noise read as one clean sentence again. Returns (clean_text,
    removed_word_count); removed_word_count is 0 (and clean_text is `text` unchanged) for the
    common case where no noise was found."""
    spans = find_caption_panel_runs(text)
    if not spans:
        return text, 0
    removed_words = 0
    out = []
    pos = 0
    for start, end in spans:
        out.append(text[pos:start])
        removed_words += len(text[start:end].split())
        pos = end
    out.append(text[pos:])
    cleaned = re.sub(r"\s+", " ", "".join(out)).strip()
    return cleaned, removed_words


def strip_figure_caption_noise_from_blocks(blocks):
    """Applies strip_figure_caption_noise to every paragraph/list block's text (headings/tables are
    never run-on sentences with wedged captions, so they're left untouched). Returns (new_blocks,
    total_words_removed)."""
    new_blocks = []
    removed_total = 0
    for btype, text in blocks:
        if btype in ("para", "list"):
            cleaned, removed = strip_figure_caption_noise(text)
            removed_total += removed
            new_blocks.append((btype, cleaned))
        else:
            new_blocks.append((btype, text))
    return new_blocks, removed_total


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
        blocks, _caption_words_removed = strip_figure_caption_noise_from_blocks(blocks)
        if _caption_words_removed:
            cleaned_text = clean_mod.render(chapter_num, chapter_title, blocks)
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
