#!/usr/bin/env python
"""Deterministic text cross-check between an NCERT chapter PDF and a set of cleaned sections --
no model calls, no Supabase/network calls of its own. Shared by server.py's upload-time preview
(before Apply), its "Verify against PDF" action for already-loaded chapters, and ncert_ingest.py's
CLI (which treats a FAIL the same as any other validate_chapter flag).

Reuses ncert_clean_manual.py's own junk-line rules (is_junk_line, build_blocks,
pdf_text_without_junk) and ncert_ingest_core.py's merge_wrapped_headings, so a line is only ever
reported "dropped" here if the real cleaning pipeline would actually drop it too.

Two independent checks, both against the PDF's text layer after whitespace/quote/dash
normalization (ncert_clean_manual.normalize_ws_quotes):
  (a) every sentence in the given sections' text must appear in the PDF text (minus junk lines) --
      catches invented/altered text that was never in the source.
  (b) every PDF paragraph of more than MISSING_PARAGRAPH_MIN_WORDS words, once junk lines are
      ignored, must appear in the sections' combined text -- catches real content that got dropped.
FAILs if either check finds anything; otherwise PASSes. Coverage/dropped-line reporting runs
either way, so even a PASS result has useful numbers for the page to show.
"""
import re
from collections import Counter

import fitz

import ncert_clean_manual as clean_mod
import ncert_ingest_core as core_mod

MAX_EXCERPT_CHARS = 200
MISSING_PARAGRAPH_MIN_WORDS = 8
OTHER_DROP_PCT_MAX = 1.0
OTHER_DROP_LONG_WORDS = 25

# Heuristics beyond is_junk_line's own categories (running header / page number / stray letter /
# lone caps phrase) -- best-effort line classification for the admin page's dropped-lines report,
# not used to decide what actually gets dropped (that's still entirely is_junk_line's call).
UNIT_INTRO_RE = re.compile(r"^UNIT\b", re.IGNORECASE)
FIGURE_LABEL_RE = re.compile(r"^(Fig\.?|Figure)\s*\d", re.IGNORECASE)
BOX_MARKER_RE = re.compile(r"^(do you know|activity|point to ponder|think it over|box\b)", re.IGNORECASE)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.?!])\s+")

DROP_REASONS = (
    "running header", "page number", "figure caption or label", "box or sidebar",
    "unit intro", "repeated contents", "other",
)


def classify_dropped_line(line: str) -> str:
    """Best-effort reason for why a line was dropped. Order matters -- more specific patterns are
    checked before the generic ones is_junk_line itself uses."""
    s = line.strip()
    if UNIT_INTRO_RE.match(s):
        return "unit intro"
    if clean_mod.REPRINT_RE.match(s) or any(p.match(s) for p in clean_mod.RUNNING_HEADER_RES):
        return "running header"
    if clean_mod.LONE_PAGE_NUM_RE.match(s):
        return "page number"
    if BOX_MARKER_RE.match(s):
        return "box or sidebar"
    if FIGURE_LABEL_RE.match(s) or clean_mod.STRAY_LETTER_RE.match(s):
        return "figure caption or label"
    if clean_mod.HEADING_RE.match(s):
        return "repeated contents"
    return "other"


def _collect_dropped(front_lines, body_lines):
    """front_lines: raw lines before the 'CHAPTER n' marker (title, contents listing, unit intro --
    never stored in any section). body_lines: raw lines after it, as strip_front_matter returns
    them. Returns [(text, reason, word_count), ...]."""
    dropped = []
    for raw in front_lines:
        s = raw.strip()
        if not s:
            continue
        dropped.append((s, classify_dropped_line(s), len(s.split())))

    heading_positions = {}
    for raw in body_lines:
        s = raw.strip()
        if not s:
            continue
        if clean_mod.is_junk_line(s):
            dropped.append((s, classify_dropped_line(s), len(s.split())))
            continue
        m = clean_mod.HEADING_RE.match(s)
        if m:
            heading_positions.setdefault(m.group(1), []).append(s)

    # dedupe_headings (in the real pipeline) keeps only the LAST occurrence of a repeated heading
    # number -- everything earlier is a contents-listing duplicate, dropped from the stored text.
    for num, occurrences in heading_positions.items():
        for s in occurrences[:-1]:
            dropped.append((s, "repeated contents", len(s.split())))

    return dropped


def _find_missing_paragraphs(body_lines, sections):
    """PDF paragraphs (build_blocks' own grouping, so headings/junk lines are already excluded)
    of more than MISSING_PARAGRAPH_MIN_WORDS words that don't appear anywhere in the sections'
    combined text."""
    blocks = clean_mod.build_blocks(body_lines)
    combined = clean_mod.normalize_ws_quotes(" ".join(s.get("text") or "" for s in sections))
    missing = []
    for btype, text in blocks:
        if btype == "heading":
            continue
        if len(text.split()) <= MISSING_PARAGRAPH_MIN_WORDS:
            continue
        if text not in combined:
            missing.append(text)
    return missing


def _find_sentence_mismatches(sections, full_pdf_text):
    """Sentences in the sections' text that don't appear in the PDF's (junk-stripped) text layer."""
    mismatches = []
    for s in sections:
        text = s.get("text") or ""
        for sent in SENTENCE_SPLIT_RE.split(text):
            sent = clean_mod.normalize_ws_quotes(sent)
            if sent and sent not in full_pdf_text:
                mismatches.append(sent)
    return mismatches


def _build_summary(status, coverage_pct, dropped_count, mismatch_count, missing_count):
    parts = [f"{status} - {coverage_pct:.1f}% coverage", f"{dropped_count} dropped line(s)"]
    if status == "FAIL":
        parts.append(f"{mismatch_count} sentence mismatch(es)")
        parts.append(f"{missing_count} missing paragraph(s)")
    return ", ".join(parts)[:500]


def verify_text_coverage(pdf_bytes: bytes, sections: list) -> dict:
    """sections: [{"heading": ..., "text": ...}, ...] -- either a fresh preview's sections (not
    yet written) or a chapter's existing ncert_sections rows (the "Verify against PDF" action).
    Never raises for a readable PDF; returns a dict, never writes anything."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pages = [page.get_text() for page in doc]
        raw_text = "\n\n".join(pages)
        raw_lines = core_mod.merge_wrapped_headings(raw_text.replace("#", "").split("\n"))
        _, _, body_lines = clean_mod.strip_front_matter(raw_lines)
        front_lines = raw_lines[: len(raw_lines) - len(body_lines)]

        dropped = _collect_dropped(front_lines, body_lines)
        missing_paragraphs = _find_missing_paragraphs(body_lines, sections)
        full_pdf_text = clean_mod.pdf_text_without_junk(doc)
        sentence_mismatches = _find_sentence_mismatches(sections, full_pdf_text)
        total_words = sum(len(p.split()) for p in pages)
    finally:
        doc.close()

    kept_words = sum(len((s.get("text") or "").split()) for s in sections)
    coverage_pct = round(kept_words / total_words * 100, 2) if total_words else 0.0
    word_diff = total_words - kept_words
    word_diff_pct = round(abs(word_diff) / total_words * 100, 2) if total_words else 0.0

    other_words = sum(wc for _, reason, wc in dropped if reason == "other")
    other_pct = (other_words / total_words * 100) if total_words else 0.0
    other_bucket_flagged = other_pct > OTHER_DROP_PCT_MAX

    dropped_by_reason = Counter()
    dropped_out = []
    for text, reason, wc in dropped:
        dropped_by_reason[reason] += 1
        check_this = (reason == "other" and other_bucket_flagged) or (
            wc > OTHER_DROP_LONG_WORDS and reason not in ("figure caption or label", "running header")
        )
        dropped_out.append({
            "text": text[:MAX_EXCERPT_CHARS], "reason": reason, "words": wc, "check_this": check_this,
        })

    status = "PASS" if not sentence_mismatches and not missing_paragraphs else "FAIL"
    summary = _build_summary(status, coverage_pct, len(dropped_out), len(sentence_mismatches), len(missing_paragraphs))

    return {
        "status": status,
        "summary": summary,
        "total_words": total_words,
        "kept_words": kept_words,
        "coverage_pct": coverage_pct,
        "word_diff": word_diff,
        "word_diff_pct": word_diff_pct,
        "dropped_lines": dropped_out,
        "dropped_lines_count": len(dropped_out),
        "dropped_by_reason": dict(dropped_by_reason),
        "sentence_mismatches": [s[:MAX_EXCERPT_CHARS] for s in sentence_mismatches],
        "missing_paragraphs": [p[:MAX_EXCERPT_CHARS] for p in missing_paragraphs],
    }
