#!/usr/bin/env python
"""Deterministic text cross-check between an NCERT chapter PDF and a set of cleaned sections --
no model calls, no Supabase/network calls of its own. Shared by server.py's upload-time preview
(before Apply), its "Verify against PDF" action for already-loaded chapters, and ncert_ingest.py's
CLI (which treats a FAIL the same as any other validate_chapter flag).

Reuses ncert_clean_manual.py's own junk-line rules (is_junk_line, build_blocks,
pdf_text_without_junk) and ncert_ingest_core.py's merge_wrapped_headings, so the PDF side of every
comparison is cleaned through the exact same pipeline that would have produced the stored sections
if everything had gone perfectly.

Every difference between the PDF and the stored sections is classified into one of two buckets so
an admin reading the result doesn't have to guess which lines matter:
  - harmless: a hyphenation/line-break join, a dash or quote style difference, extra or missing
    whitespace, a leftover running header or page number, or a ligature/soft-hyphen difference --
    cosmetic/extraction noise, never a sign of lost or altered content.
  - real: a paragraph in the PDF missing from storage ("missing"), text in storage not in the PDF
    ("extra"), a word-level change inside an otherwise-matching paragraph ("changed"), a heading
    missing from or extra in storage, headings in a different order than the PDF, or a section
    whose word count differs substantially from its PDF region ("word_count_mismatch").
Each difference carries a `diff_key` -- a hash of its type plus its (already-truncated) excerpts,
never the excerpt text itself -- so the admin page can let an admin mark one "OK" and have that
decision persist (see ncert_verify_decisions.sql) without ever storing the chapter's actual text.

Figure captions, box/sidebar text, unit intros and repeated contents-listing lines are dropped by
is_junk_line/classify_dropped_line the same way the real ingest pipeline drops them -- they are
intentionally excluded from the PDF side before any comparison runs, so they never show up as a
"difference" at all (they were correctly supposed to go). Only leftover running-header/page-number
fragments that still land inside a compared paragraph are surfaced, as a harmless difference.

verify_text_coverage(pdf_bytes, sections) -> dict never raises for a readable PDF, never writes
anything, and never logs or stores the excerpts it returns -- callers (server.py) decide whether to
persist a diff_key and never persist the text alongside it.
"""
import difflib
import hashlib
import re
from collections import Counter

import fitz

import ncert_clean_manual as clean_mod
import ncert_ingest_core as core_mod

MAX_EXCERPT_CHARS = 200
NEAR_MATCH_MIN_RATIO = 0.4  # below this, two paragraphs are unrelated (separate missing + extra)
WORD_COUNT_MISMATCH_PCT = 15.0
WORD_COUNT_MISMATCH_MIN_WORDS = 5

# Heuristics beyond is_junk_line's own categories (running header / page number / stray letter /
# lone caps phrase) -- best-effort line classification, not used to decide what actually gets
# dropped (that's still entirely is_junk_line's call).
UNIT_INTRO_RE = re.compile(r"^UNIT\b", re.IGNORECASE)
FIGURE_LABEL_RE = re.compile(r"^(Fig\.?|Figure)\s*\d", re.IGNORECASE)
BOX_MARKER_RE = re.compile(r"^(do you know|activity|point to ponder|think it over|box\b)", re.IGNORECASE)

DROP_REASONS = (
    "running header", "page number", "figure caption or label", "box or sidebar",
    "unit intro", "repeated contents", "other",
)

HARMLESS_TYPES = (
    "hyphenation or line-break join", "dash or quote style", "extra or missing whitespace",
    "running header or page number", "ligature or soft-hyphen difference",
)
REAL_TYPES = (
    "missing", "extra", "changed", "heading_missing", "heading_extra", "heading_reordered",
    "word_count_mismatch",
)

_WS_RE = re.compile(r"\s+")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\s+(\w)")
_LIGATURE_MAP = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "\xad": "", "­": "",
}
_DASH_RE = re.compile("[‐‑‒–—−]")
_QUOTE_SINGLE_RE = re.compile("[‘’‚‛]")
_QUOTE_DOUBLE_RE = re.compile("[“”„‟]")


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
    them. Returns [(text, reason, word_count, heading_or_None), ...] -- heading is whichever
    numbered heading most recently preceded this line in body_lines (None for front matter), used
    only to attribute a section to the harmless running-header/page-number differences built from
    this list."""
    dropped = []
    for raw in front_lines:
        s = raw.strip()
        if not s:
            continue
        if clean_mod.CHAPTER_TITLE_RE.match(s):
            # The "CHAPTER n" marker itself incidentally matches RUNNING_HEADER_RES' all-caps-
            # plus-trailing-number shape -- it's the structural signal strip_front_matter uses to
            # find the chapter, not extraction noise, so it is never reported as a dropped line.
            continue
        dropped.append((s, classify_dropped_line(s), len(s.split()), None))

    current_heading = None
    heading_positions = {}
    for raw in body_lines:
        s = raw.strip()
        if not s:
            continue
        m = clean_mod.HEADING_RE.match(s)
        if m:
            current_heading = s
            heading_positions.setdefault(m.group(1), []).append(s)
            continue
        if clean_mod.is_junk_line(s):
            dropped.append((s, classify_dropped_line(s), len(s.split()), current_heading))

    # dedupe_headings (in the real pipeline) keeps only the LAST occurrence of a repeated heading
    # number -- everything earlier is a contents-listing duplicate, dropped from the stored text.
    for num, occurrences in heading_positions.items():
        for s in occurrences[:-1]:
            dropped.append((s, "repeated contents", len(s.split()), None))

    return dropped


def _strip_ws(s):
    return _WS_RE.sub("", s)


def _dehyphenate(s):
    return _HYPHEN_BREAK_RE.sub(r"\1\2", s)


def _delig(s):
    for ch, rep in _LIGATURE_MAP.items():
        s = s.replace(ch, rep)
    return s


def _normalize_dash_quote(s):
    s = _DASH_RE.sub("-", s)
    s = _QUOTE_SINGLE_RE.sub("'", s)
    s = _QUOTE_DOUBLE_RE.sub('"', s)
    return s


def _harmless_subtype(a, b):
    """a, b: already run through clean_mod.normalize_ws_quotes and confirmed NOT identical. Tries
    progressively specific explanations for the gap, most specific first, and returns the matching
    harmless type label, or None if the gap looks like real content rather than formatting noise."""
    if _strip_ws(a) == _strip_ws(b):
        return "extra or missing whitespace"
    a2, b2 = _dehyphenate(a), _dehyphenate(b)
    if _strip_ws(a2) == _strip_ws(b2):
        return "hyphenation or line-break join"
    a3, b3 = _delig(a2), _delig(b2)
    if _strip_ws(a3) == _strip_ws(b3):
        return "ligature or soft-hyphen difference"
    a4, b4 = _normalize_dash_quote(a3), _normalize_dash_quote(b3)
    if _strip_ws(a4) == _strip_ws(b4):
        return "dash or quote style"
    return None


def _word_diff_spans(a, b):
    """Word-level opcodes between two (already-normalized) strings, for the "changed" real type --
    [{"op": "equal"|"removed"|"added", "text": "..."}, ...]. "removed" = words only on the PDF
    side, "added" = words only on the stored side."""
    aw, bw = a.split(), b.split()
    sm = difflib.SequenceMatcher(None, aw, bw, autojunk=False)
    spans = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            spans.append({"op": "equal", "text": " ".join(aw[i1:i2])})
        elif tag == "delete":
            spans.append({"op": "removed", "text": " ".join(aw[i1:i2])})
        elif tag == "insert":
            spans.append({"op": "added", "text": " ".join(bw[j1:j2])})
        elif tag == "replace":
            spans.append({"op": "removed", "text": " ".join(aw[i1:i2])})
            spans.append({"op": "added", "text": " ".join(bw[j1:j2])})
    return spans


def _diff_key(diff_type, pdf_text, stored_text):
    """A hash of the type plus the (truncated, normalized) excerpts actually shown to the admin --
    never the full text. Two different long paragraphs that share a type and a 200-char prefix can
    collide; that's fine, since the decision is about what the admin can SEE, not the underlying
    text."""
    pdf_part = clean_mod.normalize_ws_quotes(pdf_text or "")[:MAX_EXCERPT_CHARS]
    stored_part = clean_mod.normalize_ws_quotes(stored_text or "")[:MAX_EXCERPT_CHARS]
    basis = f"{diff_type}|{pdf_part}|{stored_part}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:24]


def _page_for_text(pages_norm, text):
    """Best-effort PDF page number for a piece of already-cleaned text: finds the first page whose
    normalized text contains the text's first ~60 characters. Approximate by design (a repeated
    running header matches whichever page it first appears on; a paragraph split across a page
    break matches its first page) -- good enough for "which page should I look at", not a precise
    citation."""
    if not text:
        return None
    snippet = clean_mod.normalize_ws_quotes(text)[:60]
    if not snippet:
        return None
    for pno, page_text in enumerate(pages_norm, start=1):
        if snippet in page_text:
            return pno
    return None


def _classify_pair(pdf_text, stored_text, pdf_norm, stored_norm):
    label = _harmless_subtype(pdf_norm, stored_norm)
    if label:
        return {"type": label, "category": "harmless", "pdf_text": pdf_text, "stored_text": stored_text}
    return {
        "type": "changed", "category": "real", "pdf_text": pdf_text, "stored_text": stored_text,
        "diff_spans": _word_diff_spans(pdf_norm, stored_norm),
    }


def _compare_paragraphs(pdf_paras, stored_paras):
    """Greedy best-match alignment between one section's PDF-cleaned paragraphs and its stored
    paragraphs. Paragraphs equal after normalize_ws_quotes are matched for free (via
    SequenceMatcher, so repeated identical paragraphs line up in order); everything left over is
    paired by similarity and classified harmless/changed, or left unmatched (missing/extra) when no
    similar counterpart exists on the other side."""
    pdf_norm = [clean_mod.normalize_ws_quotes(p) for p in pdf_paras]
    stored_norm = [clean_mod.normalize_ws_quotes(p) for p in stored_paras]

    sm = difflib.SequenceMatcher(None, pdf_norm, stored_norm, autojunk=False)
    matched_pdf, matched_stored = set(), set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            matched_pdf.update(range(i1, i2))
            matched_stored.update(range(j1, j2))

    pdf_left = [i for i in range(len(pdf_paras)) if i not in matched_pdf]
    stored_left = {j for j in range(len(stored_paras)) if j not in matched_stored}

    diffs = []
    for i in pdf_left:
        best_j, best_ratio = None, 0.0
        for j in stored_left:
            ratio = difflib.SequenceMatcher(None, pdf_norm[i], stored_norm[j], autojunk=False).ratio()
            if ratio > best_ratio:
                best_ratio, best_j = ratio, j
        if best_j is not None and best_ratio >= NEAR_MATCH_MIN_RATIO:
            stored_left.discard(best_j)
            diffs.append(_classify_pair(pdf_paras[i], stored_paras[best_j], pdf_norm[i], stored_norm[best_j]))
        else:
            diffs.append({"type": "missing", "category": "real", "pdf_text": pdf_paras[i], "stored_text": None})
    for j in sorted(stored_left):
        diffs.append({"type": "extra", "category": "real", "pdf_text": None, "stored_text": stored_paras[j]})
    return diffs


def _check_word_count(pdf_paras, stored_paras, heading_display, pdf_heading_text, pages_norm):
    pdf_wc = sum(len(p.split()) for p in pdf_paras)
    stored_wc = sum(len(p.split()) for p in stored_paras)
    if pdf_wc == 0:
        return None
    diff = abs(pdf_wc - stored_wc)
    pct = diff / pdf_wc * 100
    if diff < WORD_COUNT_MISMATCH_MIN_WORDS or pct <= WORD_COUNT_MISMATCH_PCT:
        return None
    pdf_excerpt = f"{pdf_wc} word(s) in this section in the PDF"
    stored_excerpt = f"{stored_wc} word(s) stored"
    return {
        "type": "word_count_mismatch", "category": "real", "section": heading_display,
        "page": _page_for_text(pages_norm, pdf_heading_text), "pdf_excerpt": pdf_excerpt,
        "stored_excerpt": stored_excerpt, "diff_spans": None,
        "diff_key": _diff_key("word_count_mismatch", pdf_excerpt, stored_excerpt),
    }


def _match_key(heading):
    return clean_mod.normalize_ws_quotes(heading or "").lower()


def _group_pdf_blocks_by_heading(pdf_blocks):
    """[(heading_text, [paragraph_text, ...]), ...], in PDF order. Anything before the first
    heading (chapter title, intro) is dropped, matching parse_sections' own behavior -- it is
    never stored as a section either, so it's not a real "difference"."""
    groups = []
    current = None
    paras = []
    for btype, text in pdf_blocks:
        if btype == "heading":
            if current is not None:
                groups.append((current, paras))
            current = text
            paras = []
        elif current is not None:
            paras.append(text)
    if current is not None:
        groups.append((current, paras))
    return groups


def _compare_headings_and_sections(pdf_blocks, sections, pages_norm):
    differences = []
    groups = _group_pdf_blocks_by_heading(pdf_blocks)

    pdf_key_to_heading, pdf_paras_by_key, pdf_order_keys = {}, {}, []
    for heading, paras in groups:
        key = _match_key(heading)
        if key not in pdf_key_to_heading:
            pdf_key_to_heading[key] = heading
            pdf_paras_by_key[key] = paras
            pdf_order_keys.append(key)

    stored_key_to_heading, stored_paras_by_key, stored_order_keys = {}, {}, []
    for s in sections:
        key = _match_key(s.get("heading"))
        if key not in stored_key_to_heading:
            stored_order_keys.append(key)
        stored_key_to_heading[key] = s.get("heading")
        stored_paras_by_key[key] = [ln for ln in (s.get("text") or "").split("\n") if ln.strip()]

    for key in pdf_order_keys:
        if key not in stored_key_to_heading:
            heading = pdf_key_to_heading[key]
            differences.append({
                "type": "heading_missing", "category": "real", "section": heading,
                "page": _page_for_text(pages_norm, heading),
                "pdf_excerpt": heading[:MAX_EXCERPT_CHARS], "stored_excerpt": None,
                "diff_spans": None, "diff_key": _diff_key("heading_missing", heading, None),
            })
    for key in stored_order_keys:
        if key not in pdf_key_to_heading:
            heading = stored_key_to_heading[key]
            differences.append({
                "type": "heading_extra", "category": "real", "section": heading, "page": None,
                "pdf_excerpt": None, "stored_excerpt": heading[:MAX_EXCERPT_CHARS],
                "diff_spans": None, "diff_key": _diff_key("heading_extra", None, heading),
            })

    pdf_common = [k for k in pdf_order_keys if k in stored_key_to_heading]
    stored_common = [k for k in stored_order_keys if k in pdf_key_to_heading]
    if pdf_common != stored_common:
        sm = difflib.SequenceMatcher(None, pdf_common, stored_common, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            pdf_slice = [pdf_key_to_heading[k] for k in pdf_common[i1:i2]]
            stored_slice = [stored_key_to_heading[k] for k in stored_common[j1:j2]]
            if not pdf_slice and not stored_slice:
                continue
            pdf_excerpt = " -> ".join(pdf_slice)[:MAX_EXCERPT_CHARS] or None
            stored_excerpt = " -> ".join(stored_slice)[:MAX_EXCERPT_CHARS] or None
            differences.append({
                "type": "heading_reordered", "category": "real",
                "section": stored_slice[0] if stored_slice else (pdf_slice[0] if pdf_slice else None),
                "page": _page_for_text(pages_norm, pdf_slice[0]) if pdf_slice else None,
                "pdf_excerpt": pdf_excerpt, "stored_excerpt": stored_excerpt,
                "diff_spans": None, "diff_key": _diff_key("heading_reordered", pdf_excerpt, stored_excerpt),
            })

    for key in pdf_common:
        heading_display = stored_key_to_heading[key]
        pdf_paras = pdf_paras_by_key[key]
        stored_paras = stored_paras_by_key[key]

        wc_diff = _check_word_count(pdf_paras, stored_paras, heading_display, pdf_key_to_heading[key], pages_norm)
        if wc_diff:
            differences.append(wc_diff)

        for raw in _compare_paragraphs(pdf_paras, stored_paras):
            pdf_text, stored_text = raw.get("pdf_text"), raw.get("stored_text")
            differences.append({
                "type": raw["type"], "category": raw["category"], "section": heading_display,
                "page": _page_for_text(pages_norm, pdf_text) if pdf_text else None,
                "pdf_excerpt": pdf_text[:MAX_EXCERPT_CHARS] if pdf_text else None,
                "stored_excerpt": stored_text[:MAX_EXCERPT_CHARS] if stored_text else None,
                "diff_spans": raw.get("diff_spans"),
                "diff_key": _diff_key(raw["type"], pdf_text, stored_text),
            })
    return differences


def _make_harmless_dropped_diff(line_text, heading, pages_norm):
    return {
        "type": "running header or page number", "category": "harmless",
        "section": heading, "page": _page_for_text(pages_norm, line_text),
        "pdf_excerpt": line_text[:MAX_EXCERPT_CHARS], "stored_excerpt": None,
        "diff_spans": None, "diff_key": _diff_key("running header or page number", line_text, None),
    }


def _build_summary(status, coverage_pct, real_count, harmless_count):
    parts = [
        f"{status} - {coverage_pct:.1f}% coverage",
        f"{real_count} real difference(s)",
        f"{harmless_count} harmless difference(s)",
    ]
    return ", ".join(parts)[:500]


def verify_text_coverage(pdf_bytes: bytes, sections: list) -> dict:
    """sections: [{"heading": ..., "text": ...}, ...] -- either a fresh preview's sections (not
    yet written) or a chapter's existing ncert_sections rows (the "Verify against PDF" action).
    Never raises for a readable PDF; returns a dict, never writes anything, never stores or logs
    any excerpt. `status`/`counts` here are the RAW pre-decisions picture (every real difference
    counts against PASS); the caller applies any persisted "Mark as OK" decisions on top of
    `differences` before deciding the final verdict shown to the admin."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pages_raw = [page.get_text() for page in doc]
        pages_norm = [clean_mod.normalize_ws_quotes(p) for p in pages_raw]
        raw_text = "\n\n".join(pages_raw)
        raw_lines = core_mod.merge_wrapped_headings(raw_text.replace("#", "").split("\n"))
        _, _, body_lines = clean_mod.strip_front_matter(raw_lines)
        front_lines = raw_lines[: len(raw_lines) - len(body_lines)]

        dropped = _collect_dropped(front_lines, body_lines)
        pdf_blocks = clean_mod.dedupe_headings(clean_mod.build_blocks(body_lines))
        total_words = sum(len(p.split()) for p in pages_raw)

        differences = [
            _make_harmless_dropped_diff(text, heading, pages_norm)
            for text, reason, _wc, heading in dropped
            if reason in ("running header", "page number")
        ]
        differences += _compare_headings_and_sections(pdf_blocks, sections, pages_norm)
    finally:
        doc.close()

    kept_words = sum(len((s.get("text") or "").split()) for s in sections)
    coverage_pct = round(kept_words / total_words * 100, 2) if total_words else 0.0
    word_diff = total_words - kept_words
    word_diff_pct = round(abs(word_diff) / total_words * 100, 2) if total_words else 0.0

    real_count = sum(1 for d in differences if d["category"] == "real")
    harmless_count = len(differences) - real_count
    status = "PASS" if real_count == 0 else "FAIL"
    summary = _build_summary(status, coverage_pct, real_count, harmless_count)
    by_type = dict(Counter(d["type"] for d in differences))

    return {
        "status": status,
        "summary": summary,
        "total_words": total_words,
        "kept_words": kept_words,
        "coverage_pct": coverage_pct,
        "word_diff": word_diff,
        "word_diff_pct": word_diff_pct,
        "dropped_lines_count": len(dropped),
        "differences": differences,
        "counts": {"real": real_count, "harmless": harmless_count, "by_type": by_type},
    }
