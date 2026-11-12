#!/usr/bin/env python
"""Clean a manually-pasted NCERT chapter text file into normalized plain text.

Usage:
    py ncert_clean_manual.py <input.txt> [--pdf <source.pdf>] [--out <output.txt>]

Writes the cleaned file to ncert_manual/clean/<input basename> by default,
and prints a validation report against the source PDF's text layer (if
--pdf is given).
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

SMALL_WORDS = {
    "a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "into",
    "nor", "of", "on", "onto", "or", "over", "the", "to", "with", "within",
    "without",
}

SENTENCE_ENDERS = (".", "?", "!")

REPRINT_RE = re.compile(r"^Reprint\s+\d{4}\s*-\s*\d{2,4}\b", re.IGNORECASE)
RUNNING_HEADER_RES = [
    re.compile(r"^[A-Z][A-Z\s'\-]{3,}\s+\d{1,4}$"),   # "THE LIVING WORLD 5"
    re.compile(r"^\d{1,4}\s+[A-Z][A-Z\s'\-]{2,}$"),   # "4 BIOLOGY"
]
LONE_PAGE_NUM_RE = re.compile(r"^\d{1,4}$")
STRAY_LETTER_RE = re.compile(r"^[A-Za-z]$")
# A short all-caps line with no digits, on its own (e.g. a running header's
# chapter-name/subject fragment left over after the PDF text layer splits
# "THE LIVING WORLD 5" or "4 BIOLOGY" across two separate lines).
LONE_CAPS_PHRASE_RE = re.compile(r"^[A-Z][A-Z\s']{1,40}$")
CHAPTER_TITLE_RE = re.compile(r"^CHAPTER\s+(\d+)\s*$")
HEADING_RE = re.compile(r"^(\d+(?:\.\d+)+)\s+(.+)$")
SECTION_WORD_RE = re.compile(r"^(SUMMARY|EXERCISES)\s*$", re.IGNORECASE)
LIST_ITEM_RE = re.compile(r"^(?:\d+\.\s+\D|\([a-zA-Z0-9]+\)\s+)")
TABLE_REF_RE = re.compile(r"\bTable\s+(\d+\.\d+)\b")


def title_case(s: str) -> str:
    words = s.split()
    out = []
    for i, w in enumerate(words):
        lw = w.lower()
        if i > 0 and lw in SMALL_WORDS:
            out.append(lw)
        else:
            out.append(lw[:1].upper() + lw[1:] if lw else lw)
    return " ".join(out)


def normalize_ws_quotes(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def is_junk_line(line: str) -> bool:
    if REPRINT_RE.match(line):
        return True
    for pat in RUNNING_HEADER_RES:
        if pat.match(line):
            return True
    if LONE_PAGE_NUM_RE.match(line):
        return True
    if STRAY_LETTER_RE.match(line):
        return True
    if SECTION_WORD_RE.match(line):
        return False
    words = line.split()
    if 1 <= len(words) <= 4 and LONE_CAPS_PHRASE_RE.match(line):
        return True
    return False


def strip_front_matter(raw_lines):
    """Return (chapter_num, title_text, body_lines) dropping everything
    before the real ALL-CAPS 'CHAPTER n' title and the contents lines
    directly under it."""
    start_idx = None
    chapter_num = None
    for i, ln in enumerate(raw_lines):
        m = CHAPTER_TITLE_RE.match(ln.strip())
        if m:
            start_idx = i
            chapter_num = m.group(1)
            break
    if start_idx is None:
        return None, None, raw_lines

    j = start_idx + 1
    title_parts = []
    while j < len(raw_lines):
        stripped = raw_lines[j].strip()
        if not stripped:
            j += 1
            continue
        if HEADING_RE.match(stripped) or SECTION_WORD_RE.match(stripped):
            break
        title_parts.append(stripped)
        j += 1
    chapter_title = normalize_ws_quotes(" ".join(title_parts))
    return chapter_num, chapter_title, raw_lines[j:]


def build_blocks(body_lines):
    """Turn junk-filtered body lines into (type, text) blocks: 'heading',
    'list', or 'para'. Rejoins lines broken mid-sentence (including across
    blank page-break lines) without merging separate paragraphs, list
    items or headings."""
    blocks = []
    block = []
    block_type = None

    def ends_terminal(text):
        return text.rstrip().endswith(SENTENCE_ENDERS)

    def flush():
        nonlocal block, block_type
        if block:
            text = normalize_ws_quotes(" ".join(block))
            blocks.append((block_type or "para", text))
        block = []
        block_type = None

    for raw in body_lines:
        line = raw.strip()
        if not line:
            if block and ends_terminal(block[-1]):
                flush()
            continue
        if is_junk_line(line):
            continue

        if SECTION_WORD_RE.match(line):
            flush()
            blocks.append(("heading", title_case(line)))
            continue

        if HEADING_RE.match(line):
            flush()
            blocks.append(("heading", line))
            continue

        if LIST_ITEM_RE.match(line):
            flush()
            block = [line]
            block_type = "list"
            continue

        if block and ends_terminal(block[-1]):
            flush()
            block = [line]
            block_type = "para"
        else:
            block.append(line)
            if block_type is None:
                block_type = "para"
    flush()
    return blocks


def dedupe_headings(blocks):
    """Numbered headings that appear twice (once in a contents-style
    listing right under the chapter title, once as the real body heading)
    collapse to a single heading at the later position, using whichever
    casing is not all-uppercase."""
    occurrences = defaultdict(list)
    for idx, (btype, text) in enumerate(blocks):
        if btype != "heading":
            continue
        m = HEADING_RE.match(text)
        if m:
            occurrences[m.group(1)].append((idx, m.group(2)))

    best_text = {}
    keep_idx = {}
    for num, occ in occurrences.items():
        candidates = [t for _, t in occ]
        non_caps = [t for t in candidates if t != t.upper()]
        best_text[num] = non_caps[0] if non_caps else title_case(candidates[-1])
        keep_idx[num] = occ[-1][0]

    new_blocks = []
    for idx, (btype, text) in enumerate(blocks):
        if btype == "heading":
            m = HEADING_RE.match(text)
            if m:
                num = m.group(1)
                if idx != keep_idx[num]:
                    continue
                text = f"{num} {best_text[num]}"
        new_blocks.append((btype, text))
    return new_blocks


def split_threshold(gaps):
    if not gaps:
        return 1e9
    sorted_gaps = sorted(gaps)
    if len(sorted_gaps) == 1:
        return sorted_gaps[0] + 1
    best_split = sorted_gaps[0] / 2
    best_jump = -1
    for i in range(1, len(sorted_gaps)):
        jump = sorted_gaps[i] - sorted_gaps[i - 1]
        if jump > best_jump:
            best_jump = jump
            best_split = (sorted_gaps[i] + sorted_gaps[i - 1]) / 2
    return best_split


def cluster_1d(values):
    vals = sorted(set(round(v, 1) for v in values))
    if len(vals) <= 1:
        return vals
    gaps = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    threshold = split_threshold(gaps)
    clusters = [[vals[0]]]
    for v, g in zip(vals[1:], gaps):
        if g <= threshold:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters]


def extract_table_from_pdf(doc, table_number):
    """Best-effort reconstruction of a simple grid table from PyMuPDF word
    positions: cluster words into rows by y-gap, derive column anchors
    from the header row's x0 positions, assign every other word to its
    nearest anchor. Returns (title, [row_cells, ...]) or None."""
    caption_re = re.compile(r"^TABLE\s+" + re.escape(table_number) + r"\b\s*(.*)$")
    for page in doc:
        words = page.get_text("words")
        if not words:
            continue
        line_groups = defaultdict(list)
        for w in words:
            line_groups[(w[5], w[6])].append(w)

        caption_y1 = None
        caption_title = None
        for ws in line_groups.values():
            ws_sorted = sorted(ws, key=lambda w: w[0])
            text = " ".join(w[4] for w in ws_sorted)
            m = caption_re.match(text)
            if m:
                caption_y1 = max(w[3] for w in ws_sorted)
                caption_title = m.group(1).strip()
                break
        if caption_y1 is None:
            continue

        region_words = sorted((w for w in words if w[1] > caption_y1 + 2), key=lambda w: w[1])
        if not region_words:
            continue

        rows_raw = []
        cur_y = None
        cur_line = []
        for w in region_words:
            if cur_y is None or abs(w[1] - cur_y) > 3:
                if cur_line:
                    rows_raw.append(cur_line)
                cur_line = [w]
                cur_y = w[1]
            else:
                cur_line.append(w)
        if cur_line:
            rows_raw.append(cur_line)
        if len(rows_raw) < 2:
            continue

        row_ys = [min(w[1] for w in line) for line in rows_raw]
        gaps = [row_ys[i + 1] - row_ys[i] for i in range(len(row_ys) - 1)]
        row_threshold = split_threshold(gaps)

        logical_rows = []
        cur_group = []
        prev_y = None
        for line_words, y in zip(rows_raw, row_ys):
            if prev_y is not None and (y - prev_y) > row_threshold:
                logical_rows.append(cur_group)
                cur_group = list(line_words)
            else:
                cur_group.extend(line_words)
            prev_y = y
        if cur_group:
            logical_rows.append(cur_group)
        if len(logical_rows) < 2:
            continue

        anchors = sorted(cluster_1d(w[0] for w in logical_rows[0]))

        def nearest_anchor(x):
            return min(anchors, key=lambda a: abs(a - x))

        def build_row(row_words):
            cells = defaultdict(list)
            for w in row_words:
                cells[nearest_anchor(w[0])].append(w)
            out = []
            for a in anchors:
                ws_sorted = sorted(cells.get(a, []), key=lambda w: (round(w[1]), w[0]))
                text = ""
                for w in ws_sorted:
                    text = text if not text else (text if text.endswith("/") else text + " ")
                    text += w[4]
                out.append(text)
            return out

        return caption_title, [build_row(r) for r in logical_rows]
    return None


def insert_tables(blocks, doc):
    seen = set()
    new_blocks = []
    for btype, text in blocks:
        new_blocks.append((btype, text))
        if btype != "para":
            continue
        m = TABLE_REF_RE.search(text)
        if not m or m.group(1) in seen:
            continue
        seen.add(m.group(1))
        result = extract_table_from_pdf(doc, m.group(1))
        if result is None:
            continue
        title, rows = result
        new_blocks.append(("table", f"Table {m.group(1)} {title}".strip()))
        for row in rows:
            new_blocks.append(("table", " | ".join(row)))
    return new_blocks


def render(chapter_num, chapter_title, blocks):
    lines = []
    if chapter_num and chapter_title:
        # Single "#" (not "##") deliberately: load_ncert_sections.py's parse_sections() only
        # treats a "## " line as a section heading. A "## Chapter N: Title" line here used to be
        # parsed as a real section -- an empty one, since the chapter's actual first "## " heading
        # follows right after it -- which then failed validate_chapter's empty-section check on
        # every chapter that has a detected title, in both the manual-paste and PDF pipelines.
        lines.append(f"# Chapter {chapter_num}: {title_case(chapter_title)}")
        lines.append("")
    prev_type = None
    for btype, text in blocks:
        if btype == "heading":
            if lines and lines[-1] != "":
                lines.append("")
            lines.append(f"## {text}")
            lines.append("")
        else:
            if prev_type == "heading" and lines and lines[-1] == "":
                pass
            lines.append(text)
        prev_type = btype
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def clean_text(raw_text, pdf_doc=None):
    raw_lines = raw_text.replace("#", "").split("\n")
    chapter_num, chapter_title, body_lines = strip_front_matter(raw_lines)
    blocks = build_blocks(body_lines)
    blocks = dedupe_headings(blocks)
    if pdf_doc is not None:
        blocks = insert_tables(blocks, pdf_doc)
    return render(chapter_num, chapter_title, blocks), blocks


def pdf_text_without_junk(pdf_doc):
    pages_text = []
    for page in pdf_doc:
        kept = [ln for ln in page.get_text().split("\n") if not is_junk_line(ln.strip())]
        pages_text.append(" ".join(kept))
    return normalize_ws_quotes(" ".join(pages_text))


def validate(blocks, pdf_doc):
    full_pdf_text = pdf_text_without_junk(pdf_doc)
    flagged = []
    clean_word_count = 0
    for btype, text in blocks:
        clean_word_count += len(text.split())
        if btype != "para" and btype != "list":
            continue
        sentences = re.split(r"(?<=[.?!])\s+", text)
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if sent not in full_pdf_text:
                flagged.append(sent)

    pdf_word_count = len(full_pdf_text.split())
    diff_pct = (
        abs(clean_word_count - pdf_word_count) / pdf_word_count * 100
        if pdf_word_count else 0.0
    )
    return flagged, clean_word_count, pdf_word_count, diff_pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_path")
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    in_path = Path(args.input_path)
    raw_text = in_path.read_text(encoding="utf-8")

    pdf_doc = None
    if args.pdf:
        import fitz
        pdf_doc = fitz.open(args.pdf)

    cleaned, blocks = clean_text(raw_text, pdf_doc)

    out_path = Path(args.out) if args.out else Path("ncert_manual/clean") / in_path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(cleaned, encoding="utf-8")
    print(f"Wrote {out_path}")

    headings = [t for b, t in blocks if b == "heading"]
    print(f"Headings ({len(headings)}):")
    for h in headings:
        print(f"  - {h}")

    if pdf_doc is not None:
        flagged, clean_wc, pdf_wc, diff_pct = validate(blocks, pdf_doc)
        print(f"\nClean word count: {clean_wc}")
        print(f"PDF word count:   {pdf_wc}")
        print(f"Word count diff:  {diff_pct:.2f}%" + (" [FLAGGED >3%]" if diff_pct > 3 else ""))
        if flagged:
            print(f"\nSentences NOT found in PDF text layer ({len(flagged)}):")
            for s in flagged:
                print(f"  ! {s}")
        else:
            print("\nAll sentences found in PDF text layer.")


if __name__ == "__main__":
    main()
