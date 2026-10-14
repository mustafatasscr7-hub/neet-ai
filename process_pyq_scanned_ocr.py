"""Free, no-AI extraction mode for SCANNED, image-only PYQ PDFs: local OCR + rules, $0 per page.

A sibling of process_pyq_scanned.py (the Gemini "Scanned pages (image-only)" mode) -- same input,
same output shape, never auto-selected. Instead of a model reading each page, a local OCR engine
returns text lines with bounding boxes, and plain rules here turn those lines into the SAME item
list the Gemini mode's model returns ({"kind": "question" | "continuation" | "answer_key" |
"chapter_start" | "solutions_start", "box_2d": [...]} -- see build_scanned_page_prompt). From there
everything is reused unchanged from process_pyq_scanned: band/column reading order, joining a
question across a column or page break, numbering gaps and duplicates, answer-key matching by
question number, cropped-edge detection, and the exact flag wording. Text helpers come from
process_pyq_book_3col (_normalize_ws, the year-digit regex).

Why not process_pyq_book_3col's own top-level parser: it reads PyMuPDF text blocks and expects
that book's conventions (zero-padded "01" numbers, "[NEET 2016]" bracketed tags, a printed
"Ans. (x)" per question). DISHA-style scans have none of those -- "1." numbers, unbracketed
right-aligned year tags, and one answer-key table -- and changing that module would put its
byte-identical output guarantee at risk. The item format above is the shared, layout-neutral
contract both scanned modes now feed.

OCR engine: RapidOCR (PaddleOCR's PP-OCR detection/recognition models run on onnxruntime, which
this project already depends on for diagram detection). Chosen over Tesseract because Tesseract is
a native binary with no pip package on Windows (system installer only, and an OS package on
Railway), and over full PaddleOCR / Surya because those pull in paddlepaddle / PyTorch (hundreds of
MB to GBs). The models (~31MB) ship inside the wheel -- no download at runtime.

Built against DISHA_36_YEARS_BIOLOGY_11thTEST.pdf; every layout rule below cites the real line it
was written for.
"""
import re
import threading
import time

import fitz

from process_pyq_book_3col import _normalize_ws
from process_pyq_scanned import (
    order_page_items,
    merge_document_items,
    finalize_segments,
    pdf_cropped_edges,
    cropped_edge_flags,
    split_option_markers,
    _OPTION_MARKER_RE,
    _LETTERS,
)

OCR_ENGINE_NAME = "RapidOCR (PP-OCR, onnxruntime)"
OCR_DPI = 200  # same render resolution as the Gemini scanned mode
# A word/line the OCR engine itself scores below this is surfaced for review. Measured on DISHA:
# 469 of 472 lines scored >= 0.93; the 3 below 0.8 were a hanging "1." (0.74) and an answer-key
# "5" (0.59) -- both read correctly, but exactly the kind of thing worth a second look.
LOW_CONFIDENCE = 0.80

_engine = None
_engine_lock = threading.Lock()


def _get_engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            from rapidocr import RapidOCR  # lazy: the server imports this module even where it isn't installed
            # The angle classifier only matters for upside-down text lines; a scanned book page is
            # upright, so skipping it is pure speed.
            _engine = RapidOCR(params={"Global.use_cls": False})
        return _engine


def ocr_available():
    try:
        import rapidocr  # noqa: F401
        return True
    except ImportError:
        return False


def ocr_page(page):
    """One rendered page -> {"width", "height", "lines": [{"text", "score", "box": [x0,y0,x1,y1] px}]}."""
    import numpy as np
    pix = page.get_pixmap(dpi=OCR_DPI)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = img[:, :, :3]
    result = _get_engine()(img)
    lines = []
    for box, text, score in zip(result.boxes if result.boxes is not None else [], result.txts or [], result.scores or []):
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        lines.append({"text": text, "score": float(score), "box": [min(xs), min(ys), max(xs), max(ys)]})
    return {"width": pix.width, "height": pix.height, "lines": lines}


def figure_boxes_1000(page):
    """Figure regions on the page in box_2d's 0-1000 space, via process_pyq_vision's existing
    DocLayout-YOLO helpers (unmodified, the same ones 3-column mode uses). Never fatal: without
    them has_diagram just stays false, which the review UI's checkbox covers."""
    try:
        import numpy as np
        import cv2
        from PIL import Image
        from process_pyq_vision import _get_yolo_session, _yolo_infer_figure_boxes, _YOLO_DPI
        pix = page.get_pixmap(dpi=_YOLO_DPI)
        arr = np.array(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        return [(y0 * 1000 / pix.height, x0 * 1000 / pix.width, y1 * 1000 / pix.height, x1 * 1000 / pix.width)
                for x0, y0, x1, y1 in _yolo_infer_figure_boxes(_get_yolo_session(), arr, cv2, np)]
    except Exception as e:
        print(f"    Free OCR mode: diagram detection unavailable for a page ({e}) -- non-fatal")
        return []


# ---------------------------------------------------------------- page -> items (pure rules)

# Running headers and page furniture, only ever dropped from the top strip of a page. Real lines:
# "A-3", "Botany" (and the edge-cropped "Botal"), "Class-XI/Chapter-1/The Living World".
_HEADER_RE = re.compile(r'^(?:A\s*-\s*\d+|Bota\w*|Zool\w*|Class\s*-?\s*X?I{1,2}\b.*/.*)$', re.I)
_HANDLE_RE = re.compile(r'^@\S+$')  # "@sachasanatani" (uploader watermark)
_HEADER_ZONE = 120  # top 12% of the page, in 0-1000 units

_TREND_RE = re.compile(r'Trend\s*Analysis', re.I)
_LEGEND_RE = re.compile(r'Level\s*of\s*Difficulty|No\.?\s*of\s*Questions', re.I)
_ANSWER_KEY_RE = re.compile(r'^\s*ANSWER\s*KEYS?\s*$', re.I)
_SOLUTIONS_RE = re.compile(r'^\s*(?:Hints?\s*(?:&|and)?\s*Solutions?|Solutions?|Explanations?)\s*$', re.I)
_TOPIC_RE = re.compile(r'^\s*Topic\s*[-–]?\s*\d+\s*[:.]', re.I)
# One answer-key cell or several merged by OCR: "13", "(a)", "13 (a)", "a)".
_KEY_CELLS_RE = re.compile(r'^(?:\s*(?:\d{1,3}|\(?[a-dA-D]\)))+\s*$')
_KEY_TOKEN_RE = re.compile(r'(\d{1,3})|\(?([a-dA-D])\)')

# "1.", "23. A taxon is", "38.", and "123.Satellite RNAs" (OCR dropped the space -- DISHA
# Biological Classification page 9; only before a capital, so "1.5 g" never matches).
_QSTART_RE = re.compile(r'^(\d{1,3})\s*[.)](?:(?:\s+|(?=[A-Z]))(.*))?$')
# A hanging number OCR couldn't read as digits ("l.", "I.") -- still a question start, number unknown.
_QSTART_UNREADABLE_RE = re.compile(r'^[\dIlO|!]{1,3}[.,]$')
_NUMBER_HANG = 20  # a question number sits at the column's left edge; wrapped text is indented past this
# The column's left edge is taken from rows within this distance above/below, not the whole
# column: a tilted scan drifts. Seen live on DISHA Biological Classification pages 3-4, the left
# edge runs from x=91 at the top to x=70 at the bottom, so against the column-wide minimum the
# top rows' hanging "39." / "40." / "49." sat 21 units in and were read as wrapped text.
_EDGE_WINDOW = 250
# "NCERT | Page-8 / N-6 | 1990,91,92,96", "NCERT | Page-10 / N-8|2022", "NCERT | Page-12 |2010 M"
_NCERT_RE = re.compile(r'NCERT\s*\|?\s*(Page[^|]*?)\s*\|\s*([^|]+?)\s*$', re.I)
# Right-aligned year tags as printed: "2016", "2003, 04", "1990,91,92,96", "2010 M", "Ph-II 2022".
_YEAR_TAG_RE = re.compile(r'^(?:Ph\s*-?\s*I{1,3}\s*)?(?:19|20)\d{2}(?:\s*,\s*(?:19|20)?\d{2})*(?:\s*[A-Z])?$')
_TAG_RIGHT_SLACK = 40  # a year tag's right edge sits at the column's right edge
_OPEN_MARKER_RE = re.compile(r'(?<!\S)\(([a-d])(?=\s)')       # OCR dropped the ")" -> "(a"
_BARE_MARKER_RE = re.compile(r'(?:^|(?<=\s))([a-d])\)(?=\s)')  # OCR dropped the "(" -> "a)"
# OCR dropped the space before a marker -> "increase in number(b) disappear" (DISHA Biological
# Classification Q64); the option split only sees a marker after whitespace.
_GLUED_MARKER_RE = re.compile(r'(?<=[A-Za-z.,])\(([a-d])\)(?=\s|[A-Z])')
_SUBSTATEMENT_START_RE = re.compile(r'^(?:\((?:[ivx]+|[A-D])\)|Column\s+I\b)')
# The "(A) (B) (C) (D)" header row over a match-the-column answer grid (DISHA page 4, top left).
# Dropped: the grid rows below it get the labels back via _expand_match_grid_options.
_GRID_HEADER_RE = re.compile(r'^(?:\(\s*[A-D]\s*\)\s*){2,4}$')

# Column split is measured per page, not fixed: DISHA page 2 sits ~80 units further right than
# pages 1 and 3 (left-column year tags end at x=546 there, while page 3's right column STARTS at
# 481), so any single fixed gutter misfiles one or the other -- seen live, a fixed x=470 pulled
# page 2's left-column tags into the right column and hid Q16's hanging number.
# Searched well past the midpoint either side, for books whose gutter isn't centred.
_GUTTER_SEARCH = (250, 750)
# A question number OCR read as its own box ("3."). These are left out when measuring the gutter
# and then go with the text to their right: seen live on DISHA Biological Classification page 1,
# the right column's hanging "3." / "4." / "5." sit at x=505-531, closer to the left column's
# text (ends 487) than the gap between them and their own text (starts 549) is wide -- so with
# them counted, the widest empty band was 531-549 and all three were filed into the LEFT column.
_NUMBER_BOX_RE = re.compile(r'^[\dIlO|!]{1,3}\s*[.,)]$')
_NUMBER_TO_TEXT_GAP = 80  # a hanging number's own text starts within this distance to its right
# A real gutter is an empty vertical band at least this wide, crossed by (almost) no lines; a page
# with none is read as one column.
_MIN_GUTTER_WIDTH = 3
_MAX_GUTTER_CROSSING = 0.05


def _norm_lines(ocr):
    w, h = ocr["width"], ocr["height"]
    out = []
    for ln in ocr["lines"]:
        x0, y0, x1, y1 = ln["box"]
        out.append({"text": ln["text"].strip(), "score": ln["score"],
                    "x0": x0 * 1000 / w, "y0": y0 * 1000 / h, "x1": x1 * 1000 / w, "y1": y1 * 1000 / h})
    return [l for l in out if l["text"]]


def _rows(lines):
    """Lines -> rows (lists of lines left-to-right sharing one baseline). DISHA's line pitch is
    ~16 units with ~19-unit-tall boxes, so neighbours overlap vertically -- grouping is by centre
    distance, not overlap."""
    if not lines:
        return []
    heights = sorted(l["y1"] - l["y0"] for l in lines)
    tol = 0.45 * heights[len(heights) // 2]
    rows = []
    for l in sorted(lines, key=lambda l: (l["y0"] + l["y1"]) / 2):
        cy = (l["y0"] + l["y1"]) / 2
        if rows and abs(cy - rows[-1]["cy"]) <= tol:
            r = rows[-1]
            r["lines"].append(l)
            r["cy"] = sum((m["y0"] + m["y1"]) / 2 for m in r["lines"]) / len(r["lines"])
        else:
            rows.append({"cy": cy, "lines": [l]})
    return [sorted(r["lines"], key=lambda l: l["x0"]) for r in rows]


def _is_number_box(l):
    return bool(_NUMBER_BOX_RE.match(l["text"]))


def find_gutter(lines):
    """The x (0-1000) between the two columns, or None for a single-column page: centre of the
    widest vertical band in the page's middle that the fewest text lines cross (normally none).
    Full-width zones are already removed. Hanging question-number boxes are ignored here (see
    _NUMBER_BOX_RE) and assigned to a column afterwards by assign_columns."""
    text = [l for l in lines if not _is_number_box(l)]
    if not text:
        return None
    lo, hi = _GUTTER_SEARCH
    cover = [0] * (hi - lo + 1)
    for l in text:
        for x in range(max(lo, int(l["x0"]) + 1), min(hi, int(l["x1"])) + 1):
            cover[x - lo] += 1
    best = min(cover)
    run_start, best_run = None, (0, -1)
    for i, c in enumerate(cover + [best + 1]):
        if c == best and run_start is None:
            run_start = i
        elif c != best and run_start is not None:
            if i - run_start > best_run[1] - best_run[0]:
                best_run = (run_start, i)
            run_start = None
    gutter = lo + (best_run[0] + best_run[1]) / 2
    # Single column: no clean empty band (most lines run straight across the middle), or no
    # real text on one side of it.
    if best > max(1, _MAX_GUTTER_CROSSING * len(text)) or best_run[1] - best_run[0] < _MIN_GUTTER_WIDTH:
        return None
    left = sum(1 for l in text if l["x1"] <= gutter)
    right = sum(1 for l in text if l["x0"] >= gutter)
    if min(left, right) < 2:
        return None
    return gutter


def assign_columns(lines, gutter):
    """[(line, "left" | "right")]. Lines go by their centre; a hanging question-number box goes
    with the first text line to its right on the same row (its own question's text), wherever
    the gutter fell -- that's the whole fix for a right-column "3." sitting just past the
    midpoint. A number with no text beside it falls back to its centre."""
    if gutter is None:
        return [(l, "left") for l in lines]

    def by_centre(l):
        return "left" if (l["x0"] + l["x1"]) / 2 < gutter else "right"

    heights = sorted(l["y1"] - l["y0"] for l in lines)
    tol = 0.45 * heights[len(heights) // 2] if heights else 0
    out = []
    for l in lines:
        col = by_centre(l)
        if _is_number_box(l):
            cy = (l["y0"] + l["y1"]) / 2
            beside = [m for m in lines if m is not l and not _is_number_box(m)
                      and abs((m["y0"] + m["y1"]) / 2 - cy) <= tol
                      and -5 <= m["x0"] - l["x1"] <= _NUMBER_TO_TEXT_GAP]
            if beside:
                col = by_centre(min(beside, key=lambda m: m["x0"]))
        out.append((l, col))
    return out


def _box(lines):
    return [min(l["y0"] for l in lines), min(l["x0"] for l in lines), max(l["y1"] for l in lines), max(l["x1"] for l in lines)]


def _parse_answer_key(cell_lines):
    """Cells -> ({number: letter}, notes). A letter is paired ONLY with the number cell directly
    to its left in the same row -- never inferred from grid position. Seen live: DISHA's key cell
    "1" (white on dark green) was never read at all, so its "(b)" stays unpaired and Q1 is left
    blank and flagged rather than assumed to be 1."""
    tokens = []
    for ln in cell_lines:
        width = max(ln["x1"] - ln["x0"], 1)
        n = max(len(ln["text"]), 1)
        for m in _KEY_TOKEN_RE.finditer(ln["text"]):
            x = ln["x0"] + width * (m.start() / n)
            tok = {"x": x, "y0": ln["y0"], "y1": ln["y1"], "score": ln["score"]}
            if m.group(1):
                tok["num"] = int(m.group(1))
            else:
                tok["letter"] = m.group(2).lower()
            tokens.append(tok)
    entries, notes = {}, []
    for row in _rows([{**t, "x0": t["x"], "x1": t["x"] + 1, "text": ""} for t in tokens]):
        prev = None
        for t in row:
            if "letter" in t:
                if prev is not None and "num" in prev:
                    entries[prev["num"]] = t["letter"]
                    if min(prev["score"], t["score"]) < LOW_CONFIDENCE:
                        notes.append(f"answer-key entry Q{prev['num']} ({t['letter']}) was read with low OCR confidence -- check it against the printed key")
                else:
                    notes.append(f"an answer-key cell ({t['letter']}) has no readable question number beside it, so it was not attached to any question")
            prev = t
    return entries, notes


# Shape of a right-aligned tag box regardless of whether OCR read its digits right: short, mostly
# digits/commas. Seen live: DISHA Q7's "1991" came back as "1661".
_TAG_SHAPE_RE = re.compile(r'^(?:Ph\s*-?\s*I{1,3}\s*)?[\dOIl|]{4}(?:\s*,\s*[\dOIl|]{2,4})*(?:\s*[A-Z])?$')


_MERGED_YEAR_TAG_RE = re.compile(r'(?<=[\s?.:])\s*((?:19|20)\d{2}(?:\s*,\s*(?:19|20)?\d{2})*(?:\s*M)?)$')


def _strip_tags(row, col_right, tags, issues):
    """Removes an NCERT tag and/or a right-aligned year tag from one row (fills `tags`), returning
    the row's remaining text. A tag-shaped box whose digits aren't a plausible year is still
    removed from the text, but its year is left blank and reported in `issues` -- never corrected."""
    kept = []
    for i, ln in enumerate(row):
        text = ln["text"]
        if tags.get("year_tag") is None and not tags.get("_tag_seen"):
            m = _NCERT_RE.search(text)
            if m:
                tags["ncert_tag"] = re.sub(r'\s*/\s*', ' / ', _normalize_ws(m.group(1)))
                tags["_tag_seen"] = True
                year = _normalize_ws(m.group(2))
                if _YEAR_TAG_RE.match(year):
                    tags["year_tag"] = year
                else:
                    issues.append(f"year in the NCERT tag was read as {year!r}, which isn't a valid year -- left blank")
                text = text[:m.start()].strip()
                if not text:
                    continue
            elif (i == len(row) - 1 and _TAG_SHAPE_RE.match(text) and ln["x1"] >= col_right - _TAG_RIGHT_SLACK
                  and not (i > 0 and re.search(r'\([a-d]\)$', row[i - 1]["text"]))):
                # Not when the box right before it is a bare option marker -- then this "1992" is
                # that option's own text, not a tag.
                tags["_tag_seen"] = True
                if _YEAR_TAG_RE.match(text):
                    tags["year_tag"] = _normalize_ws(text)
                else:
                    issues.append(f"year tag was read as {text!r}, which isn't a valid year -- left blank")
                continue
            elif i == len(row) - 1 and ln["x1"] >= col_right - _TAG_RIGHT_SLACK and not tags.get("_options_started"):
                # OCR sometimes glues the tag onto the stem's own box (seen live: "...written by
                # 1999", "...described?2012", "...the statement of 1994"). Only 19xx/20xx exam
                # years, only at the column's right edge, only before the options begin.
                m = _MERGED_YEAR_TAG_RE.search(text)
                if m:
                    tags["_tag_seen"] = True
                    tags["year_tag"] = _normalize_ws(m.group(1))
                    text = text[:m.start()].rstrip()
        if re.match(r'^\(?[a-d]\)', text):
            tags["_options_started"] = True
        kept.append(text)
    return " ".join(kept)


_CELL_START_RE = re.compile(r'^\(?[a-d][)\s]')


def _merge_grid_wraps(rows):
    """2x2 option grids: "(a) X  (b) Y" on one row, "(c) Z  (d) W" on the next. Reading each row
    left to right already gives a, b, c, d; what breaks is a cell whose text WRAPS onto a line of
    its own -- read row-wise, "(a) long text (b) Y" + "wrapped" would put "wrapped" into (b).
    A markerless row right under a row of two or more option cells is split by x: each box goes
    straight after the cell it sits under. Anything else is returned unchanged."""
    out = []
    for r in rows:
        prev = out[-1] if out else None
        cells = [l for l in prev if _CELL_START_RE.match(l["text"])] if prev else []
        if (len(cells) >= 2 and not any(_CELL_START_RE.match(l["text"]) or _QSTART_RE.match(l["text"]) for l in r)
                and min(l["x0"] for l in r) >= cells[0]["x0"] - 5):
            merged = []
            for i, cell in enumerate(cells):
                nxt = cells[i + 1]["x0"] if i + 1 < len(cells) else float("inf")
                merged.extend(l for l in prev if cell["x0"] <= l["x0"] < nxt)
                merged.extend(l for l in r if (cell["x0"] - 5 if i else float("-inf")) <= l["x0"] < nxt - 5)
            if len(merged) == len(prev) + len(r):
                out[-1] = merged
                continue
        out.append(r)
    return out


def _fix_ocr_markers(text):
    # U+2212 minus / en dash -> the plain hyphen the book prints in "Class - Phylum - Tribe".
    text = text.replace("\u2212", "-").replace("\u2013", "-")
    text = _GLUED_MARKER_RE.sub(r' (\1)', text)
    return _BARE_MARKER_RE.sub(r'(\1)', _OPEN_MARKER_RE.sub(r'(\1)', text))


def _join_rows(texts):
    out = ""
    for t in texts:
        t = t.strip()
        if not t:
            continue
        if not out:
            out = t
        elif _SUBSTATEMENT_START_RE.match(t):
            out += "\n" + t
        elif re.search(r'[A-Za-z]-$', out) and t[:1].islower():
            out = out[:-1] + t  # "unexcep-" + "tionally"
        else:
            out += " " + t
    return out


# OCR misreads that come back with HIGH confidence, so the score alone never catches them -- both
# seen live on DISHA: "Îarge" for "large" (0.993) and "scientificallycorrectlynamed,correctly"
# with its spaces dropped (1.0). An English NEET book has no accented Latin letters, and no real
# word in it runs 25+ letters.
_ODD_CHAR_RE = re.compile(r'[À-ÖØ-öø-ɏ]')
_RUN_TOGETHER_RE = re.compile(r'[A-Za-z][A-Za-z,]{24,}')


def _suspicious_text(text):
    out = []
    odd = sorted(set(_ODD_CHAR_RE.findall(text)))
    if odd:
        out.append(f"unusual character(s) {', '.join(repr(c) for c in odd)} -- likely an OCR misread")
    for m in _RUN_TOGETHER_RE.finditer(text):
        out.append(f"words look run together ({m.group(0)!r}) -- OCR may have dropped spaces")
    return out


def _question_type(stem):
    low = stem.lower()
    if re.search(r'\bmatch\b', low) and "column" in low:
        return "match_column"
    if "assertion" in low and "reason" in low:
        return "assertion_reason"
    if re.search(r'\(ii\)', stem) or re.search(r'\bstatement\s+(?:i|1)\b', low):
        return "statement_based"
    return "mcq"


def _overlaps(box, fig, min_frac=0.3):
    y0, x0, y1, x1 = box
    fy0, fx0, fy1, fx1 = fig
    iw, ih = min(x1, fx1) - max(x0, fx0), min(y1, fy1) - max(y0, fy0)
    area = max((fx1 - fx0) * (fy1 - fy0), 1e-6)
    return iw > 0 and ih > 0 and (iw * ih) / area >= min_frac


def segment_page(ocr, figures=()):
    """One page's OCR lines -> (items in the Gemini mode's format, page notes). Pure function of
    its input, so the whole layout logic is unit-testable with synthetic lines.

    Every question/continuation item carries "_ocr_issues" (low OCR confidence, an unreadable
    year tag), flagged later against whichever question the item ends up joined to."""
    lines = _norm_lines(ocr)
    items, notes = [], []

    # 1. Page furniture: running headers in the top strip, uploader handles anywhere.
    lines = [l for l in lines if not ((l["y0"] < _HEADER_ZONE and _HEADER_RE.match(l["text"])) or _HANDLE_RE.match(l["text"]))]

    # 2. Full-width zones, found by their anchor lines; everything inside a zone leaves the
    #    column flow. Each zone also splits the page into bands, same as order_page_items does.
    zones = []  # (y_top, y_bottom)
    trend = next((l for l in lines if _TREND_RE.search(l["text"])), None)
    if trend:
        legend = [l for l in lines if _LEGEND_RE.search(l["text"]) and l["y0"] > trend["y0"]]
        bottom = max(l["y1"] for l in legend) if legend else trend["y1"]
        if not legend:
            notes.append("A trend-analysis table was found but not its legend row, so only its heading was skipped -- check that no table cells leaked into a question.")
        # A trend table only appears on a chapter's opening page: everything above it is the
        # chapter banner (title, chapter number, subject).
        above = [l for l in lines if l["y1"] <= trend["y0"] + 2]
        titled = [l for l in above if sum(c.isalpha() for c in l["text"]) >= 4]
        title = max(titled, key=lambda l: l["y1"] - l["y0"])["text"] if titled else None
        items.append({"kind": "chapter_start", "title": title, "box_2d": [0, 0, trend["y0"], 1000]})
        zones.append((0, bottom))

    for anchor in [l for l in lines if _ANSWER_KEY_RE.match(l["text"])]:
        below = sorted((l for l in lines if l["y0"] > anchor["y0"] + 1), key=lambda l: l["y0"])
        cells = []
        for l in below:
            if not _KEY_CELLS_RE.match(l["text"]):
                break
            cells.append(l)
        entries, key_notes = _parse_answer_key(cells)
        notes.extend(key_notes)
        bottom = max([anchor["y1"]] + [c["y1"] for c in cells])
        items.append({"kind": "answer_key", "entries": {str(k): v for k, v in entries.items()}, "box_2d": [anchor["y0"], 0, bottom, 1000]})
        zones.append((anchor["y0"], bottom))

    sol = next((l for l in sorted(lines, key=lambda l: l["y0"]) if _SOLUTIONS_RE.match(l["text"])), None)
    if sol:
        items.append({"kind": "solutions_start", "box_2d": [sol["y0"], 0, sol["y1"], 1000]})
        zones.append((sol["y0"], 1001))  # explanations run to the end of the page

    def in_zone(l):
        cy = (l["y0"] + l["y1"]) / 2
        return any(top - 1 <= cy <= bottom + 1 for top, bottom in zones)

    flow = [l for l in lines if not in_zone(l)]
    bounds = sorted({b for z in zones for b in z})

    def band_of(l):
        return sum(1 for b in bounds if b <= l["y0"])

    groups = {}
    for l, col in assign_columns(flow, find_gutter(flow)):
        groups.setdefault((band_of(l), col), []).append(l)
    for it in items:  # the zone items above
        it["_col"] = "full"

    # 3. Within one band of one column: drop topic headings, then cut into questions at each
    #    hanging question number; anything above the first number continues an earlier question.
    for (_, col), glines in sorted(groups.items()):
        rows = _rows(glines)
        col_right = max(l["x1"] for l in glines)

        kept_rows, heading = [], None
        for r in rows:
            first = r[0]
            if _TOPIC_RE.match(first["text"]):
                heading = _box(r)
                continue
            # "Topic-2: Diversity in the Living World," wraps to a centred second line
            # ("Taxonomic Categories") inside the heading's own box -- indented past the
            # heading's start and no wider than it.
            if heading and first["x0"] >= heading[1] + 30 and r[-1]["x1"] <= heading[3] + 10 \
                    and first["y0"] - heading[2] < 15 and not _QSTART_RE.match(first["text"]):
                heading = _box(r)
                continue
            heading = None
            if _GRID_HEADER_RE.match(" ".join(l["text"] for l in r)):
                continue
            kept_rows.append(r)

        chunks = []  # (number or None, is_question, rows)
        for r in kept_rows:
            first = r[0]
            m = _QSTART_RE.match(first["text"])
            cy = (first["y0"] + first["y1"]) / 2
            col_left = min(o[0]["x0"] for o in kept_rows if abs((o[0]["y0"] + o[0]["y1"]) / 2 - cy) <= _EDGE_WINDOW)
            hanging = first["x0"] <= col_left + _NUMBER_HANG
            if hanging and m:
                chunks.append((int(m.group(1)), True, [r]))
            elif hanging and _QSTART_UNREADABLE_RE.match(first["text"]) and len(r) > 1:
                chunks.append((None, True, [r]))
            elif chunks:
                chunks[-1][2].append(r)
            else:
                chunks.append((None, False, [r]))

        for number, is_question, crows in chunks:
            all_lines = [l for r in crows for l in r]
            issues = []
            low = [l["text"] for l in all_lines if l["score"] < LOW_CONFIDENCE]
            if low:
                issues.append(f"low OCR confidence on {', '.join(repr(t) for t in low[:3])} -- check against the printed page")
            box = _box(all_lines)
            if not is_question:
                text = _fix_ocr_markers(_join_rows(" ".join(l["text"] for l in r) for r in crows))
                issues.extend(_suspicious_text(text))
                items.append({"kind": "continuation", "text": text, "options": {}, "box_2d": box, "_col": col, "_ocr_issues": issues})
                continue
            tags = {"year_tag": None, "ncert_tag": None}
            texts = [_strip_tags(r, col_right, tags, issues) for r in _merge_grid_wraps(crows)]
            m = _QSTART_RE.match(texts[0]) or re.match(r'^\S+\s*(.*)$', texts[0])
            texts[0] = (m.group(2) or "") if m.re is _QSTART_RE else m.group(1)
            full = _fix_ocr_markers(_join_rows(texts))
            issues.extend(_suspicious_text(full))
            first_opt = _OPTION_MARKER_RE.search(full)
            # Stem split here rather than via split_option_markers' lead, which collapses
            # whitespace: a match-the-column table or (i)-(v) statement list needs its line breaks.
            stem_raw = full[:first_opt.start()] if first_opt else full
            stem = "\n".join(_normalize_ws(ln) for ln in stem_raw.split("\n") if ln.strip())
            options = split_option_markers(full[first_opt.start():])[1] if first_opt else {}
            items.append({
                "kind": "question", "number": number, "question": stem, "options": options,
                "year_tag": tags["year_tag"], "ncert_tag": tags["ncert_tag"],
                "question_type": _question_type(stem),
                "has_diagram": any(_overlaps(box, f) for f in figures),
                "box_2d": box, "_col": col, "_ocr_issues": issues,
            })
    return items, notes


def order_ocr_items(items):
    """process_pyq_scanned.order_page_items' reading order (bands split by full-width items; left
    column, then right; a continuation first in its column), but using the column segment_page
    actually measured ("_col") instead of that function's fixed page-midpoint split -- which
    would misfile a question on a page whose gutter isn't centred. That function is shared with
    the Gemini mode and left as is."""
    if any("_col" not in it for it in items):
        return order_page_items(items)[0]
    full_tops = sorted(it["box_2d"][0] for it in items if it["_col"] == "full")

    def key(idx):
        it = items[idx]
        ymin = it["box_2d"][0]
        if it["_col"] == "full":
            return (full_tops.index(ymin) * 2 + 1, 0, 0, ymin, idx)
        band = sum(1 for t in full_tops if t < ymin)
        return (band * 2, 0 if it["_col"] == "left" else 1, 0 if it["kind"] == "continuation" else 1, ymin, idx)

    return [items[i] for i in sorted(range(len(items)), key=key)]


# ---------------------------------------------------------------- safety net: polluted options

# Things that never belong inside an option's text -- each means the next question (or its tags)
# got glued onto this one. Seen live on DISHA Biological Classification page 1: Q2's options ended
# up holding "5.", "NCERT |Page-19 & 20 /N-13 | 1998, 2003" and "3." before the column fix.
_EMBEDDED_QNUM_RE = re.compile(r'(?:^|(?<=\s))(\d{1,3})\.(?=\s|$|[A-Z])')
_EMBEDDED_NCERT_RE = re.compile(r'NCERT\s*\|[^|]*\|\s*(?:Ph\s*-?\s*I{1,3}\s*)?(?:19|20)\d{2}(?:\s*,\s*(?:19|20)?\d{2})*(?:\s*[A-Z]\b)?')
_EMBEDDED_YEAR_RE = re.compile(r'(?:^|(?<=\s))(?:19|20)\d{2}(?:\s*,\s*(?:19|20)?\d{2})+(?=\s|$)')
# A single state-exam year tag glued onto an option's end: "They are phagotrophs Kar.2013"
# (DISHA Biological Classification Q90). Flagged only -- one year alone can be real option text.
_TRAILING_TAG_RE = re.compile(r'\s*\b[A-Z][a-z]{1,5}\.?\s*(?:19|20)\d{2}$')
# An option this many times longer than the longest of its siblings (and long in absolute terms)
# is almost always carrying another question's text.
_LONG_OPTION_RATIO = 3
_LONG_OPTION_MIN = 80


def _expected_numbers(seg, q):
    """Numbers that could legitimately start right after q: from q+1 up to (not including) the
    next number actually found in this segment, capped at q+5."""
    if q["number"] is None:
        return set()
    later = [o["number"] for o in seg["questions"] if o["number"] is not None and o["number"] > q["number"]]
    stop = min(later + [q["number"] + 6])
    return set(range(q["number"] + 1, stop))


def _resplit_polluted(seg):
    """Runs after merge_document_items and before finalize_segments, on one segment. For each
    option holding another question's start ("N." with N a missing next number), the text from
    there on is cut out and parsed as question N -- stem, tags and its own (a)-(d) -- and
    inserted right after. NCERT / multi-year tags left inside an option are stripped (and given to
    the question if it has none). Every question touched is flagged; an option still far longer
    than its siblings is flagged too. Never drops text: whatever can't be placed stays where it was."""
    out = []
    for q in seg["questions"]:
        out.append(q)
        expected = _expected_numbers(seg, q)
        for letter in _LETTERS:
            text = q["options"].get(letter, "")
            m = next((m for m in _EMBEDDED_QNUM_RE.finditer(text) if int(m.group(1)) in expected), None)
            if not m:
                continue
            tail = text[m.end():].strip()
            first = _OPTION_MARKER_RE.search(tail)
            new_opts = split_option_markers(tail[first.start():])[1] if first else {}
            stem = tail[:first.start()] if first else tail
            if not stem.strip() and not new_opts:
                continue
            q["options"][letter] = text[:m.start()].strip()
            num = int(m.group(1))
            expected.discard(num)
            new_q = {
                "number": num, "question": "", "options": {}, "year_tag": None, "ncert_tag": None,
                "question_type": "mcq", "has_diagram": False, "page": q["page"], "continued": False,
                "issues": [f"split out of Q{q['number']}'s option ({letter}) automatically -- check the stem and options against the printed page"],
            }
            new_q["question"] = _strip_option_tags(new_q, _normalize_ws(stem))
            new_q["options"] = {l: _strip_option_tags(new_q, v) for l, v in new_opts.items()}
            new_q["question_type"] = _question_type(new_q["question"])
            q["issues"].append(f"option ({letter}) contained the start of Q{num}, which was split out into its own question -- check both")
            out.append(new_q)
    seg["questions"] = out
    for q in out:
        for letter in _LETTERS:
            if q["options"].get(letter):
                before = q["options"][letter]
                q["options"][letter] = _strip_option_tags(q, before)
                if q["options"][letter] != before:
                    q["issues"].append(f"option ({letter}) contained an NCERT or year tag, which was removed -- check the option")
                stray = [n.group(1) for n in _EMBEDDED_QNUM_RE.finditer(q["options"][letter])]
                if stray:
                    q["issues"].append(f"option ({letter}) contains what looks like a question number ({', '.join(s + '.' for s in stray)}) -- check it")
                tail = _TRAILING_TAG_RE.search(q["options"][letter])
                if tail and tail.start() > 0:
                    q["issues"].append(f"option ({letter}) ends with what looks like a year tag ({tail.group(0).strip()!r}) -- check it")
        lens = {l: len(v) for l, v in q["options"].items() if v}
        for letter, n in lens.items():
            others = [v for k, v in lens.items() if k != letter]
            if others and n >= _LONG_OPTION_MIN and n > _LONG_OPTION_RATIO * max(others):
                q["issues"].append(f"option ({letter}) is much longer than the other options -- it may hold another question's text")


def _strip_option_tags(q, text):
    m = _EMBEDDED_NCERT_RE.search(text)
    if m:
        tag = _NCERT_RE.search(m.group(0))
        if tag and not q.get("ncert_tag"):
            q["ncert_tag"] = re.sub(r'\s*/\s*', ' / ', _normalize_ws(tag.group(1)))
        if tag and not q.get("year_tag") and _YEAR_TAG_RE.match(_normalize_ws(tag.group(2))):
            q["year_tag"] = _normalize_ws(tag.group(2))
        text = _normalize_ws(text[:m.start()] + " " + text[m.end():])
    m = _EMBEDDED_YEAR_RE.search(text)
    # Not when the years are the option's whole text ("(a) 1953, 1958" is a real answer).
    if m and _normalize_ws(text[:m.start()] + text[m.end():]):
        if not q.get("year_tag"):
            q["year_tag"] = _normalize_ws(m.group(0))
        text = _normalize_ws(text[:m.start()] + " " + text[m.end():])
    return text


# ---------------------------------------------------------------- entry point

def scan_ocr_pdf_bytes(pdf_bytes, subject, ocr_fn=None, figures_fn=None):
    """Entry point for /admin/scan-pdf with layout="scanned_ocr". Same return shape as the other
    modes, plus "cost" (always $0) with timing. ocr_fn/figures_fn exist for tests."""
    if ocr_fn is None and not ocr_available():
        return {"error": "The free OCR engine isn't installed on this server (pip install rapidocr)."}
    ocr_fn = ocr_fn or ocr_page
    figures_fn = figures_fn or figure_boxes_1000

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    t_start = time.time()
    flagged, pages_items = [], []
    try:
        for idx, page in enumerate(doc):
            try:
                items, notes = segment_page(ocr_fn(page), figures_fn(page))
            except Exception as e:
                print(f"    Free OCR mode: error on page {idx + 1}: {e}")
                flagged.append({"page": idx + 1, "reason": f"OCR failed on this page: {e}. Nothing from this page is below."})
                items, notes = [], []
            flagged.extend({"page": idx + 1, "reason": n[0].upper() + n[1:]} for n in notes)
            ordered = order_ocr_items(items)
            pages_items.append((idx + 1, ordered))
            if not {it.get("kind") for it in ordered} & {"question", "continuation", "answer_key"} \
                    and not any(f["page"] == idx + 1 for f in flagged):
                flagged.append({"page": idx + 1, "reason": "No questions, continuations or answer key were found on this scanned page. If it does have questions, check it manually."})
        page_count = len(doc)
    finally:
        doc.close()
    secs = time.time() - t_start

    segments, page_flags = merge_document_items(pages_items)
    for seg in segments:
        _resplit_polluted(seg)
    questions, q_flags = finalize_segments(segments)

    # OCR-specific issues, attributed to whichever question each item ended up in (a
    # continuation's issue belongs to the question it was joined onto).
    issues_by_q = {}
    for page, items in pages_items:
        for it in items:
            if it.get("_ocr_issues") and "_question_number" in it:
                issues_by_q.setdefault((it["_question_number"], page), []).extend(it["_ocr_issues"])
    for (num, page), found in sorted(issues_by_q.items(), key=lambda kv: (kv[0][1], kv[0][0] or 0)):
        reason = "; ".join(found)
        for q in questions:
            if q["question_number"] == num and num is not None:
                q["needs_review"] = True
                q["review_reason"] = f"{q['review_reason']}; {reason}" if q["review_reason"] else reason
        label = f"Q{num}" if num is not None else "Unnumbered question"
        q_flags.append({"page": page, "reason": f"{label}: {reason}."})

    flagged.extend(page_flags)
    flagged.extend(q_flags)
    # Same detector as the Gemini mode, but the consequence differs: OCR never completes a cut-off
    # word (DISHA Q38 keeps the printed "photograpl"), so that mode's "the model may have filled
    # in cut-off words" wording would be wrong here.
    for f in cropped_edge_flags(pdf_cropped_edges(pdf_bytes), pages_items):
        f["reason"] = f["reason"].replace(
            "The model may have filled in cut-off words itself",
            "Words at that edge are cut off in the scan and were kept exactly as printed (not completed)")
        flagged.append(f)
    flagged.sort(key=lambda f: f["page"])

    cost = {"engine": OCR_ENGINE_NAME, "usd": 0.0, "pages_called": page_count, "pages_cached": 0,
            "seconds": round(secs, 1), "seconds_per_page": round(secs / page_count, 1) if page_count else 0}
    print(f"FREE OCR MODE: pages={page_count} secs={secs:.1f} cost=$0", flush=True)
    return {"questions": questions, "pages_scanned": page_count, "flagged_pages": flagged, "cost": cost}
