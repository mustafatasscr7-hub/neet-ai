"""Extraction mode for the "3-column coaching book" PDF layout (e.g. Disha/MTG-style chapter-wise
PYQ books), as opposed to process_pyq_vision.py's "Standard" mode (official NEET/AIPMT question
papers, 1-2 columns, numbered 1-4 options).

Deliberately a SEPARATE module, not a change to process_pyq_vision.py -- the two layouts need
different detection logic (3-way column split vs. 2-way; lettered (a)-(d) options living mostly in
ONE block per question vs. options genuinely fragmented across blocks; a page-printed "Ans. (x)"
to read directly vs. an answer that's never in the source at all) and mixing them risks exactly
the kind of cross-contamination already found and fixed once this session (see
process_pyq_vision.py's _BARE_OPTION_MARKER_RE comment for that history).

No Gemini/Claude call anywhere in this module. Diagnosis (2026-10-11/12, Chapter 3 - Plant
Kingdom Pages 18-30.pdf) found this layout's structure is mechanical, not something that needs a
model's judgement: every question's own number, stem, year tag and "Ans. (x)" are findable by
plain pattern. The two real wrinkles are both pure geometry, not ambiguous reads:
  1. A question's own lettered options sometimes don't fit in its own text frame and spill into
     one or more follow-on blocks, in strict reading order -- bottom of one column continues at
     the top of the next column, and the bottom of a page's last column continues at the top of
     the next page's first column.
  2. PyMuPDF occasionally merges an explanation's own tail directly onto the very next question's
     stem as ONE block (no blank-line signal strong enough to split them) -- confirmed live, Q21
     and Q22 both arrive this way, completely invisible to a check that only looks at a block's
     own first line.
"""
import re
import fitz

from process_pyq_vision import (
    _block_text_from_dict,
    _reconstruct_line_text,
    _cluster_columns,
    _MIN_DIAGRAM_DIM,
)

# A real question-start block is exactly as wide as ONE column's text frame (~155pt on the real
# PDF this was built against). The book's own running header/footer ("Plant Kingdom\n03" at the
# top of every page, "Plant Kingdom\n19" / "20\nNEET Chapterwise Topicwise Biology" alternating)
# spans the FULL page width instead (~300-500pt) -- and a bare running page number directly
# followed by a newline matches the exact same "zero-padded number then whitespace" shape
# _QNUM_RE looks for, so width is what tells them apart. Confirmed live: without this check, every
# page's own header/footer block got misread as an extra, bogus "question".
_MAX_QUESTION_BLOCK_WIDTH = 250

_QNUM_RE = re.compile(r'^(0[1-9]|[1-9][0-9])\s+(.*)$', re.S)
_QNUM_LINE_RE = re.compile(r'^(0[1-9]|[1-9][0-9])\s+\S')
# "[NEET 2021]", "[NEET (Oct.) 2020]", "[CBSE AIPMT 2014]", "[NEET 2016, Phase II]" -- captured
# as a whole (source_tag, minus the brackets) and the first real 4-digit year inside it kept
# separately (year). Confirmed live to appear ANYWHERE in a question's own block -- including
# mid-way through the lettered options, not just right after the stem -- so this is searched
# across the whole accumulated text, never assumed to sit in one fixed position.
_YEAR_TAG_RE = re.compile(r'\[([^\[\]]*)\]')
_YEAR_DIGITS_RE = re.compile(r'(?:19|20)\d{2}')
# \b before "Ans" is load-bearing, not decorative -- confirmed live, Q14: its own option (a)
# ends ".....mannans" and is immediately followed by "\n(b)...", and without the boundary this
# matched "ans" (the tail of "mannans") + the following newline + "(b)" as if it were a genuine
# "Ans. (b)" marker, truncating option (a) mid-word and fabricating a wrong answer letter.
_ANS_RE = re.compile(r'\bAns\.?\s*\(([a-dA-D])\)', re.I)
_OPTION_MARKER_RE = re.compile(r'\(([a-dA-D])\)')
# A follow-on block is recognized as "this question's own options/answer, just spilled into
# another text frame" by SHAPE alone -- it starts with its own lettered marker, with "Ans.", or
# with a year tag (confirmed live: a follow-on block can lead with "[CBSE AIPMT 1995, 2006]"
# before its own options, not just the options directly) -- never by position relative to the
# question.
_CONTINUATION_START_RE = re.compile(r'^\(?[a-dA-D]\)|^Ans\.?\s*\(|^\[', re.I)
# How many extra blocks a single question is allowed to pull in -- bounds the lookahead so a
# genuinely broken/empty stretch of the document can't make this scan every remaining block
# looking for a continuation that was never coming.
_MAX_FOLLOWUP_BLOCKS = 6


def _split_embedded_question_starts(block):
    """Returns a list of block-like dicts (each with its own 'bbox' and 'lines') -- normally just
    [block] unchanged, but if a NEW question's own number+stem starts MID-BLOCK, splits at each
    such internal start so the question underneath is still reachable by the normal
    question-start check. Confirmed live: PyMuPDF sometimes merges an explanation's own tail
    directly onto the very next question's stem as one block, with no blank-line signal strong
    enough to split them on its own (Plant Kingdom Q21 and Q22 both arrive this way -- Q21's
    entire number+stem+options+"Ans." sits glued onto the end of Q20's explanation, in the SAME
    block, invisible to any check that only looks at the block's own first line)."""
    lines = block.get("lines", [])
    if not lines:
        return [block]
    split_at = []
    for idx, line in enumerate(lines):
        if idx == 0:
            continue  # the block's own first line is what the normal top-level check already sees
        if _QNUM_LINE_RE.match(_reconstruct_line_text(line).strip()):
            split_at.append(idx)
    if not split_at:
        return [block]

    segments = []
    bounds = [0] + split_at + [len(lines)]
    for start, end in zip(bounds[:-1], bounds[1:]):
        seg_lines = lines[start:end]
        if not seg_lines:
            continue
        x0 = min(l["bbox"][0] for l in seg_lines)
        y0 = min(l["bbox"][1] for l in seg_lines)
        x1 = max(l["bbox"][2] for l in seg_lines)
        y1 = max(l["bbox"][3] for l in seg_lines)
        segments.append({"bbox": (x0, y0, x1, y1), "lines": seg_lines})
    return segments


def _cluster_all_columns(blocks, threshold=40):
    """Clusters ALL of a page's text blocks (not just question-starts) into left-to-right column
    groups by x0, each sorted top-to-bottom -- reuses process_pyq_vision's own generic
    column-clustering helper, unmodified. threshold=40 sits comfortably between the real
    within-column x0 jitter observed (a question at ~64pt, its own explanation at ~81pt, a
    diagram's scattered text labels anywhere up to ~470pt within the SAME 155pt-wide column) and
    the real gap between adjacent columns (~149pt on the PDF this was built against)."""
    if not blocks:
        return []
    cols = _cluster_columns(blocks, x_key=lambda b: b["bbox"][0], threshold=threshold)
    return [sorted(col, key=lambda b: b["bbox"][1]) for col in cols]


def _question_start_info(block):
    """Returns (qnum, text) if `block` looks like a real question start, else None. Width-filtered
    before the text pattern is even checked, so a wide running header never gets a chance to
    false-positive on its own leading page number (see _MAX_QUESTION_BLOCK_WIDTH)."""
    bbox = block["bbox"]
    if (bbox[2] - bbox[0]) > _MAX_QUESTION_BLOCK_WIDTH:
        return None
    text = _block_text_from_dict(block).strip()
    m = _QNUM_RE.match(text)
    return (int(m.group(1)), text) if m else None


def _parse_answer(text):
    m = _ANS_RE.search(text)
    if not m:
        return None, text
    letter = m.group(1).lower()
    # Nothing legitimate follows "Ans. (x)" in this layout (confirmed live across every sampled
    # question) -- drop it and anything after, not just the match itself.
    return letter, text[:m.start()]


def _parse_year_tag(text):
    m = _YEAR_TAG_RE.search(text)
    if not m:
        return None, None, text
    tag_full = m.group(1).strip()
    ym = _YEAR_DIGITS_RE.search(tag_full)
    year = int(ym.group(0)) if ym else None
    cleaned = text[:m.start()] + " " + text[m.end():]
    return tag_full, year, cleaned


def _normalize_ws(text):
    return re.sub(r'\s+', ' ', text).strip()


def _parse_options(options_blob):
    """Scans the WHOLE blob for every (a)-(d) marker regardless of line breaks, rather than
    parsing line-by-line -- this is what makes a 2x2 grid ("(a) X (b) Y" on one line, "(c) Z (d)
    W" on the next) and a wrapped option (continuation text with no marker of its own) both just
    fall out of the same logic for free: whatever text sits between one marker and the next
    belongs to the first one, line breaks or not."""
    matches = list(_OPTION_MARKER_RE.finditer(options_blob))
    options = {}
    for i, m in enumerate(matches):
        letter = m.group(1).lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(options_blob)
        content = _normalize_ws(options_blob[start:end])
        if letter not in options or len(content) > len(options[letter]):
            options[letter] = content
    return options


def _parse_question_text(full_text):
    """full_text is the question's own block PLUS any merged follow-on blocks, already joined.
    Returns a dict with qnum, question (stem), options (dict letter->text, may be missing keys),
    answer (letter or None), year, source_tag. Never raises on malformed input; a missing piece
    just comes back None/empty, left for the caller's flagging to catch."""
    m = _QNUM_RE.match(full_text)
    qnum = int(m.group(1))
    rest = m.group(2)

    answer, rest = _parse_answer(rest)
    source_tag, year, rest = _parse_year_tag(rest)

    first_marker = _OPTION_MARKER_RE.search(rest)
    if not first_marker:
        return {"qnum": qnum, "question": _normalize_ws(rest), "options": {}, "answer": answer, "year": year, "source_tag": source_tag}

    stem = _normalize_ws(rest[:first_marker.start()])
    options = _parse_options(rest[first_marker.start():])
    return {"qnum": qnum, "question": stem, "options": options, "answer": answer, "year": year, "source_tag": source_tag}


def _is_complete(parsed):
    return parsed["answer"] is not None and all(parsed["options"].get(l) for l in ("a", "b", "c", "d"))


def _build_question_dict(parsed, page_num, has_diagram):
    opts = parsed["options"]
    missing = [l for l in ("a", "b", "c", "d") if not opts.get(l)]
    needs_review = False
    review_reason = None
    if missing:
        needs_review = True
        review_reason = f"Option(s) {', '.join(l.upper() for l in missing)} came back empty for question {parsed['qnum']:02d} -- 3-column book parsing may have missed a marker or its continuation block."
    elif parsed["answer"] is None:
        needs_review = True
        review_reason = f"Could not find 'Ans. (x)' for question {parsed['qnum']:02d} -- the answer letter may be missing, in an unexpected format, or on a continuation block this scan didn't reach."
    elif not parsed["question"]:
        needs_review = True
        review_reason = f"Question {parsed['qnum']:02d}'s own stem came back empty -- parsing may have consumed the whole block as options."

    return {
        "question": parsed["question"],
        "option_a": opts.get("a", ""),
        "option_b": opts.get("b", ""),
        "option_c": opts.get("c", ""),
        "option_d": opts.get("d", ""),
        "correct_answer": parsed["answer"] or "",
        "question_type": "mcq",
        "has_diagram": has_diagram,
        "year": parsed["year"],
        "source_tag": parsed["source_tag"],
        "source_page": page_num,
        "needs_review": needs_review,
        "review_reason": review_reason,
    }


def _get_diagram_bboxes(page):
    """Raster + vector diagram bounding boxes for ONE page, in PDF point coordinates. Reuses
    process_pyq_vision's own DocLayout-YOLO primitives (unmodified) for the vector pass -- this
    book's diagrams (life-cycle diagrams, structural drawings) are drawn with native PDF vector
    primitives, confirmed live (zero raster images on most pages, 10-220 vector drawing objects
    on every page), so the raster-only pass most PDFs would need is not enough here on its own.
    Never fatal: any failure (model unavailable, inference error) just means diagram attribution
    falls back to raster-only for this page, same fallback contract as the function being reused
    from."""
    bboxes = []
    for xref in {img[0] for img in page.get_images(full=True)}:
        for rect in page.get_image_rects(xref):
            if min(rect.width, rect.height) >= _MIN_DIAGRAM_DIM:
                bboxes.append((rect.x0, rect.y0, rect.x1, rect.y1))

    try:
        from process_pyq_vision import _get_yolo_session, _yolo_infer_figure_boxes, _YOLO_DPI
        from PIL import Image
        import numpy as np
        import cv2

        session = _get_yolo_session()
        pix = page.get_pixmap(dpi=_YOLO_DPI)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        arr = np.array(img)
        scale = 72.0 / _YOLO_DPI
        for x0, y0, x1, y1 in _yolo_infer_figure_boxes(session, arr, cv2, np):
            bboxes.append((x0 * scale, y0 * scale, x1 * scale, y1 * scale))
    except Exception as e:
        print(f"    3-column book mode: vector diagram detection unavailable for this page, raster-only: {e}")

    return bboxes


def _bbox_overlaps(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0


def _union_bbox(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _find_numbering_gaps(qnum_page):
    """qnum_page: {question number -> the page it was found on}. Returns a list of
    {"page", "reason"} dicts, one per integer missing from the 1..max(found) sequence -- the
    book numbers questions with no gaps, so any missing integer means a question was never even
    found as a block anywhere in the document (confirmed live: Q21 and Q22, before the
    embedded-question-start fix in _split_embedded_question_starts, were silently absent with no
    trace in flagged_pages at all). Never guesses the missing question's own content -- only
    where to go look for it: the nearest-following found question's own page (a gap sits right
    before whatever comes next in numbering order), or the last found page if the gap is at the
    very end, since the admin UI's banner needs a real page number and that's exactly where a
    human would look for the missing question anyway."""
    if not qnum_page:
        return []
    found_sorted = sorted(qnum_page)
    gaps = []
    for missing_num in sorted(set(range(1, found_sorted[-1] + 1)) - set(found_sorted)):
        next_found = next((num for num in found_sorted if num > missing_num), None)
        nearby_page = qnum_page[next_found] if next_found is not None else qnum_page[found_sorted[-1]]
        gaps.append({
            "page": nearby_page,
            "reason": f"Question {missing_num:02d} was never found anywhere in the PDF's text layer -- its own number may be merged into another block, or it may be missing from this scan range. Use \"+ Add Missed Question\" to add it manually.",
        })
    return gaps


def _build_reading_order(pdf_bytes):
    """Returns (items, diagram_bboxes_by_page, page_count). `items` is the WHOLE document's text
    blocks flattened into one true reading-order sequence: page by page, column by column
    (left-to-right), top-to-bottom within each column -- so a question's own options/answer that
    spill past the bottom of one column, the bottom of the page's last column, or even onto the
    next page are all just "the next few items in this one sequence," with no separate
    same-column/next-column/cross-page special-casing needed. Each item is
    {"page": 1-indexed page number, "block": block-like dict with bbox+lines}.

    Blocks are run through _split_embedded_question_starts first, so a question whose own number
    got merged into the tail of the PRECEDING explanation's block (see that function's docstring)
    is still its own reachable item here, in its correct position."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = len(doc)
    items = []
    diagram_bboxes_by_page = []
    for page_num in range(page_count):
        page = doc[page_num]
        raw_blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type") == 0]
        expanded = [seg for b in raw_blocks for seg in _split_embedded_question_starts(b)]
        for col in _cluster_all_columns(expanded):
            for block in col:
                items.append({"page": page_num + 1, "block": block})
        diagram_bboxes_by_page.append(_get_diagram_bboxes(page))
    doc.close()
    return items, diagram_bboxes_by_page, page_count


def scan_book_pdf_bytes(pdf_bytes, subject):
    """Callable entry point for the admin review pipeline's "3-column book" layout option
    (server.py's /admin/scan-pdf with layout="3col_book"). Returns the exact same shape as
    process_pyq_vision.scan_pdf_bytes -- {"questions": [...], "pages_scanned": N,
    "flagged_pages": [...]} -- so it drops straight into the existing review-before-save UI with
    no changes there. No model call anywhere: see this module's own docstring for why."""
    items, diagram_bboxes_by_page, page_count = _build_reading_order(pdf_bytes)

    questions = []
    flagged_pages = []
    qnum_page = {}  # qnum -> the page it was found on, also used to place a gap's flag sensibly
    n = len(items)
    i = 0

    while i < n:
        block = items[i]["block"]
        info = _question_start_info(block)
        if not info:
            i += 1
            continue

        qnum, text_acc = info
        q_page = items[i]["page"]
        qnum_page[qnum] = q_page
        q_bbox = block["bbox"]
        parsed = _parse_question_text(text_acc)

        j = i + 1
        followups = 0
        while not _is_complete(parsed) and j < n and followups < _MAX_FOLLOWUP_BLOCKS:
            cand_block = items[j]["block"]
            if _question_start_info(cand_block):
                break  # the next real question -- whatever's still missing stays missing, flagged below
            cand_bbox = cand_block["bbox"]
            if (cand_bbox[2] - cand_bbox[0]) > _MAX_QUESTION_BLOCK_WIDTH:
                # A running header/footer between this question and its own continuation
                # (confirmed live: page 2's "Plant Kingdom\n19" sits directly before Q07's
                # actual "Ans. (b)" continuation) -- skipped over, not treated as a stopping
                # point and not counted against the follow-up budget.
                j += 1
                continue
            cand_text = _block_text_from_dict(cand_block).strip()
            if not _CONTINUATION_START_RE.match(cand_text):
                break  # genuine explanation prose -- stop, leave it for the outer loop to skip
            text_acc = text_acc + " " + cand_text
            q_bbox = _union_bbox(q_bbox, cand_bbox)
            parsed = _parse_question_text(text_acc)
            j += 1
            followups += 1

        diagram_bboxes = diagram_bboxes_by_page[q_page - 1]
        has_diagram = any(_bbox_overlaps(q_bbox, d) for d in diagram_bboxes)
        q = _build_question_dict(parsed, q_page, has_diagram)
        questions.append(q)
        if q["needs_review"]:
            flagged_pages.append({"page": q_page, "reason": q["review_reason"]})

        i = j

    # Final safety net -- never guesses a missing question's content, just surfaces the gap by
    # number so "+ Add Missed Question" can be used instead of it going unnoticed.
    flagged_pages.extend(_find_numbering_gaps(qnum_page))

    flagged_pages.sort(key=lambda f: f["page"])
    return {"questions": questions, "pages_scanned": page_count, "flagged_pages": flagged_pages}
