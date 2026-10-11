"""Extraction mode for the "3-column coaching book" PDF layout (e.g. Disha/MTG-style chapter-wise
PYQ books), as opposed to process_pyq_vision.py's "Standard" mode (official NEET/AIPMT question
papers, 1-2 columns, numbered 1-4 options).

Deliberately a SEPARATE module, not a change to process_pyq_vision.py -- the two layouts need
different detection logic (3-way column split vs. 2-way; lettered (a)-(d) options living mostly in
ONE block per question vs. options genuinely fragmented across blocks; a page-printed "Ans. (x)"
to read directly vs. an answer that's never in the source at all) and mixing them risks exactly
the kind of cross-contamination already found and fixed once this session (see
process_pyq_vision.py's _BARE_OPTION_MARKER_RE comment for that history).

No Gemini/Claude call anywhere in this module. Diagnosis (2026-10-11, Chapter 3 - Plant Kingdom
Pages 18-30.pdf) found this layout's structure is mechanical, not something that needs a model's
judgement: every question's own number, stem, year tag and "Ans. (x)" are findable by plain
pattern, and the publisher's explanation paragraph is always a separate block that follows. The
one genuine wrinkle (see _looks_like_continuation below) is that a question's own lettered options
sometimes don't fit in its own text frame and spill into one or more follow-on blocks -- sometimes
onto the very next page -- which a model call wouldn't actually help with either, since it's a
layout fact, not an ambiguous read.
"""
import re
import fitz

from process_pyq_vision import (
    _block_text_from_dict,
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
# question (confirmed live: this can be the very next block in the same column, one of several
# such blocks in a row, or the first block at the top of the NEXT page's same column after the
# current page's column ran out of room).
_CONTINUATION_START_RE = re.compile(r'^\(?[a-dA-D]\)|^Ans\.?\s*\(|^\[', re.I)
# How many extra blocks a single question is allowed to pull in across the page boundary --
# bounds the cross-page lookahead so a genuinely broken/empty page can't make this scan every
# block of the rest of the document looking for a continuation that was never coming.
_MAX_FOLLOWUP_BLOCKS = 6


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


def _consume_followups(text_acc, q_bbox, parsed, blocks, start_idx, budget):
    """Appends blocks[start_idx:] one at a time into text_acc as long as each next one is both
    continuation-shaped (see _CONTINUATION_START_RE) and not itself a new question start,
    re-parsing after each -- stopping as soon as the result is complete, the shape check fails,
    or `budget` is exhausted. Returns (text_acc, q_bbox, parsed, consumed_count)."""
    k = 0
    while not _is_complete(parsed) and start_idx + k < len(blocks) and k < budget:
        block = blocks[start_idx + k]
        candidate_text = _block_text_from_dict(block).strip()
        if _question_start_info(block) or not _CONTINUATION_START_RE.match(candidate_text):
            break
        text_acc = text_acc + " " + candidate_text
        bbox = block["bbox"]
        q_bbox = (min(q_bbox[0], bbox[0]), min(q_bbox[1], bbox[1]), max(q_bbox[2], bbox[2]), max(q_bbox[3], bbox[3]))
        parsed = _parse_question_text(text_acc)
        k += 1
    return text_acc, q_bbox, parsed, k


def _bbox_overlaps(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0


def scan_book_pdf_bytes(pdf_bytes, subject):
    """Callable entry point for the admin review pipeline's "3-column book" layout option
    (server.py's /admin/scan-pdf with layout="3col_book"). Returns the exact same shape as
    process_pyq_vision.scan_pdf_bytes -- {"questions": [...], "pages_scanned": N,
    "flagged_pages": [...]} -- so it drops straight into the existing review-before-save UI with
    no changes there. No model call anywhere: see this module's own docstring for why."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = len(doc)

    # Pass 1: per page, per column (left-to-right, each top-to-bottom) -- every block, not just
    # question-starts, so a follow-on options/answer block or a diagram's own stray text label
    # is available for the merge step below in its own correct reading-order position.
    page_columns = []
    page_diagram_bboxes = []
    for page_num in range(page_count):
        page = doc[page_num]
        blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type") == 0]
        page_columns.append(_cluster_all_columns(blocks))
        page_diagram_bboxes.append(_get_diagram_bboxes(page))
    doc.close()

    questions = []
    flagged_pages = []
    # Tracks, per (page_index, column_index), how many of that column's LEADING blocks have
    # already been consumed as a cross-page continuation by a question from the previous page --
    # so this page's own first question doesn't also try to read them as part of ITS stem.
    consumed_leading = {}

    for page_num in range(page_count):
        columns = page_columns[page_num]
        diagram_bboxes = page_diagram_bboxes[page_num]

        for col_idx, col_blocks in enumerate(columns):
            start_at = consumed_leading.get((page_num, col_idx), 0)
            i = start_at
            while i < len(col_blocks):
                info = _question_start_info(col_blocks[i])
                if not info:
                    i += 1
                    continue
                qnum, text_acc = info
                q_bbox = col_blocks[i]["bbox"]
                parsed = _parse_question_text(text_acc)
                j = i + 1

                # Same-page, same-column follow-on blocks (Plant Kingdom Q05/Q10's shape: the
                # question's own block has only the stem, and one or more later blocks in the
                # same column hold "(a)...(d)\nAns.(x)", sometimes split across 2 blocks).
                text_acc, q_bbox, parsed, used = _consume_followups(text_acc, q_bbox, parsed, col_blocks, j, _MAX_FOLLOWUP_BLOCKS)
                j += used

                # Cross-page: this question ran out of blocks in its own column on this page and
                # STILL isn't complete. Which column of the NEXT page the reader's eye actually
                # continues into turns out NOT to be consistent -- confirmed live, Plant Kingdom:
                # some questions continue at the same column index on the next page, but Q07
                # specifically continues at the next page's FIRST column instead (its own column
                # was the page's last, so the reader's eye goes to the top of the next page
                # entirely). Rather than guess one rule, try the same index first, then column 0
                # (when different), and use whichever one's leading block actually looks like a
                # continuation -- a real question start there blocks the wrong guess immediately.
                if not _is_complete(parsed) and j >= len(col_blocks) and page_num + 1 < page_count:
                    next_cols = page_columns[page_num + 1]
                    for candidate_idx in dict.fromkeys([col_idx, 0]):  # dedupe, preserve order
                        if candidate_idx >= len(next_cols):
                            continue
                        next_blocks = next_cols[candidate_idx]
                        # The next page's own running header/footer ("Plant Kingdom\n19") sits
                        # at the very top of column 0 and matches neither the question-start nor
                        # the continuation shape -- it would otherwise block this search before
                        # it even reaches the real continuation block right after it (confirmed
                        # live: Q07's "Ans. (b)" is the SECOND block in page 2's column 0, not
                        # the first). Skipped here by the same width test as a real question
                        # start (a header spans the full page width; a real continuation block
                        # is exactly one column wide), not by position alone.
                        skip = 0
                        while skip < len(next_blocks) and (next_blocks[skip]["bbox"][2] - next_blocks[skip]["bbox"][0]) > _MAX_QUESTION_BLOCK_WIDTH:
                            skip += 1
                        text_acc, q_bbox, parsed, used = _consume_followups(text_acc, q_bbox, parsed, next_blocks, skip, _MAX_FOLLOWUP_BLOCKS)
                        if used > 0:
                            consumed_leading[(page_num + 1, candidate_idx)] = max(consumed_leading.get((page_num + 1, candidate_idx), 0), skip + used)
                            break

                has_diagram = any(_bbox_overlaps(q_bbox, d) for d in diagram_bboxes)
                q = _build_question_dict(parsed, page_num + 1, has_diagram)
                questions.append(q)
                if q["needs_review"]:
                    flagged_pages.append({"page": page_num + 1, "reason": q["review_reason"]})

                i = j

    flagged_pages.sort(key=lambda f: f["page"])
    return {"questions": questions, "pages_scanned": page_count, "flagged_pages": flagged_pages}
