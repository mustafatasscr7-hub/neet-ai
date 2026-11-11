"""Deterministic (NO model call, ever) extraction for the Mock exam PDF layout -- a full
NEET-pattern mock paper (Physics 1-45 / Chemistry 1-45 / Biology 1-90, each subject's own
numbering restarting at 1, two text columns per page) uploaded as one PDF with no known page
ranges up front.

Replaces the old process_pyq_vision.scan_mock_exam_pdf, which fed each page's text to
extract_questions_from_text (a model call) for the actual question/option split. Diagnosis
(2026-11-11, MockTest1.pdf) found that call routinely read half a question or skipped whole
questions on this layout's heavy inline math -- and a model call is also simply not needed here:
every question's own number and lettered (a)-(d) options are findable by plain geometry/regex,
the same conclusion process_pyq_book_3col.py reached for the "3-column coaching book" layout. This
module is that same approach, built fresh for THIS layout's own shape (2 columns not 3, letter
options "a." not "(a)", no inline answer to parse, numbering restarts per subject, a repeated
instructions page and a per-subject "Answers and Solutions" block to skip).

Design, in one pass over the whole document:
  1. Reject up front (no model fallback) if the PDF has no real selectable text layer at all --
     a scanned mock exam needs a different tool, not a silent wrong answer here.
  2. Walk every page's text BLOCKS (PyMuPDF get_text("dict"), type 0 only) in column order (left
     half of the page top-to-bottom, then right half) -- a running header/footer block is
     recognized and skipped purely by its WIDTH (spans close to the full page, every real
     question/option block does not), same trick process_pyq_book_3col.py uses for its own
     header/footer.
  3. A block starting with "<number>." is a new question IF that number is strictly greater than
     the highest number already accepted for the CURRENT subject -- this one rule does triple
     duty: it is what makes numbering restart per subject work, what silently absorbs an embedded
     numbered sub-list (a Match-the-following/Assertion-Reason item list reusing small numbers)
     into the question it belongs to instead of misreading it as a new question, and what leaves
     a real gap (a question whose own marker this scan never recognized) as a genuine, reportable
     gap instead of silently renumbering everything after it.
  4. Every block from one accepted question-start up to (not including) the next is that
     question's own content, no exceptions -- there is no inline explanation/answer text in this
     layout to skip over mid-question (unlike the coaching-book layout), so this stays simple.
  5. Within a question's accumulated text, (a)-(d) is found by scanning for every line starting
     "<letter>. " and assigning whatever follows (up to the next such line) to that letter -- the
     same "scan the whole blob for markers, letter->content" shape process_pyq_book_3col.py uses
     for its own "(a)"-"(d)" markers.
  6. Three accuracy gates (requirement 2): question numbers run 1..N with no gaps (reported as
     "missing"); every question has exactly 4 non-empty options; and word coverage -- a block is
     "covered" the instant it becomes part of SOME question's content in step 4, so coverage loss
     only ever comes from content sitting before a subject's very first recognized question-start
     (the one place content can still go unclaimed) or from a whole subject never being detected.
     A question failing any gate, or whose own text reads as leftover fragmented math (a high
     ratio of stray single-character tokens -- the signature this layout's heaviest formulas leave
     behind when a marker/fraction gets split across many tiny spans), is flagged needs_review
     with a human-readable reason and a cropped image of its own source region, rather than ever
     dropped silently.
"""
import re
import io
import base64

import fitz

from process_pyq_vision import (
    _reconstruct_line_text,
    _block_text_from_dict,
    _flag_suspicious_mcq_options,
    MOCK_TEST_SUBJECT_ORDER,
)
from process_pyq_book_3col import _get_diagram_bboxes, _bbox_overlaps, _union_bbox
from process_mock_exam_math import build_option_zones, assign_spans_to_zones, reconstruct_zone

MOCK_EXAM_SUBJECT_COUNTS = {"Physics": 45, "Chemistry": 45, "Biology": 90}
MOCK_EXAM_SUBJECT_OFFSET = {"Physics": 0, "Chemistry": 45, "Biology": 90}

REJECT_NO_TEXT_LAYER = "This PDF has no selectable text. Mock exams must be text PDFs."
# A whole document's real text-layer pages run well into the thousands of characters even for a
# single page of this layout (see MockTest1.pdf: 1500-3600 chars/page) -- a genuinely scanned PDF
# (image pages, no text layer at all) comes back with ZERO characters from get_text() on every
# page, not just few. 50 total chars across the WHOLE document is comfortably above the stray
# watermark/metadata text a scanned PDF can still carry, and comfortably below any real page.
_MIN_DOC_TEXT_CHARS = 50

# Same reasoning and same regexes process_pyq_vision.py's own mock-exam section used (kept
# unchanged here on purpose -- requirement 1 says keep the existing ignore rules): a repeated
# instructions/cover page, a per-subject "Answers and Solutions" block, and the running
# Physics/Chemistry/Biology header that may or may not have a space before its own page number.
_INSTRUCTIONS_RE = re.compile(
    r"PAPER CONSISTS|Do not open this Test Booklet|read the instructions carefully", re.IGNORECASE)
_ANSWERS_HEADING_RE = re.compile(r"Answers\s+and\s+Solutions", re.IGNORECASE)
_SUBJECT_HEADER_RE = re.compile(r"\b(Physics|Chemistry|Biology)", re.IGNORECASE)

# A real question-start or option-content block is one column wide. The running header/footer
# (subject name + page number, or page number + "Mock Test-1") spans close to the full page width
# instead -- confirmed live on MockTest1.pdf: ~497pt wide vs. a real column's own blocks topping
# out well under 300pt on a ~595pt-wide page.
_MAX_HEADER_FOOTER_WIDTH = 270

_QSTART_RE = re.compile(r'^(\d{1,3})[.)]\s*(.*)$', re.S)
# Matches a marker at the true start of the accumulated text OR anywhere preceded by whitespace
# (not just a line start) -- so "a. X b. Y c. Z d. W" all on one physical line (requirement 1's
# "inline on one line" option layout) splits correctly too, not just the one-per-line/2x2-grid
# shapes a line-start-only anchor would have been limited to.
_OPTION_MARKER_RE = re.compile(r'(?:^|(?<=\s))([a-dA-D])[.)]\s+')
_OPTION_LETTERS = ("a", "b", "c", "d")

# Fragmented math (a fraction/exponent/vector split by PyMuPDF into many tiny single-glyph spans,
# with no marker-aware reconstruction attempted for THIS layout's letter options -- see this
# module's own docstring) leaves behind a text run with an unusually high share of stray
# single-character tokens ("2", "1", "-", "W", "m", "K" instead of "W m^-2 K^-4"). A genuine short
# sentence/option almost never crosses this ratio; a real sample of garbled vs. clean questions on
# MockTest1.pdf confirmed 0.25 cleanly separates them.
_GARBLED_TOKEN_RATIO = 0.25
_GARBLED_MIN_TOKENS = 6


def _normalize_ws(text):
    return re.sub(r'\s+', ' ', text or '').strip()


def _page_blocks(page):
    return [b for b in page.get_text("dict")["blocks"] if b.get("type") == 0]


# Besides the full-width running header/footer block, this layout also drops a SECOND, narrow
# "Mock Test-1" caption (and sometimes a bare page number alone) as its own small text object
# floating partway down the page, near the column gutter -- confirmed live, MockTest1.pdf page 16
# (0-indexed 15): a "Mock Test-1 " block at x0=264, sorted by y0 to land BETWEEN question 1 and
# question 2 in the left column, which silently glued it onto question 1's own last option with
# no width signal to catch it (that block is nowhere near the full-page width the running
# header/footer is). Caught by content instead: a block whose ENTIRE text is just that caption
# and/or a bare 1-3 digit page number is never real question/option content either way.
_STRAY_CAPTION_RE = re.compile(r'^(?:mock\s*test-?\s*1\.?\s*)?\d{0,3}\s*(?:mock\s*test-?\s*1\.?\s*)?$', re.IGNORECASE)

# A real page-number caption is set at ordinary running-text size, same as the rest of the page's
# own prose -- a markedly smaller block that ALSO happens to be bare digits is far more likely a
# superscript/subscript exponent that PyMuPDF isolated into its own tiny block (confirmed live via
# a synthetic math-reconstruction test: a raised "2" for "m^2" got silently discarded here,
# matching _STRAY_CAPTION_RE's own bare "\d{0,3}" branch, before this guard existed). Set below
# every real exponent size confirmed live on the real sample (5.8-6.2) and below this layout's own
# body text (9.9-10.1), with headroom for a smaller-but-still-real caption font on some other PDF.
_MIN_CAPTION_FONT_SIZE = 8.0


def _is_header_footer(block):
    bbox = block["bbox"]
    if (bbox[2] - bbox[0]) > _MAX_HEADER_FOOTER_WIDTH:
        return True
    text = _normalize_ws(_block_text_from_dict(block))
    if not text or not _STRAY_CAPTION_RE.match(text):
        return False
    if text.isdigit():
        sizes = [s["size"] for line in block.get("lines", []) for s in line.get("spans", [])
                 if (s.get("text") or "").strip()]
        if sizes and max(sizes) < _MIN_CAPTION_FONT_SIZE:
            return False  # too small to be a real page-number caption -- likely an exponent
    return True


# A new question's own number+stem sometimes starts MID-BLOCK: PyMuPDF merges the tail of one
# question's last option directly onto the very next question's own opening line as ONE block
# when there's no blank-line gap wide enough for it to split them on its own -- confirmed live,
# MockTest1.pdf page 5 (0-indexed 4): one block holds "...d. L / 2 \n44. Choke coil is used to
# control \na. AC\n...d. Neither AC nor DC\n45. In a series circuit...", i.e. question 43's last
# option, ALL of question 44, and the start of question 45. Same failure shape
# process_pyq_book_3col.py's own _split_embedded_question_starts fixes for its layout -- adapted
# here with this layout's own marker punctuation ("44." not "44 ").
# No trailing \S required: a marker often sits completely alone on its own line, with the stem
# only starting on the NEXT line (confirmed live, Physics Q21 and Biology Q9 on MockTest1.pdf --
# both merged invisibly into the PRECEDING question's block until this was relaxed to match).
_QSTART_LINE_RE = re.compile(r'^(\d{1,3})[.)](?:\s|$)')


def _split_embedded_question_starts(block):
    lines = block.get("lines", [])
    if not lines:
        return [block]
    split_at = [idx for idx, line in enumerate(lines) if idx > 0 and _QSTART_LINE_RE.match(_reconstruct_line_text(line).strip())]
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


def _trim_block_before_heading(block):
    """The very last question before a subject's own "Answers and Solutions" heading routinely
    has its own last option(s) merged into the SAME block as that heading text (confirmed live,
    MockTest1.pdf: "...d. 400 Ω \\n \\nAnswers and Solutions" is one block) -- dropping that whole
    block, as a plain per-block cutoff would, silently throws away that question's real option
    text along with the heading. Returns a copy of `block` with only the lines BEFORE the heading
    kept, or None if the heading is the block's own first line (nothing worth keeping)."""
    lines = block.get("lines", [])
    keep = []
    for line in lines:
        if _ANSWERS_HEADING_RE.search(_reconstruct_line_text(line)):
            break
        keep.append(line)
    if not keep:
        return None
    x0 = min(l["bbox"][0] for l in keep)
    y0 = min(l["bbox"][1] for l in keep)
    x1 = max(l["bbox"][2] for l in keep)
    y1 = max(l["bbox"][3] for l in keep)
    return {"bbox": (x0, y0, x1, y1), "lines": keep}


def _doc_has_text_layer(doc):
    total = 0
    for page in doc:
        total += len(page.get_text().strip())
        if total >= _MIN_DOC_TEXT_CHARS:
            return True
    return False


_NUM_MARKER_SPAN_RE = re.compile(r'^(\d{1,3})[.)]$')


def _find_number_marker_span_id(block):
    """The question-number marker ("1.") is confirmed live to always be its own span (same shape
    as a lettered option marker, see process_mock_exam_math.MARKER_RE) -- found here so the math
    reconstruction pass can exclude it from the stem's own glyph pool the same way _question_start
    already excludes it from the plain-text pool."""
    for line in block.get("lines", []):
        for s in line.get("spans", []):
            if _NUM_MARKER_SPAN_RE.match((s.get("text") or "").strip()):
                return id(s)
    return None


def _question_start(block):
    """Returns (number, rest_of_text_after_the_marker) if `block`'s own text starts with a bare
    "<number>." marker, else None. Never matched against a decimal fraction mid-sentence -- this
    only ever looks at a BLOCK's own first characters, and a real question-start block in this
    layout's PDF structure always opens its own dedicated block (confirmed live, see this module's
    own docstring)."""
    text = _block_text_from_dict(block).strip()
    m = _QSTART_RE.match(text)
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def _cluster_two_columns(blocks, mid_x):
    left = sorted([b for b in blocks if b["bbox"][0] < mid_x], key=lambda b: b["bbox"][1])
    right = sorted([b for b in blocks if b["bbox"][0] >= mid_x], key=lambda b: b["bbox"][1])
    return left + right


def _word_count(block):
    return len(_block_text_from_dict(block).split())


def _parse_options(text_acc):
    """Scans the WHOLE accumulated question text for every "<letter>. " marker at the start of a
    line, regardless of how many line breaks separate them -- same shape as
    process_pyq_book_3col.py's own _parse_options for "(a)"-"(d)" markers: whatever text sits
    between one marker and the next belongs to the first one. On a duplicate letter (seen live,
    e.g. a stray "b." line from a fragmented math option), the LONGER candidate wins -- the
    shorter one is far more likely to be a bare leftover marker with nothing real after it."""
    matches = list(_OPTION_MARKER_RE.finditer(text_acc))
    if not matches:
        return _normalize_ws(text_acc), {}
    stem = _normalize_ws(text_acc[:matches[0].start()])
    options = {}
    for i, m in enumerate(matches):
        letter = m.group(1).lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text_acc)
        content = _normalize_ws(text_acc[start:end])
        if letter not in options or len(content) > len(options[letter]):
            options[letter] = content
    return stem, options


def _flatten_question_spans(active):
    """All of this question's own glyphs (across every block it accumulated, possibly spanning
    more than one page if it started near a page's bottom), minus the question-number marker
    itself -- the raw material for math reconstruction. Each span is tagged with its own page
    index so reconstruct_question_math can look up that page's own fraction-bar drawings."""
    flat = []
    for page_index, block in active.get("raw_blocks", []):
        for line in block.get("lines", []):
            for s in line.get("spans", []):
                if not s.get("text"):
                    continue
                if id(s) == active.get("number_marker_span_id"):
                    continue
                tagged = dict(s)
                tagged["page_index"] = page_index
                flat.append(tagged)
    return flat


def reconstruct_question_math(doc, active, drawings_cache):
    """Tries the geometry-aware zone reconstruction (process_mock_exam_math) first; falls back to
    the old plain-text _parse_options whenever build_option_zones can't unambiguously find this
    question's own 4 lettered markers (a genuinely different layout, or a question this module's
    own marker detection doesn't recognize -- always safer left to the existing text-based path
    and its own existing needs_review gates than forced through a zone split built on a wrong
    assumption, per process_mock_exam_math's own docstring)."""
    flat_spans = _flatten_question_spans(active)
    result = build_option_zones(flat_spans, active["bbox"])
    if result is None:
        return _parse_options(active["text_acc"])
    stem_zone, option_zones, marker_ids = result
    buckets = assign_spans_to_zones(flat_spans, stem_zone, option_zones, marker_ids)

    def _drawings_for(spans):
        pages = {s["page_index"] for s in spans}
        out = []
        for pi in pages:
            if pi not in drawings_cache:
                drawings_cache[pi] = doc[pi].get_drawings()
            out.extend(drawings_cache[pi])
        return out

    stem_text = reconstruct_zone(buckets["stem"], _drawings_for(buckets["stem"]), stem_zone)
    options = {
        letter: reconstruct_zone(buckets[letter], _drawings_for(buckets[letter]), option_zones[letter])
        for letter in ("a", "b", "c", "d")
    }
    return stem_text, options


def _looks_math_garbled(text):
    tokens = (text or "").split()
    if len(tokens) < _GARBLED_MIN_TOKENS:
        return False
    stray = sum(1 for t in tokens if len(t) <= 2 and not t.isalpha())
    return (stray / len(tokens)) > _GARBLED_TOKEN_RATIO


def _crop_question_image(doc, page_index, bbox, margin=8):
    """Base64 PNG crop of a flagged question's own source region, for the reviewer to compare
    directly against the parsed text -- only ever called for a flagged question (requirement 2),
    never for every question, so a clean scan's response stays small."""
    try:
        page = doc[page_index]
        rect = (fitz.Rect(bbox) + (-margin, -margin, margin, margin)) & page.rect
        pix = page.get_pixmap(dpi=150, clip=rect)
        return base64.standard_b64encode(pix.tobytes("png")).decode("ascii")
    except Exception as e:
        print(f"    Mock exam scan: could not crop a flagged question's image ({e})")
        return None


def _finalize_question(doc, active, mock_name, drawings_cache):
    stem, opts = reconstruct_question_math(doc, active, drawings_cache)
    missing = [l for l in _OPTION_LETTERS if not opts.get(l)]
    needs_review = False
    review_reason = None
    if missing:
        needs_review = True
        review_reason = (
            f"Option(s) {', '.join(l.upper() for l in missing)} came back empty for "
            f"{active['subject']} question {active['printed_number']} -- a marker may have been "
            f"missed or its content fragmented across the page."
        )
    elif not stem:
        needs_review = True
        review_reason = f"{active['subject']} question {active['printed_number']}'s own stem came back empty."
    else:
        garbled_fields = [name for name, val in (("stem", stem), *((f"option {l.upper()}", opts.get(l, "")) for l in _OPTION_LETTERS)) if _looks_math_garbled(val)]
        if garbled_fields:
            needs_review = True
            review_reason = (
                f"{active['subject']} question {active['printed_number']}: leftover fragmented "
                f"math glyphs detected in {', '.join(garbled_fields)} -- please clean up against the PDF."
            )

    q = {
        "question": stem,
        "option_a": opts.get("a", ""),
        "option_b": opts.get("b", ""),
        "option_c": opts.get("c", ""),
        "option_d": opts.get("d", ""),
        "correct_answer": "",
        "question_type": "mcq",
        "has_diagram": bool(active.get("has_diagram")),
        "year": None,
        "source_tag": None,
        "source_page": active["start_page"],
        "needs_review": needs_review,
        "review_reason": review_reason,
        "subject": active["subject"],
        "source": "mock",
        "mock_name": mock_name,
        "mock_position": MOCK_EXAM_SUBJECT_OFFSET[active["subject"]] + active["printed_number"],
        "printed_number": active["printed_number"],
    }
    _flag_suspicious_mcq_options(q)  # belt-and-suspenders: catches merged/duplicate options too
    if q["needs_review"]:
        q["crop_image_b64"] = _crop_question_image(doc, active["start_page"] - 1, active["bbox"])
    return q


def _missing_numbers(found, expected):
    return sorted(set(range(1, expected + 1)) - found)


def scan_mock_exam_pdf(pdf_bytes, mock_name):
    """Callable entry point for /admin/scan-pdf's layout="mock_exam". Same output shape as the
    old process_pyq_vision.scan_mock_exam_pdf -- {"questions", "pages_scanned", "flagged_pages",
    "completeness", "mock_name"} -- so it drops straight into the existing review UI. No model
    call anywhere in this module or anything it imports is ever invoked -- see this module's own
    docstring for why one isn't needed."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if not _doc_has_text_layer(doc):
        doc.close()
        raise ValueError(REJECT_NO_TEXT_LAYER)

    current_subject = None
    last_num = {s: 0 for s in MOCK_TEST_SUBJECT_ORDER}
    found_numbers = {s: set() for s in MOCK_TEST_SUBJECT_ORDER}
    coverage = {s: {"assigned": 0, "total": 0} for s in MOCK_TEST_SUBJECT_ORDER}
    per_subject_questions = {s: [] for s in MOCK_TEST_SUBJECT_ORDER}
    flagged_pages = []
    active = None  # the question currently being accumulated, or None between subjects/pages
    # Once a subject's own "Answers and Solutions" heading is seen, EVERY later page still
    # carrying that same subject's running header is solutions too (key grid, then narrative
    # answers) -- confirmed live, MockTest1.pdf: the heading prints once per subject, never
    # repeats on the pages after it, so a per-page-only cutoff would read those later pages as if
    # they were real questions (a solution's own "21. (c) ..." line reads identically to a real
    # question-start marker). A one-way latch is enough: a subject passes through "before" to
    # "after" its own heading exactly once, in document order, and never returns to "before".
    past_answers_subjects = set()
    drawings_cache = {}  # page_index -> page.get_drawings(), fetched at most once per page

    def close_active():
        nonlocal active
        if active is None:
            return
        q = _finalize_question(doc, active, mock_name, drawings_cache)
        per_subject_questions[active["subject"]].append(q)
        if q["needs_review"]:
            flagged_pages.append({"page": active["start_page"], "reason": q["review_reason"]})
        active = None

    page_count = len(doc)
    for page_index in range(page_count):
        page = doc[page_index]
        all_blocks = [seg for b in _page_blocks(page) for seg in _split_embedded_question_starts(b)]
        header_blocks = [b for b in all_blocks if _is_header_footer(b)]
        blocks = [b for b in all_blocks if not _is_header_footer(b)]
        if not blocks:
            continue
        full_page_text = "\n".join(_block_text_from_dict(b) for b in blocks)
        if _INSTRUCTIONS_RE.search(full_page_text):
            close_active()  # a question never continues across a repeated instructions page
            continue

        # The subject name lives in the running header block itself ("Physics\n2 "), which was
        # just excluded from `blocks` above as a header/footer -- searched separately here so
        # stripping that block doesn't also blind subject detection to it.
        header_text = "\n".join(_block_text_from_dict(b) for b in header_blocks)
        header_match = _SUBJECT_HEADER_RE.search(header_text[:200])
        if header_match:
            new_subject = header_match.group(1).capitalize()
            if new_subject != current_subject:
                close_active()
            current_subject = new_subject
        if not current_subject:
            flagged_pages.append({
                "page": page_index + 1,
                "reason": "Could not detect a subject (no Physics/Chemistry/Biology header seen yet) -- skipped.",
            })
            continue
        if current_subject in past_answers_subjects:
            continue  # this whole page is that subject's own answer key/solutions, already past

        mid_x = page.rect.width / 2
        ordered = _cluster_two_columns(blocks, mid_x)

        cutoff = next((i for i, b in enumerate(ordered) if _ANSWERS_HEADING_RE.search(_block_text_from_dict(b))), None)
        if cutoff is not None:
            trimmed = _trim_block_before_heading(ordered[cutoff])
            ordered = ordered[:cutoff] + ([trimmed] if trimmed else [])
            past_answers_subjects.add(current_subject)

        diagram_bboxes = _get_diagram_bboxes(page)

        for block in ordered:
            coverage[current_subject]["total"] += _word_count(block)
            start = _question_start(block)
            if start and start[0] > last_num[current_subject]:
                close_active()
                qnum, rest = start
                last_num[current_subject] = qnum
                found_numbers[current_subject].add(qnum)
                active = {
                    "subject": current_subject, "printed_number": qnum,
                    "start_page": page_index + 1, "text_acc": rest, "bbox": block["bbox"],
                    "has_diagram": any(_bbox_overlaps(block["bbox"], d) for d in diagram_bboxes),
                    "raw_blocks": [(page_index, block)],
                    "number_marker_span_id": _find_number_marker_span_id(block),
                }
                coverage[current_subject]["assigned"] += _word_count(block)
                continue
            if active is None:
                continue  # content before this subject's own first recognized question-start
            active["text_acc"] += "\n" + _block_text_from_dict(block)
            active["bbox"] = _union_bbox(active["bbox"], block["bbox"])
            active["raw_blocks"].append((page_index, block))
            if any(_bbox_overlaps(block["bbox"], d) for d in diagram_bboxes):
                active["has_diagram"] = True
            coverage[current_subject]["assigned"] += _word_count(block)

    close_active()
    doc.close()

    questions = []
    for s in MOCK_TEST_SUBJECT_ORDER:
        questions.extend(sorted(per_subject_questions[s], key=lambda q: q["printed_number"]))

    completeness = {}
    for s in MOCK_TEST_SUBJECT_ORDER:
        expected = MOCK_EXAM_SUBJECT_COUNTS[s]
        total = coverage[s]["total"]
        pct = round(100.0 * coverage[s]["assigned"] / total, 1) if total else 0.0
        red_flags = sum(1 for q in per_subject_questions[s] if q["needs_review"])
        completeness[s] = {
            "expected": expected,
            "found_count": len(found_numbers[s]),
            "missing": _missing_numbers(found_numbers[s], expected),
            "duplicates": [],
            "out_of_range": [],
            "coverage_pct": pct,
            "red_flag_count": red_flags,
        }

    flagged_pages.sort(key=lambda f: f["page"])
    return {
        "questions": questions,
        "pages_scanned": page_count,
        "flagged_pages": flagged_pages,
        "completeness": completeness,
        "mock_name": mock_name,
    }
