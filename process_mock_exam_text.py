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
import difflib

import fitz

from process_pyq_vision import (
    _reconstruct_line_text,
    _block_text_from_dict,
    _flag_suspicious_mcq_options,
    MOCK_TEST_SUBJECT_ORDER,
)
from process_pyq_book_3col import _get_diagram_bboxes, _bbox_overlaps, _union_bbox
from process_mock_exam_math import (
    build_option_zones, assign_spans_to_zones, reconstruct_zone, reclaim_fraction_denominators,
    detect_baseline_sensitive_shift, detect_drawn_root_glyphs,
    MARKER_RE, _join_plain_spans, _is_hat_candidate, _is_vec_candidate, _is_plain_stroke,
)

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

# A fraction's own denominator digit sits at ordinary BODY size (confirmed live, MockTest1.pdf
# Physics Q6: a "2a" denominator's "2" is 9.9pt, the very same size as the rest of that option's
# text) -- _MIN_CAPTION_FONT_SIZE's own guard above only protects a SMALL exponent/subscript, so a
# body-sized lone digit that happens to be its own standalone block (confirmed live: PyMuPDF splits
# a font-change mid-expression into its own block) was still read as a real page-number caption and
# silently dropped before it ever reached this question's own glyph pool -- the SAME zone-split
# math reconstruction that correctly rebuilds everything else around it then had nothing left to
# rebuild this digit from at all. A real page-number caption never sits directly under a drawn
# fraction bar; a denominator digit always does -- checked here with the page's own drawings
# (already fetched once per page for has_diagram/modulus-bar use elsewhere, never a second time).
_DENOM_BAR_REACH = 20
_DENOM_BAR_MARGIN = 6


def _has_fraction_bar_above(block, drawings):
    if not drawings:
        return False
    bx0, by0, bx1, by1 = block["bbox"]
    for d in drawings:
        rect = d.get("rect")
        if rect is None:
            continue
        x0, y0, x1, y1 = rect
        if abs(y1 - y0) > 0.75:
            continue
        width = x1 - x0
        if width < 4 or width > 220:
            continue
        if not _is_plain_stroke(d):
            continue
        if y1 > by0 + 1 or by0 - y1 > _DENOM_BAR_REACH:
            continue  # not above this block (within reach), or sitting below/overlapping it
        if x1 < bx0 - _DENOM_BAR_MARGIN or x0 > bx1 + _DENOM_BAR_MARGIN:
            continue
        return True
    return False


def _is_header_footer(block, drawings=None):
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
        if _has_fraction_bar_above(block, drawings):
            return False  # a fraction's own denominator digit, not a page-number caption
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


# A mid-sentence fragment that PyMuPDF splits into its own block purely because of a font/style
# change (confirmed live, MockTest1.pdf: a degree-symbol "60 C deg 50 C deg" value embedded inside
# a sentence -- "Hot water cools from <this> to <this> in the first 10 minutes...") can end up with
# its own bbox top sitting a point or two ABOVE the sentence block it is actually embedded inside,
# even though it reads well after that block's own opening words -- a naive pure-y0 sort then puts
# it BEFORE its own question's "<number>." start block, so it gets accumulated into whatever
# question was still active (the PRECEDING one) instead of the question it actually belongs to.
# Same "same visual row, order by x not by a few points of y jitter" idea
# process_mock_exam_math._rows_in_reading_order already uses for spans, applied here one level up,
# to whole blocks -- two blocks this close in y are never two genuinely different lines (confirmed
# live: ordinary sequential blocks sit a full line height apart, ~10pt+, on this layout).
_COLUMN_ROW_TOL = 3.0


def _cluster_two_columns(blocks, mid_x):
    def _ordered(bucket):
        by_y = sorted(bucket, key=lambda b: b["bbox"][1])
        rows = []
        for b in by_y:
            y = b["bbox"][1]
            if rows and y - rows[-1][-1]["bbox"][1] <= _COLUMN_ROW_TOL:
                rows[-1].append(b)
            else:
                rows.append([b])
        out = []
        for row in rows:
            out.extend(sorted(row, key=lambda b: b["bbox"][0]))
        return out
    left = _ordered([b for b in blocks if b["bbox"][0] < mid_x])
    right = _ordered([b for b in blocks if b["bbox"][0] >= mid_x])
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


def _rawdict_spans_by_page(doc, page_index, cache):
    """(bbox, chars) for every span of `page_index`'s own rawdict (character-level) text, fetched
    at most once per page -- see _fix_leading_space_origins's own docstring for why this is
    needed. bbox is rounded to absorb the float jitter that otherwise keeps an identical-looking
    span from matching itself across the two extraction modes."""
    if page_index not in cache:
        entries = {}
        rd = doc[page_index].get_text("rawdict")
        for block in rd.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for s in line.get("spans", []):
                    key = tuple(round(v, 1) for v in s["bbox"])
                    entries[key] = s.get("chars") or []
        cache[page_index] = entries
    return cache[page_index]


def _fix_leading_space_origins(doc, flat_spans, rawdict_cache):
    """A span's own reported "origin" is its FIRST character's origin (PyMuPDF's own convention)
    -- confirmed live (MockTest1.pdf Physics Q19 option D) that when PyMuPDF glues a leading
    inter-word SPACE onto the front of a new styled run (a font/size change starts a new span
    mid-sentence), that space's own origin can sit a full line-height away from where this SAME
    span's own real (visible) characters actually sit: here, a trailing "yx" run's own reported
    origin coincided with an unrelated fraction numerator's own baseline one row up (confirmed via
    this page's own rawdict character data: the leading space's origin was the numerator's row,
    while 'y' and 'x' themselves shared a completely different, lower baseline) -- which silently
    pulled process_mock_exam_math._base_size_and_baseline's own "most common baseline" vote onto
    the WRONG row, and from there classified this run's own trailing exponent as a subscript
    instead of a superscript. The exponent's own real position never changed; only the reference
    baseline it was measured against did. Recovered here, once per question, by looking up this
    span's own real characters via the page's rawdict text and using the first NON-space
    character's own origin instead -- a span whose own text has no leading whitespace (the
    overwhelming majority) is returned completely unchanged, and a page this lookup can't match
    (should never happen, since both extraction modes read the same content stream) leaves that
    one span's own origin as-is rather than guessing."""
    out = []
    for s in flat_spans:
        text = s.get("text") or ""
        if not text or not text[0].isspace() or not text.strip():
            out.append(s)
            continue
        entries = _rawdict_spans_by_page(doc, s["page_index"], rawdict_cache)
        key = tuple(round(v, 1) for v in s["bbox"])
        chars = entries.get(key)
        fixed_origin = None
        if chars:
            for ch in chars:
                if not ch["c"].isspace():
                    fixed_origin = ch["origin"]
                    break
        if fixed_origin is None:
            out.append(s)
            continue
        ns = dict(s)
        ns["origin"] = fixed_origin
        out.append(ns)
    return out


def reconstruct_question_math(doc, active, drawings_cache, diagram_bboxes_cache, rawdict_cache):
    """Tries the geometry-aware zone reconstruction (process_mock_exam_math) first; falls back to
    the old plain-text _parse_options whenever build_option_zones can't unambiguously find this
    question's own 4 lettered markers (a genuinely different layout, or a question this module's
    own marker detection doesn't recognize -- always safer left to the existing text-based path
    and its own existing needs_review gates than forced through a zone split built on a wrong
    assumption, per process_mock_exam_math's own docstring).

    Returns (stem_text, options, bar_issues, buckets) -- bar_issues is a dict {location_label:
    reason} for the stem and/or any option whose own modulus/magnitude bars could not be
    confidently paired (see process_mock_exam_math._normalize_modulus_bars) OR whose own Symbol-
    font glyphs included one this module's font-aware table doesn't recognize (see
    process_mock_exam_math._collapse_symbol_font_glyphs), empty whenever nothing needed flagging.
    buckets is the SAME {stem/a/b/c/d: [spans]} this function itself used to reconstruct each
    zone, handed back for _check_order_mismatch's own use (see its own docstring for why it
    reuses this instead of an independently-computed reading order) -- None on the plain-text
    fallback path below, which has no zones to hand back. Both bar_issues and the geometry-aware
    reconstruction itself are always empty/absent on that fallback path -- it never runs this
    module's own bar/glyph reconstruction at all, same as it never runs the rest of
    process_mock_exam_math either."""
    flat_spans_raw = _flatten_question_spans(active)
    flat_spans = _fix_leading_space_origins(doc, flat_spans_raw, rawdict_cache)

    def _drawings_for(spans):
        pages = {s["page_index"] for s in spans}
        out = []
        for pi in pages:
            if pi not in drawings_cache:
                drawings_cache[pi] = doc[pi].get_drawings()
            out.extend(drawings_cache[pi])
        return out

    def _diagram_bboxes_for(spans):
        pages = {s["page_index"] for s in spans}
        out = []
        for pi in pages:
            if pi not in diagram_bboxes_cache:
                diagram_bboxes_cache[pi] = _get_diagram_bboxes(doc[pi])
            out.extend(diagram_bboxes_cache[pi])
        return out

    # Conservative rule (requirement 2 / pass bar): a drawn root glyph (see detect_drawn_root_
    # glyphs' own docstring) leaves zero trace in the text layer once dropped -- caught here,
    # once per question and BEFORE the zone-split-vs-fallback branch below (a question that falls
    # back to the plain-text path can still have a drawn root silently vanish from it exactly the
    # same way), rather than ever risk it passing through silently.
    root_glyph_issue = None
    _content_y_centers = [(s["bbox"][1] + s["bbox"][3]) / 2.0 for s in flat_spans
                           if (s.get("text") or "").strip()]
    if detect_drawn_root_glyphs(_drawings_for(flat_spans), active["bbox"], _content_y_centers,
                                _diagram_bboxes_for(flat_spans)):
        root_glyph_issue = "a drawn root (square root) symbol may not have been fully reconstructed"

    result = build_option_zones(flat_spans, active["bbox"])
    if result is None:
        stem, opts = _parse_options(active["text_acc"])
        return stem, opts, ({"root geometry": root_glyph_issue} if root_glyph_issue else {}), None
    stem_zone, option_zones, marker_ids = result
    buckets = assign_spans_to_zones(flat_spans, stem_zone, option_zones, marker_ids)
    # Conservative rule (requirement 2): bbox -> this SAME span's own PRE-origin-fix copy, used by
    # detect_baseline_sensitive_shift below -- keyed globally (not per zone) so it still matches
    # correctly even after reclaim_fraction_denominators moves a span from one zone's bucket into
    # another (see that function's own docstring): the span's bbox never changes, only which
    # zone's list it lives in.
    _raw_by_bbox = {tuple(round(v, 1) for v in s["bbox"]): s for s in flat_spans_raw}

    def _uncorrected_equivalent(spans):
        out = []
        for s in spans:
            u = _raw_by_bbox.get(tuple(round(v, 1) for v in s["bbox"]))
            if u is not None:
                out.append(u)
        return out

    # Fraction-denominator reclaim (requirement 1a): a denominator row has no marker of its own to
    # anchor assign_spans_to_zones's reading-order-sequence rule, so a tightly-packed zone's own
    # marker-midpoint boundary can leave a denominator digit's own center just past its TRUE
    # option's own boundary -- see reclaim_fraction_denominators's own docstring. Run once per
    # question, on every zone's buckets together (a bar's own numerator/denominator can live in
    # either sibling zone's bucket depending on which way assign_spans_to_zones' plain-rectangle
    # fallback happened to lean), before any single zone is reconstructed on its own.
    buckets, reclaim_issue = reclaim_fraction_denominators(
        buckets, flat_spans, marker_ids, _drawings_for(flat_spans), active["bbox"],
        _diagram_bboxes_for(flat_spans))

    bar_issues = {}
    if reclaim_issue:
        bar_issues["fraction geometry"] = reclaim_issue
    if root_glyph_issue:
        bar_issues["root geometry"] = root_glyph_issue
    stem_text, stem_issue = reconstruct_zone(
        buckets["stem"], _drawings_for(buckets["stem"]), stem_zone, _diagram_bboxes_for(buckets["stem"]),
        fraction_scope_bbox=active["bbox"])
    if stem_issue:
        bar_issues["stem"] = stem_issue
    shift_issue = detect_baseline_sensitive_shift(buckets["stem"], _uncorrected_equivalent(buckets["stem"]))
    if shift_issue:
        bar_issues.setdefault("stem", shift_issue)
    options = {}
    for letter in ("a", "b", "c", "d"):
        text, issue = reconstruct_zone(
            buckets[letter], _drawings_for(buckets[letter]), option_zones[letter],
            _diagram_bboxes_for(buckets[letter]), fraction_scope_bbox=active["bbox"])
        options[letter] = text
        if issue:
            bar_issues[f"option {letter.upper()}"] = issue
        shift_issue = detect_baseline_sensitive_shift(buckets[letter], _uncorrected_equivalent(buckets[letter]))
        if shift_issue:
            bar_issues.setdefault(f"option {letter.upper()}", shift_issue)
    return stem_text, options, bar_issues, buckets


def _looks_math_garbled(text):
    tokens = (text or "").split()
    if len(tokens) < _GARBLED_MIN_TOKENS:
        return False
    # A bare "$" is never a sign of leftover fragmented math -- it is this module's OWN math-run
    # delimiter (process_mock_exam_math.reconstruct_zone), which can legitimately land as its own
    # whitespace-split token (e.g. a Greek/symbol substitution's own padding puts a space right
    # next to the $ that wraps it, confirmed live: "$ \Delta $ H"). Counting it here would flag a
    # short, perfectly correct option for no reason other than how many TIMES a real symbol like
    # "Delta" happens to repeat in it.
    stray = sum(1 for t in tokens if t != "$" and len(t) <= 2 and not t.isalpha())
    return (stray / len(tokens)) > _GARBLED_TOKEN_RATIO


# Conservative rule (requirement 2 / pass bar): an arithmetic operator (confirmed live,
# MockTest1.pdf Chemistry Q1: a "-" between "29.2" and "20.2", and a "x" between "1.79" and
# "10^5") between two decimal numbers can be ENTIRELY missing from the text layer -- not a glyph
# this module's own font-aware table could ever recognize (see process_mock_exam_math.py's own
# module docstring, the SAME "<" comparison-operator gap documented there for Chemistry Q20 --
# confirmed live here as the SAME font-decoding limitation, just a different operator), and unlike
# a reordering, a genuinely MISSING character never moves _check_order_mismatch's own token order
# at all, so that check cannot catch it either. Detected instead, purely from the TEXT itself: two
# real decimal numbers glued directly together with no operator or space between them read as one
# run with 2+ decimal points ("29.220.2") -- impossible for any single real number, so this is a
# specific, low-false-positive signal that a separator was lost, not a guess at which operator it
# was.
_MERGED_DECIMALS_RE = re.compile(r'\d+\.\d+\.\d+')


def _has_merged_decimal_numbers(text):
    return bool(_MERGED_DECIMALS_RE.search(text or ""))


# Silent-error check (requirement 3): strips every LaTeX command/delimiter and leaves only alnum
# RUNS, lowercased -- "ignore math symbols and whitespace" means a fraction/root/diacritic's own
# markup is never compared, only the real letters and digits underneath it, so two representa-
# tions of the SAME content that merely mark it up differently (one side wraps a Greek letter in
# "\Delta", the other leaves a bare reading-order run with no LaTeX at all) still line up. A
# letter run and a digit run are kept as SEPARATE tokens ("K4" -> "k","4") so an exponent digit
# still lines up against its own raw digit even when only one side's own LaTeX wrapping sits
# between them.
_LATEX_MARKUP_RE = re.compile(r'\\[a-zA-Z]+|[{}$^_\\]')
_ALNUM_TOKEN_RE = re.compile(r'[a-zA-Z]+|[0-9]+')


def _content_tokens(text):
    stripped = _LATEX_MARKUP_RE.sub(" ", text or "")
    return [t.lower() for t in _ALNUM_TOKEN_RE.findall(stripped)]


def _strip_hat_vec(spans):
    """A hat/vec diacritic glyph (a bare circumflex, or the real "MT Extra" accent font's own
    coincidental 'r'/'ur'/'Q' text -- see process_mock_exam_math's own module docstring) is
    dropped entirely rather than joined in plain reading order: _join_plain_spans has no concept
    of pairing it with its own base letter (that pairing is reconstruct_zone's own job) -- left
    in, it glues onto its own base letter's token ("a" + accent-font "r" -> "ar"), a purely
    REPRESENTATIONAL mismatch against the production side's own correctly-unwrapped "a" that
    would otherwise drown out a genuine order problem on every vector-heavy question."""
    return [s for s in spans if not _is_hat_candidate(s) and not _is_vec_candidate(s)]


def _question_reading_order_tokens(buckets):
    """Ground truth for _check_order_mismatch on the geometry-aware zone-split path: reuses the
    SAME stem/a/b/c/d buckets reconstruct_question_math itself already computed (assign_spans_to_
    zones), joining each bucket's own spans via process_mock_exam_math's plain row-based join
    (_join_plain_spans) -- NOT reconstruct_zone's own fraction/root/diacritic-aware logic -- then
    concatenating stem+a+b+c+d IN THAT FIXED ORDER, the same order the final stem+options string
    itself is built in.

    Confirmed live this is necessary, not just convenient: a 2x2 grid's own TRUE page reading
    order genuinely reads "every row-1 numerator, then every row-1 denominator" (left cell then
    right cell, top to bottom) rather than "option A's own numerator+denominator, then option
    B's" -- a real structural difference from the stem+a+b+c+d concatenation order, not a
    reconstruction mistake, and comparing against raw whole-page reading order there flagged a
    perfectly correct 2x2-grid reconstruction (confirmed live via this module's own synthetic
    test suite). Reusing the same zone buckets as production makes this check somewhat LESS able
    to catch a pure zone-ASSIGNMENT mistake (both sides would inherit the identical mistake) --
    but it stays fully independent of reconstruct_zone's own fraction/root/diacritic logic, where
    the real problems this check has caught live (a false fraction-bar match scrambling a
    stem's own reading order, confirmed live on the real sample's own Biology Q34)."""
    parts = []
    for name in ("stem", "a", "b", "c", "d"):
        text = _join_plain_spans(_strip_hat_vec(buckets.get(name) or []))
        if text.strip():
            parts.append(text)
    return _content_tokens(" ".join(parts))


def _question_reading_order_tokens_fallback(doc, flat_spans):
    """Ground truth for _check_order_mismatch on the plain-text fallback path (no zone buckets
    exist there at all): the same whole-question, zone-agnostic reading-order reconstruction the
    geometry-aware path used before buckets were available for reuse (see
    _question_reading_order_tokens's own docstring for why that path moved on) -- still fully
    independent here, since the fallback path's own _parse_options works from linear TEXT, not
    2D geometry, so it never faces the SAME grid-ordering difference in the first place.

    The per-span Bold+MARKER_RE exclusion below only ever catches a marker that is its OWN whole
    span in a BOLD font -- the geometry-aware path's own markers always are (build_option_zones
    requires it), but this fallback path's own production side (_parse_options) strips a marker
    from the middle of an ordinary, non-bold run via _OPTION_MARKER_RE instead (confirmed live: a
    plain "a. alpha b. beta c. gamma d. delta" line, one single non-bold span, is never caught by
    the per-span check at all) -- left unstripped here, the bare marker letters ("a", "b", "c",
    "d") show up as extra tokens only on THIS side, which a real, perfectly-ordered reconstruction
    can never match against. _OPTION_MARKER_RE is applied the same way _parse_options itself
    already splits on it, so both sides compare the same underlying content.

    Grouped by page_index, each page's own spans joined independently and the pages concatenated
    in page order -- confirmed live (a question whose own options spill across a column break AND
    a page break): a plain y-only row clustering across the WHOLE flat_spans pool has no concept
    of a page boundary at all, so a later page's own content starting back at a small y (e.g. this
    PDF layout's own "d. continuation option d" printed near the TOP of page 2) can sort as if it
    were physically ABOVE page 1's own later rows -- the production side never makes this mistake,
    since _parse_options accumulates block text strictly in the main scan loop's own per-page
    order. WITHIN one page, also split left/right by that page's own horizontal midpoint and join
    the LEFT half entirely before the right (same column-major order _cluster_two_columns already
    uses) -- a plain y-row clustering has no concept of COLUMNS either, so a right-column block
    sitting at a smaller y than a later left-column row (confirmed live, the same spilled-options
    question: its own right-column options sit at the same y as question 1's own left-column
    stem, well above this question's own left-column stem two rows down) would otherwise sort
    before content that reads first on the real page. doc: used only to read each page's own width
    for the midpoint split -- a page_index this doc doesn't have (should never happen) falls back
    to a plain row join for that page's own spans, no column split.

    A leading "<number>." question-marker prefix is stripped once, from the very start of the
    whole joined text -- confirmed live: when the WHOLE first line (number and stem together) is
    PyMuPDF's own single span (common in a short synthetic fixture with no font break at the
    marker), _flatten_question_spans' own number_marker_span_id exclusion (an exact "^\\d+[.)]$"
    span match) never catches it, the same way the production side's own _question_start strips it
    from TEXT instead of ever relying on a dedicated marker span existing at all."""
    non_marker = [s for s in flat_spans
                  if not ("Bold" in (s.get("font") or "") and MARKER_RE.match(s["text"].strip()))]
    non_marker = _strip_hat_vec(non_marker)
    by_page = {}
    for s in non_marker:
        by_page.setdefault(s.get("page_index"), []).append(s)
    parts = []
    for page_index in sorted(by_page, key=lambda p: (p is None, p)):
        page_spans = by_page[page_index]
        mid_x = None
        if doc is not None and page_index is not None and 0 <= page_index < len(doc):
            mid_x = doc[page_index].rect.width / 2
        if mid_x is None:
            page_text = _join_plain_spans(page_spans)
        else:
            left = [s for s in page_spans if s["bbox"][0] < mid_x]
            right = [s for s in page_spans if s["bbox"][0] >= mid_x]
            page_text = " ".join(t for t in (_join_plain_spans(left), _join_plain_spans(right)) if t.strip())
        if page_text.strip():
            parts.append(page_text)
    text = " ".join(parts)
    text = _QSTART_RE.sub(lambda m: m.group(2), text, count=1)
    text = _OPTION_MARKER_RE.sub(" ", text)
    return _content_tokens(text)


# Tuned against the real sample (MockTest1.pdf), via a full 180-question calibration pass (every
# question's own ratio, not just the already-flagged ones) plus a page-image spot-check of every
# question that landed near the boundary either way, REDONE after _question_reading_order_tokens
# moved to a bucket-based reference (see that function's own docstring for why): every question
# confirmed CORRECT against its own page image now measures >= 0.86 (several genuinely correct
# questions still cost a little on pure representational noise -- see _content_tokens's own
# docstring for why -- so this is not 1.0), while the one confirmed real problem below this
# sample's own clean floor measures 0.793 -- a chemical-ordering option's own "<" comparison signs
# silently decoded to blank spaces, a font-specific glyph-mapping gap this module's own font-aware
# table does not have enough evidence to safely resolve on its own (see
# process_mock_exam_math.py's own module docstring). The bucket-based reference fix that was
# needed to stop a 2x2 grid's own true page-reading order from reading as a false mismatch (see
# _question_reading_order_tokens) also costs this check some of its own former reach -- an
# isotope-notation option's own stacked superscript+subscript losing its ^/_ markup entirely
# (confirmed live, a real but DIFFERENT bug than the one still caught here) no longer moves the
# ratio, since both sides of the comparison now share the same zone bucket and its same body-size
# tie-break mistake; that bug is real and documented, just no longer this check's own catch.
_ORDER_MISMATCH_MIN_RATIO = 0.80


def _check_order_mismatch(doc, buckets, flat_spans, stem, opts):
    """Silent-error check (requirement 3): every question, flagged or not, gets its own FINAL
    stem+options (concatenated in that fixed a/b/c/d order) compared against an independent
    reconstruction of the very same glyphs -- _question_reading_order_tokens(buckets) when the
    geometry-aware zone split succeeded (buckets is not None), else
    _question_reading_order_tokens_fallback(flat_spans) -- as an ORDER-SENSITIVE token sequence
    (difflib.SequenceMatcher -- deliberately not the order-INsensitive word-overlap ratio
    process_mock_exam_vision.py's own gate already uses elsewhere: a reordering keeps the exact
    same bag of words, so only an order-sensitive comparison can catch it at all). Returns a
    human-readable reason the moment the match ratio drops below _ORDER_MISMATCH_MIN_RATIO, else
    None -- never raises, never touches stem/opts itself."""
    produced = _content_tokens(" ".join([stem] + [opts.get(l, "") for l in _OPTION_LETTERS]))
    reference = (_question_reading_order_tokens(buckets) if buckets is not None
                 else _question_reading_order_tokens_fallback(doc, flat_spans))
    if not produced or not reference:
        return None
    ratio = difflib.SequenceMatcher(None, reference, produced).ratio()
    if ratio < _ORDER_MISMATCH_MIN_RATIO:
        return f"order mismatch -- only {ratio:.0%} order-aligned against the page's own reading order"
    return None


def _crop_question_image(doc, page_index, bbox, margin=8, dpi=200):
    """Base64 PNG crop of a flagged question's own source region, for the reviewer to compare
    directly against the parsed text -- only ever called for a flagged question (requirement 2),
    never for every question, so a clean scan's response stays small. 200 DPI (not this project's
    more common 150): this same crop also doubles as the image sent to process_mock_exam_vision's
    model-assisted re-read, where the extra resolution measurably helps it read a small exponent
    or unit character correctly."""
    try:
        page = doc[page_index]
        rect = (fitz.Rect(bbox) + (-margin, -margin, margin, margin)) & page.rect
        pix = page.get_pixmap(dpi=dpi, clip=rect)
        return base64.standard_b64encode(pix.tobytes("png")).decode("ascii")
    except Exception as e:
        print(f"    Mock exam scan: could not crop a flagged question's image ({e})")
        return None


def _finalize_question(doc, active, mock_name, drawings_cache, diagram_bboxes_cache, rawdict_cache):
    stem, opts, bar_issues, buckets = reconstruct_question_math(
        doc, active, drawings_cache, diagram_bboxes_cache, rawdict_cache)
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
    elif bar_issues:
        needs_review = True
        review_reason = (
            f"{active['subject']} question {active['printed_number']}: "
            + "; ".join(f"{loc} has {reason}" for loc, reason in bar_issues.items())
            + " -- please check against the PDF."
        )
    elif not stem:
        needs_review = True
        review_reason = f"{active['subject']} question {active['printed_number']}'s own stem came back empty."
    else:
        garbled_fields = [name for name, val in (("stem", stem), *((f"option {l.upper()}", opts.get(l, "")) for l in _OPTION_LETTERS)) if _looks_math_garbled(val)]
        merged_decimal_fields = [name for name, val in (("stem", stem), *((f"option {l.upper()}", opts.get(l, "")) for l in _OPTION_LETTERS)) if _has_merged_decimal_numbers(val)]
        if garbled_fields:
            needs_review = True
            review_reason = (
                f"{active['subject']} question {active['printed_number']}: leftover fragmented "
                f"math glyphs detected in {', '.join(garbled_fields)} -- please clean up against the PDF."
            )
        elif merged_decimal_fields:
            needs_review = True
            review_reason = (
                f"{active['subject']} question {active['printed_number']}: an operator between "
                f"two decimal numbers may be missing in {', '.join(merged_decimal_fields)} -- "
                f"please check against the PDF."
            )
        else:
            # Silent-error check (requirement 3): runs on EVERY question that would otherwise come
            # back clean -- the other gates above all look for a SYMPTOM (an empty field, a bar
            # that won't pair, a stray short token); this one instead cross-checks the final
            # result's own ORDER against an independently-reconstructed reading order, catching a
            # zone-assignment mistake that happens to leave no such symptom behind at all.
            order_issue = _check_order_mismatch(doc, buckets, _flatten_question_spans(active), stem, opts)
            if order_issue:
                needs_review = True
                review_reason = f"{active['subject']} question {active['printed_number']}: {order_issue}."

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
        # Internal-only: process_mock_exam_vision's model-assisted re-read uses this question's own
        # raw (pre-reconstruction) text as both its prompt hint and its word-match gate's ground
        # truth -- popped back off before any scan result ever reaches the browser (see that
        # module's apply_vision_to_flagged_questions). Never set for a clean question.
        q["_text_layer_hint"] = _normalize_ws(active["text_acc"])
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
    diagram_bboxes_cache = {}  # page_index -> _get_diagram_bboxes(page), fetched at most once per page
    rawdict_cache = {}  # page_index -> {bbox: chars}, fetched at most once per page

    def close_active():
        nonlocal active
        if active is None:
            return
        q = _finalize_question(doc, active, mock_name, drawings_cache, diagram_bboxes_cache, rawdict_cache)
        per_subject_questions[active["subject"]].append(q)
        if q["needs_review"]:
            flagged_pages.append({"page": active["start_page"], "reason": q["review_reason"]})
        active = None

    page_count = len(doc)
    for page_index in range(page_count):
        page = doc[page_index]
        if page_index not in drawings_cache:
            drawings_cache[page_index] = page.get_drawings()
        page_drawings = drawings_cache[page_index]
        all_blocks = [seg for b in _page_blocks(page) for seg in _split_embedded_question_starts(b)]
        header_blocks = [b for b in all_blocks if _is_header_footer(b, page_drawings)]
        blocks = [b for b in all_blocks if not _is_header_footer(b, page_drawings)]
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
        diagram_bboxes_cache[page_index] = diagram_bboxes  # reused below for modulus-bar exclusion,
        # never recomputed -- this call is the only one _get_diagram_bboxes ever makes per page.

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
