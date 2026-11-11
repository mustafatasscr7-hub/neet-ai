"""Deterministic (NO model call, ever) math reconstruction for the Mock exam PDF layout, from
PyMuPDF glyph geometry and vector-drawn fraction bars -- see process_mock_exam_text.py for the
calling context (one already-accepted question's own accumulated region).

Confirmed live against the real sample (a NEET-pattern mock paper, heavy Physics/Chemistry inline
math): PyMuPDF's own line-clustering splits a superscript/subscript exponent, a fraction's
numerator/denominator, or a vector/hat diacritic into SEPARATE "lines" purely because they sit at
a different baseline than the surrounding body text -- plain block/line-text extraction then
either scrambles the reading order or leaves the fragments as stray tokens, which is exactly what
process_mock_exam_text.py's own needs_review gates (missing option, leftover fragmented math
glyphs) were catching. This module re-derives the real structure from each GLYPH's own font size,
baseline position and page-drawn fraction bars instead of trusting PyMuPDF's line grouping:

  - Superscript/subscript: a span whose font size is meaningfully smaller than the zone's own body
    size AND whose baseline (origin y) sits above (superscript) or below (subscript) the body
    baseline -- same signal process_pyq_vision._reconstruct_line_text already uses for
    superscript-only, extended here to also cover subscript and to work across a whole option's
    zone rather than one single PyMuPDF line.
  - Fraction: a vector-drawn horizontal line (near-zero height) with real content both above
    (numerator) and below (denominator) it, within its own x-range -- confirmed live, e.g. a
    car-collision option "s >= (V_A^2 - V_B^2)/(2a)" where PyMuPDF's own text layer never puts a
    "/" anywhere; the only signal a real fraction bar exists at all is the drawn line.
  - Hat (lines, vectors-as-unit-vectors): a bare circumflex character (U+02C6) positioned directly
    above a base letter.
  - Vector arrow: a narrow glyph in the "MTExtra" accent font (confirmed live: its own text
    content is coincidentally 'r'/'ur'/'Q' -- an artifact of that font's own glyph-index mapping,
    not meaningful) positioned directly above a base letter.
  - Greek letters and common math symbols: already correct Unicode characters in the text layer
    (confirmed live: sigma/pi/rho/mu/lambda all extract correctly as-is) -- just mapped to the
    LaTeX command names process_pyq_vision.py's own $...$ convention expects
    (_KNOWN_LATEX_COMMANDS), not reconstructed from geometry at all.

Confidence rule (requirement: a wrong reconstruction is worse than a flagged one): a diacritic
with no uniquely-close base letter, or a fraction bar whose numerator/denominator is empty or
already claimed by another bar, is left COMPLETELY ALONE -- its glyphs stay in the plain pool as
literal text. This never crashes and never silently drops content; it just means that stray
fragment stays visible in the output, which process_mock_exam_text.py's own existing
_looks_math_garbled gate (unusually many short stray tokens) already catches and flags for
manual review -- most of this module's own confidence rules add no new flagging logic of their
own, they only reduce how often that existing gate has something real to catch. The one
exception is modulus/magnitude bars ("|x|", a vector norm "||v||"): an unpaired bar is reported
back as its own dedicated issue rather than left to that generic gate, see this module's own
"Modulus/magnitude bars" section further down for why a bar needs a confidence rule of its own.
"""
import re

MARKER_RE = re.compile(r'^([a-dA-D])[.)]$')

# Smaller than this fraction of the zone's own body size, AND meaningfully off the body baseline,
# is a superscript/subscript -- same 0.85 ratio process_pyq_vision._reconstruct_line_text already
# uses live. 1.0pt baseline-offset tolerance absorbs ordinary sub-pixel rendering jitter without
# accepting a same-row span that merely has a slightly different origin.
_SIZE_RATIO = 0.85
_BASELINE_TOL = 1.0

# A real question/option's own content never sits more than this far below a fraction bar's own
# drawn line and still belongs to it -- confirmed live, e.g. a 2-row option (numerator ~10pt tall,
# small margin) with its denominator ending within ~10pt below the bar. Set generously (20pt) to
# tolerate a tall multi-token denominator, while still well short of the ~30-40pt gap to a
# genuinely unrelated next row in these option zones.
_FRAC_VERTICAL_REACH = 20
_FRAC_HORIZONTAL_MARGIN = 4

# A hat/vec diacritic sits directly above the letter it modifies and is horizontally centered on
# it (confirmed live: well within 4pt). Its own BBOX, though, routinely overlaps the base letter's
# bbox by several points rather than sitting cleanly above with a visible gap -- confirmed live, a
# circumflex's own glyph box extends a good ~7pt below its visual mark (ordinary font-metric
# padding), so "gap" (base top minus diacritic bottom) comes out clearly NEGATIVE, not just
# slightly so. _DIACRITIC_MAX_OVERLAP bounds how much of that overlap is still trusted as "this
# diacritic sits on this base" rather than "this is actually some unrelated glyph below it."
_DIACRITIC_MAX_GAP = 5
_DIACRITIC_MAX_OVERLAP = 10
_DIACRITIC_MAX_DX = 6

# "MT Extra" (MathType's own accent-glyph font, bundled with Equation Editor/Office) is this
# sample's own font for the vector-arrow mark (confirmed live: its own glyph text is a
# coincidental 'r'/'ur'/'Q', not meaningful -- see module docstring). Matched by NORMALIZED name
# (lowercased, punctuation stripped) rather than an exact string -- confirmed live on this very
# machine that two real installs of the same typeface report different PDF font names for it
# ("MTExtra" with no separator vs. "MT-Extra" with a hyphen), so an exact-match set would silently
# stop working on a PDF built with the other naming convention. Matched by font name rather than
# glyph content for the same reason the content is unreliable in the first place. A real base-
# sized letter is never this narrow.
_ACCENT_FONT_NAMES = {"mtextra"}
_MAX_ACCENT_WIDTH = 8
_NON_ALNUM_RE = re.compile(r'[^a-z0-9]')


def _normalized_font_name(font):
    return _NON_ALNUM_RE.sub('', (font or '').lower())
_HAT_CHAR = "ˆ"

# Direct Unicode-character -> LaTeX command-name mapping for symbols that already extract
# correctly as real Unicode (confirmed live) and need no geometry at all -- just wrapping in the
# site's own $...$ convention (process_pyq_vision._KNOWN_LATEX_COMMANDS). Degree, times, and pm
# are handled separately below since they render as a plain symbol rather than "\command".
_GREEK_COMMANDS = {
    "α": "alpha", "β": "beta", "γ": "gamma", "Γ": "Gamma",
    "δ": "delta", "Δ": "Delta", "ε": "epsilon", "ζ": "zeta",
    "η": "eta", "θ": "theta", "Θ": "Theta", "ι": "iota",
    "κ": "kappa", "λ": "lambda", "Λ": "Lambda", "μ": "mu",
    "µ": "mu",  # MICRO SIGN -- used interchangeably with GREEK SMALL LETTER MU in these PDFs
    "ν": "nu", "ξ": "xi", "Ξ": "Xi", "π": "pi", "Π": "Pi",
    "ρ": "rho", "σ": "sigma", "Σ": "Sigma", "τ": "tau",
    "υ": "upsilon", "φ": "phi", "Φ": "Phi", "χ": "chi",
    "ψ": "psi", "Ψ": "Psi", "ω": "omega", "Ω": "Omega",
}
_SYMBOL_COMMANDS = {
    "±": "pm", "×": "times", "÷": "div", "≈": "approx",
    "≠": "neq", "≤": "leq", "≥": "geq", "∝": "propto",
    "→": "rightarrow", "←": "leftarrow", "∞": "infty",
    "°": "circ", "≡": "equiv", "∂": "partial", "∇": "nabla",
}
_ALL_SYMBOL_COMMANDS = {**_GREEK_COMMANDS, **_SYMBOL_COMMANDS}
_SYMBOL_CHARS_RE = re.compile("[" + "".join(re.escape(c) for c in _ALL_SYMBOL_COMMANDS) + "]")

# A literal U+221A square-root character already in the text layer is reconstructed with
# confidence (requirement: handle roots) -- but ONLY this clean case. This sample's own roots are
# built from drawn paths with no reliable glyph marking where the radicand ends (confirmed live:
# no U+221A anywhere in the sample's text layer at all), so that shape is deliberately left
# unhandled rather than guessed -- see module docstring's confidence rule.
_ROOT_CHAR = "√"


def _is_fraction_bar_rect(rect, zone_bbox):
    x0, y0, x1, y1 = rect
    if abs(y1 - y0) > 0.75:
        return False
    width = x1 - x0
    if width < 4 or width > 220:
        return False
    zx0, zy0, zx1, zy1 = zone_bbox
    return zx0 - 2 <= x0 and x1 <= zx1 + 2 and zy0 - 2 <= y0 <= zy1 + 2


def get_zone_fraction_bars(drawings, zone_bbox):
    """drawings: page.get_drawings() output (fetched once per page by the caller). Returns a list
    of (x0, y0, x1) for every thin horizontal line drawing that falls inside zone_bbox -- a
    fraction bar never spans outside the option/stem zone it belongs to, which is what keeps this
    from ever claiming content across an option boundary (see this module's own docstring)."""
    bars = []
    for d in drawings:
        rect = d.get("rect")
        if rect is None:
            continue
        rect = (rect[0], rect[1], rect[2], rect[3])
        if _is_fraction_bar_rect(rect, zone_bbox):
            bars.append((rect[0], rect[1], rect[2]))
    return bars


# ---------------------------------------------------------------------------------------------
# Modulus/magnitude bars: "|x|", "|a + b + c|", a vector norm "||v||". A bar is either a literal
# "|" already in the text layer, a look-alike character (DIVIDES U+2223, PARALLEL TO U+2225 --
# both anticipated; DOUBLE VERTICAL LINE U+2016, confirmed live in the real sample, see below), or
# a short vertical vector-drawn stroke with no character at all -- the same "no reliable glyph"
# shape this module's own root handling already anticipates for a drawn-overline root. Two bar
# occurrences sitting immediately next to each other on the same row -- two single characters
# glued together (e.g. "||"), or one inherently-double glyph like "‖" -- are ONE logical delimiter
# and rebuilt as "||", never as two separate single bars.
#
# Confirmed live (MockTest1.pdf): every "|"/"‖" occurrence actually present in the real sample's
# text layer turned out to be a CHEMISTRY double-bond mark (a structural formula's own C=O/C=NH
# drawn with one or two short vertical strokes connecting an atom to the one directly above it on
# a DIFFERENT text row), not a real modulus bar -- confirmed by rendering the source pages and
# reading them directly, not just by the extracted text. This is exactly the shape the confidence
# rule below rejects on its own: a bond mark is a single, lone bar-like occurrence on its own row
# with no partner (an odd count), or a pair with nothing real between the two strokes (the gap a
# double bond's own two near-parallel lines leave behind) -- never a case this module forces
# through as if it were a real modulus pair.
#
# Confidence rule (same "wrong is worse than flagged" requirement as the rest of this module): a
# ROW's own bar occurrences are only rebuilt if they pair up EXACTLY -- an even count, each pair
# enclosing real, non-empty content on that same row -- checked for every row in the zone before
# anything is rewritten. The whole zone is rewritten together, all-or-nothing: by construction,
# either every occurrence maps 1:1 onto an output "|"/"||" (so a bar can never be silently lost by
# this rewrite -- there is no code path that drops one without ALSO leaving the zone's text
# untouched and reporting the problem back), or none of them are touched at all and the zone's own
# bar characters stay exactly as literal text, with a reason reported back for the caller to flag
# needs_review with -- never a partial rewrite, never a guess at which side of an unpaired bar is
# "inside".
_BAR_CHARS_SINGLE = {"|", "∣"}
_BAR_CHARS_DOUBLE = {"∥", "‖"}
_BAR_CHARS_ALL = _BAR_CHARS_SINGLE | _BAR_CHARS_DOUBLE
_BAR_RUN_RE = re.compile("[" + "".join(re.escape(c) for c in _BAR_CHARS_ALL) + "]+")

# A drawn vertical bar stroke -- same near-zero-thickness idea _is_fraction_bar_rect already uses
# for the horizontal fraction case, transposed to the vertical axis. Bounded to roughly one text
# row tall: a real modulus bar is never as short as a stray rendering artifact nor as tall as a
# multi-row bracket (which this module does not attempt to reconstruct at all).
_VBAR_MAX_WIDTH = 0.75
_VBAR_MIN_HEIGHT = 4.0
_VBAR_MAX_HEIGHT = 22.0
# Two bar occurrences on the same row this close together (or closer) are the SAME logical
# delimiter (a vector norm's own two strokes), not two independent bars each needing their own
# partner -- confirmed live, the one glued-together "||" span in the real sample's own text layer
# has its own two strokes well under 1pt apart.
_BAR_ADJACENT_GAP = 2.0
_BAR_ROW_TOL = 4.0
# How far a drawn stroke may sit from the nearest REAL glyph in its own zone and still be
# trusted as a candidate bar at all -- see _near_real_content's own docstring for why this check
# exists in the first place. One generous text row (not just _BAR_ROW_TOL's tight same-row
# pairing tolerance): the stroke only needs to be IN THE NEIGHBORHOOD of real content here, the
# tighter tolerance below still decides whether it actually PAIRS with anything.
_VBAR_CONTENT_REACH = 14.0


def _is_vertical_bar_rect(rect, zone_bbox):
    x0, y0, x1, y1 = rect
    if (x1 - x0) > _VBAR_MAX_WIDTH:
        return False
    height = y1 - y0
    if not (_VBAR_MIN_HEIGHT <= height <= _VBAR_MAX_HEIGHT):
        return False
    zx0, zy0, zx1, zy1 = zone_bbox
    return zx0 - 2 <= x0 <= zx1 + 2 and zy0 - 2 <= y0 and y1 <= zy1 + 2


def get_zone_vertical_bars(drawings, zone_bbox):
    """Same shape as get_zone_fraction_bars (see its own docstring), transposed to vertical
    strokes -- a drawn modulus bar never spans outside its own option/stem zone, for the same
    reason a fraction bar never does."""
    bars = []
    for d in drawings:
        rect = d.get("rect")
        if rect is None:
            continue
        rect = (rect[0], rect[1], rect[2], rect[3])
        if _is_vertical_bar_rect(rect, zone_bbox):
            bars.append(rect)
    return bars


def _bar_runs(text):
    """Every maximal run of 1+ bar-like characters in `text`, as (start, end, is_double) --
    is_double whenever the run is 2+ characters (two bars glued together) or its own single
    character is already visually double (PARALLEL TO / DOUBLE VERTICAL LINE)."""
    out = []
    for m in _BAR_RUN_RE.finditer(text):
        run = m.group(0)
        is_double = len(run) >= 2 or any(c in _BAR_CHARS_DOUBLE for c in run)
        out.append((m.start(), m.end(), is_double))
    return out


def _near_real_content(rect, content_y_centers):
    """A question's own stem/option zone_bbox is occasionally much larger than its own real
    content -- confirmed live, a trailing accumulated block (e.g. a page's own running
    watermark/footer, swept in because this question happened to be the last one recognized on
    its own page) can stretch zone_bbox across most of the page, FAR past where this zone's own
    text actually sits. Plain zone_bbox containment alone is what the rest of this module already
    uses for a drawn fraction bar too, but a fraction is self-checking (it needs real numerator
    AND denominator content within reach before it is ever trusted, see _FRAC_VERTICAL_REACH) --
    a single drawn stroke has no such built-in check, so a stray drawing anywhere in an oversized
    zone_bbox would otherwise count as a "bar" no matter how far it sits from anything real. This
    requires a drawn stroke to sit within one ordinary text row of SOME real glyph actually in
    this zone before it is trusted as a candidate bar at all."""
    rect_y_center = (rect[1] + rect[3]) / 2.0
    return any(abs(rect_y_center - cy) <= _VBAR_CONTENT_REACH for cy in content_y_centers)


def _normalize_modulus_bars(spans, drawings, zone_bbox, body_size):
    """Finds every modulus/magnitude bar occurrence in this zone (text-layer character or drawn
    vertical stroke), pairs them per row, and either (a) -- every row's own bars pair up cleanly
    -- returns new spans with every bar occurrence rewritten to the canonical "|"/"||" and no
    issue, or (b) -- any row's own bars do not -- returns the ORIGINAL spans completely untouched
    plus a human-readable reason, for the caller to flag needs_review with (see this module's own
    docstring above for why this is all-or-nothing)."""
    has_text_bar = any(s.get("text") and _BAR_RUN_RE.search(s["text"]) for s in spans)
    content_y_centers = [_span_center(s)[1] for s in spans if s.get("text") and s["text"].strip()]
    drawn_rects = [r for r in (get_zone_vertical_bars(drawings, zone_bbox) if drawings else [])
                   if _near_real_content(r, content_y_centers)]
    if not has_text_bar and not drawn_rects:
        return spans, None

    # content_items only ever need to answer "is there something real at roughly (x, y)" for the
    # non-empty-between check further down -- not the text itself, just a position. A bar-
    # containing span's own LEFTOVER text (e.g. the "x" in one single "|x|" span, glued with no
    # gap on either side -- confirmed live, PyMuPDF's own insert_text keeps same-font same-size
    # adjacent characters in ONE span) is real content exactly the same as a whole separate span
    # free of any bar character would be; excluding a span just because it ALSO happens to carry
    # the bar itself would wrongly treat "|x|" as if nothing were between its own two bars.
    occurrences = []  # {bbox, is_double, src_span (None for a drawn stroke), match (start, end)}
    content_items = []  # (x, y) -- position only, see above
    for s in spans:
        text = s.get("text") or ""
        if not text:
            continue
        runs = _bar_runs(text)
        if not runs:
            if text.strip():
                content_items.append(_span_center(s))
            continue
        x0, y0, x1, y1 = s["bbox"]
        width = x1 - x0
        n = len(text)

        def _frac_x(i, x0=x0, width=width, n=n):
            return x0 + width * (i / n) if n else x0

        cursor = 0
        for start, end, is_double in runs:
            if text[cursor:start].strip():
                content_items.append(((_frac_x(cursor) + _frac_x(start)) / 2.0, (y0 + y1) / 2.0))
            occurrences.append({"bbox": (_frac_x(start), y0, _frac_x(end), y1), "is_double": is_double,
                                 "src_span": s, "match": (start, end)})
            cursor = end
        if text[cursor:].strip():
            content_items.append(((_frac_x(cursor) + _frac_x(n)) / 2.0, (y0 + y1) / 2.0))
    for rect in drawn_rects:
        occurrences.append({"bbox": rect, "is_double": False, "src_span": None, "match": None})

    rows = _cluster_rows(occurrences, key=lambda o: (o["bbox"][1] + o["bbox"][3]) / 2.0, tol=_BAR_ROW_TOL)

    span_replacements = {}  # id(span) -> list of (start, end, bar_text), applied left-to-right
    new_bar_spans = []
    for row in rows:
        row.sort(key=lambda o: o["bbox"][0])
        units = []
        for occ in row:
            if units and occ["bbox"][0] - units[-1]["x1"] <= _BAR_ADJACENT_GAP:
                last = units[-1]
                last["members"].append(occ)
                last["is_double"] = True
                last["x1"] = max(last["x1"], occ["bbox"][2])
            else:
                units.append({"members": [occ], "is_double": occ["is_double"],
                              "x0": occ["bbox"][0], "x1": occ["bbox"][2],
                              "y0": occ["bbox"][1], "y1": occ["bbox"][3]})
        if len(units) % 2 != 0:
            return spans, "an unpaired modulus bar"
        for i in range(0, len(units), 2):
            open_u, close_u = units[i], units[i + 1]
            y_center = (open_u["y0"] + open_u["y1"]) / 2.0
            between = [ci for ci in content_items
                       if open_u["x1"] - 0.5 <= ci[0] <= close_u["x0"] + 0.5
                       and abs(ci[1] - y_center) <= _BAR_ROW_TOL]
            if not between:
                return spans, "an unpaired modulus bar"
            for u in (open_u, close_u):
                bar_text = "||" if u["is_double"] else "|"
                for occ in u["members"]:
                    if occ["src_span"] is not None:
                        span_replacements.setdefault(id(occ["src_span"]), []).append(
                            (occ["match"][0], occ["match"][1], bar_text))
                    else:
                        new_bar_spans.append({
                            "text": bar_text, "bbox": occ["bbox"],
                            "origin": (occ["bbox"][0], occ["bbox"][3]),
                            "size": body_size, "font": None,
                        })

    new_spans = []
    for s in spans:
        edits = span_replacements.get(id(s))
        if not edits:
            new_spans.append(s)
            continue
        text = s["text"]
        out = []
        pos = 0
        for start, end, bar_text in sorted(edits):
            out.append(text[pos:start])
            out.append(bar_text)
            pos = end
        out.append(text[pos:])
        ns = dict(s)
        ns["text"] = "".join(out)
        new_spans.append(ns)
    new_spans.extend(new_bar_spans)
    return new_spans, None


def _span_center(span):
    x0, y0, x1, y1 = span["bbox"]
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _is_hat_candidate(span):
    return span["text"].strip() == _HAT_CHAR


def _is_vec_candidate(span):
    x0, y0, x1, y1 = span["bbox"]
    return _normalized_font_name(span.get("font")) in _ACCENT_FONT_NAMES and (x1 - x0) <= _MAX_ACCENT_WIDTH


def _sub_symbols(text):
    """Replaces every known Greek/math Unicode character in `text` with its LaTeX command (space-
    separated so it never glues onto adjacent letters, e.g. 'rho' + 'B' never reading as one
    token). Returns (new_text, replaced_any)."""
    if not text or not _SYMBOL_CHARS_RE.search(text):
        return text, False

    def _repl(m):
        return " \\" + _ALL_SYMBOL_COMMANDS[m.group(0)] + " "

    return _SYMBOL_CHARS_RE.sub(_repl, text), True


def _base_size_and_baseline(spans):
    """The zone's own 'body' size/baseline: the size value carrying the most total characters
    (not just the most common span, so one long sentence outweighs several short exponents), and
    the most common origin-y among spans at that size."""
    if not spans:
        return 10.0, 0.0
    char_count_by_size = {}
    for s in spans:
        n = len(s["text"].strip())
        if n == 0:
            continue
        key = round(s["size"], 1)
        char_count_by_size[key] = char_count_by_size.get(key, 0) + n
    if not char_count_by_size:
        return spans[0]["size"], spans[0]["origin"][1]
    # Tie-broken toward the LARGER size -- confirmed live (a minimal option with just one base
    # character and one exponent character, e.g. "m" + "2", a real but rare shape) that an exact
    # character-count tie otherwise resolves by dict insertion order, which depends on nothing
    # more meaningful than which span PyMuPDF happened to yield first. A real body/exponent pair
    # is never the SMALLER one carrying more or equal weight -- typesetting always makes the
    # exponent the smaller of the two.
    base_size = max(char_count_by_size, key=lambda sz: (char_count_by_size[sz], sz))
    baselines = [s["origin"][1] for s in spans if round(s["size"], 1) == base_size]
    base_baseline = max(set(baselines), key=baselines.count) if baselines else spans[0]["origin"][1]
    return base_size, base_baseline


def _local_baseline(spans, base_size):
    """base_size is reliable as a GLOBAL (zone-wide) reference -- a literal font-size value means
    the same thing anywhere on the page. A single absolute baseline y does not: a fraction's own
    numerator/denominator sits at a totally different page position than the zone's own dominant
    line, so reusing the zone-wide baseline there would compare an exponent's position against
    the wrong row entirely. This recomputes just the baseline, locally, from whichever spans in
    THIS pool are "normal"-sized by the (still global) base_size cutoff."""
    baselines = [s["origin"][1] for s in spans if s["size"] >= base_size * _SIZE_RATIO]
    if not baselines:
        return _base_size_and_baseline(spans)[1]
    return max(set(baselines), key=baselines.count)


def _classify_shift(span, base_size, base_baseline):
    if not span["text"].strip():
        return "normal"
    if span["size"] >= base_size * _SIZE_RATIO:
        return "normal"
    origin_y = span["origin"][1]
    if origin_y < base_baseline - _BASELINE_TOL:
        return "super"
    if origin_y > base_baseline + _BASELINE_TOL:
        return "sub"
    return "normal"


def _row_key(span):
    x0, y0, x1, y1 = span["bbox"]
    return (y0 + y1) / 2.0


def _rows_in_reading_order(spans, row_tol=6.0):
    """Clusters spans into visual rows by y-center (within row_tol of each other), sorts rows
    top-to-bottom, and sorts each row's spans left-to-right -- the same greedy 1D clustering
    process_pyq_vision._cluster_columns already uses for its own x-axis case, applied here to the
    y-axis so a genuinely multi-line option (ordinary text wrap, not a stacked fraction) still
    reads in the right order instead of being flattened by a single global x-sort. Returns a list
    of ROWS (each its own list of spans), not a flattened list -- a caller joining text across
    rows needs the row boundary itself: PyMuPDF's own per-line span text already carries whatever
    space belongs INSIDE one visual line, but two spans that only end up adjacent because they're
    the last token of one row and the first token of the next (ordinary paragraph wrap, not a
    stacked fraction) are never guaranteed to have a space between them without one deliberately
    inserted at that boundary -- confirmed live: a wrapped option's second line routinely starts
    with a word that has no leading space of its own at all."""
    by_y = sorted(spans, key=_row_key)
    rows = []
    for s in by_y:
        y = _row_key(s)
        if rows and y - rows[-1][-1][1] <= row_tol:
            rows[-1].append((s, y))
        else:
            rows.append([(s, y)])
    return [sorted((s for s, _ in row), key=lambda s: s["bbox"][0]) for row in rows]


def _join_plain_spans(spans, base_size=None, base_baseline=None):
    """Joins a flat span list in natural reading order with the same superscript/subscript
    wrapping _reconstruct_line_text uses, generalized to also cover subscript and to work across
    PyMuPDF's own separate "line" objects -- the whole point of this module (see its own
    docstring).

    base_size is normally the CALLER's own zone-wide value (passed down from reconstruct_zone),
    not recomputed from just `spans` -- a fraction's own numerator or denominator is often only
    1-2 spans (confirmed live: "144" over "pi^2"), where _base_size_and_baseline's own "most total
    characters" tie-break has nothing real to go on and can pick the SMALL exponent span as if it
    were the body size, silently erasing the exponent instead of marking it. base_baseline is
    always recomputed LOCALLY regardless (see _local_baseline) -- a fraction's own numerator and
    denominator sit at a page position far from the zone's own single dominant baseline, so
    reusing that one absolute y value here would compare against the wrong row entirely."""
    if not spans:
        return ""
    if base_size is None or not any(s["size"] >= base_size * _SIZE_RATIO for s in spans):
        base_size, base_baseline = _base_size_and_baseline(spans)
    else:
        base_baseline = _local_baseline(spans, base_size)
    rows = _rows_in_reading_order(spans)

    out = []
    shift = None  # None | "super" | "sub"
    for ri, row in enumerate(rows):
        if ri > 0:
            if shift:
                out.append("}")
                shift = None
            if out and not out[-1].endswith((" ", "{")):
                out.append(" ")
        for s in row:
            text, _ = _sub_symbols(s["text"])
            if not text:
                continue
            this_shift = _classify_shift(s, base_size, base_baseline)
            this_shift = this_shift if this_shift != "normal" else None
            if this_shift != shift:
                if shift:
                    out.append("}")
                if this_shift:
                    out.append("^{" if this_shift == "super" else "_{")
                shift = this_shift
            out.append(text)
    if shift:
        out.append("}")
    return "".join(out)


def _cluster_rows(items, key, tol=4.0):
    """Greedy 1D clustering into rows by y, same shape as _rows_in_reading_order's own row pass --
    kept separate since this one clusters (letter, marker_span) pairs, not bare spans."""
    by_y = sorted(items, key=key)
    rows = []
    for it in by_y:
        y = key(it)
        if rows and y - key(rows[-1][-1]) <= tol:
            rows[-1].append(it)
        else:
            rows.append([it])
    return rows


def build_option_zones(flat_spans, question_bbox):
    """Finds this question's own 4 bold "a."-"d." option markers and, from their own positions,
    builds a stem zone plus one zone per option -- covering the one-per-line, inline-on-one-line,
    and 2x2-grid layouts all from the same marker-row/marker-column geometry (requirement: group
    an option's fragments, including a stacked fraction in a grid cell, by its own label
    position). Returns (stem_zone, {letter: zone}, marker_span_ids) on success.

    Returns None -- the caller's signal to fall back to the old plain-text split entirely --
    whenever marker detection isn't unambiguous (not exactly one bold "a."/"b."/"c."/"d." span
    each): a genuinely different layout this module hasn't seen is far safer left to the existing
    text-based path (and its own existing needs_review gates) than forced through a zone split
    built on a wrong assumption."""
    qx0, qy0, qx1, qy1 = question_bbox
    markers = [s for s in flat_spans
               if "Bold" in (s.get("font") or "") and MARKER_RE.match(s["text"].strip())]
    by_letter = {}
    for m in markers:
        letter = m["text"].strip()[0].lower()
        by_letter.setdefault(letter, []).append(m)
    if set(by_letter.keys()) != {"a", "b", "c", "d"} or any(len(v) != 1 for v in by_letter.values()):
        return None
    marker_list = [(letter, by_letter[letter][0]) for letter in ("a", "b", "c", "d")]

    rows = _cluster_rows(marker_list, key=lambda it: it[1]["bbox"][1])
    for row in rows:
        row.sort(key=lambda it: it[1]["bbox"][0])
    rows.sort(key=lambda row: row[0][1]["bbox"][1])
    row_top_y = [row[0][1]["bbox"][1] for row in rows]

    # 3-4 markers sharing one row (every option inline on a single line, e.g. "a. 0.1  b. 0.5
    # c. 1  d. 5", or a 2x2 grid's own "a. ...  b. ...") usually has plenty of room -- the per-
    # option budget is still generous relative to a short answer. It only breaks down when an
    # option's own content is wide enough to reach the midpoint to its neighbor's own marker --
    # confirmed live three times, in three different shapes: a unit expression like "W m^-2 K^-4"
    # whose small exponent glyph straddles the midpoint outright; a same-size inline fraction
    # (numerator/denominator stacked above/below the marker's own single line) whose own glyphs
    # straddle it the same way, at ordinary body-text size; and a trailing small exponent (e.g.
    # "kg/m^3"'s own unit exponent) that doesn't straddle the midpoint at all but lands entirely
    # on the WRONG side of it, severed from the normal-size text it's actually attached to (here
    # "kg/m" itself, immediately to its left with no gap) because that preceding text already used
    # up most of the gap on its own. The THIRD shape needs its own check, not a wider distance
    # margin from the boundary -- a margin wide enough to catch it also caught a same-sized
    # exponent that legitimately sits well inside its own option's zone (confirmed live: a false
    # positive on an otherwise-fine reconstruction). Checked instead by adjacency: a small glyph
    # with an unclaimed, touching (near-zero gap) neighbor on the SAME row is only unsafe if a
    # boundary falls somewhere between that neighbor and itself -- i.e. the boundary would sever
    # two glyphs that belong together, not merely sit somewhere nearby. A same-size glyph
    # (ordinary prose, e.g. a wide "is:") straddling the midpoint by a hair is NOT this failure and
    # reconstructs fine, so it keeps the stricter straddle-only check.
    body_size, _ = _base_size_and_baseline(flat_spans)
    non_marker_ids = {id(m) for _, m in marker_list}
    content_spans = [s for s in flat_spans if id(s) not in non_marker_ids and s["text"].strip()]
    internal_boundaries = []  # (x, row_y0, row_y1, marker_top, marker_bottom)
    for ri, row in enumerate(rows):
        if len(row) < 2:
            continue
        row_y0 = min(it[1]["bbox"][1] for it in row) - 6
        row_y1 = (row_top_y[ri + 1] - 2) if ri + 1 < len(rows) else qy1 + 4
        marker_top = min(it[1]["bbox"][1] for it in row)
        marker_bottom = max(it[1]["bbox"][3] for it in row)
        for ci in range(len(row) - 1):
            boundary_x = (row[ci][1]["bbox"][0] + row[ci + 1][1]["bbox"][0]) / 2.0
            internal_boundaries.append((boundary_x, row_y0, row_y1, marker_top, marker_bottom))
    if internal_boundaries:
        for s in content_spans:
            sx0, sy0, sx1, sy1 = s["bbox"]
            s_cy = (sy0 + sy1) / 2.0
            is_small = s["size"] < body_size * _SIZE_RATIO
            left_neighbor = None
            if is_small:
                best_gap = None
                for other in content_spans:
                    if other is s:
                        continue
                    oy0, oy1 = other["bbox"][1], other["bbox"][3]
                    if abs((oy0 + oy1) / 2.0 - s_cy) > 6:
                        continue
                    gap = sx0 - other["bbox"][2]
                    if 0 <= gap < 3 and (best_gap is None or gap < best_gap):
                        left_neighbor, best_gap = other, gap
            for bx, by0, by1, m_top, m_bottom in internal_boundaries:
                if not (by0 <= s_cy <= by1):
                    continue
                straddles = sx0 < bx < sx1
                off_marker_line = sy0 < m_top - 3 or sy1 > m_bottom + 3
                severed = left_neighbor is not None and left_neighbor["bbox"][0] <= bx <= sx1
                if (straddles and (is_small or off_marker_line)) or severed:
                    return None  # too close to call -- fall back rather than guess the side

    zones = {}
    for ri, row in enumerate(rows):
        row_y0 = min(it[1]["bbox"][1] for it in row) - 6
        row_y1 = (row_top_y[ri + 1] - 2) if ri + 1 < len(rows) else qy1 + 4
        for ci, (letter, m) in enumerate(row):
            mx0 = m["bbox"][0]
            zone_x0 = (qx0 - 4) if ci == 0 else (row[ci - 1][1]["bbox"][0] + mx0) / 2.0
            zone_x1 = ((mx0 + row[ci + 1][1]["bbox"][0]) / 2.0) if ci + 1 < len(row) else (qx1 + 4)
            zones[letter] = (zone_x0, row_y0, zone_x1, row_y1)

    stem_zone = (qx0 - 4, qy0 - 4, qx1 + 4, min(row_top_y) - 2)
    marker_ids = {id(m) for _, m in marker_list}
    return stem_zone, zones, marker_ids


def _dist_to_rect(cx, cy, rect):
    x0, y0, x1, y1 = rect
    dx = max(x0 - cx, 0, cx - x1)
    dy = max(y0 - cy, 0, cy - y1)
    return (dx * dx + dy * dy) ** 0.5


def assign_spans_to_zones(flat_spans, stem_zone, option_zones, marker_ids):
    """Every remaining glyph (the 4 markers themselves are excluded by marker_ids) goes to
    whichever zone's own rect contains its center, or -- for a glyph sitting just outside every
    zone's own margin (e.g. a numerator poking slightly further above its option's marker than
    the zone's own margin allows for) -- to whichever zone's rect is nearest. Zones never overlap
    by construction (each one is bounded by the midpoint to its neighbor), so this never lets one
    option's content bleed into another (requirement: never cross an option boundary)."""
    buckets = {"stem": [], "a": [], "b": [], "c": [], "d": []}
    all_zones = {"stem": stem_zone, **option_zones}
    for s in flat_spans:
        if id(s) in marker_ids:
            continue
        cx, cy = _span_center(s)
        placed = False
        for name, rect in all_zones.items():
            x0, y0, x1, y1 = rect
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                buckets[name].append(s)
                placed = True
                break
        if not placed:
            best_name = min(all_zones, key=lambda name: _dist_to_rect(cx, cy, all_zones[name]))
            buckets[best_name].append(s)
    return buckets


# Safety cap on how many further spans a balanced-parenthesis radicand scan will walk before
# giving up -- a real radicand is never this long; reaching the cap means something else is going
# on (e.g. a stray "(" that was never really part of a root at all), so the whole root is left
# unresolved rather than guessed (see _extract_roots's own docstring).
_MAX_ROOT_SCAN = 12


def _extract_roots(spans, base_size, base_baseline):
    """Finds every literal U+221A (square-root) glyph already in the text layer (confirmed live:
    the ONLY root shape this sample's text layer ever carries a real signal for -- see this
    module's own docstring) and confidently pairs it with its own immediately-adjacent radicand.
    The √ itself is sometimes its own span and sometimes glued directly onto the start of its own
    radicand's span (e.g. "√2" as one span, confirmed via PyMuPDF's own insert_text behavior) --
    both are handled the same way by treating "whatever follows the √ character, in THIS span if
    anything remains after it, else in the very next span" as the start of the radicand. From
    there: that text alone if its own parens (if any) already balance, OR -- when it opens an
    UNbalanced parenthesis -- everything up to the matching close paren in a later span (stripped
    off the result, since \\sqrt{...} already supplies its own delimiter), capped at
    _MAX_ROOT_SCAN further spans. A bare √ with nothing immediately after it, or an unbalanced/
    too-long parenthesis run, is left completely alone -- still literal text in the returned list,
    so it reads as a stray glyph and falls through to the existing needs_review gates instead of
    ever being guessed at."""
    rows = _rows_in_reading_order(spans)
    consumed_ids = set()
    replacements = []
    for row in rows:
        i = 0
        while i < len(row):
            s = row[i]
            text = s["text"]
            stripped = text.strip()
            if not stripped.startswith(_ROOT_CHAR):
                i += 1
                continue
            glued_remainder = text[text.index(_ROOT_CHAR) + 1:]
            if glued_remainder.strip():
                first_text = glued_remainder
                tail_start = i + 1
                radicand_bits_text = [glued_remainder]
                radicand_bits_spans = []
            elif i + 1 < len(row) and row[i + 1]["bbox"][0] - s["bbox"][2] <= 2:
                first_text = row[i + 1]["text"]
                tail_start = i + 2
                radicand_bits_text = [first_text]
                radicand_bits_spans = [row[i + 1]]
            else:
                i += 1
                continue  # nothing follows -- leave the bare root as literal text
            depth = first_text.count("(") - first_text.count(")")
            k = tail_start
            while depth > 0 and k < len(row) and (k - i) < _MAX_ROOT_SCAN:
                t = row[k]["text"]
                depth += t.count("(") - t.count(")")
                radicand_bits_text.append(t)
                radicand_bits_spans.append(row[k])
                k += 1
            if depth != 0:
                i += 1
                continue  # never balanced within the scan cap -- don't guess where it ends
            radicand = "".join(radicand_bits_text).strip()
            if radicand.startswith("(") and radicand.endswith(")"):
                radicand = radicand[1:-1]
            radicand, _ = _sub_symbols(radicand)
            all_bits = [s] + radicand_bits_spans
            consumed_ids.add(id(s))
            for g in radicand_bits_spans:
                consumed_ids.add(id(g))
            replacements.append({
                "text": "\\sqrt{" + radicand + "}",
                "bbox": (min(b["bbox"][0] for b in all_bits), min(b["bbox"][1] for b in all_bits),
                         max(b["bbox"][2] for b in all_bits), max(b["bbox"][3] for b in all_bits)),
                "origin": s["origin"], "size": s["size"], "font": s.get("font"), "is_math": True,
            })
            i = k  # past the root span itself and every span consumed into the radicand
    kept = [s for s in spans if id(s) not in consumed_ids]
    return kept + replacements


def reconstruct_zone(spans, drawings, zone_bbox):
    """Main entry point: reconstructs one question-stem or option's own text from its raw glyph
    spans (list of PyMuPDF span dicts restricted to this zone) plus the page's own vector
    drawings (for fraction-bar detection, restricted to this zone by get_zone_fraction_bars).
    Returns (text, bar_issue) -- text already wrapped in $...$ around every math run -- never
    raises, never drops a glyph: anything not confidently reconstructed is passed through as
    literal text. bar_issue is None unless this zone's own modulus/magnitude bars could not be
    confidently paired (see _normalize_modulus_bars), in which case text is returned with those
    bar characters completely unchanged and the caller should flag the question needs_review with
    bar_issue as the reason."""
    spans = [s for s in spans if s.get("text")]
    if not spans:
        return "", None

    # Computed ONCE from the zone's own full span pool (before anything is claimed by a fraction)
    # and reused everywhere below -- a fraction's numerator/denominator is often only 1-2 spans,
    # too few for this same "most total characters" heuristic to reliably tell body text apart
    # from an exponent on its own (see _join_plain_spans's own docstring).
    base_size, base_baseline = _base_size_and_baseline(spans)
    spans, bar_issue = _normalize_modulus_bars(spans, drawings, zone_bbox, base_size)

    claimed_ids = set()
    pool = []  # list of dicts: {text, bbox, origin, size, font, is_math}

    # ---- Fractions first (the most structurally confident signal: an actual drawn bar) ----
    bars = get_zone_fraction_bars(drawings, zone_bbox)
    for bar_x0, bar_y0, bar_x1 in bars:
        num_spans, den_spans = [], []
        for s in spans:
            if id(s) in claimed_ids:
                continue
            cx, cy = _span_center(s)
            if not (bar_x0 - _FRAC_HORIZONTAL_MARGIN <= cx <= bar_x1 + _FRAC_HORIZONTAL_MARGIN):
                continue
            if bar_y0 - _FRAC_VERTICAL_REACH <= cy < bar_y0:
                num_spans.append(s)
            elif bar_y0 < cy <= bar_y0 + _FRAC_VERTICAL_REACH:
                den_spans.append(s)
        if not num_spans or not den_spans:
            continue  # no confident fraction here -- leave every span in the plain pool
        for s in num_spans + den_spans:
            claimed_ids.add(id(s))
        num_text = _join_plain_spans(num_spans, base_size, base_baseline)
        den_text = _join_plain_spans(den_spans, base_size, base_baseline)
        pool.append({
            "text": "\\frac{" + num_text + "}{" + den_text + "}",
            "bbox": (bar_x0, bar_y0 - _FRAC_VERTICAL_REACH, bar_x1, bar_y0 + _FRAC_VERTICAL_REACH),
            "origin": (bar_x0, bar_y0),
            "size": 0, "font": "", "is_math": True,
        })

    remaining = _extract_roots([s for s in spans if id(s) not in claimed_ids], base_size, base_baseline)

    # ---- Hat/vector diacritics: pair each candidate with its own unique nearest base letter ----
    diacritics = [s for s in remaining if _is_hat_candidate(s) or _is_vec_candidate(s)]
    bases = [s for s in remaining if s not in diacritics]
    base_wraps = {}  # id(base_span) -> "hat" | "vec"
    used_diacritic_ids = set()
    for d in diacritics:
        dx, _ = _span_center(d)
        _, _, _, d_bottom = d["bbox"]
        candidates = []
        for b in bases:
            if id(b) in base_wraps:
                continue
            bx, _ = _span_center(b)
            b_top = b["bbox"][1]
            gap = b_top - d_bottom
            if -_DIACRITIC_MAX_OVERLAP <= gap <= _DIACRITIC_MAX_GAP and abs(bx - dx) <= _DIACRITIC_MAX_DX:
                candidates.append((abs(bx - dx) + max(0, gap), b))
        if not candidates:
            continue  # no confident pairing -- leave the diacritic glyph as literal stray text
        candidates.sort(key=lambda c: c[0])
        if len(candidates) >= 2 and candidates[1][0] - candidates[0][0] < 0.5:
            continue  # ambiguous (two equally-close bases) -- stay unresolved, don't guess
        best = candidates[0][1]
        base_wraps[id(best)] = "hat" if _is_hat_candidate(d) else "vec"
        used_diacritic_ids.add(id(d))

    plain_pool = []
    for s in remaining:
        if id(s) in used_diacritic_ids:
            continue
        wrap = base_wraps.get(id(s))
        if wrap:
            plain_pool.append({
                "text": "\\" + wrap + "{" + s["text"].strip() + "}",
                "bbox": s["bbox"], "origin": s["origin"], "size": s["size"], "font": s["font"],
                "is_math": True,
            })
        else:
            text, replaced = _sub_symbols(s["text"])
            plain_pool.append({
                "text": text, "bbox": s["bbox"], "origin": s["origin"], "size": s["size"],
                # `or s.get("is_math")`: a root's own replacement entry (see _extract_roots) is
                # already a fully-formed "\sqrt{...}" string with its is_math flag pre-set --
                # _sub_symbols finds nothing new to substitute inside it and would otherwise
                # silently overwrite that flag back to False, losing its own $...$ wrap below.
                "font": s.get("font"), "is_math": replaced or bool(s.get("is_math")),
            })

    # ---- Superscript/subscript classification for whatever's left, then final assembly ----
    # (base_size/base_baseline: the same zone-wide values computed at the top of this function)
    rows = _rows_in_reading_order(pool + plain_pool)

    out = []
    shift = None
    in_math = False
    for ri, row in enumerate(rows):
        if ri > 0:
            if shift:
                out.append("}")
                shift = None
            if in_math:
                out.append("$")
                in_math = False
            if out and not out[-1].endswith((" ", "{")):
                out.append(" ")

        # Pre-pass: classify every span in this row first, then absorb a PLAIN token backward
        # into its immediate math neighbor's run whenever they touch with no gap (e.g. the base
        # letter "V" immediately before its own "_{A}" subscript) -- same "glued directly, no
        # space" shape process_pyq_vision.wrap_bare_latex_notation's own docstring already uses
        # for this, confirmed live: without it, "V" stayed plain and only the subscript got its
        # own separate $...$, rendering as two adjacent expressions ("V$_{A}$") instead of one
        # ("$V_{A}$").
        classified = []
        for s in row:
            text = s["text"]
            if not text:
                continue
            is_frac_or_wrap = s.get("is_math") and ("\\frac" in text or "\\hat" in text or "\\vec" in text or "\\sqrt" in text)
            this_shift = None
            if not is_frac_or_wrap and s.get("size"):
                this_shift = _classify_shift(s, base_size, base_baseline)
                this_shift = this_shift if this_shift != "normal" else None
            is_math_token = bool(s.get("is_math")) or this_shift is not None
            classified.append([s, is_math_token, this_shift])
        for idx in range(len(classified) - 2, -1, -1):
            cur_s, cur_math, cur_shift = classified[idx]
            next_s, next_math, _ = classified[idx + 1]
            if not cur_math and next_math and (next_s["bbox"][0] - cur_s["bbox"][2]) <= 1:
                classified[idx][1] = True

        for s, is_math_token, this_shift in classified:
            text = s["text"]
            # Close, then open -- in that order, and shift nested INSIDE math both ways: closing
            # must end "}$" (the sub/superscript wrap ends before the math run it lives inside
            # does), and opening must start "$^{"/"$_{" (math opens before the wrap it's about to
            # hold). Doing this as one combined "if this_shift != shift" toggle before even
            # checking is_math_token (as an earlier version of this function did) emits the wrap
            # before the $ that should enclose it -- confirmed live: a subscript opening a fresh
            # math run came out "_{$A}$" instead of "$_{A}$".
            if shift is not None and this_shift != shift:
                out.append("}")
                shift = None
            if in_math and not is_math_token:
                out.append("$")
                in_math = False
            if is_math_token and not in_math:
                out.append("$")
                in_math = True
            if this_shift is not None and this_shift != shift:
                out.append("^{" if this_shift == "super" else "_{")
                shift = this_shift
            out.append(text)

    if shift:
        out.append("}")
    if in_math:
        out.append("$")

    result = "".join(out)
    result = re.sub(r"\$\s*\$", " ", result)  # two adjacent math runs with nothing real between
    result = _space_after_closing_dollars(result)
    result = re.sub(r"[ \t]+", " ", result).strip()
    return result, bar_issue


def _space_after_closing_dollars(text):
    """A math run's own CLOSING "$" is followed directly by the next word with no space between
    them whenever that word's own leading space lived inside the just-closed run instead (e.g. a
    Greek symbol substitution's own trailing space, added by _sub_symbols, ends up just before the
    closing $ rather than after it) -- confirmed live: "$\\propto$is" instead of "$\\propto$ is".
    Walks the string tracking $ parity (every even one is a close, same convention
    process_pyq_vision._fix_unescaped_json_backslashes already uses for $$ pairs) and inserts one
    space after a closing $ immediately followed by a letter or digit."""
    out = []
    dollar_count = 0
    for i, ch in enumerate(text):
        out.append(ch)
        if ch == "$":
            dollar_count += 1
            if dollar_count % 2 == 0 and i + 1 < len(text) and text[i + 1].isalnum():
                out.append(" ")
    return "".join(out)
