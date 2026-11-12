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
manual review -- this module adds no new flagging logic of its own beyond the font-aware symbol
table's own unrecognized-glyph case (see that section), it only reduces how often the existing
gates have something real to catch.

Known, deliberately UNFIXED gap (found live via process_mock_exam_text.py's own order-mismatch
check, not guessed): the real sample's own "Fewness,Bold" font -- its own embedded subset is
mostly an ordinary BOLD TEXT face (every bold a-d marker and bold question number on much of the
document uses it) -- decodes a "<"/">" comparison-operator glyph to a bare U+0020 SPACE in this
PDF's own ToUnicode table (confirmed live, Chemistry Q20's own "NaCl < NH4Cl < NaCN < HCl"-style
option). This is NOT in the font-aware table above: PyMuPDF's own text-extraction API (get_text
"dict"/"rawdict") only ever exposes the ALREADY-DECODED (here wrong) Unicode character, never the
underlying raw glyph index/CID a (font, glyph-code) mapping would need to tell a real "<" apart
from an ordinary bold-text space using this SAME font (confirmed live: hundreds of genuine bold-
marker spaces share the exact same font/size/width signature as the handful of real "<" signs --
no geometry-only rule tried could separate them without risking a wrong guess elsewhere). Per this
module's own confidence rule, guessing was rejected -- process_mock_exam_text.py's own order-
mismatch check (_check_order_mismatch) is what actually catches a question built from this gap.
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

# A SMALL (sub/superscript-sized) candidate gets a tighter margin than an ordinary body-sized one
# -- confirmed live, MockTest1.pdf Physics Q39: the trailing "1" of an inverse-trig "tan^{-1}"
# exponent sitting immediately before a fraction's own bar had its own center just 0.4pt inside
# the ordinary 4pt margin, so the fraction wrongly claimed it as part of its own numerator while
# its own immediately-preceding sibling character ("-", the same exponent run) stayed outside and
# was correctly left as plain superscript text -- splitting one exponent run in half. The wider
# 4pt margin exists for a genuinely wide BODY-sized glyph (a tall paren, confirmed live, Q6)
# whose own bbox legitimately pokes past the bar's drawn extent; a small glyph has no such
# legitimate reason to sit outside the bar's own narrower true span in the first place.
_FRAC_HORIZONTAL_MARGIN_SMALL = 1

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
    "∆": "Delta",  # U+2206 INCREMENT -- this PDF's own Symbol font uses this, not U+0394, for
    # a capital delta "change in X" (confirmed live: a page already past the answers heading
    # uses it repeatedly as "∆" standing alone, never once as U+0394 anywhere in the sample).
    "Ω": "Omega",  # U+2126 OHM SIGN -- this PDF's own Symbol font prints a literal ohm unit
    # ("3Ω", "R = 300 Ω") with this codepoint, not U+03A9 (confirmed live, real option text on
    # the sample's own Physics Q45) -- visually identical to capital Greek omega, same LaTeX.
}
_SYMBOL_COMMANDS = {
    "±": "pm", "×": "times", "÷": "div", "≈": "approx",
    "≠": "neq", "≤": "leq", "≥": "geq", "∝": "propto",
    "→": "rightarrow", "←": "leftarrow", "∞": "infty",
    "°": "circ", "≡": "equiv", "∂": "partial", "∇": "nabla",
    "⇒": "Rightarrow", "∴": "therefore", "∠": "angle", "⋅": "cdot",
    "′": "prime",
}
_ALL_SYMBOL_COMMANDS = {**_GREEK_COMMANDS, **_SYMBOL_COMMANDS}
_SYMBOL_CHARS_RE = re.compile("[" + "".join(re.escape(c) for c in _ALL_SYMBOL_COMMANDS) + "]")

# A literal U+221A square-root character already in the text layer is reconstructed with
# confidence (requirement: handle roots) -- but ONLY this clean case. This sample's own roots are
# built from drawn paths with no reliable glyph marking where the radicand ends (confirmed live:
# no U+221A anywhere in the sample's text layer at all), so that shape is deliberately left
# unhandled rather than guessed -- see module docstring's confidence rule.
_ROOT_CHAR = "√"

# ---------------------------------------------------------------------------------------------
# Font-aware symbol table. Every character above is mapped by itself, font-independent, because
# it already extracts as correct, universal Unicode (confirmed live) -- a real Greek letter means
# the same thing no matter which font drew it. This PDF's own embedded "Symbol" font subset ALSO
# emits characters in the Private Use Area (U+F8E5-U+F8FB, confirmed live via get_text("dict")'s
# own span "font" field) that have NO such universal meaning: a bare PUA codepoint is only
# meaningful alongside the FONT that produced it. Inspecting this sample's own "Symbol" font
# usage directly (not guessed from rendered shape) turned up Microsoft Equation Editor/MathType's
# own long-standing convention for drawing a TALL parenthesis/bracket out of several vertically-
# stacked pieces (a top hook, a repeatable middle extension, a bottom hook) around a multi-row
# expression, and for extending an arrow's own shaft to an arbitrary printed length -- confirmed
# live by the exact evidence below, not assumed from the codepoints' names:
#   - every LEFT-bracket-piece codepoint U+F8EB-U+F8F0 has its own RIGHT-bracket-piece partner at
#     EXACTLY +0x0B (11) -- checked against every real occurrence in the sample, which always
#     uses a left codepoint and its own +0x0B partner together, in matched same-size sets.
#   - U+F8E7 is always immediately followed, on the same row, by a real arrowhead character ("->"
#     U+2192 or "=>" U+21D2, both already in _SYMBOL_COMMANDS above) after possibly more of
#     itself -- a longer printed arrow repeats this shaft piece more times, confirmed live
#     ("C H I [x3]-> Intermediate").
# This table is keyed on (normalized font name, character) and checked ONLY for a span whose own
# font is a known Symbol-style math font (_SYMBOL_FONT_NAMES) -- the same codepoint in an ordinary
# text font is just an unrelated private-use character, never looked up here at all.
_SYMBOL_FONT_NAMES = {"symbol"}
_LEFT_BRACKET_PIECE_CHARS = frozenset(chr(c) for c in range(0xF8EB, 0xF8F1))   # F8EB..F8F0
_RIGHT_BRACKET_PIECE_CHARS = frozenset(chr(c) for c in range(0xF8F6, 0xF8FC))  # F8F6..F8FB
_ARROW_SHAFT_EXTENDER_CHAR = chr(0xF8E7)
_ARROWHEAD_CHARS = {"→", "⇒"}
# The full Private Use Area range this sample's own "Symbol" font subset is confirmed to draw
# from (see above) -- any character in here, in a Symbol-font span, that is NOT one of the
# specific pieces this table already recognizes is an unknown glyph (a different equation-editor
# construct this sample never happened to use) rather than a guess: per this module's own
# confidence rule, _collapse_symbol_font_glyphs leaves it as literal text AND reports an issue for
# the caller to flag needs_review with, rather than ever letting an un-renderable private-use
# character silently reach the output unflagged.
_SYMBOL_FONT_PUA_RANGE = (0xF8E0, 0xF8FF)


def _collapse_symbol_font_glyphs(spans):
    """Collapses every run of this zone's own Symbol-font bracket-piece glyphs (one character per
    vertical piece -- top/extension/bottom, see this section's own docstring) into a single "("
    or ")" character, and drops an arrow-shaft-extender run that leads into a real arrowhead
    character. Returns (new_spans, issue) -- issue is a human-readable reason the moment ANY
    Symbol-font Private Use Area character is left over afterwards unrecognized (never silently
    passed through as a literal, un-renderable character); new_spans is returned completely
    UNCHANGED (not even reordered) whenever this zone has no Symbol-font content at all, the
    overwhelmingly common case, so this is cheap to call for every zone."""
    if not any(_normalized_font_name(s.get("font")) in _SYMBOL_FONT_NAMES
               and any(_SYMBOL_FONT_PUA_RANGE[0] <= ord(c) <= _SYMBOL_FONT_PUA_RANGE[1] for c in (s.get("text") or ""))
               for s in spans):
        return spans, None

    rows = _rows_in_reading_order(spans)
    row_of_id = {id(s): row for row in rows for s in row}

    out = []
    issue = None
    # A bracket construct TALL enough to need more than one size-tier of pieces per side has each
    # piece on its own different baseline -- confirmed live, the real sample's own Q40 (Biot-
    # Savart's law, a fraction wrapped in a tall paren around ANOTHER fraction): unlike a short
    # single-row construct (confirmed glued into one span's own text, see the within-span loop
    # below), PyMuPDF never merges characters across different baselines into one span, so each
    # vertical piece lands in its OWN separate span. Collapsing each one independently would emit
    # one "(" per PIECE instead of one for the whole construct -- confirmed live, "((((" -- so a
    # left/right side with piece characters split across more than one span is counted here and
    # flagged below rather than guessed at.
    left_piece_span_count = 0
    right_piece_span_count = 0
    for row in rows:
        for s in row:
            font = _normalized_font_name(s.get("font"))
            text = s.get("text") or ""
            if font not in _SYMBOL_FONT_NAMES or not any(
                    _SYMBOL_FONT_PUA_RANGE[0] <= ord(c) <= _SYMBOL_FONT_PUA_RANGE[1] for c in text):
                out.append(s)
                continue
            if any(c in _LEFT_BRACKET_PIECE_CHARS for c in text):
                left_piece_span_count += 1
            if any(c in _RIGHT_BRACKET_PIECE_CHARS for c in text):
                right_piece_span_count += 1
            # Rebuild this ONE span's own text, character by character -- confirmed live, a
            # SHORT construct's own pieces sit glued into a single span's text already (PyMuPDF
            # merges adjacent same-font/size characters into one span regardless of which
            # individual glyph each one is).
            new_text = []
            j, n = 0, len(text)
            while j < n:
                ch = text[j]
                if ch in _LEFT_BRACKET_PIECE_CHARS:
                    while j < n and text[j] in _LEFT_BRACKET_PIECE_CHARS:
                        j += 1
                    new_text.append("(")
                elif ch in _RIGHT_BRACKET_PIECE_CHARS:
                    while j < n and text[j] in _RIGHT_BRACKET_PIECE_CHARS:
                        j += 1
                    new_text.append(")")
                elif ch == _ARROW_SHAFT_EXTENDER_CHAR:
                    k = j
                    while k < n and text[k] == _ARROW_SHAFT_EXTENDER_CHAR:
                        k += 1
                    followed_by_arrowhead = k < n and text[k] in _ARROWHEAD_CHARS
                    if not followed_by_arrowhead:
                        row_list = row_of_id[id(s)]
                        pos = next(idx for idx, x in enumerate(row_list) if x is s)
                        for later in row_list[pos + 1:]:
                            later_text = (later.get("text") or "").lstrip()
                            if later_text:
                                followed_by_arrowhead = later_text[0] in _ARROWHEAD_CHARS
                                break
                    if followed_by_arrowhead:
                        j = k  # the shaft contributes nothing -- the arrowhead carries the meaning
                    else:
                        issue = f"an unrecognized Symbol-font glyph (U+{ord(ch):04X})"
                        new_text.append(ch)  # left as literal -- never silently dropped either
                        j += 1
                elif _SYMBOL_FONT_PUA_RANGE[0] <= ord(ch) <= _SYMBOL_FONT_PUA_RANGE[1]:
                    issue = f"an unrecognized Symbol-font glyph (U+{ord(ch):04X})"
                    new_text.append(ch)
                    j += 1
                else:
                    new_text.append(ch)
                    j += 1
            ns = dict(s)
            ns["text"] = "".join(new_text)
            out.append(ns)
    if left_piece_span_count > 1 or right_piece_span_count > 1:
        issue = "a tall Symbol-font bracket spanning multiple lines could not be confidently collapsed"
    return out, issue


def _is_fraction_bar_rect(rect, zone_bbox):
    x0, y0, x1, y1 = rect
    if abs(y1 - y0) > 0.75:
        return False
    width = x1 - x0
    if width < 4 or width > 220:
        return False
    zx0, zy0, zx1, zy1 = zone_bbox
    return zx0 - 2 <= x0 and x1 <= zx1 + 2 and zy0 - 2 <= y0 <= zy1 + 2


def get_zone_fraction_bars(drawings, zone_bbox, diagram_bboxes=()):
    """drawings: page.get_drawings() output (fetched once per page by the caller). Returns a list
    of (x0, y0, x1) for every thin horizontal line drawing that falls inside zone_bbox -- a
    fraction bar never spans outside the option/stem zone it belongs to, which is what keeps this
    from ever claiming content across an option boundary (see this module's own docstring).

    Same two exclusions get_zone_vertical_bars already applies to a modulus bar, confirmed live
    this section has the identical exposure (see that function's own docstring for how each was
    found): _is_plain_stroke rejects a filled rectangle (a table's own cell border/shading, type
    "f" -- a different drawing primitive than a real bar's stroked "l" line entirely), and
    diagram_bboxes rejects a stroke that is really part of a diagram's own line art. Confirmed
    live on a REAL false match (the real sample's own Biology Q34): a question whose own content
    runs to near a page's own bottom accumulates a stray near-empty block from the TOP of the
    NEXT page (harmless on its own -- just whitespace) that pulls that next page's own geometry
    into this zone's own drawings lookup; zone_bbox's own y-range, a naive union across the two
    pages, then coincidentally overlaps that NEXT page's own unrelated table border, which this
    check rejects outright rather than ever risking a match this far from anything the zone's own
    content could possibly mean."""
    bars = []
    for d in drawings:
        rect = d.get("rect")
        if rect is None:
            continue
        rect = (rect[0], rect[1], rect[2], rect[3])
        if not _is_fraction_bar_rect(rect, zone_bbox):
            continue
        if not _is_plain_stroke(d):
            continue
        if diagram_bboxes and _in_any_bbox(rect, diagram_bboxes):
            continue
        bars.append((rect[0], rect[1], rect[2]))
    return bars


# ---------------------------------------------------------------------------------------------
# Modulus/magnitude bars: "|x|", "|a + b + c|", a vector norm "||v||". A bar is either a literal
# "|" already in the text layer, a look-alike character (DIVIDES U+2223, PARALLEL TO U+2225 -
# both anticipated; DOUBLE VERTICAL LINE U+2016, confirmed live in the real sample), or a short
# vertical vector-drawn stroke with no character at all - the same "no reliable glyph" shape this
# module's own root handling already anticipates for a drawn-overline root. Two bar occurrences
# sitting immediately next to each other on the same row - two single characters glued together
# (e.g. "||"), or one inherently-double glyph like "‖" - are ONE logical delimiter and rebuilt as
# "||", never as two separate single bars.
#
# Regression (2026-11-11, fixed here): an earlier version of this section pushed the real
# sample's own flagged count from 31 to 42 of 180 (the reconstruction path's own engagement rate
# was never affected either way - the regression was purely extra flags, not a fallback). Root
# cause, confirmed by rendering the actual source pages rather than trusting geometry alone:
# almost every false-positive candidate had nothing to do with modulus bars at all. Counted across
# the whole real sample, 453 of 482 drawn-stroke candidates were a Match-the-Following table's own
# cell border/shading, drawn as a thin FILLED RECTANGLE (PyMuPDF drawing type "f", a single "re"
# item) - a different drawing primitive than a real bar's stroked "l" line, not a geometry edge
# case worth tuning thresholds around. Most of the rest were a genuine drawn "l" stroke that is
# really part of a circuit diagram's own battery symbol, a structural formula's own ring bond, or
# a dumbbell/graph figure's own line art - these DO share a real bar's own stroke primitive, so
# they are only told apart by checking the page's own already-detected diagram bboxes (the same
# detector process_mock_exam_text.py already calls for has_diagram). A single genuine TEXT double-
# bond character was also found, in "CH3-C(=NH)(NH2)" - a lone, unpaired mark with no partner
# anywhere nearby, not a drawn stroke at all.
#
# get_zone_vertical_bars (see its own docstring) excludes a candidate drawn stroke outright for
# either reason above, before it is ever treated as a "bar occurrence" worth considering at all.
# _normalize_modulus_bars itself never acts on a stray, partner-less mark of any kind (text or
# drawn): a row with fewer than two bar-like units has nothing to pair and is left completely
# alone, no rewrite and no flag, exactly like any other glyph this module does not confidently
# reconstruct - "never reject or bail a zone for a stray stroke elsewhere" is a correctness
# requirement here, not just a performance nicety. Only once a row genuinely has two or more units
# left standing does the original "wrong is worse than flagged" rule below apply.
#
# Confidence rule (same "wrong is worse than flagged" requirement as the rest of this module): a
# ROW's own bar occurrences are only rebuilt if they pair up EXACTLY - an even count, each pair
# enclosing real, non-empty content on that same row - checked for every row in the zone before
# anything is rewritten. The whole zone is rewritten together, all-or-nothing: by construction,
# either every occurrence maps 1:1 onto an output "|"/"||" (so a bar can never be silently lost by
# this rewrite - there is no code path that drops one without ALSO leaving the zone's text
# untouched and reporting the problem back), or none of them are touched at all and the zone's own
# bar characters stay exactly as literal text, with a reason reported back for the caller to flag
# needs_review with - never a partial rewrite, never a guess at which side of an unpaired bar is
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


def _is_plain_stroke(d):
    """True only for a genuine drawn LINE stroke (PyMuPDF drawing type "s", every item a straight
    line segment "l") -- confirmed live (MockTest1.pdf, see this section's own "Regression"
    comment above): a real modulus bar is drawn exactly this way, while a table's own cell
    border/shading is a FILLED RECTANGLE ("re", type "f") instead - 453 of 482 candidates on the
    real sample, a different drawing primitive entirely, not a geometry edge case."""
    if d.get("type") != "s":
        return False
    items = d.get("items") or []
    return bool(items) and all(it[0] == "l" for it in items)


def _in_any_bbox(rect, bboxes):
    cx, cy = (rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0
    return any(b[0] - 2 <= cx <= b[2] + 2 and b[1] - 2 <= cy <= b[3] + 2 for b in bboxes)


# ---------------------------------------------------------------------------------------------
# Drawn root (square root) glyph detection -- requirement 2/pass bar: this module's own confidence
# rule already refuses to guess where a DRAWN root's radicand ends (_ROOT_CHAR's own docstring --
# this sample's root sign is never a real U+221A character, only ever a drawn checkmark+overline
# path, confirmed live: zero occurrences of the real character anywhere in the sample's text
# layer). A drawn root that is silently dropped leaves NO trace at all in the text layer for any
# other gate here to catch -- confirmed live, MockTest1.pdf Physics Q4 and Q11: "2{root}5" reads
# as a perfectly ordinary, clean "25" once the root vanishes, no stray glyph, nothing a word-count
# or garbled-token check could ever flag. This section's only job is to confirm the drawn shape
# EXISTS somewhere in scope, so the caller can flag needs_review instead of ever losing it
# silently -- it never attempts to reconstruct the radicand itself.
#
# Signature (confirmed live against every one of this sample's own 5 known root occurrences --
# Q4 options a-d and Q11 option d, all independently verified against the real page image): a
# stroke ("s") drawing of EXACTLY 4 straight "l" segments tracing the checkmark+overline shape,
# matched by a near-identical-bbox FILL ("f") drawing (that same glyph's own bold outline, more
# segments) within _ROOT_GLYPH_PAIR_TOL -- this redundant stroke+fill pairing at a small, body-
# text-ish size is specific enough that nothing else on this sample (table borders, circuit/ring
# diagrams -- already told apart elsewhere by bbox/stroke-primitive, see get_zone_vertical_bars's
# own docstring) matches it by accident.
_ROOT_GLYPH_STROKE_ITEMS = 4
_ROOT_GLYPH_FILL_MIN_ITEMS = 8
_ROOT_GLYPH_MIN_SIZE = 6.0
_ROOT_GLYPH_MAX_SIZE = 16.0
_ROOT_GLYPH_PAIR_TOL = 1.0


def _is_root_checkmark_stroke(d):
    if d.get("type") != "s":
        return False
    items = d.get("items") or []
    return len(items) == _ROOT_GLYPH_STROKE_ITEMS and all(it[0] == "l" for it in items)


def detect_drawn_root_glyphs(drawings, scope_bbox, content_y_centers, diagram_bboxes=()):
    """True the moment a drawn root checkmark+overline glyph (see this section's own docstring) is
    found within scope_bbox -- never reconstructs anything, only detects. diagram_bboxes: excludes
    a candidate sitting inside an already-detected diagram region, same exclusion
    get_zone_vertical_bars/get_zone_fraction_bars already apply to their own drawn shapes.

    content_y_centers: this question's own real glyphs' own y-centers -- a candidate is also
    required to sit within _near_real_content's own reach of ONE of them, the same protection
    _normalize_modulus_bars already relies on for its own drawn strokes (see that function's own
    docstring): a question's own accumulated bbox is occasionally far larger than its own real
    content (confirmed live, MockTest1.pdf Physics Q7 -- a PRE-EXISTING block-accumulation
    artifact unrelated to this detector's own job, its own bbox stretches across nearly this
    page's entire left column) -- scope_bbox containment ALONE would then wrongly match a root
    glyph that actually belongs to a completely different, unrelated question sharing the same
    inflated bbox."""
    strokes = [d for d in drawings if _is_root_checkmark_stroke(d)]
    if not strokes:
        return False
    fills = [d for d in drawings if d.get("type") == "f"
             and len(d.get("items") or []) >= _ROOT_GLYPH_FILL_MIN_ITEMS]
    sx0, sy0, sx1, sy1 = scope_bbox
    for s in strokes:
        rect = s.get("rect")
        if rect is None:
            continue
        rx0, ry0, rx1, ry1 = rect
        w, h = rx1 - rx0, ry1 - ry0
        if not (_ROOT_GLYPH_MIN_SIZE <= w <= _ROOT_GLYPH_MAX_SIZE
                and _ROOT_GLYPH_MIN_SIZE <= h <= _ROOT_GLYPH_MAX_SIZE):
            continue
        cx, cy = (rx0 + rx1) / 2.0, (ry0 + ry1) / 2.0
        if not (sx0 - 2 <= cx <= sx1 + 2 and sy0 - 2 <= cy <= sy1 + 2):
            continue
        if not _near_real_content(rect, content_y_centers):
            continue
        if diagram_bboxes and _in_any_bbox(rect, diagram_bboxes):
            continue
        for f in fills:
            frect = f.get("rect")
            if frect is None:
                continue
            if all(abs(a - b) <= _ROOT_GLYPH_PAIR_TOL for a, b in zip(rect, frect)):
                return True
    return False


def get_zone_vertical_bars(drawings, zone_bbox, diagram_bboxes=()):
    """Same shape as get_zone_fraction_bars (see its own docstring), transposed to vertical
    strokes -- a drawn modulus bar never spans outside its own option/stem zone, for the same
    reason a fraction bar never does. Two further confirmed-live exclusions (see this section's
    own "Regression" comment above for how each was found): _is_plain_stroke rejects a filled
    rectangle (a table's own cell border/shading), and diagram_bboxes - this zone's own page(s)'
    already-detected diagram regions, same detector process_mock_exam_text.py already calls for
    has_diagram - rejects a stroke that is really part of a circuit/ring/figure's own line art.
    Neither is a geometry edge case: a battery symbol's own plate, or a structural formula's own
    ring bond, passes every width/height bound a real modulus bar does, and is only told apart by
    one of these two checks."""
    bars = []
    for d in drawings:
        rect = d.get("rect")
        if rect is None:
            continue
        rect = (rect[0], rect[1], rect[2], rect[3])
        if not _is_vertical_bar_rect(rect, zone_bbox):
            continue
        if not _is_plain_stroke(d):
            continue
        if diagram_bboxes and _in_any_bbox(rect, diagram_bboxes):
            continue
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


def _normalize_modulus_bars(spans, drawings, zone_bbox, body_size, diagram_bboxes=()):
    """Finds every modulus/magnitude bar occurrence in this zone (text-layer character or drawn
    vertical stroke - see get_zone_vertical_bars for how a drawn one is told apart from an
    unrelated table border or diagram line), pairs them per row, and either (a) - every row with
    two or more bar-like units pairs up cleanly - returns new spans with every such occurrence
    rewritten to the canonical "|"/"||" and no issue, or (b) - some row with two or more units does
    NOT pair up cleanly - returns the ORIGINAL spans completely untouched plus a human-readable
    reason, for the caller to flag needs_review with (see this module's own docstring above for
    why this is all-or-nothing). A row with fewer than two units - a single stray mark with no
    possible partner anywhere nearby - is never acted on and never flagged either way: see this
    section's own "Regression" comment above for why a lone mark must stay silent."""
    has_text_bar = any(s.get("text") and _BAR_RUN_RE.search(s["text"]) for s in spans)
    content_y_centers = [_span_center(s)[1] for s in spans if s.get("text") and s["text"].strip()]
    drawn_rects = [r for r in (get_zone_vertical_bars(drawings, zone_bbox, diagram_bboxes) if drawings else [])
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
        if len(units) < 2:
            continue  # a lone bar-like mark has no possible partner -- never act on it, never flag it
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


def base_size_is_ambiguous(spans):
    """Conservative rule (requirement 2): "most total characters wins" (see _base_size_and_
    baseline's own docstring) silently picks the WRONG size as this zone's own body text whenever
    a SHORT option's own exponent/subscript happens to have MORE characters than its own (very
    short) base text -- confirmed live, MockTest1.pdf Physics Q14 option D: "{prop}x^{1/2}" has
    only 2 real body characters ("{prop}" and "x") against the exponent's own 3 ("1","/","2"), so
    5.75pt (the EXPONENT's own size) won the vote over 8.63pt (the TRUE body size) -- every
    character at that wrongly-chosen "body" size then reads as "normal" sized text instead of a
    superscript, silently erasing the exponent (same failure process_mock_exam_math.
    _join_plain_spans's own docstring already anticipated for a FRACTION's num/den sub-pool, here
    newly confirmed live at the top-level ZONE call too, where a short option's own body text can
    be just as thin). Detected by checking whether any OTHER size actually present is GENUINELY
    competitive with the chosen size's own count (at least half of it, not just a couple of
    characters) while being LARGER than the size _base_size_and_baseline actually picked. The "at
    least half" bar matters live: an ordinary enclosing parenthesis pair is routinely rendered a
    point or two TALLER than its own surrounding body text (equation-editor bracket-stretching,
    confirmed live on this same sample's own Q6 -- a real, correct, unrelated zone, "(" ")" both
    11.48pt against a 9.9pt body of 7 real characters) -- counting that alone as "a competing
    body-size candidate" would false-positive on any such zone; a genuine exponent/subscript
    conflict (confirmed live, Q14 option D: a 5.75pt "1/2" exponent with 3 characters against an
    8.63pt body of only 2) is never this lopsided. A real body/exponent pair's own body text is
    never smaller than its own exponent, so this conflict only ever arises in exactly this
    pathological shape."""
    char_count_by_size = {}
    for s in spans:
        n = len(s["text"].strip())
        if n == 0:
            continue
        key = round(s["size"], 1)
        char_count_by_size[key] = char_count_by_size.get(key, 0) + n
    if len(char_count_by_size) < 2:
        return False
    base_size = max(char_count_by_size, key=lambda sz: (char_count_by_size[sz], sz))
    base_count = char_count_by_size[base_size]
    return any(sz > base_size and count * 2 >= base_count for sz, count in char_count_by_size.items())


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


def detect_baseline_sensitive_shift(corrected_spans, uncorrected_spans):
    """Conservative rule (requirement 2): a "base glyph with two candidate exponents" -- here, two
    candidate DIRECTIONS (super vs sub) for the very same small glyph, depending on which of two
    legitimately-computed baselines is trusted. Recomputes this SAME zone's own base_size/
    base_baseline from `uncorrected_spans` (this zone's own spans BEFORE
    process_mock_exam_text._fix_leading_space_origins ran) and compares every span's own
    _classify_shift result against the SAME span classified under the CORRECTED pipeline's own
    base_size/base_baseline -- confirmed live, MockTest1.pdf Physics Q19 option D: this exact
    disagreement (super under the corrected baseline, sub under the uncorrected one) is what
    silently produced "x_2" instead of "x^2" before that fix existed (see that function's own
    docstring for the full story). A zone where the origin fix never actually changed anything
    (the overwhelming majority, confirmed live -- only a span with its own literal leading
    whitespace ever differs between the two span lists at all) always agrees here and is never
    flagged; this is deliberately NOT a blanket "small offset" margin (tried first, confirmed live
    via a full 180-question real-sample run: ordinary legitimate super/sub offsets on this sample
    vary too widely across different font/size combinations for any single fixed-point cutoff to
    separate "confident" from "ambiguous" without drowning the real signal in false positives --
    29 of 180 on the real sample, against this check's own single-digit true positive)."""
    if not corrected_spans:
        return None
    base_size_c, baseline_c = _base_size_and_baseline(corrected_spans)
    base_size_u, baseline_u = _base_size_and_baseline(uncorrected_spans)
    by_bbox = {}
    for s in uncorrected_spans:
        by_bbox[tuple(round(v, 1) for v in s["bbox"])] = s
    for s in corrected_spans:
        if not s["text"].strip():
            continue
        u = by_bbox.get(tuple(round(v, 1) for v in s["bbox"]))
        if u is None:
            continue
        if _classify_shift(s, base_size_c, baseline_c) != _classify_shift(u, base_size_u, baseline_u):
            return ("a superscript/subscript classification that depends on an ambiguous "
                    "baseline reference")
    return None


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
    # c. 1  d. 5", or a 2x2 grid's own "a. ...  b. ...") used to need its own rectangle-boundary
    # safety check here: a plain marker-midpoint x-split breaks down the moment one option's own
    # content is wider than half its own gap to the next marker -- confirmed live three times, a
    # unit expression's own exponent straddling the midpoint outright, an inline fraction doing
    # the same at ordinary body-text size, and a trailing exponent landing entirely on the WRONG
    # side of a midpoint it never even straddles (severed from the normal-size text it is really
    # attached to). Reconstruct-before-splitting requirement: rather than guard against each of
    # these shapes with its own geometric margin, assign_spans_to_zones no longer uses these
    # rectangles to decide a SHARED row's own content at all -- it walks a 2+-marker row by
    # READING-ORDER marker sequence instead (see that function's own docstring), which has no
    # straddle/sever failure mode in the first place: "whatever comes after marker A and before
    # marker B" stays correct no matter how far right option A's own content runs. This is what
    # finally lets a tightly-packed shared line (the real sample's own Q1, four options' worth of
    # unit exponents sitting right on top of each other) engage this module's own reconstruction
    # at all, instead of bailing the moment any one of them sits near a boundary. The rectangles
    # built below are kept only for a row with FEWER than 2 markers (a genuine x-column question,
    # e.g. a grid's own fraction-denominator continuation row with no marker of its own) and for
    # scoping this zone's own fraction-bar/modulus-bar drawing lookups -- both already proven,
    # unrelated to this requirement, and deliberately left untouched.
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
    """Every remaining glyph (the 4 markers themselves are excluded by marker_ids, consumed only
    to update `current_zone` below) goes to whichever zone's own rect contains its center, or --
    for a glyph sitting just outside every zone's own margin (e.g. a numerator poking slightly
    further above its option's marker than the zone's own margin allows for) -- to whichever
    zone's rect is nearest. This rectangle rule is used for a row carrying FEWER than 2 of this
    question's own bold a-d markers (every option on its own row, or a grid's own fraction-
    denominator continuation row with no marker of its own) -- a genuine x-column question the
    rectangles already answer correctly (confirmed live: unrelated to, and unaffected by, the
    rule below).

    Reconstruct-before-splitting requirement: a row carrying 2 OR MORE markers together (every
    option packed inline on one shared physical line, or a 2x2 grid's own marker row) is resolved
    by READING-ORDER marker sequence instead -- walking that row's own spans left to right (rows
    from _rows_in_reading_order, which already groups a raised/lowered glyph with its own base
    line) and assigning EVERY span on it, base-sized or small, to whichever marker was most
    recently crossed -- the same "whatever follows marker K up to marker K+1" rule the plain-text
    _parse_options already uses, just generalized to glyph geometry here. Confirmed live this is
    necessary even for a BASE-sized glyph, not only a superscript: a rectangle zone's own marker-
    midpoint boundary can fall INSIDE one option's own real content whenever that option's own
    content is simply wider than half its own gap to the next marker (the real sample's own Q1 --
    option A's own base letter "K" sits past the straight midpoint toward option B's marker,
    purely because A's own content is short and B's marker follows close behind). Reading-order
    sequence has no such failure mode: "whatever comes after marker A and before marker B" stays
    correct no matter how far right it runs. `current_zone` persists ACROSS rows (not reset per
    row) so a marker crossed on one row still governs any later row with no marker of its own at
    all (an ordinary wrapped continuation line)."""
    buckets = {"stem": [], "a": [], "b": [], "c": [], "d": []}
    all_zones = {"stem": stem_zone, **option_zones}

    def _plain_zone_for(cx, cy):
        for name, rect in all_zones.items():
            x0, y0, x1, y1 = rect
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                return name
        return min(all_zones, key=lambda name: _dist_to_rect(cx, cy, all_zones[name]))

    current_zone = "stem"
    for row in _rows_in_reading_order(flat_spans):
        marker_count = sum(1 for s in row if id(s) in marker_ids)
        for s in row:
            if id(s) in marker_ids:
                current_zone = s["text"].strip()[0].lower()
                continue
            if marker_count >= 2:
                buckets[current_zone].append(s)
            else:
                cx, cy = _span_center(s)
                buckets[_plain_zone_for(cx, cy)].append(s)
    return buckets


def reclaim_fraction_denominators(buckets, flat_spans, marker_ids, drawings, question_bbox, diagram_bboxes=()):
    """Fixes a real, confirmed-live bug (MockTest1.pdf Physics Q10, a 4-options-inline row): a
    fraction's own denominator sits on a row with NO marker of its own (the marker row is the
    numerator's own row -- see assign_spans_to_zones's own docstring), so assign_spans_to_zones
    falls back to plain rectangle-containment/nearest-zone for it -- but a tightly-packed option
    column's own zone boundary (a marker x MIDPOINT) routinely sits just a hair to the LEFT of
    where a denominator digit's own center actually falls (confirmed live: a "4" denominator's own
    center sat 0.3pt past its true option's own zone boundary, purely because that option's
    numerator -- "d", directly above it -- happens to be WIDER than the denominator, same
    "content wider than the gap to the next marker" shape the reading-order-marker-sequence fix
    already handles for a MARKER row, but this row has no marker to anchor that fix to at all). The
    digit gets assigned to the WRONG, NEIGHBORING zone instead -- never dropped, just misplaced.

    Fixed here by finding every fraction bar in the WHOLE QUESTION (not any one zone's own narrow
    rectangle -- a bar whose own drawn width pokes a couple of points past a tight zone boundary
    must never be measured against that boundary at all, same root cause as the digit above:
    confirmed live, MockTest1.pdf Physics Q6, a bar 11pt wider than its own 2x2-grid zone), then
    using whichever zone ALREADY correctly holds this bar's own numerator (a numerator always sits
    on the marker's own row, already resolved correctly by assign_spans_to_zones's reading-order
    sequence) as this bar's owner, and moving any denominator-reach span sitting in a DIFFERENT
    zone's bucket into the owner's bucket instead -- never a guess: if a bar's own numerator reach
    turns up spans in more than one zone at once (a genuinely ambiguous case this sample never
    produces, but never assumed safe either), nothing is moved for that bar at all.

    Only a span from a row with NO marker of its own at all is ever eligible to move -- a span
    that already sits on the SAME row as its own zone's own marker (an ordinary short option next
    to a sibling option's fraction, confirmed live via this module's own test suite: a "plain" word
    sitting right next to its own "b." marker, geometrically within a NEIGHBORING fraction's own
    denominator-reach window purely by coincidence) is never touched, no matter how close it sits
    to another zone's own bar -- it is already correctly anchored by the reading-order-marker-
    sequence rule, and this pass must never second-guess that.

    Returns (buckets, issue) -- issue (requirement 2's conservative rule) is a human-readable
    reason the moment some bar's own numerator-reach content spreads across 2+ zones at once (a
    glyph genuinely claimed by more than one zone), else None; buckets is always returned, whether
    or not an issue was found, every one of its keys' own lists possibly mutated in place."""
    markerless_ids = set()
    for row in _rows_in_reading_order(flat_spans):
        if any(id(s) in marker_ids for s in row):
            continue
        markerless_ids.update(id(s) for s in row)

    issue = None
    bars = get_zone_fraction_bars(drawings, question_bbox, diagram_bboxes)
    for bar_x0, bar_y0, bar_x1 in bars:
        zones_hit = set()
        for zone_name, spans in buckets.items():
            for s in spans:
                cx, cy = _span_center(s)
                if not (bar_x0 - _FRAC_HORIZONTAL_MARGIN <= cx <= bar_x1 + _FRAC_HORIZONTAL_MARGIN):
                    continue
                if bar_y0 - _FRAC_VERTICAL_REACH <= cy < bar_y0:
                    zones_hit.add(zone_name)
        if len(zones_hit) >= 2:
            # Conservative rule (requirement 2): this bar's own numerator-reach content spreads
            # across more than one zone at once -- a glyph genuinely claimed by two zones -- never
            # guessed which one owns it, flagged instead.
            issue = issue or "a fraction bar whose own numerator is claimed by more than one option"
            continue
        if len(zones_hit) != 1:
            continue  # no numerator found at all for this bar -- not this module's problem to flag
        owner = next(iter(zones_hit))
        for zone_name in list(buckets.keys()):
            if zone_name == owner:
                continue
            moving_ids = set()
            for s in buckets[zone_name]:
                if id(s) not in markerless_ids:
                    continue  # anchored to its own zone's own marker row -- never reclaimed
                cx, cy = _span_center(s)
                if not (bar_x0 - _FRAC_HORIZONTAL_MARGIN <= cx <= bar_x1 + _FRAC_HORIZONTAL_MARGIN):
                    continue
                if bar_y0 < cy <= bar_y0 + _FRAC_VERTICAL_REACH:
                    moving_ids.add(id(s))
            if moving_ids:
                moved = [s for s in buckets[zone_name] if id(s) in moving_ids]
                buckets[zone_name] = [s for s in buckets[zone_name] if id(s) not in moving_ids]
                buckets[owner].extend(moved)
    return buckets, issue


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


def reconstruct_zone(spans, drawings, zone_bbox, diagram_bboxes=(), fraction_scope_bbox=None):
    """Main entry point: reconstructs one question-stem or option's own text from its raw glyph
    spans (list of PyMuPDF span dicts restricted to this zone) plus the page's own vector
    drawings (for fraction-bar and modulus-bar detection, restricted to this zone by
    get_zone_fraction_bars/get_zone_vertical_bars). diagram_bboxes: this zone's own page(s)'
    already-detected diagram regions (see get_zone_vertical_bars's own docstring) - pass () when
    the caller has none computed (e.g. a synthetic test with no diagram at all).

    fraction_scope_bbox: the bbox a candidate FRACTION bar's own width/position is checked against
    -- defaults to this zone's own zone_bbox (the original, narrower behavior, unchanged for any
    caller/test that omits it) but the real caller (process_mock_exam_text.reconstruct_question_
    math) passes the WHOLE QUESTION's own bbox instead. Confirmed live (MockTest1.pdf Physics Q6,
    a 2x2 grid): a real fraction bar's own drawn width can poke several points past a tightly-
    packed zone's own marker-midpoint boundary -- rejecting it outright at that narrow boundary
    (the original behavior) silently left its numerator/denominator as unformatted stray text
    instead of a "\\frac{}{}" even once every one of those glyphs reaches this zone's own span
    pool correctly. Num/den CLAIMING itself still only ever searches `spans` (this zone's own
    bucket, unaffected by this param) -- widening where the bar's own geometry is checked can
    never pull in a sibling option's own content, only stop rejecting a bar that genuinely belongs
    to THIS zone purely because it is wider than this zone's own narrow rectangle. Modulus bars
    (_normalize_modulus_bars, a couple of lines below) keep using zone_bbox unchanged -- this
    sample never showed that same exposure live, so that path is deliberately left as-is.
    Returns (text, issue) -- text already wrapped in $...$ around every math run -- never raises,
    never drops a glyph: anything not confidently reconstructed is passed through as literal
    text. issue is None unless this zone's own modulus/magnitude bars could not be confidently
    paired (see _normalize_modulus_bars) OR a Symbol-font glyph could not be recognized (see
    _collapse_symbol_font_glyphs) -- either way text is returned with that content completely
    unchanged and the caller should flag the question needs_review with issue as the reason."""
    spans = [s for s in spans if s.get("text")]
    if not spans:
        return "", None

    spans, glyph_issue = _collapse_symbol_font_glyphs(spans)

    # Computed ONCE from the zone's own full span pool (before anything is claimed by a fraction)
    # and reused everywhere below -- a fraction's numerator/denominator is often only 1-2 spans,
    # too few for this same "most total characters" heuristic to reliably tell body text apart
    # from an exponent on its own (see _join_plain_spans's own docstring).
    base_size, base_baseline = _base_size_and_baseline(spans)
    size_issue = ("this zone's own body-text size could not be confidently told apart from a "
                  "superscript/subscript's own size") if base_size_is_ambiguous(spans) else None
    spans, bar_issue = _normalize_modulus_bars(spans, drawings, zone_bbox, base_size, diagram_bboxes)
    bar_issue = bar_issue or glyph_issue or size_issue

    claimed_ids = set()
    pool = []  # list of dicts: {text, bbox, origin, size, font, is_math}

    # ---- Fractions first (the most structurally confident signal: an actual drawn bar) ----
    bars = get_zone_fraction_bars(drawings, fraction_scope_bbox or zone_bbox, diagram_bboxes)
    for bar_x0, bar_y0, bar_x1 in bars:
        num_spans, den_spans = [], []
        for s in spans:
            if id(s) in claimed_ids:
                continue
            cx, cy = _span_center(s)
            margin = (_FRAC_HORIZONTAL_MARGIN_SMALL if s.get("size") and s["size"] < base_size * _SIZE_RATIO
                      else _FRAC_HORIZONTAL_MARGIN)
            if not (bar_x0 - margin <= cx <= bar_x1 + margin):
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
            # Conservative rule (requirement 2): a hat/vec CANDIDATE glyph (this sample's own
            # narrow MTExtra-font accent mark, see _is_vec_candidate's own docstring) with no base
            # letter anywhere near it to attach to is left as literal stray text either way (never
            # guessed), but flagged here too -- confirmed live, MockTest1.pdf Physics Q39 option C:
            # an orphaned "ur" (MTExtra) sits well below this option's own real content with
            # nothing plausible above it, reading as a visible, unexplained suffix that nothing
            # else here was catching (it is alphabetic, so _looks_math_garbled's own stray-token
            # ratio never counts it).
            bar_issue = bar_issue or "an unattached hat/vector accent glyph with no nearby base letter"
            continue  # no confident pairing -- leave the diacritic glyph as literal stray text
        candidates.sort(key=lambda c: c[0])
        if len(candidates) >= 2 and candidates[1][0] - candidates[0][0] < 0.5:
            # Conservative rule (requirement 2): a diacritic with two (or more) equally-close base
            # letters is a glyph genuinely claimed by more than one possible attachment -- flagged
            # here instead of just silently staying unresolved, so a human reviewer sees WHY.
            bar_issue = bar_issue or "an ambiguous hat/vector attachment (two equally-close base letters)"
            continue
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
    # (base_size: the same zone-wide value computed at the top of this function)
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

        # A multi-LINE zone (an ordinary wrapped stem/option, not a stacked fraction) has a
        # DIFFERENT absolute baseline per physical line -- confirmed live, MockTest1.pdf Physics
        # Q18's own stem: the zone-wide base_baseline (computed once, dominated by whichever line
        # has the most characters) sat 24.7pt away from a LATER line's own real baseline, so that
        # later line's own genuine superscript ("10^3") measured as confidently BELOW it instead
        # of above -- confidently, not just borderline, so no margin-based ambiguity check could
        # catch it either; it needed the right reference instead. Recomputed per row via
        # _local_baseline (same mechanism _join_plain_spans already uses for a fraction's own
        # numerator/denominator sub-pool, generalized here to the top-level zone's own multi-line
        # case) -- falls back to the zone-wide base_baseline whenever this row has no normal-sized
        # span of its own to anchor to (a row that's ENTIRELY a lone exponent, never confirmed
        # live but not ruled out either).
        row_baseline = _local_baseline(row, base_size) if row else base_baseline

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
                this_shift = _classify_shift(s, base_size, row_baseline)
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
