import os
import json
import re
import base64
import concurrent.futures
import fitz
from anthropic import Anthropic
from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv
import requests as http_requests

load_dotenv()

ANTHROPIC_KEY = os.getenv("ANTHROPIC_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

client = Anthropic(api_key=ANTHROPIC_KEY)
# Text-only extraction (extract_questions_from_text, below) runs on Gemini 3.5 Flash-Lite --
# moved off DeepSeek V4 Flash, which wasn't extracting accurately enough for this task (real
# comparison on 3 PDF pages, see the migration commit). The vision path
# (extract_questions_from_page, image bytes sent to Claude) stays on `client` above -- it's a
# genuine vision task and the actual scanned-PDF path, untouched by this swap.
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

VISION_MODEL = "claude-haiku-4-5-20251001"

def build_extraction_prompt(subject):
    return f"""You are extracting NEET {subject} questions from one page of a scanned PDF.

For EACH complete question visible on this page, extract:
- question (full question text, including any sub-statements A/B/C/D or i/ii/iii if part of the question)
- option_a, option_b, option_c, option_d (exact text of each option)
- question_type: one of "mcq", "match_column", "assertion_reason", "statement_based"
- has_diagram: true if the question references a figure, diagram, image, or shows a chemical structure/graph/apparatus, otherwise false
- year (the 4-digit exam year, extracted from any bracketed exam-source tag near this question
  regardless of exact wording or exam name -- e.g. [NEET-2024], [NEET 2018], [AIPMT 1990],
  [AIPMT Screening 2008], [Re-NEET 2024] all count. Extract just the year as an integer. If no
  such tag appears, use null -- do not guess a year from context.)
- source_tag (the FULL exam-source tag text near this question, verbatim, without the brackets
  -- e.g. "AIPMT 1990", "NEET 2018", "AIPMT Screening 2008", "Re-NEET 2024". This is the same
  tag `year` was extracted from, just kept in full instead of reduced to a number. If no such
  tag appears, use null.)

Do NOT guess or infer a chapter name - that is handled separately. Do NOT attempt to determine the correct answer, even if you can solve the question - leave that to a human reviewer.

If a question is cut off at the top or bottom of this page (incomplete), SKIP it entirely - do not guess missing parts.
If there are no complete questions on this page (e.g. this page is a cover page, instructions, or an answer key), return an empty array.

Return ONLY a JSON array, no other text. Example:
[
  {{
    "question": "...",
    "option_a": "...",
    "option_b": "...",
    "option_c": "...",
    "option_d": "...",
    "question_type": "mcq",
    "has_diagram": false,
    "year": 2024,
    "source_tag": "AIPMT Screening 2008"
  }}
]
"""

# Kept for backward compatibility with the standalone Biology CLI batch job below.
EXTRACTION_PROMPT = build_extraction_prompt("Biology")

def pdf_to_page_images(pdf_source):
    """pdf_source may be a file path (str) or raw PDF bytes."""
    doc = fitz.open(pdf_source) if isinstance(pdf_source, str) else fitz.open(stream=pdf_source, filetype="pdf")
    images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        images.append(base64.standard_b64encode(img_bytes).decode("utf-8"))
    doc.close()
    return images

def extract_questions_from_page(image_b64, page_num, subject="Biology", model=VISION_MODEL):
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64
                        }
                    },
                    {
                        "type": "text",
                        "text": build_extraction_prompt(subject)
                    }
                ]
            }
        ]
    )

    text = message.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    return _parse_extraction_json(text, page_num)

TEXT_MODEL = "gemini-3.5-flash-lite"
MIN_TEXT_LENGTH = 40
MIN_ALNUM_RATIO = 0.3
DIAGRAM_MARKER = "<<<DIAGRAM_HERE>>>"
# Shared by extract_pages_text_and_diagrams (per-block, via .match()) and scan_pdf_bytes's
# under-extraction safety net (per-line across a whole page, via .findall() -- re.M makes "^"
# match after every "\n" too, which .match() ignores since it only ever tries position 0).
_QUESTION_START_RE = re.compile(r"^\d+\s*[.)]", re.M)

def build_text_extraction_prompt(subject):
    return f"""You are extracting NEET {subject} questions from raw text extracted programmatically
from one page of a PDF. The text may have minor formatting artifacts (missing line breaks,
irregular spacing) from the extraction process - use your judgement to reconstruct the
original question structure.

The literal marker "{DIAGRAM_MARKER}" has been inserted into the text at the exact position
where a real image/figure/diagram appears in the original PDF page (based on its actual
position on the page, not a guess). Use this to determine has_diagram PER QUESTION: if the
marker falls within a question's own text, or immediately after its stem/options and before
the next question starts, set has_diagram=true for THAT question only - not for other
questions on the same page that have no marker near them.

A number or unit immediately followed by "^{{...}}" (e.g. "10^{{24}}", "m^{{3}}") marks a
genuine superscript from the original PDF, detected from the source's actual font size and
vertical position, not a guess. Preserve these as LaTeX exponent notation wrapped in single
dollar signs in your output -- e.g. source text "6.023 × 10^{{24}}" should become "$6.023 \\times
10^{{24}}$" in the extracted question/option. Never strip the ^{{...}} markup or flatten it back
into plain concatenated digits (e.g. "1024"). This font-size detection occasionally
mis-highlights an unrelated glyph as superscript (e.g. a degree symbol ° rendered in a broken
font shows up as a raised "^{{1}}" mid-word) -- if a ^{{...}} span clearly isn't a real
mathematical exponent or unit power given the surrounding context, use your own scientific
judgement to write what the notation most likely actually is instead of outputting it literally.

More generally, whenever you write ANY mathematical notation in the extracted question/option
text -- subscripts (e.g. mu-naught as $\\mu_{{0}}$), superscripts/exponents, Greek letters used
as symbols (e.g. $\\varepsilon$, $\\theta$), fractions, or any other LaTeX command -- always wrap
the whole math expression in single dollar signs ($...$), covering just the mathematical part,
not the surrounding plain-English words. Never output raw LaTeX (backslash commands, ^, _, {{}})
outside of $...$ delimiters -- unwrapped LaTeX displays as literal text with visible braces and
backslashes to the student, not as rendered math. Example: source text "dimensions of
(mu0 epsilon0)^-1/2 are" should become "dimensions of $(\\mu_{{0}}\\varepsilon_{{0}})^{{-1/2}}$
are", not left as plain "(mu0 epsilon0)^-1/2" or only partially wrapped.

A numbered option (1./2./3./4.) may be split across several lines in this raw text -- e.g. a
fraction's numerator on one line and its denominator on the next, or a two-term expression like
"2Q/4πε0R - 2q/4πε0R" with each term on its own line. Treat ALL text between one option's own
number and the start of the next option's number as belonging to that ONE option, however many
lines it spans. Every MCQ question has exactly 4 options (1-4) -- never merge two different
option numbers' content into a single option field, never split one option's own content across
two different option fields, and never invent/fabricate an option's content from unrelated
leftover text. If the raw text for one of the 4 options is genuinely missing or unrecoverable,
leave that option_X field as an empty string rather than guessing or fabricating a value.

For EACH complete question in this text, extract:
- question (full question text, including any sub-statements A/B/C/D or i/ii/iii if part of the question) - do not include the marker itself in the question text
- option_a, option_b, option_c, option_d (exact text of each option)
- question_type: one of "mcq", "match_column", "assertion_reason", "statement_based"
- has_diagram: true only if the {DIAGRAM_MARKER} marker is positioned within or immediately next to THIS question, otherwise false
- year (the 4-digit exam year, extracted from any bracketed exam-source tag near this question
  regardless of exact wording or exam name -- e.g. [NEET-2024], [NEET 2018], [AIPMT 1990],
  [AIPMT Screening 2008], [Re-NEET 2024] all count. Extract just the year as an integer. If no
  such tag appears, use null -- do not guess a year from context.)
- source_tag (the FULL exam-source tag text near this question, verbatim, without the brackets
  -- e.g. "AIPMT 1990", "NEET 2018", "AIPMT Screening 2008", "Re-NEET 2024". This is the same
  tag `year` was extracted from, just kept in full instead of reduced to a number. If no such
  tag appears, use null.)

Do NOT guess or infer a chapter name - that is handled separately. Do NOT attempt to determine
the correct answer, even if an answer key or answer text appears elsewhere in this text or you
can solve the question yourself - leave that to a human reviewer.

If a question is cut off at the top or bottom of this page (incomplete), SKIP it entirely - do not guess missing parts.
If there are no complete questions in this text (e.g. this is a cover page, instructions, or an answer key), return an empty array.

Return ONLY a JSON array, no other text. Example:
[
  {{
    "question": "...",
    "option_a": "...",
    "option_b": "...",
    "option_c": "...",
    "option_d": "...",
    "question_type": "mcq",
    "has_diagram": false,
    "year": 2024,
    "source_tag": "AIPMT Screening 2008"
  }}
]

TEXT FROM THE PDF PAGE:
---
{{page_text}}
---
"""

def looks_garbled_or_empty(text, min_length=MIN_TEXT_LENGTH, min_alnum_ratio=MIN_ALNUM_RATIO):
    """Heuristic for 'this page probably has no real extractable text' - e.g. a scanned/image
    page slipped into a batch that's supposed to be text-layer PDFs only. Tuned against real
    text-layer PYQ pages, which measured 0.42-0.76 alnum ratio with 1000+ characters."""
    stripped = (text or "").strip()
    if len(stripped) < min_length:
        return True
    alnum_count = sum(c.isalnum() for c in stripped)
    return (alnum_count / len(stripped)) < min_alnum_ratio

def _reconstruct_line_text(line):
    """Joins a line's spans into plain text, wrapping any span that's genuinely superscript-
    formatted (meaningfully smaller font AND raised above the line's own baseline, vs. the
    line's own dominant-size spans) in ^{...}. get_text("blocks")/plain get_text("text") only
    return character content with no font-size/position info, so a real exponent like "10"
    followed by a raised, smaller "24" span gets silently concatenated into the wrong number
    "1024" with no trace two separate spans were ever superscripted -- confirmed via
    get_text("dict") on a real NEET PDF (QP_2019.pdf p1: normal-text spans share one origin y
    at font size 10.08, the exponent digit span sits ~4pt higher at size 7.44, ~74% as tall)."""
    spans = [s for s in line.get("spans", []) if s.get("text")]
    if not spans:
        return ""
    sizes = [s["size"] for s in spans if s["text"].strip()]
    if not sizes:
        return "".join(s["text"] for s in spans)
    base_size = max(sizes)
    baseline_y = min(s["origin"][1] for s in spans if s["size"] >= base_size - 0.1)

    out = ""
    in_super = False
    for s in spans:
        text = s["text"]
        if not text:
            continue
        is_super = bool(text.strip()) and s["size"] < base_size * 0.85 and s["origin"][1] < baseline_y - 1
        if is_super and not in_super:
            out += "^{"
            in_super = True
        elif not is_super and in_super:
            out += "}"
            in_super = False
        out += text
    if in_super:
        out += "}"
    return out

def _block_text_from_dict(block):
    line_texts = [_reconstruct_line_text(line) for line in block.get("lines", [])]
    return "\n".join(t for t in line_texts if t)

# Matches a LINE whose ENTIRE text is just a numbered-option marker ("1.", "2.", ... "4.") and
# nothing else -- see _reconstruct_fragmented_options below for why this specific shape is the
# signal that a whole option list needs special handling.
_BARE_OPTION_MARKER_RE = re.compile(r'^([1-4])\.\s*$')
# An option short enough to sit entirely on the same line as its own marker (e.g. "3. vd =
# constant") needs no zone-reconstruction for its own text, but it still has to be recognized as
# marker "3" for the sequence-continuity check below, or a genuinely fragmented neighbor (marker
# "1"/"2"/"4") breaks its run the moment it hits this "gap" and silently loses every marker after
# it (confirmed live: Current Electricity Q5, "vd ~ E" / "vd ~ 1/E" fragmented across two blocks,
# "vd = constant" complete on one line -- without this, only options 1-2 ever got reconstructed
# and 3-4 were left dangling, out of order, in the stem block's own leftover text).
_SELF_CONTAINED_OPTION_RE = re.compile(r'^([1-4])\.\s+(\S.*)$')
# A decimal number that happens to start with 1-4 ("2.5 x 10^6") looks IDENTICAL to a real
# marker+answer under the regex above ("2." + " 5 x 10^6") -- confirmed live: Current Electricity
# Q4's four mobility values all literally begin "2.5"/"2.5"/"2.25"/"2.25", so all four got
# mis-tagged as bogus "marker 2" candidates, which then broke the REAL marker run for the
# unrelated Q5 immediately below it (a mismatch of ANY kind stops the greedy adjacency scan, so
# the real run past the false candidate was never found). Every genuine self-contained option
# observed (units, words, symbols) contains at least one letter; every false decimal-continuation
# case observed contains none -- used as a cheap, reliable filter rather than trying to parse "is
# this really a new sentence" from a single line in isolation.
_HAS_LETTER_RE = re.compile(r'[A-Za-zͰ-Ͽ]')
# Deliberately broader than _QUESTION_START_RE (which requires a literal "."/")" right after the
# number, e.g. "4)") -- a real question's own stem frequently starts with just "NN " and no
# punctuation at all (confirmed live, e.g. "55 An electric field exists..."), which the stricter
# regex never matches. Used below only to find the next safe stopping point for an option zone,
# never to identify question boundaries anywhere else.
_ANY_NUMBER_START_RE = re.compile(r'^\d{1,3}\b')

def _cluster_columns(items, x_key, threshold=8):
    """Greedy 1D clustering of items into left-to-right columns by x-position."""
    sorted_items = sorted(items, key=x_key)
    clusters = []
    for item in sorted_items:
        x = x_key(item)
        if clusters and x - x_key(clusters[-1][-1]) <= threshold:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    return clusters

def _reconstruct_fragmented_options(blocks, mid_x, page_width):
    """Detects a numbered option list (1.-4.) whose markers PyMuPDF split away from math content
    (fractions, operators) that got fragmented -- and sometimes MERGED ACROSS DIFFERENT OPTIONS
    into one indivisible block -- by PyMuPDF's own block-clustering. Confirmed via a real PDF
    (NEET Physics "Electrostatic Potential and Capacitance", Q27: a thin spherical shell question
    with a fraction-heavy option list) where this silently scrambled option values across all 4
    options -- one option's value corrupted (numerator/denominator from two different options
    merged), and the rest shifted, with a fabricated leftover 4th option. A pure sort-order fix
    cannot recover this: once PyMuPDF's own block boundaries have merged two different options'
    content into one block, that information is already gone at the block level. The fix instead
    goes down to LINE-level bboxes (each line keeps its own accurate position even when its
    parent block merged multiple options together, or merged a marker onto the tail of the
    question stem's own block -- confirmed live on Current Electricity, where PyMuPDF fused a
    bare "1." marker onto the same block as the stem's last line), assigns each content line to
    its NEAREST option marker by vertical distance, then reconstructs each option's own reading
    order via left-to-right sub-column clustering (for side-by-side fraction/operator/fraction
    layouts, e.g. "2Q/4pi*e0*R - 2q/4pi*e0*R") using each line's CENTER-x rather than its left
    edge -- a narrow single-glyph numerator/operator (e.g. "q") sits centered over its own wider
    denominator, not left-aligned with it, since real math typesetting centers a fraction's parts
    around a shared midline.

    A marker can also be "self-contained" -- already carrying its own answer text on the same
    line (e.g. "3. vd = constant") -- rather than "bare" (just "3." with the real content
    fragmented elsewhere). Both are detected and merged into one marker sequence so a page mixing
    both shapes in the same option list (confirmed live) still forms one continuous run.

    Returns (excluded_block_ids, reconstructed_entries, modified_block_text): the caller should
    skip normal block-level handling for excluded_block_ids, use modified_block_text in place of
    a block's normal full text for any block id present in that dict (its own marker line(s)
    stripped out, real stem text kept), and add reconstructed_entries (x0, y0, text) in their
    place. Returns (set(), [], {}) when no fragmented option list is detected -- the
    overwhelmingly common case, left completely untouched by this function."""
    all_markers = []
    for b in blocks:
        for line in b.get("lines", []):
            text = _reconstruct_line_text(line).strip()
            m = _BARE_OPTION_MARKER_RE.match(text)
            if m:
                all_markers.append({"num": int(m.group(1)), "bbox": line["bbox"], "block": b, "line": line, "self_text": None})
                continue
            m2 = _SELF_CONTAINED_OPTION_RE.match(text)
            if m2 and _HAS_LETTER_RE.search(m2.group(2)):
                all_markers.append({"num": int(m2.group(1)), "bbox": line["bbox"], "block": b, "line": line, "self_text": m2.group(2)})

    # Grouped PER PAGE-COLUMN, not by a single page-wide y0 sort -- on a real 2-column page,
    # sorting every marker on the page together interleaves two DIFFERENT questions' markers
    # (left column vs right column often share near-identical y0 ranges), which silently breaks
    # the run-building the moment marker detection is done at line-level (an unrelated
    # same-numbered marker from the other column landing between two real same-column markers in
    # the sort derails the "next number, same x0" check). Must be a consistent x-column (all
    # markers left-aligned to each other) and start at 1 to be confident this is a real, in-order
    # option list, not a coincidental digit-period line.
    groups = []
    for column_markers in (
        sorted([m for m in all_markers if m["bbox"][0] < mid_x], key=lambda m: m["bbox"][1]),
        sorted([m for m in all_markers if m["bbox"][0] >= mid_x], key=lambda m: m["bbox"][1]),
    ):
        i = 0
        while i < len(column_markers):
            run = [column_markers[i]]
            j = i + 1
            while (j < len(column_markers) and column_markers[j]["num"] == run[-1]["num"] + 1
                   and abs(column_markers[j]["bbox"][0] - run[0]["bbox"][0]) <= 3):
                run.append(column_markers[j])
                j += 1
            if run[0]["num"] == 1 and len(run) >= 2:
                groups.append(run)
            i = j

    if not groups:
        return set(), [], {}

    excluded_ids = set()
    reconstructed_entries = []
    modified_block_text = {}

    for markers in groups:
        marker_x0 = markers[0]["bbox"][0]
        zone_x0 = marker_x0 - 2
        # Never cross into the OTHER page-column -- otherwise a totally unrelated question's own
        # option list sitting in the other column (same page, only coincidentally Y-overlapping)
        # bleeds straight in.
        zone_x1 = mid_x if marker_x0 < mid_x else page_width
        # Small margin -- just enough to catch an option's own content starting a few points
        # above its marker (a fraction's numerator commonly sits slightly above the marker's own
        # baseline), without reaching far enough to also catch the previous option's tail or the
        # question stem's own inline math sitting almost flush against it.
        zone_y0 = markers[0]["bbox"][1] - 8
        last_y1 = markers[-1]["bbox"][3]
        # Bound the zone's bottom at whatever comes next in this same column (the next question's
        # own stem, or another option list's own "1." for a short, closely-spaced option block) --
        # a fixed trailing margin alone bled into whatever followed whenever options were short.
        next_boundary_ys = [
            b["bbox"][1] for b in blocks
            if b["bbox"][1] > last_y1 and abs(b["bbox"][0] - marker_x0) <= 3
            and _ANY_NUMBER_START_RE.match(_block_text_from_dict(b).strip())
        ]
        zone_y1 = (min(next_boundary_ys) - 2) if next_boundary_ys else (last_y1 + 15)

        marker_line_ids = {id(m["line"]) for m in markers}

        # Handle each marker's parent block: fully exclude it if the block is ENTIRELY marker
        # line(s), otherwise keep the block but with just the marker line(s) stripped out -- a
        # marker fused onto the tail of the question stem's own block (confirmed live) must keep
        # that block's real stem text, not lose it entirely.
        parent_block_ids = {id(m["block"]) for m in markers}
        for pb_id in parent_block_ids:
            block = next(b for b in blocks if id(b) == pb_id)
            remaining_lines = [l for l in block.get("lines", []) if id(l) not in marker_line_ids]
            if remaining_lines:
                kept_text = "\n".join(t for t in (_reconstruct_line_text(l) for l in remaining_lines) if t.strip())
                if kept_text.strip():
                    modified_block_text[pb_id] = kept_text
                else:
                    excluded_ids.add(pb_id)
            else:
                excluded_ids.add(pb_id)

        content_lines = []
        for b in blocks:
            bx0, by0, bx1, by1 = b["bbox"]
            if bx1 < zone_x0 or bx0 > zone_x1 or by0 > zone_y1 or by1 < zone_y0:
                continue
            block_contributed = False
            for line in b.get("lines", []):
                if id(line) in marker_line_ids:
                    continue
                lx0, ly0, lx1, ly1 = line["bbox"]
                # Compare the line's OWN top edge against the zone (not "does this line's bbox
                # overlap at all") -- a line whose bbox straddles the boundary needs a clean,
                # unambiguous side to land on, not a fuzzy overlap test.
                if lx1 < zone_x0 or lx0 > zone_x1 or ly0 < zone_y0 or ly0 > zone_y1:
                    continue
                text = _reconstruct_line_text(line).strip()
                if not text:
                    continue
                # Real math-fragment content sits to the right of the marker's own x-column and
                # is short -- but "short" can't be as tight as a bare symbolic fragment
                # ("4pi*e0*R"), since a legitimate single-line option can read like "VAB = 75 V ,
                # VBC = 25 V" (23 chars, Current Electricity Q142) or a wrapped self-contained
                # option's continuation line can run even longer (electrostatics Q62's "no excess
                # charge in the static situation.", 41 chars) -- both confirmed live to need this
                # raised well past the original 20-char cutoff. 50 was verified empirically (not
                # just reasoned about) to still correctly exclude the original electrostatics
                # Q27 regression case's own stem leak ("from the centre of the shell is:", 33
                # chars) -- some other check in this zone, not the length cutoff alone, is what
                # actually keeps that stem fragment out.
                if lx0 < marker_x0 + 10 or len(text) > 50:
                    continue
                content_lines.append({"bbox": line["bbox"], "text": text})
                block_contributed = True
            if block_contributed:
                excluded_ids.add(id(b))

        if not content_lines:
            continue  # markers with no matching zone content -- leave this page's block-level path untouched

        marker_y_centers = [(m["num"], (m["bbox"][1] + m["bbox"][3]) / 2) for m in markers]
        by_num = {m["num"]: [] for m in markers}
        for line in content_lines:
            ly_center = (line["bbox"][1] + line["bbox"][3]) / 2
            nearest_num = min(marker_y_centers, key=lambda mc: abs(mc[1] - ly_center))[0]
            by_num[nearest_num].append(line)

        for m in markers:
            # A self-contained marker's own line is already excluded from content_lines (via
            # marker_line_ids), so it's never double-counted here -- but a self-contained option
            # can still legitimately WRAP to a second line (a plain sentence continuation, not a
            # stacked fraction), which this same nearest-marker zone-gathering also finds
            # correctly (confirmed live: Current Electricity Q62's "1. The interior...can have" +
            # its own next line "no excess charge in the static situation." both nearest-assign
            # to marker 1) -- so self_text is simply prepended as the option's own first part
            # rather than skipping zone-gathering entirely, which previously left any wrapped
            # continuation stranded in the stem block's own leftover text instead of the option.
            columns = _cluster_columns(by_num[m["num"]], x_key=lambda l: (l["bbox"][0] + l["bbox"][2]) / 2)
            columns.sort(key=lambda col: min((l["bbox"][0] + l["bbox"][2]) / 2 for l in col))
            parts = [m["self_text"]] if m["self_text"] is not None else []
            for col in columns:
                parts.extend(l["text"] for l in sorted(col, key=lambda l: l["bbox"][1]))
            option_text = f"{m['num']}. " + " ".join(parts)
            reconstructed_entries.append((marker_x0, m["bbox"][1], option_text))

    return excluded_ids, reconstructed_entries, modified_block_text

# A raster image narrower or shorter than this many pixels is near-certainly a repeated fill-
# tile (table/answer-box shading), not real diagram content -- confirmed on a real PDF
# (AIPMT_2015.pdf) where a 2x2px image was placed 9 times on one page, each placement previously
# counted as its own diagram. Every genuine diagram/graph checked during the DocLayout-YOLO
# benchmark was at least ~90px on its short side.
_MIN_DIAGRAM_DIM = 20

def extract_pages_text_and_diagrams(pdf_bytes):
    """Free/local, no API call: raw text per page (PyMuPDF), with a marker inserted at the
    actual on-page position of each real (non-watermark) image, so the extraction model can attribute
    has_diagram to the correct individual question rather than the whole page. Also preserves
    real superscript exponents (see _reconstruct_line_text) that plain block-mode text would
    silently flatten into wrong plain-digit numbers.

    Images that repeat identically (same xref) across multiple pages are treated as a
    watermark/logo/letterhead, not a real per-question diagram - confirmed empirically on
    real PYQ PDFs, where the branding image on every page would otherwise flag every question.
    get_image_rects() (not the richer get_text("dict") image blocks, whose position/identity
    fields turned out unreliable on real files) gives the real display bbox for a confirmed xref.

    Two additional passes on top of the original raster/xref detection above, added after a
    DocLayout-YOLO benchmark against real PDFs (2026-09-02):

    1. Tiny fill-tile images (see _MIN_DIAGRAM_DIM) are excluded, and a real image is only ever
       counted ONCE per page regardless of how many times it's placed -- multiple placements of
       the SAME xref is the repeated-shading pattern, not multiple distinct diagrams.
    2. A DocLayout-YOLO pass (see _detect_vector_diagrams_yolo) runs after the raster pass, to
       catch diagrams drawn with native PDF vector primitives (lines/circles/curves) -- these
       have no embedded image at all, so the xref-based pass above is structurally blind to them.
       Confirmed live: a real coaching-material PDF had 494 vector drawing objects (the actual
       geometry diagrams) vs. only 38 raster images (mostly a per-page logo). This pass is
       strictly additive (only adds markers for regions the raster pass didn't already find) and
       never fatal to the scan -- any failure (model unavailable, inference error) just falls
       back to raster-only detection, exactly as before this existed."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_images_meta = [page.get_images(full=True) for page in doc]
    page_xrefs = [set(img[0] for img in imgs) for imgs in page_images_meta]

    xref_page_count = {}
    for xrefs in page_xrefs:
        for xref in xrefs:
            xref_page_count[xref] = xref_page_count.get(xref, 0) + 1
    template_xrefs = {xref for xref, count in xref_page_count.items() if count > 1}

    xref_dims = {}  # xref -> (width, height), from get_images(full=True)'s own metadata
    for imgs in page_images_meta:
        for img in imgs:
            xref_dims[img[0]] = (img[2], img[3])

    pages_meta = []
    for i, page in enumerate(doc):
        real_diagram_xrefs = {
            xref for xref in (page_xrefs[i] - template_xrefs)
            if min(xref_dims.get(xref, (0, 0))) >= _MIN_DIAGRAM_DIM
        }
        mid_x = page.rect.width / 2
        page_blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type") == 0]
        excluded_block_ids, reconstructed_entries, modified_block_text = _reconstruct_fragmented_options(page_blocks, mid_x, page.rect.width)

        text_entries = []  # (x0, y0, text)
        for block in page_blocks:
            if id(block) in excluded_block_ids:
                continue
            # A block that had just its marker line(s) stripped out (see modified_block_text's
            # docstring above) keeps its own real stem text here, sorted at its own normal
            # position -- only the marker itself moved into reconstructed_entries.
            block_text = modified_block_text.get(id(block), _block_text_from_dict(block))
            if block_text.strip():
                x0, y0 = block["bbox"][0], block["bbox"][1]
                text_entries.append((x0, y0, block_text.strip()))
        text_entries.extend(reconstructed_entries)

        # A one-time (non-repeating) image sitting ABOVE the first real question on the page -
        # e.g. a chapter-title banner - is a decorative header, not a per-question diagram.
        # Repeating images are already excluded above; this catches the one-time-only case.
        question_start_ys = [y0 for _, y0, text in text_entries if _QUESTION_START_RE.match(text)]
        content_start_y = min(question_start_ys) if question_start_ys else 0

        entries = list(text_entries)  # (x0, y0, text) - text blocks and diagram markers
        raster_rects = []  # kept per-page so the YOLO pass below can avoid double-flagging these
        for xref in real_diagram_xrefs:
            rects = [r for r in page.get_image_rects(xref) if r.y0 >= content_start_y]
            if not rects:
                continue
            entries.append((rects[0].x0, rects[0].y0, DIAGRAM_MARKER))
            raster_rects.extend(rects)

        pages_meta.append({
            "entries": entries,
            "mid_x": mid_x,
            "content_start_y": content_start_y,
            "raster_rects": raster_rects,
        })

    try:
        yolo_markers = _detect_vector_diagrams_yolo(doc, pages_meta)
    except Exception as e:
        print(f"    DocLayout-YOLO diagram detection unavailable, using raster-only detection: {e}")
        yolo_markers = {}
    for i, markers in yolo_markers.items():
        pages_meta[i]["entries"].extend(markers)

    pages = []
    for pm in pages_meta:
        entries = pm["entries"]
        mid_x = pm["mid_x"]
        # Sort by COLUMN first (left-of-midpoint vs right-of-midpoint), then by y0 within that
        # column -- not by y0 alone. A real 2-column exam-paper layout (official NEET papers,
        # confirmed live on QP_2024/QP_2025: right column's first block sits at the same y0 as
        # the left column's, a sort tie broken by PyMuPDF's arbitrary block-discovery order) was
        # producing badly scrambled reading order -- e.g. Q57 (right column) landing before
        # Q52-56 (left column) in the extracted text, which is no longer a coherent question
        # sequence, so Gemini correctly (per its own "skip incomplete/incoherent" instruction)
        # returned zero questions for the whole page -- a real, silent root cause of pages
        # producing far fewer questions than expected, confirmed via a live 22-PDF audit (found
        # 2026-08-10). For a genuinely single-column page every block falls on the same side of
        # the midpoint, so this reduces to the previous plain y0 sort -- verified harmless there.
        entries.sort(key=lambda e: (0 if e[0] < mid_x else 1, e[1]))
        pages.append({"text": "\n".join(content for _, _, content in entries)})
    doc.close()
    return pages

_YOLO_REPO_ID = "wybxc/DocLayout-YOLO-DocStructBench-onnx"
_YOLO_FILENAME = "doclayout_yolo_docstructbench_imgsz1024.onnx"
_YOLO_DPI = 150
_YOLO_IMGSZ = 1024
_YOLO_STRIDE = 32  # fixed for this checkpoint -- hardcoded (with the class list below) instead of
                    # read from the .onnx file's own metadata, so the `onnx` package (only useful
                    # for that one read) isn't a dependency, on top of onnxruntime.
_YOLO_FIGURE_CLASS = 3  # DocStructBench class labels: 0 title, 1 plain text, 2 abandon, 3 figure, ...
_YOLO_CONF = 0.35
# Fraction of the smaller box's area that must overlap an existing raster-detected rect for a
# YOLO figure box to be considered "already flagged" and skipped (avoids double-counting one
# diagram that happens to ALSO be embedded as a raster image).
_YOLO_OVERLAP_THRESHOLD = 0.3
_yolo_session_cache = {}

def _get_yolo_session():
    """Lazily downloads (first call only, cached by huggingface_hub on disk) and loads the
    DocLayout-YOLO ONNX session. Imports onnxruntime/cv2 inside this function, not at module
    level, so the rest of this module (used by several other endpoints) still works even in an
    environment where this optional dependency isn't installed or fails to load.

    Originally integrated via the `doclayout_yolo` PyPI package (PyTorch/ultralytics-based) --
    switched to this ONNX export of the SAME checkpoint after a real speed investigation found
    plain PyTorch CPU inference was the dominant cost of the whole diagram-detection pipeline
    (~4.6s/page, >90% of total scan time). Verified live, per-page, on all 3 benchmark files used
    throughout this integration: ONNX produces IDENTICAL figure counts to the PyTorch path on
    every single page, while running noticeably faster (1.4-1.8s/page vs 2-4.8s/page measured
    back to back on the same machine) and pulling in a dramatically lighter dependency tree
    (onnxruntime+opencv-python+huggingface_hub, ~200MB, vs the full torch/torchvision/ultralytics
    stack this replaces, ~890MB) -- same weights, same accuracy, strictly better on every axis
    that was actually measured."""
    if "session" not in _yolo_session_cache:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(repo_id=_YOLO_REPO_ID, filename=_YOLO_FILENAME)
        _yolo_session_cache["session"] = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return _yolo_session_cache["session"]

def _yolo_resize_and_pad(image_bgr, new_shape, stride, cv2):
    """Matches the reference preprocessing for this exact ONNX export (letterbox resize + pad to
    a stride-aligned square, mid-grey fill) -- verified live to reproduce the PyTorch path's
    detections exactly, so kept byte-for-byte equivalent rather than simplified."""
    h, w = image_bgr.shape[:2]
    r = min(new_shape / h, new_shape / w)
    resized_h, resized_w = int(round(h * r)), int(round(w * r))
    image = cv2.resize(image_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    pad_w = (new_shape - resized_w) % stride
    pad_h = (new_shape - resized_h) % stride
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return image, r, left, top

def _yolo_infer_figure_boxes(session, image_rgb, cv2, np):
    """Returns figure-class boxes as (x0,y0,x1,y1) in the ORIGINAL image's pixel space (padding/
    scaling already undone)."""
    orig_h, orig_w = image_rgb.shape[:2]
    image_bgr = image_rgb[:, :, ::-1]  # RGB -> BGR, this checkpoint's expected channel order
    padded, scale, pad_left, pad_top = _yolo_resize_and_pad(image_bgr, _YOLO_IMGSZ, _YOLO_STRIDE, cv2)
    pix = np.transpose(padded, (2, 0, 1))
    pix = np.expand_dims(pix, axis=0).astype(np.float32) / 255.0
    preds = session.run(None, {"images": pix})[0]
    preds = preds[preds[..., 4] > _YOLO_CONF]
    boxes = []
    for row in preds:
        if int(row[5]) != _YOLO_FIGURE_CLASS:
            continue
        x0 = (row[0] - pad_left) / scale
        y0 = (row[1] - pad_top) / scale
        x1 = (row[2] - pad_left) / scale
        y1 = (row[3] - pad_top) / scale
        if x1 > 0 and y1 > 0:
            boxes.append((max(x0, 0), max(y0, 0), min(x1, orig_w), min(y1, orig_h)))
    return boxes

def _box_already_covered(box, existing_rects):
    box_area = max(box.get_area(), 1e-6)
    for r in existing_rects:
        inter = box & r
        if inter.is_empty:
            continue
        if inter.get_area() / min(box_area, max(r.get_area(), 1e-6)) >= _YOLO_OVERLAP_THRESHOLD:
            return True
    return False

def _detect_vector_diagrams_yolo(doc, pages_meta):
    """See extract_pages_text_and_diagrams's docstring for why this exists. Page rendering (fitz
    calls) happens sequentially -- PyMuPDF is not safe for concurrent access to the same Document
    object across threads.

    Inference is ALSO sequential -- a ThreadPoolExecutor calling the equivalent PyTorch model
    concurrently from multiple threads on one shared model instance was tried first (before the
    ONNX switch documented in _get_yolo_session) and silently corrupted results (confirmed live:
    concurrent calls returned ZERO detections on every page of a real 19-page file, vs. 16 for the
    identical sequential calls). A genuine batched call was also tried and, while correct, measured
    slower than sequential single-image calls on this CPU -- no parallelism win available here
    worth the complexity, so this stays sequential under ONNX Runtime too."""
    from PIL import Image
    import numpy as np
    import cv2

    session = _get_yolo_session()  # raises if unavailable -- caller catches and no-ops

    page_arrays = []
    for page in doc:
        pix = page.get_pixmap(dpi=_YOLO_DPI)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        page_arrays.append(np.array(img))
    scale = 72.0 / _YOLO_DPI  # pixel coords (at _YOLO_DPI) -> PDF point coords

    out = {}
    for i, arr in enumerate(page_arrays):
        try:
            pm = pages_meta[i]
            markers = []
            for x0, y0, x1, y1 in _yolo_infer_figure_boxes(session, arr, cv2, np):
                x0, y0, x1, y1 = x0 * scale, y0 * scale, x1 * scale, y1 * scale
                # No content_start_y (header/banner) exclusion here, unlike the raster pass just
                # above -- that heuristic (first question-number marker on the page) turned out
                # fragile on coaching-material PDFs with bare-digit question numbering and dotted
                # answer-option numbering ("1." for an option, not "1." for the question), which
                # can push it far down the page and wrongly exclude real early-page diagrams
                # (confirmed live: page 1 of a real file computed content_start_y=563pt on an
                # ~800pt-tall page). Not needed here anyway: DocStructBench's own trained classes
                # already separate a logo/header from a real figure (confirmed live on the same
                # file -- the per-page logo is consistently classified 'abandon' or 'title', never
                # 'figure', at conf 0.7-0.9), so this pass can trust the model's own classification
                # instead of re-deriving "is this in the header" from the page's text layout.
                box = fitz.Rect(x0, y0, x1, y1)
                if _box_already_covered(box, pm["raster_rects"]):
                    continue  # already flagged via the raster pass -- don't double-count
                markers.append((x0, y0, DIAGRAM_MARKER))
            if markers:
                out[i] = markers
        except Exception as e:
            print(f"    DocLayout-YOLO failed on page {i + 1}, skipping for this page: {e}")
    return out

_KNOWN_LATEX_COMMANDS = {
    "text", "mathrm", "mathbf", "mathit", "overline", "underline", "hat", "vec", "dot", "ddot",
    "binom", "frac", "sqrt", "times", "div", "pm", "mp", "cdot", "circ", "quad", "qquad", "left",
    "right", "rightarrow", "leftarrow", "Rightarrow", "Leftarrow", "to",
    "sum", "int", "prod", "infty", "partial", "nabla",
    "alpha", "beta", "gamma", "Gamma", "delta", "Delta", "epsilon", "varepsilon", "zeta", "eta",
    "theta", "Theta", "iota", "kappa", "lambda", "Lambda", "mu", "nu", "xi", "Xi", "pi", "Pi",
    "rho", "sigma", "Sigma", "tau", "upsilon", "phi", "varphi", "Phi", "chi", "psi", "Psi",
    "omega", "Omega",
    "sin", "cos", "tan", "cot", "sec", "csc", "log", "ln", "exp", "lim", "min", "max",
    "ast", "star", "prime", "dagger", "ne", "neq", "leq", "geq", "approx", "equiv", "propto",
    "in", "notin", "subset", "supset", "cup", "cap", "forall", "exists", "emptyset",
    "angle", "perp", "parallel", "triangle", "square", "therefore", "because",
}
_CMD_NAME_RE = re.compile(r'[a-zA-Z]+')

def _fix_unescaped_json_backslashes(text):
    """Gemini/Claude are inconsistent about JSON-escaping the literal backslashes in LaTeX
    commands they write (e.g. \\mu, \\varepsilon, \\times, \\text, \\frac) inside the extracted
    question/option strings -- confirmed live: the exact same prompt, same content, sometimes
    correctly emits \\\\mu and sometimes invalid raw \\mu, non-deterministically. Prompting
    harder didn't fix it (tried, including response_mime_type="application/json").

    IMPORTANT: this used to only double a backslash NOT already followed by a "valid JSON escape
    character" (one of \\"/bfnrtu), on the theory that those were probably intentional. That was
    wrong in a dangerous way -- \\t, \\b, \\f, \\n, \\r are ALSO the first two characters of common
    LaTeX commands (\\text, \\theta, \\times, \\tan, \\to, \\frac, \\beta, \\bar, \\nabla, \\rho,
    \\rightarrow...), and json.loads() treats \\t etc as VALID escapes -- it doesn't raise, so the
    old fallback-on-exception logic never even ran for these. The result was silent corruption:
    "\\text{AlF}_3" parsed as an actual TAB CHARACTER followed by the literal text "ext{AlF}_3",
    with no error anywhere, visible only much later as garbled "ext{...}" in the admin UI (real
    bug, reported live 2026-08-10).

    Simply doubling every lone backslash unconditionally isn't right either -- confirmed live on
    the same real extraction batch that reproduced the bug above: a genuinely-intended "\\n\\n"
    paragraph break between "Statement I:" and "Statement II:" in a real question also showed up
    as a lone backslash, and blanket-doubling it would turn a real line break into literal visible
    "\\n" text.

    Originally this was disambiguated purely by $...$ position (a lone backslash INSIDE an active
    $...$ span is LaTeX, one OUTSIDE is prose) -- but that broke down for the companion bug fixed
    in wrap_bare_latex_notation() below: the model sometimes writes a real LaTeX command with NO
    $ wrapper at all (e.g. bare "PCl_{3}, \\text{ } NH_{3}"), and that backslash needs protecting
    too even though it's technically "outside" any $ span (reported live 2026-08-10, same session,
    a different admin-pdf-review.html question). So a lone backslash is now protected if EITHER
    it's inside $...$, OR the word immediately following it exactly matches a known LaTeX command
    name (_KNOWN_LATEX_COMMANDS) -- checking against a specific whitelist rather than "2+ letters
    follow" avoids the false positive where a genuine "\\n" is immediately followed by the next
    sentence's first word with no space (e.g. "...heating.\\nAssertion continues" -- "nAssertion"
    has plenty of letters after the backslash, but isn't a known command, so the \\n is correctly
    left as a real newline). An already-correct \\\\ or \\" pair is consumed atomically (2 chars
    at once) wherever it appears, in or out of math mode, so its second character is never
    independently re-examined and re-doubled.

    "$$...$$" (display math) is treated as ONE atomic 2-char delimiter, not two independent "$"
    toggles -- toggling on each "$" individually flips in_math ON then immediately back OFF for
    "$$", silently breaking display-math detection (found via a live full-table audit before this
    was ever used to write data, not from a user report)."""
    out = []
    in_math = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == '$':
            if i + 1 < n and text[i + 1] == '$':
                out.append('$$')
                in_math = not in_math
                i += 2
                continue
            in_math = not in_math
            out.append(ch)
            i += 1
            continue
        if ch == '\\' and i + 1 < n:
            nxt = text[i + 1]
            if nxt in ('\\', '"'):
                out.append(ch); out.append(nxt); i += 2; continue
            cmd_match = _CMD_NAME_RE.match(text, i + 1)
            is_known_cmd = bool(cmd_match) and cmd_match.group(0) in _KNOWN_LATEX_COMMANDS
            if in_math or is_known_cmd:
                out.append('\\\\'); i += 1; continue
            out.append(ch); i += 1; continue
        out.append(ch)
        i += 1
    return ''.join(out)

_LATEX_TOKEN_RE = re.compile(r'(?:[A-Za-z0-9]|_\{[^{}]*\}|\^\{[^{}]*\}|\\[a-zA-Z]+(?:\{[^{}]*\})*)+')
_MATH_MARKER_RE = re.compile(r'_\{[^{}]*\}|\^\{[^{}]*\}|\\([a-zA-Z]+)')

def _token_has_math_marker(token):
    """A _{...} or ^{...} group is always a real math marker. A \\command is only a real marker
    if it's a KNOWN LaTeX command -- accepting ANY \\[a-zA-Z]+ wrapped stray non-command
    backslash-letter debris (e.g. a literal "\\n" left over from other corrupted content) as if
    it were legitimate LaTeX (found via the same live audit as the $$ fix above)."""
    for m in _MATH_MARKER_RE.finditer(token):
        if m.group(1) is None or m.group(1) in _KNOWN_LATEX_COMMANDS:
            return True
    return False

def wrap_bare_latex_notation(text):
    """Separate bug from the backslash-escaping one above: the model sometimes writes correctly-
    formed LaTeX notation (_{...} subscripts, ^{...} superscripts, \\text{...}) but forgets to
    wrap it in $...$ AT ALL -- e.g. option text literally "SF_{6}" or "PCl_{3}, \\text{ } NH_{3}"
    with no dollar signs anywhere, so KaTeX never renders it and the raw notation shows as-is
    (reported live 2026-08-10, a different admin-pdf-review.html report from the same session as
    the backslash bug). This is a prompt-adherence gap, not an encoding bug -- no amount of
    re-parsing JSON fixes it, the text is already valid Python/JSON, just not valid *rendering*
    input. Runs on each already-extracted field's plain text (NOT on raw JSON before parsing --
    unlike the backslash fix, this would misfire on JSON's own quote/brace structure), walking
    $-parity the same way (with the same "$$" atomic-pair handling), and wraps any maximal run of
    letters/digits glued directly (no spaces) to at least one _{...}/^{...}/known \\command in a
    fresh $...$ pair. Never touches text already inside an existing $...$ span, and plain prose
    with no math markers is left untouched."""
    out = []
    in_math = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == '$':
            if i + 1 < n and text[i + 1] == '$':
                out.append('$$')
                in_math = not in_math
                i += 2
                continue
            in_math = not in_math
            out.append(ch)
            i += 1
            continue
        if not in_math:
            m = _LATEX_TOKEN_RE.match(text, i)
            if m and _token_has_math_marker(m.group(0)):
                out.append('$' + m.group(0) + '$')
                i = m.end()
                continue
        out.append(ch)
        i += 1
    return ''.join(out)

_NUMBERED_MARKER_RE = re.compile(r'(\d{1,2})\.\s')

def format_numbered_substatements(text):
    """"Which of the following is/is not correct" style questions often extract as one dense
    run-on block with no separation between numbered sub-statements ("...correct? 1. X is zero.
    2. X is positive. 3. X is negative."). Inserts a line break before each such marker so it
    displays one point per line -- purely whitespace, never touches the actual wording/LaTeX.

    Only fires when the digits found form a genuine ascending sequence starting at 1 (1, 2, 3,
    ...) with at least two markers. A single isolated "N. " match is left alone -- shape alone
    can't tell a list marker apart from an ordinary sentence that happens to end in a bare number
    ("the pH will be 7. This means neutral."), so the sequence itself is what has to prove it's a
    real list, not the regex. "2 moles of gas" never matches at all: there's no period directly
    after the digit, which is exactly what distinguishes a marker from a plain quantity.

    Only meant to run on the question stem (see _parse_extraction_json's call site) -- the 4
    option fields are separate UI fields already, not numbered sub-points within a block of text."""
    if not isinstance(text, str) or not text:
        return text
    candidates = []
    for m in _NUMBERED_MARKER_RE.finditer(text):
        start = m.start()
        if start > 0 and not text[start - 1].isspace():
            continue  # glued to a preceding word/digit, e.g. mid-token -- not a real marker
        candidates.append(m)
    if len(candidates) < 2:
        return text

    expected = 1
    confirmed = []
    for m in candidates:
        if int(m.group(1)) == expected:
            confirmed.append(m)
            expected += 1
        elif confirmed:
            break  # sequence broken after at least one hit -- don't extend past the real list
    if len(confirmed) < 2:
        return text

    out = []
    last_end = 0
    for m in confirmed:
        start = m.start()
        ws_start = start
        while ws_start > last_end and text[ws_start - 1].isspace():
            ws_start -= 1
        out.append(text[last_end:ws_start])
        if ws_start > 0:
            out.append('\n')
        out.append(text[start:m.end()])
        last_end = m.end()
    out.append(text[last_end:])
    return ''.join(out)

_EXTRACTED_TEXT_FIELDS = ("question", "option_a", "option_b", "option_c", "option_d")

def _normalize_option_for_dedupe(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r'\s+', '', text.replace('$', '')).lower()

def _flag_suspicious_mcq_options(q):
    """Belt-and-suspenders check, not a fix on its own: doesn't correct anything, just makes a
    bad result visible instead of silently saving it. Catches the two failure shapes the option-
    list fragmentation bug (see _reconstruct_fragmented_options) actually produced on a real PDF
    even before that fix existed -- two options merged into identical text, and an empty/missing
    option left behind by the merge -- so a case _reconstruct_fragmented_options's own detection
    doesn't cover (a different PDF's layout quirk, a genuinely low-confidence model guess) still
    surfaces for manual review instead of saving straight through unflagged."""
    if q.get("question_type") != "mcq":
        return
    opts = {label: q.get(f"option_{label}", "") for label in ("a", "b", "c", "d")}
    empties = [label for label, val in opts.items() if not isinstance(val, str) or not val.strip()]
    if empties:
        q["needs_review"] = True
        q["review_reason"] = f"Option(s) {', '.join(l.upper() for l in empties)} came back empty for an MCQ question -- extraction may have merged or dropped an option."
        return
    seen = {}
    dupes = []
    for label, val in opts.items():
        norm = _normalize_option_for_dedupe(val)
        if norm in seen:
            dupes.append((seen[norm], label))
        else:
            seen[norm] = label
    if dupes:
        pairs = ", ".join(f"{a.upper()}/{b.upper()}" for a, b in dupes)
        q["needs_review"] = True
        q["review_reason"] = f"Option(s) {pairs} came back identical -- extraction may have scrambled this question's options, please review manually."

def _parse_extraction_json(text, page_num):
    """Shared by both extraction paths (Gemini text-only and Claude Vision). Always runs the
    backslash fix BEFORE parsing rather than only as an exception fallback -- see
    _fix_unescaped_json_backslashes's docstring for why "did the first parse succeed" can't be
    trusted as a signal here. The fix is a verified no-op on already-correct JSON, so this is
    strictly safer than the old try-original-then-fallback order, never worse.

    IMPORTANT: raises on a genuine parse failure instead of swallowing it and returning [] --
    the old behavior made a page that failed to parse indistinguishable from a page that
    legitimately had zero questions, so scan_pdf_bytes's flagged_pages (which DOES catch an
    exception from this call, via future.result() in its executor loop) never saw it: the page
    just silently contributed zero questions with no trace anywhere (a real, if not the only,
    cause of "questions getting skipped" -- see extract_pages_text_and_diagrams's 2-column-layout
    fix above for the other, bigger one found in the same investigation, 2026-08-10)."""
    try:
        questions = json.loads(_fix_unescaped_json_backslashes(text))
    except Exception:
        questions = None
    if questions is None:
        try:
            questions = json.loads(text)  # last resort, in case the fix broke something unexpected
        except Exception as e:
            raise ValueError(f"Could not parse model response as JSON: {e}") from e
    for q in questions:
        for field in _EXTRACTED_TEXT_FIELDS:
            if isinstance(q.get(field), str):
                q[field] = wrap_bare_latex_notation(q[field])
        # Question stem only -- see format_numbered_substatements's docstring for why this
        # doesn't run on option_a-d too.
        if isinstance(q.get("question"), str):
            q["question"] = format_numbered_substatements(q["question"])
        _flag_suspicious_mcq_options(q)
    return questions

def extract_questions_from_text(page_text, page_num, subject="Biology", model=TEXT_MODEL):
    prompt = build_text_extraction_prompt(subject).replace("{page_text}", page_text)
    response = gemini_client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(max_output_tokens=4096)
    )

    text = (response.text or "").strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    return _parse_extraction_json(text, page_num)

def scan_pdf_bytes(pdf_bytes, subject, max_workers=4):
    """Callable entry point for the admin review pipeline (server.py's /admin/scan-pdf).
    Text-layer PDFs only (not scanned/image PDFs) - extracts real text + programmatic diagram
    detection for free, then a TEXT-ONLY (no Vision, cheaper) Gemini call per page to structure
    it into questions. Returns raw extracted questions - no chapter/class/correct_answer
    guessing, that's left to the TF-IDF classifier and manual review on the frontend.

    Pages are sent to Gemini concurrently - doing them one at a time made scanning a
    multi-page PDF take minutes.

    max_workers lowered from 5 to 4 (2026-08-26) after real profiling found this was the actual
    bottleneck, and in the opposite direction from what you'd guess: PyMuPDF extraction + garbled-
    page detection together took under 1s combined on a real 10-page PDF, while the Gemini calls
    took 220s total at max_workers=5 -- but almost all of that was retry-backoff churn from
    tripping a rate limit, not real model latency. Confirmed directly: the same pages that took
    150-209s each under 5x concurrency completed in ~3s when called alone, and a full 10-page
    PDF at max_workers=4 consistently finished in 7-9s (3 clean trials across 2 different real
    PDFs) vs. 52-220s at max_workers=5 (anomalies in 2/2 trials) -- extracting the identical
    question count both times. So this isn't a parallelize-more speedup, it's a parallelize-
    slightly-less fix: avoiding the rate limit is what makes it fast. Model, prompt, and every
    validation step are unchanged -- see the PDF-scan speed investigation for the full profiling
    data and the isolated-page tests that pinned this down before changing anything."""
    pages = extract_pages_text_and_diagrams(pdf_bytes)
    flagged_pages = []
    to_extract = {}  # page index -> page text, only for pages that pass the safety check
    for i, page in enumerate(pages):
        if looks_garbled_or_empty(page["text"]):
            flagged_pages.append({"page": i + 1, "reason": "Extraction found little or no usable text - possible scanned/image page, or a cover/blank page."})
        else:
            to_extract[i] = page["text"]

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(extract_questions_from_text, text, idx + 1, subject=subject): idx
            for idx, text in to_extract.items()
        }
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"    Error extracting page {idx + 1}: {e}")
                flagged_pages.append({"page": idx + 1, "reason": f"Extraction call failed: {e}"})
                results[idx] = []

    # General safety net, not tied to any one specific root cause: a page with several "N."
    # question-start markers in its own raw text that STILL came back with zero extracted
    # questions is a strong, cheap, false-positive-resistant signal something went wrong on that
    # page specifically (the 2-column-layout bug fixed above in extract_pages_text_and_diagrams
    # was the first real cause found this way, live, 2026-08-10 -- but this check exists to catch
    # whatever similar issue turns up next too, not just that one). Deliberately NOT flagged when
    # 0 < actual < expected -- a page legitimately ending mid-question extracts fewer than its
    # full "N." count (the prompt explicitly tells the model to skip a cut-off question), so a
    # partial shortfall is normal, expected behavior, not a sign of a problem.
    for idx, page_text in to_extract.items():
        expected = len(_QUESTION_START_RE.findall(page_text))
        actual = len(results.get(idx, []))
        if expected >= 2 and actual == 0:
            flagged_pages.append({
                "page": idx + 1,
                "reason": f"Found {expected} question-number markers on this page but extracted 0 questions -- likely an extraction issue, not a genuinely empty page. Worth checking manually."
            })

    questions = []
    for i, page in enumerate(pages):
        page_num = i + 1
        for q in results.get(i, []):
            questions.append({
                "question": q.get("question", ""),
                "option_a": q.get("option_a", ""),
                "option_b": q.get("option_b", ""),
                "option_c": q.get("option_c", ""),
                "option_d": q.get("option_d", ""),
                "correct_answer": "",
                "question_type": q.get("question_type", "mcq"),
                "has_diagram": bool(q.get("has_diagram", False)),
                "year": q.get("year"),
                "source_tag": q.get("source_tag"),
                "source_page": page_num,
                "needs_review": bool(q.get("needs_review", False)),
                "review_reason": q.get("review_reason"),
            })
            # Reuses the existing flagged_pages banner (already wired up in admin-pdf-review.html)
            # rather than adding a new UI element -- a per-question review flag is exactly the
            # same "something here needs a human look before trusting it" signal the banner
            # already exists for, just with a question-level cause instead of a page-level one.
            if q.get("needs_review"):
                snippet = (q.get("question") or "")[:80]
                flagged_pages.append({
                    "page": page_num,
                    "reason": f"“{snippet}...” -- {q.get('review_reason', 'flagged for manual review')}"
                })
    flagged_pages.sort(key=lambda f: f["page"])
    return {"questions": questions, "pages_scanned": len(pages), "flagged_pages": flagged_pages}

def slice_pdf_pages(pdf_bytes, start_page, end_page):
    """start_page/end_page are 1-indexed, inclusive, matching how an admin reads printed
    page numbers. Builds a standalone in-memory PDF for just that range via PyMuPDF, so
    scan_pdf_bytes() runs completely unmodified against it -- no changes to the proven
    single-subject PYQ scan path."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total = len(doc)
    if start_page < 1 or end_page < start_page or end_page > total:
        doc.close()
        raise ValueError(f"Invalid page range {start_page}-{end_page} for a {total}-page PDF")
    doc.select(list(range(start_page - 1, end_page)))
    out_bytes = doc.tobytes()
    doc.close()
    return out_bytes

MOCK_TEST_SUBJECT_ORDER = ["Physics", "Chemistry", "Biology"]

def scan_mock_test_pdf(pdf_bytes, ranges, max_workers=3):
    """ranges: {"Physics": (start,end), "Chemistry": (start,end), "Biology": (start,end)}.
    Slices once per subject, then calls the existing scan_pdf_bytes() 3x concurrently (each
    already parallelizes its own pages internally). question_order is assigned in fixed
    Physics->Chemistry->Biology order regardless of dict/completion order, so the served
    exam's section order can't get scrambled by an out-of-order scan completion."""
    sliced = {s: slice_pdf_pages(pdf_bytes, *ranges[s]) for s in MOCK_TEST_SUBJECT_ORDER}
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_subject = {executor.submit(scan_pdf_bytes, sliced[s], s): s for s in MOCK_TEST_SUBJECT_ORDER}
        for future in concurrent.futures.as_completed(future_to_subject):
            s = future_to_subject[future]
            try:
                results[s] = future.result()
            except Exception as e:
                results[s] = {"error": str(e), "questions": [], "pages_scanned": 0, "flagged_pages": []}

    questions = []
    flagged_pages = []
    per_subject_counts = {}
    errors = {}
    order = 1
    for s in MOCK_TEST_SUBJECT_ORDER:
        r = results[s]
        if r.get("error"):
            errors[s] = r["error"]
        for f in r.get("flagged_pages", []):
            flagged_pages.append({**f, "subject": s})
        subj_questions = r.get("questions", [])
        per_subject_counts[s] = len(subj_questions)
        for q in subj_questions:
            questions.append({**q, "subject": s, "question_order": order})
            order += 1

    return {
        "questions": questions,
        "flagged_pages": flagged_pages,
        "per_subject_counts": per_subject_counts,
        "errors": errors
    }

def insert_question(q, subject="Biology"):
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "question": q.get("question"),
        "option_a": q.get("option_a"),
        "option_b": q.get("option_b"),
        "option_c": q.get("option_c"),
        "option_d": q.get("option_d"),
        "correct_answer": "EMPTY",
        "question_type": q.get("question_type", "mcq"),
        "chapter": q.get("chapter"),
        "year": q.get("year"),
        "source_tag": q.get("source_tag"),
        "subject": subject,
        "has_diagram": q.get("has_diagram", False),
        "is_active": True
    }
    res = http_requests.post(
        f"{SUPABASE_URL}/rest/v1/pyq",
        headers=headers,
        json=payload
    )
    return res.status_code

if __name__ == "__main__":
    folder = "bio_pdfs_to_process"
    pdf_files = [f for f in os.listdir(folder) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDFs")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder, pdf_file)
        print(f"\nProcessing {pdf_file}...")
        try:
            page_images = pdf_to_page_images(pdf_path)
            print(f"  {len(page_images)} pages found")
            total_inserted = 0
            for page_num, img in enumerate(page_images, 1):
                try:
                    questions = extract_questions_from_page(img, page_num)
                except Exception as e:
                    print(f"  Page {page_num}: SKIPPED -- {e}")
                    continue
                print(f"  Page {page_num}: {len(questions)} questions")
                for q in questions:
                    status = insert_question(q)
                    if status == 201:
                        total_inserted += 1
            print(f"  TOTAL inserted for {pdf_file}: {total_inserted}")
        except Exception as e:
            print(f"  ERROR on {pdf_file}: {e}")

    print("\nAll done.")