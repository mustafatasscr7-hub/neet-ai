"""Model-assisted re-read for Mock exam questions the deterministic scan (process_mock_exam_text.py,
process_mock_exam_math.py) already flagged needs_review -- a SEPARATE module on purpose, same
reasoning process_pyq_scanned.py's own docstring gives for being separate from process_pyq_vision:
process_mock_exam_text.py's own docstring says "no model call anywhere in this module or anything
it imports is ever invoked", and that claim must stay true. This module is the one and only place
a mock-exam question is ever sent to a model, and it is only ever invoked by the caller (server.py)
for questions the deterministic scan already marked needs_review -- a clean question never reaches
here at all.

Why a model call is safe to add back in HERE specifically, when process_mock_exam_text.py's own
docstring (and process_mock_exam_math.py's) argues at length for why one isn't needed: those two
modules solve the 85%+ of questions geometry alone reconstructs correctly. The real sample
(MockTest1.pdf) still leaves ~31/180 flagged -- mostly options with fractions, roots and unit
exponents where PyMuPDF's own glyph layout is too fragmented for geometry to confidently resolve
(e.g. Q1: true options "W m^-2 K^-1" / "W m^2 K^-4" / "W m^-2 K^-4" / "W m^2 K^4" come back as a
scrambled "2 1 - - W m K" with options B and D identical). A human already has to open the PDF for
every one of these anyway (that is what needs_review means) -- reading the SAME crop image with
Gemini Vision first, gated hard against the PDF's own text layer before anything is trusted, turns
some fraction of those manual fixes into an auto-accept instead, without ever widening which
questions a model gets to decide exist (scan_mock_exam_pdf alone still owns that, unchanged).

Confidence rule, same as the rest of this feature's own modules: a model read that doesn't pass
every gate is simply never applied. The question stays exactly as the deterministic scan left it,
still flagged, still red, still carrying its own crop image -- never worse than today.
"""
import os
import re
import base64
from collections import Counter

from google.genai import types as genai_types

from process_pyq_vision import gemini_client, TEXT_MODEL, wrap_bare_latex_notation

# Same model the rest of the site's Gemini text/vision calls already use (process_pyq_vision.
# TEXT_MODEL, process_pyq_scanned.SCANNED_MODEL) -- this call just happens to also carry an image.
MOCK_VISION_MODEL = TEXT_MODEL

# Default true: this is a quality improvement on an already-flagged question, never a regression
# path (a gate failure always falls back to the existing deterministic result) -- see module
# docstring's confidence rule. Settable to "false" to turn it off entirely if ever needed.
MOCK_VISION_FLAGGED_ENABLED = os.getenv("MOCK_VISION_FLAGGED_ENABLED", "true").strip().lower() == "true"

# Isolation requirement: only flagged questions ever reach _attempt_question (the caller filters),
# and at most this many PER PDF -- a guard against a pathological PDF (hundreds of questions this
# layout can't parse at all) turning one scan into hundreds of paid model calls. The real sample
# has 31 flagged, comfortably under this; a PDF with more than 60 flagged questions has a bigger
# problem than this feature can fix and should be reviewed manually past that point anyway.
MOCK_VISION_MAX_CALLS_PER_PDF = 60

EXPECTED_OPTION_COUNT = 4

# Mirrors server.py's GEMINI_FLASH_LITE_RATE (USD per 1M tokens) -- duplicated rather than imported
# because server.py imports this module, not the other way round (same reasoning process_pyq_
# scanned.py's own GEMINI_RATE_PER_M comment gives). test_mock_exam_vision.py asserts the two stay
# equal.
GEMINI_RATE_PER_M = {"input": 0.30, "output": 2.50}

# A model read is only ever applied once its own stem+options word-overlap against the PDF's own
# text layer for this question clears this ratio (see _word_overlap_ratio) -- tuned against the
# real sample (MockTest1.pdf): every genuinely correct model-read question measured comfortably
# above this, every wrong one measured well below it. Deliberately the stricter (max-denominator)
# ratio, not a lenient recall-only one -- see _word_overlap_ratio's own docstring for why: this
# gate must catch BOTH a dropped/garbled word AND an invented one, not just one of the two.
_WORD_MATCH_MIN_RATIO = 0.6

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")
# Strips a LaTeX command name (e.g. "\frac", "\Omega") and any bare {}$^_\\ markup -- so the
# model's own "$W\\,m^{-2}$" and the PDF's own raw "W m -2" glyphs reduce to the SAME word set
# ("w", "m") instead of the model picking up spurious tokens ("frac", "Omega") the PDF's raw text
# never had a word for in the first place. The PDF's own raw text-layer hint rarely has a literal
# backslash command at all (it is pre-reconstruction text -- see _text_layer_hint's own origin in
# process_mock_exam_text._finalize_question), so this mostly matters for the MODEL's side, but is
# applied to both for the same reason the rest of this module never special-cases either side.
_LATEX_STRIP_RE = re.compile(r"\\[a-zA-Z]+|[{}$^_\\]")


def _normalize_ws(text):
    return re.sub(r"\s+", " ", text or "").strip()


def _words(text):
    stripped = _LATEX_STRIP_RE.sub(" ", text or "")
    return [w.lower() for w in _WORD_RE.findall(stripped)]


def _word_overlap_ratio(words_a, words_b):
    """Multiset overlap divided by the LARGER side's own length, not the smaller -- a lenient
    min-denominator ratio would pass a model reply that invents a long run of extra content never
    in the PDF at all, as long as it also happens to repeat every real word somewhere. Dividing by
    max() instead means EITHER missing real words OR adding invented ones pulls the ratio down -- a
    model read must actually match, not just not-contradict."""
    if not words_a or not words_b:
        return 0.0
    ca, cb = Counter(words_a), Counter(words_b)
    overlap = sum((ca & cb).values())
    return overlap / max(len(words_a), len(words_b))


def _build_prompt(subject, printed_number, text_hint):
    return f"""You are reading ONE multiple-choice question from a NEET mock exam PDF page image.

This is {subject} question {printed_number}. Below is the PDF's own text layer for this same
question, given only as a HINT to confirm wording -- it may be garbled, especially inside
fractions, roots, exponents, vectors and unit expressions, so trust the IMAGE for how it is
actually printed, not this hint's own formatting:
---
{text_hint}
---

Transcribe this question exactly as printed in the image: the question stem, then its four
options. Write every fraction, root, exponent, subscript, vector, unit and Greek letter as LaTeX
inside $...$ (for example $W\\,m^{{-2}}\\,K^{{-4}}$, $\\frac{{a}}{{b}}$, $\\sqrt{{2}}$). Do not
solve the question, do not add any explanation, and do not add or invent content that is not
actually in the image.

Reply in EXACTLY this shape and nothing else. Normalize whichever option labels are actually
printed in the image (a./b./c./d., (a)/(b)/(c)/(d), or anything else) to A)/B)/C)/D):

Q: <question stem>
A) <option A text>
B) <option B text>
C) <option C text>
D) <option D text>
"""


_REPLY_STEM_RE = re.compile(r"^\s*Q:\s*(.*)", re.S)
_REPLY_OPTION_MARKER_RE = re.compile(r"(?:^|\n)\s*([A-D])\)\s*", re.M)


def _parse_model_reply(raw_text):
    """Plain-text parse of the fixed Q:/A)/B)/C)/D) shape _build_prompt asks for -- same marker-
    scan shape process_mock_exam_text._parse_options already uses for the PDF's own lowercase
    "a."-"d." markers, here anchored to the model's own normalized uppercase "A)"-"D)" labels
    instead. Never raises: whatever options it does find (possibly fewer than 4, possibly none)
    are returned as-is for _gate to judge -- a malformed reply is a gate failure, not a crash."""
    text = (raw_text or "").strip()
    stem_match = _REPLY_STEM_RE.match(text)
    body = stem_match.group(1) if stem_match else text
    matches = list(_REPLY_OPTION_MARKER_RE.finditer(body))
    if not matches:
        return _normalize_ws(body), {}
    stem = _normalize_ws(body[: matches[0].start()])
    options = {}
    for i, m in enumerate(matches):
        letter = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        options[letter] = _normalize_ws(body[start:end])
    return stem, options


_RAW_REPLY_MAX_CHARS = 1800
# A real reply (prompt asks for the stem + 4 options and "nothing else") never approaches this on
# the real sample -- the longest of 30 real model-read questions measured well under 700 characters
# raw. Confirmed live (2026-11-12): the model occasionally ignores "nothing else" and leaks its own
# chain-of-thought ("wait, let's look at option a in the image...") ahead of or instead of the
# fixed format, repeating the Q:/A)/B)/C)/D) markers many times over -- _parse_model_reply's own
# last-occurrence-wins marker scan would then silently pick fragments from the WRONG draft, not
# the model's final answer. A plain length cap, checked before any parsing is attempted, catches
# this cheaply without needing to understand the reply's own structure at all.

# Counts a delta (charge-polarisation) marker by MEANING, not by character: the PDF's own text
# layer carries a literal "δ" glyph, but the model's own LaTeX reply almost always spells it out as
# the command "\delta" instead (confirmed live -- a literal "δ" count on 3 independent real replies
# for the same flagged question came back 0 every time, even on two replies that got everything
# else right). Counting both forms is what makes this check mean anything at all for THIS model.
_DELTA_RE = re.compile(r"\u03b4|delta", re.IGNORECASE)
# Structural-fidelity gate (2026-11-12, confirmed live): a flagged question whose own options
# differ ONLY by which atom a δ+/δ- charge marker sits on (e.g. a polarisation diagram) can still
# pass the word-overlap gate even when the model silently drops the marker that was attached to a
# single-letter atom like "O" -- that atom's own letter is still present elsewhere in the same
# option, so the word-level ratio barely moves, but the actual distinguishing content of the
# option is gone. A real PDF with this many delta markers repeats them close to once per option; a
# model reply that comes back with noticeably fewer has dropped exactly that kind of detail. -1
# tolerance only (not a ratio): confirmed live, a fully faithful reply reproduces the real count
# exactly.
_DELTA_MIN_COUNT_FOR_CHECK = 2


def _gate(stem, options, text_hint):
    """Returns None on success, or a human-readable failure reason. Five independent checks, any
    one of which fails the whole read (see module docstring's confidence rule) -- the four
    requirement 2 names (wrong option count, empty option(s), duplicate options, and word-level
    mismatch against the PDF's own text layer) plus the delta-marker structural-fidelity check
    below, added after a real failure this gate's word-overlap alone did not catch (see its own
    comment)."""
    if len(options) != EXPECTED_OPTION_COUNT:
        return f"model returned {len(options)} option(s), expected {EXPECTED_OPTION_COUNT}"
    letters = ("A", "B", "C", "D")
    empty = [l for l in letters if not options.get(l, "").strip()]
    if empty:
        return f"option(s) {', '.join(empty)} came back empty from the model"
    seen = {}
    dupes = []
    for l in letters:
        norm = _normalize_ws(options[l]).lower()
        if norm in seen:
            dupes.append(f"{seen[norm]}/{l}")
        else:
            seen[norm] = l
    if dupes:
        return f"model option(s) {', '.join(dupes)} came back identical"
    if not stem.strip():
        return "model returned an empty question stem"
    model_words = _words(stem) + [w for l in letters for w in _words(options[l])]
    hint_words = _words(text_hint)
    ratio = _word_overlap_ratio(model_words, hint_words)
    if ratio < _WORD_MATCH_MIN_RATIO:
        return f"model wording only {ratio:.0%} overlaps the PDF text layer (need >= {_WORD_MATCH_MIN_RATIO:.0%})"
    hint_deltas = len(_DELTA_RE.findall(text_hint or ""))
    if hint_deltas >= _DELTA_MIN_COUNT_FOR_CHECK:
        model_raw = stem + " " + " ".join(options[l] for l in letters)
        model_deltas = len(_DELTA_RE.findall(model_raw))
        if model_deltas < hint_deltas - 1:
            return f"model reply has {model_deltas} delta (charge) marker(s), PDF text layer has {hint_deltas} -- likely dropped a structural detail"
    return None


def _default_generate(image_bytes, prompt):
    """One real Gemini Vision call. Returns (raw_text, usage) -- usage shaped like process_pyq_
    scanned.extract_page_items's own (input_tokens/output_tokens, thinking counted as output,
    same reasoning its own docstring gives). Swapped out via apply_vision_to_flagged_questions's
    own generate_fn parameter for tests -- never called by anything in this module directly."""
    response = gemini_client.models.generate_content(
        model=MOCK_VISION_MODEL,
        contents=[
            genai_types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            prompt,
        ],
        config=genai_types.GenerateContentConfig(max_output_tokens=2048),
    )
    um = response.usage_metadata
    usage = {
        "input_tokens": (um.prompt_token_count or 0) if um else 0,
        "output_tokens": ((um.candidates_token_count or 0) + (getattr(um, "thoughts_token_count", 0) or 0)) if um else 0,
    }
    return response.text or "", usage


def _call_cost_usd(usage):
    return (usage["input_tokens"] * GEMINI_RATE_PER_M["input"] + usage["output_tokens"] * GEMINI_RATE_PER_M["output"]) / 1_000_000


def _attempt_question(q, generate_fn):
    """One flagged question, up to two model calls (one retry on any gate failure, including a
    call that raised). Mutates `q` in place ONLY on success -- question/option_a-d, needs_review,
    review_reason, model_read; printed_number (and every other identity field) is never touched
    either way, asserted explicitly below rather than just relied on by construction. Returns
    (success, reason_or_none, cost_usd, calls_made)."""
    image_bytes = base64.standard_b64decode(q["crop_image_b64"])
    text_hint = q.get("_text_layer_hint", "")
    prompt = _build_prompt(q.get("subject", ""), q.get("printed_number", "?"), text_hint)
    before_number = q.get("printed_number")

    total_cost = 0.0
    calls_made = 0
    last_reason = None
    for _attempt in range(2):  # one try + one retry, see module docstring's confidence rule
        calls_made += 1
        try:
            raw_text, usage = generate_fn(image_bytes, prompt)
        except Exception as e:
            last_reason = f"model call failed: {e}"
            continue
        total_cost += _call_cost_usd(usage)
        if len(raw_text) > _RAW_REPLY_MAX_CHARS:
            last_reason = f"model reply was {len(raw_text)} characters, over the {_RAW_REPLY_MAX_CHARS}-character cap -- likely leaked reasoning instead of the requested format"
            continue
        stem, options = _parse_model_reply(raw_text)
        reason = _gate(stem, options, text_hint)
        if reason is None:
            q["question"] = wrap_bare_latex_notation(stem)
            for letter, field in (("A", "option_a"), ("B", "option_b"), ("C", "option_c"), ("D", "option_d")):
                q[field] = wrap_bare_latex_notation(options[letter])
            assert q.get("printed_number") == before_number, "model-read path must never touch printed_number"
            q["needs_review"] = False
            q["review_reason"] = None
            q["model_read"] = True
            return True, None, total_cost, calls_made
        last_reason = reason
    return False, last_reason, total_cost, calls_made


def apply_vision_to_flagged_questions(result, max_calls=MOCK_VISION_MAX_CALLS_PER_PDF, generate_fn=None):
    """Callable entry point: takes scan_mock_exam_pdf's own return dict and, for each question it
    already flagged needs_review (up to max_calls, document order, see module-level constant's own
    docstring for why a cap exists), tries one model-assisted re-read. Mutates and returns the SAME
    dict -- questions beyond the cap, and every non-flagged question, are returned completely
    untouched (requirement 3's isolation: a clean question never gets a model call here, no matter
    how many OTHER questions in the same PDF are flagged).

    generate_fn(image_bytes, prompt) -> (raw_text, usage_dict) defaults to a real Gemini call
    (_default_generate); tests inject a fake here instead of monkeypatching the real client.
    """
    if generate_fn is None:
        generate_fn = _default_generate
    questions = result.get("questions") or []
    flagged = [q for q in questions if q.get("needs_review")]
    attempted = flagged[:max_calls]
    skipped_over_cap = len(flagged) - len(attempted)

    total_cost = 0.0
    total_calls = 0
    model_read = 0
    gate_failed = 0
    retried = 0

    for q in attempted:
        success, _reason, cost, calls_made = _attempt_question(q, generate_fn)
        total_cost += cost
        total_calls += calls_made
        if calls_made > 1:
            retried += 1
        if success:
            model_read += 1
        else:
            gate_failed += 1
            # q is untouched here -- still needs_review True, still the deterministic result, same
            # as if this module had never run at all (see module docstring's confidence rule).

    for q in questions:
        q.pop("_text_layer_hint", None)  # internal-only hint, never meant to reach the browser

    for subject, bucket in (result.get("completeness") or {}).items():
        bucket["red_flag_count"] = sum(1 for q in questions if q.get("subject") == subject and q.get("needs_review"))

    result["cost"] = {
        "usd": round(total_cost, 6),
        "calls": total_calls,
        "attempted": len(attempted),
        "model_read": model_read,
        "gate_failed": gate_failed,
        "retried": retried,
        "skipped_over_cap": skipped_over_cap,
        "model": MOCK_VISION_MODEL,
    }
    print(
        f"Mock exam scan: vision pass on {len(attempted)} flagged question(s) -- "
        f"{model_read} model-read, {gate_failed} still flagged, {retried} retried, "
        f"{skipped_over_cap} skipped over the {max_calls}-call cap, cost ${total_cost:.4f}"
    )
    return result
