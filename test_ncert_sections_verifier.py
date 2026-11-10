"""Unit tests for the "From NCERT" exact-quote verifier (NCERT_SECTIONS_ENABLED, default off) --
see _verify_ncert_quotes / _normalize_ncert_quote_text / _strip_ncert_block in server.py.

Run with: python test_ncert_sections_verifier.py
Plain pass/fail harness, same style as test_ncert_relevance.py. Fully offline -- no network, no
model call: the verifier is pure string matching, no model is involved in checking it.

2026-11-10: the model is no longer asked for a citation at all (see _NCERT_SECTIONS_RULE) --
_verify_ncert_quotes now takes `sections` (the real picked DB rows: chapter/heading/text) instead
of a plain source string plus an opt-in heading check, and always attaches the real citation of
whichever section a quote matched, ignoring whatever the model wrote for a citation entirely.

1. A real verbatim quote (>=6 words) against the source text passes, citation is script-attached.
2. Curly quotes in the model's output are accepted the same as straight quotes.
3. Extra/irregular whitespace in the quoted text still matches.
4. Case differences still match.
5. A near-miss with exactly one changed word is rejected (not a true substring).
6. A quote under 6 words is rejected even though it IS a verbatim substring.
7. A genuinely invented (not-in-source) quote is rejected.
7b-7e. A verbatim quote with no citation / a wrong citation / the literal unfilled placeholder /
    a different REAL chapter's citation all end up shown with the correct, script-attached
    citation -- the model's own citation text is never trusted or checked.
8. Multiple quotes in one block: valid ones kept, invalid ones dropped, independently.
8b. Two picked sections: a quote is attributed to whichever section's own text it was found in.
9. No start marker at all in the model's raw block -- an empty/garbage block yields nothing.
10. _strip_ncert_block: a normal answer with no block passes through byte-for-byte unchanged,
    and the student-visible chunks never contain the marker tokens for a real one either.
"""
import asyncio
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server

_passed = []
_failed = []


def check(label, condition, detail=""):
    if condition:
        _passed.append(label)
        print(f"  PASS: {label}")
    else:
        _failed.append(label)
        print(f"  FAIL: {label}" + (f" -- {detail}" if detail else ""))


SOURCE_TEXT = (
    "How wonderful is the living world! The wide range of living types is amazing. "
    "The extraordinary habitats in which we find living organisms leave us speechless. "
    "All living organisms exhibit growth, reproduction and the ability to sense their "
    "environment and respond to it."
)
SECTION = {"chapter": "Chapter 1", "heading": "The Living World", "text": SOURCE_TEXT}
SECTIONS = [SECTION]
REAL_CITATION = f"{SECTION['chapter']}, {SECTION['heading']}"

OTHER_SOURCE_TEXT = (
    "Taxonomy is the branch of biology that deals with identification, nomenclature and "
    "classification of organisms. Taxonomic studies help in differentiating useful organisms "
    "from harmful ones."
)
OTHER_SECTION = {"chapter": "Chapter 1", "heading": "Taxonomic Categories", "text": OTHER_SOURCE_TEXT}


def test_real_verbatim_quote_passes():
    print("\n1. real verbatim quote (>=6 words) passes, citation script-attached")
    raw = '"The wide range of living types is amazing." (Chapter 1, The Living World)'
    shown, dropped = server._verify_ncert_quotes(raw, SECTIONS)
    check("one quote shown", len(shown) == 1, shown)
    check("no quotes dropped", len(dropped) == 0, dropped)
    check("citation is the real script-attached one", shown and shown[0]["citation"] == REAL_CITATION, shown)


def test_curly_quotes_accepted():
    print("\n2. curly quotes accepted same as straight quotes")
    raw = "“The wide range of living types is amazing.” (Chapter 1, The Living World)"
    shown, dropped = server._verify_ncert_quotes(raw, SECTIONS)
    check("curly-quoted line still shown", len(shown) == 1, (shown, dropped))
    check("nothing dropped", len(dropped) == 0, dropped)


def test_extra_whitespace_matches():
    print("\n3. extra/irregular whitespace still matches")
    raw = '"The   wide range  of living\ntypes is   amazing." (Chapter 1, The Living World)'
    shown, dropped = server._verify_ncert_quotes(raw, SECTIONS)
    check("whitespace-mangled quote still shown", len(shown) == 1, (shown, dropped))
    check("nothing dropped", len(dropped) == 0, dropped)


def test_case_difference_matches():
    print("\n4. case difference still matches")
    raw = '"THE WIDE RANGE OF LIVING TYPES IS AMAZING." (Chapter 1, The Living World)'
    shown, dropped = server._verify_ncert_quotes(raw, SECTIONS)
    check("uppercased quote still shown", len(shown) == 1, (shown, dropped))
    check("nothing dropped", len(dropped) == 0, dropped)


def test_one_changed_word_rejected():
    print("\n5. near-miss with one changed word is rejected")
    raw = '"The wide range of living creatures is amazing." (Chapter 1, The Living World)'
    shown, dropped = server._verify_ncert_quotes(raw, SECTIONS)
    check("nothing shown", len(shown) == 0, shown)
    check("one dropped", len(dropped) == 1, dropped)


def test_under_six_words_rejected():
    print("\n6. a quote under 6 words is rejected even if verbatim")
    raw = '"How wonderful is the living" (Chapter 1, The Living World)'
    shown, dropped = server._verify_ncert_quotes(raw, SECTIONS)
    check("nothing shown", len(shown) == 0, shown)
    check("one dropped", len(dropped) == 1, dropped)
    check("5-word quote fails the word-count floor", len("How wonderful is the living".split()) < 6)


def test_invented_quote_rejected():
    print("\n7. a fully invented (not-in-source) quote is rejected")
    raw = '"Mitochondria is the powerhouse of the cell." (Chapter 1, The Living World)'
    shown, dropped = server._verify_ncert_quotes(raw, SECTIONS)
    check("nothing shown", len(shown) == 0, shown)
    check("one dropped", len(dropped) == 1, dropped)


def test_mixed_valid_and_invalid_quotes():
    print("\n8. a block with one valid and one invalid quote keeps only the valid one")
    raw = (
        '"The wide range of living types is amazing." (Chapter 1, The Living World)\n'
        '"All living things exhibit death and decay." (Chapter 1, The Living World)'
    )
    shown, dropped = server._verify_ncert_quotes(raw, SECTIONS)
    check("exactly one shown", len(shown) == 1, shown)
    check("exactly one dropped", len(dropped) == 1, dropped)
    check("the real sentence is the one shown", shown and "wide range" in shown[0]["quote"], shown)


def test_quote_with_no_citation_gets_real_one_attached():
    print("\n7b. model writes no citation at all -> script attaches the real one")
    raw = '"The wide range of living types is amazing."'
    shown, dropped = server._verify_ncert_quotes(raw, SECTIONS)
    check("shown despite no citation written", len(shown) == 1, (shown, dropped))
    check("real citation attached", shown and shown[0]["citation"] == REAL_CITATION, shown)


def test_quote_with_wrong_citation_still_shown_with_real_one():
    print("\n7c. model writes a WRONG citation -> ignored, real citation attached anyway")
    raw = '"The wide range of living types is amazing." (NCERT Class 11, Chapter 4 - Animal Kingdom)'
    shown, dropped = server._verify_ncert_quotes(raw, SECTIONS)
    check("shown (text is what's verified, not the model's citation)", len(shown) == 1, (shown, dropped))
    check("real citation attached, not the model's wrong one", shown and shown[0]["citation"] == REAL_CITATION, shown)


def test_quote_with_placeholder_citation_still_shown_with_real_one():
    print("\n7d. model writes back the literal unfilled placeholder -> ignored, real citation attached")
    raw = '"The wide range of living types is amazing." (<Chapter>, <Heading>)'
    shown, dropped = server._verify_ncert_quotes(raw, SECTIONS)
    check("shown despite literal placeholder citation", len(shown) == 1, (shown, dropped))
    check("real citation attached, not the placeholder", shown and shown[0]["citation"] == REAL_CITATION, shown)


def test_quote_citing_a_different_real_chapter_still_shown_with_real_one():
    print("\n7e. model cites a DIFFERENT REAL chapter -> ignored, real citation attached")
    raw = '"The wide range of living types is amazing." (Chapter 7, Evolution)'
    shown, dropped = server._verify_ncert_quotes(raw, SECTIONS)
    check("shown (citation never checked anymore)", len(shown) == 1, (shown, dropped))
    check("real citation attached, not the other real chapter", shown and shown[0]["citation"] == REAL_CITATION, shown)


def test_two_sections_quote_attributed_to_the_right_one():
    print("\n8b. two picked sections: a quote is attributed to whichever section it's actually from")
    sections = [SECTION, OTHER_SECTION]
    raw = (
        '"The wide range of living types is amazing." (Chapter 1, The Living World)\n'
        '"Taxonomic studies help in differentiating useful organisms from harmful ones." (Chapter 1, The Living World)'
    )
    shown, dropped = server._verify_ncert_quotes(raw, sections)
    check("both quotes shown", len(shown) == 2, (shown, dropped))
    citations = {q["citation"] for q in shown}
    check("first quote attributed to SECTION", f"{SECTION['chapter']}, {SECTION['heading']}" in citations, shown)
    check("second quote attributed to OTHER_SECTION despite the model's wrong citation", f"{OTHER_SECTION['chapter']}, {OTHER_SECTION['heading']}" in citations, shown)


def test_empty_or_garbage_block():
    print("\n9. empty/garbage block yields nothing, never raises")
    shown, dropped = server._verify_ncert_quotes("", SECTIONS)
    check("empty string -> nothing shown, nothing dropped", shown == [] and dropped == [])
    # The formatting-tolerant split treats ANY >=6-word leftover line as a candidate (that's the
    # whole point -- a real quote missing its quote marks must not be invisible). This made-up
    # prose is a candidate, but a FAILING one (not a real NCERT substring) -- so it must show
    # nothing, but it shows up in `dropped` rather than being invisible entirely.
    shown, dropped = server._verify_ncert_quotes("no quotes here at all, just prose", SECTIONS)
    check("garbage prose -> nothing shown, but caught and dropped as a failing candidate", shown == [] and len(dropped) == 1, (shown, dropped))
    shown, dropped = server._verify_ncert_quotes(None, SECTIONS)
    check("None raw_block -> nothing shown, nothing dropped, no exception", shown == [] and dropped == [])
    shown, dropped = server._verify_ncert_quotes('"The wide range of living types is amazing."', [])
    check("empty sections list -> nothing shown, nothing raises", shown == [], shown)


def test_no_quote_marks_still_shown():
    print("\n11. a verbatim line with NO quote marks at all is still caught and shown")
    raw = "The wide range of living types is amazing. (Chapter 1, The Living World)"
    shown, dropped = server._verify_ncert_quotes(raw, SECTIONS)
    check("shown despite missing quote marks", len(shown) == 1, (shown, dropped))
    check("nothing dropped", len(dropped) == 0, dropped)
    check("rendered with the source's own exact text", shown and shown[0]["quote"] == "The wide range of living types is amazing.", shown)


def test_bulleted_line_without_quotes():
    print("\n12. a bullet-prefixed line with no quote marks is still caught")
    raw = "- The wide range of living types is amazing. (Chapter 1, The Living World)"
    shown, dropped = server._verify_ncert_quotes(raw, SECTIONS)
    check("bullet-prefixed line shown", len(shown) == 1, (shown, dropped))
    check("bullet char not left in the quote", shown and not shown[0]["quote"].startswith("-"), shown)


def test_numbered_line_without_quotes():
    print("\n13. a numbered line (\"1.\") with no quote marks is still caught")
    raw = "1. The wide range of living types is amazing. (Chapter 1, The Living World)"
    shown, dropped = server._verify_ncert_quotes(raw, SECTIONS)
    check("numbered line shown", len(shown) == 1, (shown, dropped))
    check("number prefix not left in the quote", shown and not shown[0]["quote"][0].isdigit(), shown)


def test_curly_quotes_without_matching_pair_still_shown():
    print("\n14. curly-quoted line with no quote marks still matches (regression guard)")
    raw = "“The wide range of living types is amazing.” (Chapter 1, The Living World)"
    shown, dropped = server._verify_ncert_quotes(raw, SECTIONS)
    check("curly-quoted line shown", len(shown) == 1, (shown, dropped))


def test_paraphrase_without_quotes_rejected():
    print("\n16. a paraphrase with no quote marks is rejected (not a true substring)")
    raw = "The huge variety of living creatures found on Earth is astonishing. (Chapter 1, The Living World)"
    shown, dropped = server._verify_ncert_quotes(raw, SECTIONS)
    check("nothing shown", len(shown) == 0, shown)
    check("one dropped", len(dropped) == 1, dropped)


def test_one_word_changed_without_quotes_rejected():
    print("\n17. one-word-changed line with no quote marks is rejected")
    raw = "The wide range of living creatures is amazing. (Chapter 1, The Living World)"
    shown, dropped = server._verify_ncert_quotes(raw, SECTIONS)
    check("nothing shown", len(shown) == 0, shown)
    check("one dropped", len(dropped) == 1, dropped)


def test_under_six_words_without_quotes_rejected():
    print("\n18. under-6-words line with no quote marks is rejected even though verbatim")
    raw = "How wonderful is the living (Chapter 1, The Living World)"
    shown, dropped = server._verify_ncert_quotes(raw, SECTIONS)
    check("nothing shown", len(shown) == 0, shown)
    check("one dropped", len(dropped) == 1, dropped)


def test_mixed_quoted_and_unquoted_lines_in_one_block():
    print("\n20. a block mixing a quoted line and an unquoted line: both verified independently")
    raw = (
        '"The wide range of living types is amazing." (Chapter 1, The Living World)\n'
        "All living organisms exhibit growth, reproduction and the ability to sense their "
        "environment and respond to it. (Chapter 1, The Living World)"
    )
    shown, dropped = server._verify_ncert_quotes(raw, SECTIONS)
    check("both lines shown", len(shown) == 2, (shown, dropped))
    check("nothing dropped", len(dropped) == 0, dropped)


def test_two_quoted_spans_crammed_on_one_line():
    print("\n21. two quoted spans on a single physical line are split into two candidates")
    raw = (
        '"The wide range of living types is amazing." (Chapter 1, The Living World) '
        '"All living organisms exhibit growth, reproduction and the ability to sense their '
        'environment and respond to it." (Chapter 1, The Living World)'
    )
    shown, dropped = server._verify_ncert_quotes(raw, SECTIONS)
    check("both quoted spans shown", len(shown) == 2, (shown, dropped))
    check("nothing dropped", len(dropped) == 0, dropped)


async def _collect(gen):
    out = []
    async for chunk in gen:
        out.append(chunk)
    return out


async def _fake_stream(chunks):
    for c in chunks:
        yield c


def test_strip_ncert_block_passthrough_when_no_marker():
    print("\n10a. _strip_ncert_block: no start marker -> byte-for-byte passthrough")
    holder = {}
    chunks = ["Answer: ", "mitochondria is the site of respiration.", " Quick Recall: cellular powerhouse."]

    async def run():
        out = await _collect(server._strip_ncert_block(_fake_stream(chunks), holder))
        return out

    out = asyncio.run(run())
    check("output reassembles to the exact original text", "".join(out) == "".join(chunks), out)
    check("holder['raw_block'] stays None", holder.get("raw_block") is None, holder)


def test_strip_ncert_block_captures_and_hides_real_block():
    print("\n10b. _strip_ncert_block: a real block is captured, never yielded to the student")
    holder = {}
    answer_text = "Answer: mitochondria is the site of respiration."
    block_text = (
        f"\n{server._NCERT_BLOCK_START}\n"
        '"The wide range of living types is amazing."\n'
        f"{server._NCERT_BLOCK_END}"
    )
    full_text = answer_text + block_text
    # Split into small, arbitrary chunk boundaries (including one that splits the marker itself)
    # to prove the lookback buffer catches it regardless of where chunk boundaries fall.
    split_point = len(answer_text) + len(server._NCERT_BLOCK_START) // 2 + 1
    chunks = [full_text[:split_point], full_text[split_point:split_point + 5], full_text[split_point + 5:]]

    async def run():
        out = await _collect(server._strip_ncert_block(_fake_stream(chunks), holder))
        return out

    out = asyncio.run(run())
    joined = "".join(out)
    # The newline right before the start marker is part of the real answer text (a blank line
    # separating it from the block), not part of the block itself -- expected to survive.
    check("student-visible text is exactly the answer, nothing from the block", joined == answer_text + "\n", repr(joined))
    check("marker tokens never appear in what was yielded", server._NCERT_BLOCK_START not in joined and server._NCERT_BLOCK_END not in joined, joined)
    check("holder captured the raw block content (markers stripped)", holder.get("raw_block") is not None and "wide range" in holder["raw_block"], holder)
    # And the captured block verifies cleanly end-to-end.
    shown, dropped = server._verify_ncert_quotes(holder["raw_block"], SECTIONS)
    check("the captured block's quote verifies", len(shown) == 1 and len(dropped) == 0, (shown, dropped))
    check("script-attached citation present even though the model wrote none", shown and shown[0]["citation"] == REAL_CITATION, shown)


if __name__ == "__main__":
    test_real_verbatim_quote_passes()
    test_curly_quotes_accepted()
    test_extra_whitespace_matches()
    test_case_difference_matches()
    test_one_changed_word_rejected()
    test_under_six_words_rejected()
    test_invented_quote_rejected()
    test_mixed_valid_and_invalid_quotes()
    test_quote_with_no_citation_gets_real_one_attached()
    test_quote_with_wrong_citation_still_shown_with_real_one()
    test_quote_with_placeholder_citation_still_shown_with_real_one()
    test_quote_citing_a_different_real_chapter_still_shown_with_real_one()
    test_two_sections_quote_attributed_to_the_right_one()
    test_empty_or_garbage_block()
    test_no_quote_marks_still_shown()
    test_bulleted_line_without_quotes()
    test_numbered_line_without_quotes()
    test_curly_quotes_without_matching_pair_still_shown()
    test_paraphrase_without_quotes_rejected()
    test_one_word_changed_without_quotes_rejected()
    test_under_six_words_without_quotes_rejected()
    test_mixed_quoted_and_unquoted_lines_in_one_block()
    test_two_quoted_spans_crammed_on_one_line()
    test_strip_ncert_block_passthrough_when_no_marker()
    test_strip_ncert_block_captures_and_hides_real_block()
    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
