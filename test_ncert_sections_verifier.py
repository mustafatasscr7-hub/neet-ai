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


def test_cutoff_quote_extended_to_sentence_end():
    print("\n22. a quote cut off mid-sentence (no terminal punctuation) is extended to the sentence's end")
    source = (
        "Mitosis ensures that each daughter cell receives an identical set of chromosomes or "
        "an equal share of the parent cell's genetic material."
    )
    section = {"chapter": "Chapter 5", "heading": "Cell Division", "text": source}
    raw = '"Mitosis ensures that each daughter cell receives an identical set of chromosomes or"'
    shown, dropped = server._verify_ncert_quotes(raw, [section])
    check("one quote shown", len(shown) == 1, (shown, dropped))
    check("nothing dropped", len(dropped) == 0, dropped)
    check("quote extended through the rest of the sentence", shown and shown[0]["quote"] == source, shown)


def test_cutoff_quote_dropped_when_no_sentence_end_follows():
    print("\n23. a cut-off quote is dropped when the section text runs out before a sentence end")
    source = (
        "Photosynthesis occurs in the chloroplasts of plant cells and requires sunlight, water and "
        "carbon dioxide along with"
    )
    section = {"chapter": "Chapter 6", "heading": "Photosynthesis", "text": source}
    raw = '"Photosynthesis occurs in the chloroplasts of plant cells and requires sunlight, water and"'
    shown, dropped = server._verify_ncert_quotes(raw, [section])
    check("nothing shown (no terminal punctuation to extend to)", len(shown) == 0, (shown, dropped))
    check("one dropped", len(dropped) == 1, dropped)


def test_cutoff_quote_dropped_when_extension_exceeds_word_cap():
    print("\n24. a cut-off quote is dropped when extending to the sentence end would exceed 40 words")
    source = "The process begins when the cell enters the S phase and" + " filler" * 40 + "."
    section = {"chapter": "Chapter 5", "heading": "Cell Division", "text": source}
    raw = '"The process begins when the cell enters the S phase and"'
    shown, dropped = server._verify_ncert_quotes(raw, [section])
    check("nothing shown (extension would exceed the 40-word cap)", len(shown) == 0, (shown, dropped))
    check("one dropped", len(dropped) == 1, dropped)


def test_quote_already_ending_in_punctuation_not_extended():
    print("\n25. a quote that already ends with terminal punctuation is shown unchanged (regression guard)")
    source = (
        "Mitosis ensures that each daughter cell receives an identical set of chromosomes. "
        "This is followed by cytokinesis."
    )
    section = {"chapter": "Chapter 5", "heading": "Cell Division", "text": source}
    raw = '"Mitosis ensures that each daughter cell receives an identical set of chromosomes."'
    shown, dropped = server._verify_ncert_quotes(raw, [section])
    check("one quote shown", len(shown) == 1, (shown, dropped))
    check(
        "quote is exactly the first sentence only, not extended into the next one",
        shown and shown[0]["quote"] == "Mitosis ensures that each daughter cell receives an identical set of chromosomes.",
        shown,
    )


def test_figure_label_wedged_mid_sentence_rejected_unless_text_matches():
    print("\n26. a quote skipping wedged figure-label noise is rejected unless the stored text matches it")
    dirty_source = (
        "During mitosis the chromosomes condense and align at the equatorial plate "
        "Figure 9.9 (a) Prophase (b) Metaphase "
        "before being pulled apart towards opposite poles."
    )
    dirty_section = {"chapter": "Chapter 5", "heading": "Cell Division", "text": dirty_source}
    # A "clean" candidate quote that skips over the wedged figure labels entirely -- NOT a
    # verbatim substring of the actual (dirty) stored section text, so it must be rejected exactly
    # as any other non-exact candidate would be (see _verify_ncert_quotes's exact-substring rule).
    clean_raw = (
        '"During mitosis the chromosomes condense and align at the equatorial plate before being '
        'pulled apart towards opposite poles."'
    )
    shown, dropped = server._verify_ncert_quotes(clean_raw, [dirty_section])
    check("the label-skipping quote is rejected (not a true substring of the dirty stored text)", len(shown) == 0, (shown, dropped))
    check("one dropped", len(dropped) == 1, dropped)
    # Once the stored text is itself clean (no wedged labels -- what a fixed ingest would produce),
    # the identical quote now IS an exact substring and is correctly shown.
    clean_section_text = (
        "During mitosis the chromosomes condense and align at the equatorial plate "
        "before being pulled apart towards opposite poles."
    )
    clean_section = {"chapter": "Chapter 5", "heading": "Cell Division", "text": clean_section_text}
    shown2, dropped2 = server._verify_ncert_quotes(clean_raw, [clean_section])
    check("the same quote is shown once the stored text is clean", len(shown2) == 1, (shown2, dropped2))
    check("nothing dropped once clean", len(dropped2) == 0, dropped2)


def test_quote_starting_mid_sentence_extended_to_sentence_start():
    print("\n27. a quote starting mid-sentence (leftover fragment) is extended back to the sentence's start")
    source = (
        "Enzymes are biocatalysts that speed up the rate of biochemical reactions. "
        "The process of glycolysis takes place in the cytoplasm and produces pyruvate."
    )
    section = {"chapter": "Chapter 9", "heading": "Biomolecules", "text": source}
    raw = '"process of glycolysis takes place in the cytoplasm and produces pyruvate."'
    shown, dropped = server._verify_ncert_quotes(raw, [section])
    check("one quote shown", len(shown) == 1, (shown, dropped))
    check("nothing dropped", len(dropped) == 0, dropped)
    check(
        "quote extended back to the start of its own sentence",
        shown and shown[0]["quote"] == "The process of glycolysis takes place in the cytoplasm and produces pyruvate.",
        shown,
    )


def test_quote_dropped_when_no_sentence_start_precedes():
    print("\n28. a mid-sentence-start quote is dropped when no earlier sentence boundary exists")
    source = "along with the cytoplasm and produces pyruvate at the end of the pathway."
    section = {"chapter": "Chapter 9", "heading": "Biomolecules", "text": source}
    raw = '"cytoplasm and produces pyruvate at the end of the pathway."'
    shown, dropped = server._verify_ncert_quotes(raw, [section])
    check("nothing shown (no earlier sentence boundary to extend back to)", len(shown) == 0, (shown, dropped))
    check("one dropped", len(dropped) == 1, dropped)


def test_quote_dropped_when_start_extension_exceeds_word_cap():
    print("\n29. a mid-sentence-start quote is dropped when extending to the sentence start would exceed 40 words")
    source = "Enzymes are biocatalysts." + " filler" * 40 + " and the process of glycolysis takes place in the cytoplasm."
    section = {"chapter": "Chapter 9", "heading": "Biomolecules", "text": source}
    raw = '"process of glycolysis takes place in the cytoplasm."'
    shown, dropped = server._verify_ncert_quotes(raw, [section])
    check("nothing shown (extension would exceed the 40-word cap)", len(shown) == 0, (shown, dropped))
    check("one dropped", len(dropped) == 1, dropped)


def test_quote_already_at_sentence_start_not_extended():
    print("\n30. a quote that already begins at the very start of the section text is shown unchanged (regression guard)")
    source = "The process of glycolysis takes place in the cytoplasm. This is followed by the Krebs cycle."
    section = {"chapter": "Chapter 9", "heading": "Biomolecules", "text": source}
    raw = '"The process of glycolysis takes place in the cytoplasm."'
    shown, dropped = server._verify_ncert_quotes(raw, [section])
    check("one quote shown", len(shown) == 1, (shown, dropped))
    check(
        "quote is exactly the first sentence only, not extended backward",
        shown and shown[0]["quote"] == "The process of glycolysis takes place in the cytoplasm.",
        shown,
    )


def test_overlapping_quotes_only_first_kept():
    print("\n31. two quoted spans that overlap in the source text keep only the first, the second is dropped")
    source = (
        "Enzymes are biocatalysts that speed up the rate of biochemical reactions in living cells. "
        "Most enzymes are proteins that act on a specific substrate."
    )
    section = {"chapter": "Chapter 9", "heading": "Biomolecules", "text": source}
    raw = (
        '"Enzymes are biocatalysts that speed up the rate of biochemical reactions in living cells."\n'
        '"speed up the rate of biochemical reactions in living cells."'
    )
    shown, dropped = server._verify_ncert_quotes(raw, [section])
    check("only one quote shown", len(shown) == 1, (shown, dropped))
    check("the second, overlapping quote is dropped", len(dropped) == 1, dropped)
    check(
        "the first (longer) quote is the one kept",
        shown and shown[0]["quote"].startswith("Enzymes are biocatalysts"),
        shown,
    )


def test_non_overlapping_quotes_both_kept():
    print("\n32. two quotes from different, non-overlapping sentences in the same section both survive (regression guard)")
    source = (
        "Enzymes are biocatalysts that speed up the rate of biochemical reactions in living cells. "
        "Most enzymes are proteins that act on a specific substrate."
    )
    section = {"chapter": "Chapter 9", "heading": "Biomolecules", "text": source}
    raw = (
        '"Enzymes are biocatalysts that speed up the rate of biochemical reactions in living cells."\n'
        '"Most enzymes are proteins that act on a specific substrate."'
    )
    shown, dropped = server._verify_ncert_quotes(raw, [section])
    check("both quotes shown", len(shown) == 2, (shown, dropped))
    check("nothing dropped", len(dropped) == 0, dropped)


def test_quotes_from_two_chapters_only_first_chapter_kept():
    print("\n33. quotes matched against two DIFFERENT chapters: only the first quote's chapter survives")
    chapter9_section = {"chapter": "Chapter 9", "heading": "Biomolecules", "text": SOURCE_TEXT}
    chapter3_section = {"chapter": "Chapter 3", "heading": "Plant Kingdom", "text": OTHER_SOURCE_TEXT}
    raw = (
        '"The wide range of living types is amazing."\n'
        '"Taxonomic studies help in differentiating useful organisms from harmful ones."'
    )
    shown, dropped = server._verify_ncert_quotes(raw, [chapter9_section, chapter3_section])
    check("only one quote shown", len(shown) == 1, (shown, dropped))
    check("the second quote (a different chapter) is dropped", len(dropped) == 1, dropped)
    check("the surviving quote cites chapter 9 (the first quote's chapter)", shown and shown[0]["citation"].startswith("Chapter 9"), shown)


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


def test_strip_ncert_block_merges_two_separate_blocks():
    print("\n10c. _strip_ncert_block: two separate START/END blocks are merged, not just the first")
    holder = {}
    answer_text = "Answer: mitochondria is the site of respiration."
    block_text = (
        f"\n{server._NCERT_BLOCK_START}\n"
        '"The wide range of living types is amazing."\n'
        f"{server._NCERT_BLOCK_END}\n"
        f"{server._NCERT_BLOCK_START}\n"
        '"All living organisms exhibit growth, reproduction and the ability to sense their environment and respond to it."\n'
        f"{server._NCERT_BLOCK_END}"
    )
    full_text = answer_text + block_text
    chunks = [full_text[:len(answer_text) + 5], full_text[len(answer_text) + 5:]]

    async def run():
        return await _collect(server._strip_ncert_block(_fake_stream(chunks), holder))

    asyncio.run(run())
    raw_block = holder.get("raw_block") or ""
    check("first block's content survived", "wide range" in raw_block, raw_block)
    check("second block's content also survived (not dropped)", "ability to sense" in raw_block, raw_block)
    shown, dropped = server._verify_ncert_quotes(raw_block, SECTIONS)
    check("both blocks' quotes verify", len(shown) == 2 and len(dropped) == 0, (shown, dropped))


def test_strip_ncert_block_single_block_one_line():
    print("\n10d. _strip_ncert_block: a single block with exactly one line still works")
    holder = {}
    answer_text = "Answer: mitochondria is the site of respiration."
    block_text = (
        f"\n{server._NCERT_BLOCK_START}\n"
        '"The wide range of living types is amazing."\n'
        f"{server._NCERT_BLOCK_END}"
    )
    full_text = answer_text + block_text

    async def run():
        return await _collect(server._strip_ncert_block(_fake_stream([full_text]), holder))

    asyncio.run(run())
    raw_block = holder.get("raw_block") or ""
    shown, dropped = server._verify_ncert_quotes(raw_block, SECTIONS)
    check("the single line verifies", len(shown) == 1 and len(dropped) == 0, (shown, dropped))


def test_strip_ncert_block_empty_block_no_lines():
    print("\n10e. _strip_ncert_block: a block with no lines at all (empty between markers)")
    holder = {}
    answer_text = "Answer: mitochondria is the site of respiration."
    block_text = f"\n{server._NCERT_BLOCK_START}\n{server._NCERT_BLOCK_END}"
    full_text = answer_text + block_text

    async def run():
        return await _collect(server._strip_ncert_block(_fake_stream([full_text]), holder))

    asyncio.run(run())
    raw_block = holder.get("raw_block")
    shown, dropped = server._verify_ncert_quotes(raw_block or "", SECTIONS)
    check("no candidates survive an empty block", len(shown) == 0 and len(dropped) == 0, (shown, dropped))
    check("retry would trigger on an empty block", server._should_retry_ncert_block(raw_block, shown), (raw_block, shown))


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
    test_cutoff_quote_extended_to_sentence_end()
    test_cutoff_quote_dropped_when_no_sentence_end_follows()
    test_cutoff_quote_dropped_when_extension_exceeds_word_cap()
    test_quote_already_ending_in_punctuation_not_extended()
    test_figure_label_wedged_mid_sentence_rejected_unless_text_matches()
    test_quote_starting_mid_sentence_extended_to_sentence_start()
    test_quote_dropped_when_no_sentence_start_precedes()
    test_quote_dropped_when_start_extension_exceeds_word_cap()
    test_quote_already_at_sentence_start_not_extended()
    test_overlapping_quotes_only_first_kept()
    test_non_overlapping_quotes_both_kept()
    test_quotes_from_two_chapters_only_first_chapter_kept()
    test_strip_ncert_block_passthrough_when_no_marker()
    test_strip_ncert_block_captures_and_hides_real_block()
    test_strip_ncert_block_merges_two_separate_blocks()
    test_strip_ncert_block_single_block_one_line()
    test_strip_ncert_block_empty_block_no_lines()
    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
