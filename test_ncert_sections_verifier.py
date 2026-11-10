"""Unit tests for the "From NCERT" exact-quote verifier (NCERT_SECTIONS_ENABLED, default off) --
see _verify_ncert_quotes / _normalize_ncert_quote_text / _strip_ncert_block in server.py.

Run with: python test_ncert_sections_verifier.py
Plain pass/fail harness, same style as test_ncert_relevance.py. Fully offline -- no network, no
model call: the verifier is pure string matching, no model is involved in checking it.

1. A real verbatim quote (>=6 words) against the source text passes.
2. Curly quotes in the model's output are accepted the same as straight quotes.
3. Extra/irregular whitespace in the quoted text still matches.
4. Case differences still match.
5. A near-miss with exactly one changed word is rejected (not a true substring).
6. A quote under 6 words is rejected even though it IS a verbatim substring.
7. A genuinely invented (not-in-source) quote is rejected.
7b. A verbatim quote with a FABRICATED citation (real words, wrong chapter/heading) is rejected
    when valid_headings is given; the citation check is opt-in (omitting it skips the check).
8. Multiple quotes in one block: valid ones kept, invalid ones dropped, independently.
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


def test_real_verbatim_quote_passes():
    print("\n1. real verbatim quote (>=6 words) passes")
    block = '==='  # marker literals are stripped by _strip_ncert_block, not by the verifier itself
    raw = '"The wide range of living types is amazing." (Chapter 1, The Living World)'
    shown, dropped = server._verify_ncert_quotes(raw, SOURCE_TEXT)
    check("one quote shown", len(shown) == 1, shown)
    check("no quotes dropped", len(dropped) == 0, dropped)
    check("citation captured", shown and shown[0]["citation"] == "Chapter 1, The Living World", shown)


def test_curly_quotes_accepted():
    print("\n2. curly quotes accepted same as straight quotes")
    raw = "“The wide range of living types is amazing.” (Chapter 1, The Living World)"
    shown, dropped = server._verify_ncert_quotes(raw, SOURCE_TEXT)
    check("curly-quoted line still shown", len(shown) == 1, (shown, dropped))
    check("nothing dropped", len(dropped) == 0, dropped)


def test_extra_whitespace_matches():
    print("\n3. extra/irregular whitespace still matches")
    raw = '"The   wide range  of living\ntypes is   amazing." (Chapter 1, The Living World)'
    shown, dropped = server._verify_ncert_quotes(raw, SOURCE_TEXT)
    check("whitespace-mangled quote still shown", len(shown) == 1, (shown, dropped))
    check("nothing dropped", len(dropped) == 0, dropped)


def test_case_difference_matches():
    print("\n4. case difference still matches")
    raw = '"THE WIDE RANGE OF LIVING TYPES IS AMAZING." (Chapter 1, The Living World)'
    shown, dropped = server._verify_ncert_quotes(raw, SOURCE_TEXT)
    check("uppercased quote still shown", len(shown) == 1, (shown, dropped))
    check("nothing dropped", len(dropped) == 0, dropped)


def test_one_changed_word_rejected():
    print("\n5. near-miss with one changed word is rejected")
    raw = '"The wide range of living creatures is amazing." (Chapter 1, The Living World)'
    shown, dropped = server._verify_ncert_quotes(raw, SOURCE_TEXT)
    check("nothing shown", len(shown) == 0, shown)
    check("one dropped", len(dropped) == 1, dropped)


def test_under_six_words_rejected():
    print("\n6. a quote under 6 words is rejected even if verbatim")
    raw = '"How wonderful is the living" (Chapter 1, The Living World)'
    shown, dropped = server._verify_ncert_quotes(raw, SOURCE_TEXT)
    check("nothing shown", len(shown) == 0, shown)
    check("one dropped", len(dropped) == 1, dropped)
    check("5-word quote fails the word-count floor", len("How wonderful is the living".split()) < 6)


def test_invented_quote_rejected():
    print("\n7. a fully invented (not-in-source) quote is rejected")
    raw = '"Mitochondria is the powerhouse of the cell." (Chapter 1, The Living World)'
    shown, dropped = server._verify_ncert_quotes(raw, SOURCE_TEXT)
    check("nothing shown", len(shown) == 0, shown)
    check("one dropped", len(dropped) == 1, dropped)


def test_mixed_valid_and_invalid_quotes():
    print("\n8. a block with one valid and one invalid quote keeps only the valid one")
    raw = (
        '"The wide range of living types is amazing." (Chapter 1, The Living World)\n'
        '"All living things exhibit death and decay." (Chapter 1, The Living World)'
    )
    shown, dropped = server._verify_ncert_quotes(raw, SOURCE_TEXT)
    check("exactly one shown", len(shown) == 1, shown)
    check("exactly one dropped", len(dropped) == 1, dropped)
    check("the real sentence is the one shown", shown and "wide range" in shown[0]["quote"], shown)


def test_real_quote_wrong_citation_rejected():
    print("\n7b. a verbatim quote with a FABRICATED citation (real words, wrong chapter) is rejected")
    # Live-observed failure mode: the model copied a real sentence word-for-word but cited a
    # different, wrong NCERT chapter for it instead of the one it actually came from.
    raw = '"The wide range of living types is amazing." (NCERT Class 11, Chapter 4 - Animal Kingdom)'
    shown, dropped = server._verify_ncert_quotes(raw, SOURCE_TEXT, valid_headings=["Chapter 1: The Living World"])
    check("nothing shown (citation doesn't name the real heading)", len(shown) == 0, shown)
    check("one dropped", len(dropped) == 1, dropped)
    # Same quote, correct citation this time -> passes.
    raw_ok = '"The wide range of living types is amazing." (Chapter 1: The Living World)'
    shown2, dropped2 = server._verify_ncert_quotes(raw_ok, SOURCE_TEXT, valid_headings=["Chapter 1: The Living World"])
    check("shown once citation matches the real heading", len(shown2) == 1, (shown2, dropped2))
    # No valid_headings given at all (caller didn't ask for the check) -> citation is not checked.
    shown3, dropped3 = server._verify_ncert_quotes(raw, SOURCE_TEXT)
    check("citation check is opt-in: omitting valid_headings skips it", len(shown3) == 1, (shown3, dropped3))


def test_empty_or_garbage_block():
    print("\n9. empty/garbage block yields nothing, never raises")
    shown, dropped = server._verify_ncert_quotes("", SOURCE_TEXT)
    check("empty string -> nothing shown, nothing dropped", shown == [] and dropped == [])
    # Updated 2026-11-10: the formatting-tolerant split now treats ANY >=6-word leftover line as
    # a candidate (that's the whole point -- a real quote missing its quote marks must no longer
    # be invisible). This made-up prose is a candidate, but a FAILING one (not a real NCERT
    # substring) -- so it must show nothing, but it now shows up in `dropped` rather than being
    # invisible to the verifier the way it was under the old quote-mark-anchored regex.
    shown, dropped = server._verify_ncert_quotes("no quotes here at all, just prose", SOURCE_TEXT)
    check("garbage prose -> nothing shown, but caught and dropped as a failing candidate", shown == [] and len(dropped) == 1, (shown, dropped))
    shown, dropped = server._verify_ncert_quotes(None, SOURCE_TEXT)
    check("None raw_block -> nothing shown, nothing dropped, no exception", shown == [] and dropped == [])


def test_no_quote_marks_still_shown():
    print("\n11. a verbatim line with NO quote marks at all is still caught and shown")
    # Live root cause (2026-11-10, Qwen-Flash): a genuinely verbatim, correctly cited line with
    # the quote marks dropped entirely was invisible to the old quote-mark-anchored regex.
    raw = "The wide range of living types is amazing. (Chapter 1, The Living World)"
    shown, dropped = server._verify_ncert_quotes(raw, SOURCE_TEXT)
    check("shown despite missing quote marks", len(shown) == 1, (shown, dropped))
    check("nothing dropped", len(dropped) == 0, dropped)
    check("rendered with the source's own exact text", shown and shown[0]["quote"] == "The wide range of living types is amazing.", shown)


def test_bulleted_line_without_quotes():
    print("\n12. a bullet-prefixed line with no quote marks is still caught")
    raw = "- The wide range of living types is amazing. (Chapter 1, The Living World)"
    shown, dropped = server._verify_ncert_quotes(raw, SOURCE_TEXT)
    check("bullet-prefixed line shown", len(shown) == 1, (shown, dropped))
    check("bullet char not left in the quote", shown and not shown[0]["quote"].startswith("-"), shown)


def test_numbered_line_without_quotes():
    print("\n13. a numbered line (\"1.\") with no quote marks is still caught")
    raw = "1. The wide range of living types is amazing. (Chapter 1, The Living World)"
    shown, dropped = server._verify_ncert_quotes(raw, SOURCE_TEXT)
    check("numbered line shown", len(shown) == 1, (shown, dropped))
    check("number prefix not left in the quote", shown and not shown[0]["quote"][0].isdigit(), shown)


def test_curly_quotes_without_matching_pair_still_shown():
    print("\n14. curly-quoted line with no quote marks still matches (regression guard)")
    raw = "“The wide range of living types is amazing.” (Chapter 1, The Living World)"
    shown, dropped = server._verify_ncert_quotes(raw, SOURCE_TEXT)
    check("curly-quoted line shown", len(shown) == 1, (shown, dropped))


def test_citation_variant_without_quotes():
    print("\n15. a citation-variant line (different wording, same heading) with no quote marks")
    raw = "The wide range of living types is amazing. (NCERT Class 11, Chapter 1: The Living World)"
    shown, dropped = server._verify_ncert_quotes(raw, SOURCE_TEXT, valid_headings=["Chapter 1: The Living World"])
    check("shown: citation names the real heading even with extra wording", len(shown) == 1, (shown, dropped))


def test_paraphrase_without_quotes_rejected():
    print("\n16. a paraphrase with no quote marks is rejected (not a true substring)")
    raw = "The huge variety of living creatures found on Earth is astonishing. (Chapter 1, The Living World)"
    shown, dropped = server._verify_ncert_quotes(raw, SOURCE_TEXT)
    check("nothing shown", len(shown) == 0, shown)
    check("one dropped", len(dropped) == 1, dropped)


def test_one_word_changed_without_quotes_rejected():
    print("\n17. one-word-changed line with no quote marks is rejected")
    raw = "The wide range of living creatures is amazing. (Chapter 1, The Living World)"
    shown, dropped = server._verify_ncert_quotes(raw, SOURCE_TEXT)
    check("nothing shown", len(shown) == 0, shown)
    check("one dropped", len(dropped) == 1, dropped)


def test_under_six_words_without_quotes_rejected():
    print("\n18. under-6-words line with no quote marks is rejected even though verbatim")
    raw = "How wonderful is the living (Chapter 1, The Living World)"
    shown, dropped = server._verify_ncert_quotes(raw, SOURCE_TEXT)
    check("nothing shown", len(shown) == 0, shown)
    check("one dropped", len(dropped) == 1, dropped)


def test_wrong_citation_without_quotes_rejected():
    print("\n19. wrong-citation line with no quote marks is rejected")
    raw = "The wide range of living types is amazing. (NCERT Class 11, Chapter 4 - Animal Kingdom)"
    shown, dropped = server._verify_ncert_quotes(raw, SOURCE_TEXT, valid_headings=["Chapter 1: The Living World"])
    check("nothing shown", len(shown) == 0, shown)
    check("one dropped", len(dropped) == 1, dropped)


def test_mixed_quoted_and_unquoted_lines_in_one_block():
    print("\n20. a block mixing a quoted line and an unquoted line: both verified independently")
    raw = (
        '"The wide range of living types is amazing." (Chapter 1, The Living World)\n'
        "All living organisms exhibit growth, reproduction and the ability to sense their "
        "environment and respond to it. (Chapter 1, The Living World)"
    )
    shown, dropped = server._verify_ncert_quotes(raw, SOURCE_TEXT)
    check("both lines shown", len(shown) == 2, (shown, dropped))
    check("nothing dropped", len(dropped) == 0, dropped)


def test_two_quoted_spans_crammed_on_one_line():
    print("\n21. two quoted spans on a single physical line are split into two candidates")
    raw = (
        '"The wide range of living types is amazing." (Chapter 1, The Living World) '
        '"All living organisms exhibit growth, reproduction and the ability to sense their '
        'environment and respond to it." (Chapter 1, The Living World)'
    )
    shown, dropped = server._verify_ncert_quotes(raw, SOURCE_TEXT)
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
        '"The wide range of living types is amazing." (Chapter 1, The Living World)\n'
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
    shown, dropped = server._verify_ncert_quotes(holder["raw_block"], SOURCE_TEXT)
    check("the captured block's quote verifies", len(shown) == 1 and len(dropped) == 0, (shown, dropped))


if __name__ == "__main__":
    test_real_verbatim_quote_passes()
    test_curly_quotes_accepted()
    test_extra_whitespace_matches()
    test_case_difference_matches()
    test_one_changed_word_rejected()
    test_under_six_words_rejected()
    test_invented_quote_rejected()
    test_real_quote_wrong_citation_rejected()
    test_mixed_valid_and_invalid_quotes()
    test_empty_or_garbage_block()
    test_no_quote_marks_still_shown()
    test_bulleted_line_without_quotes()
    test_numbered_line_without_quotes()
    test_curly_quotes_without_matching_pair_still_shown()
    test_citation_variant_without_quotes()
    test_paraphrase_without_quotes_rejected()
    test_one_word_changed_without_quotes_rejected()
    test_under_six_words_without_quotes_rejected()
    test_wrong_citation_without_quotes_rejected()
    test_mixed_quoted_and_unquoted_lines_in_one_block()
    test_two_quoted_spans_crammed_on_one_line()
    test_strip_ncert_block_passthrough_when_no_marker()
    test_strip_ncert_block_captures_and_hides_real_block()
    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
