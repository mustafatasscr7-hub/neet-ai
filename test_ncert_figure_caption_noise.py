"""Unit tests for the figure/table caption and multi-panel letter-label noise detector in
ncert_ingest_core.py -- find_caption_panel_runs / strip_figure_caption_noise /
strip_figure_caption_noise_from_blocks, and their wiring into process_chapter_pdf_bytes for
future ingests.

Fully offline, synthetic text only -- no real NCERT text, no network, no model calls.

Run with: python test_ncert_figure_caption_noise.py
Plain pass/fail harness, same style as test_ncert_ingest.py.
"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ncert_ingest_core as core
import test_admin_ncert_content as base  # reuses make_chapter_pdf

make_chapter_pdf = base.make_chapter_pdf

_passed = []
_failed = []


def check(label, condition, detail=""):
    if condition:
        _passed.append(label)
        print(f"  PASS: {label}")
    else:
        _failed.append(label)
        print(f"  FAIL: {label}" + (f" -- {detail}" if detail else ""))


def test_figure_prefix_with_two_panels_wedged_mid_sentence_is_dropped():
    print("\n1. 'Figure N.N (a) X (b) Y' wedged mid-sentence is dropped, sentence reads clean")
    text = (
        "The chromosomes condense and align at the equatorial plate "
        "Figure 9.9 (a) Prophase (b) Metaphase "
        "before being pulled apart towards opposite poles."
    )
    cleaned, removed = core.strip_figure_caption_noise(text)
    check(
        "clean sentence reads as one continuous sentence",
        cleaned == "The chromosomes condense and align at the equatorial plate before being "
        "pulled apart towards opposite poles.",
        cleaned,
    )
    check("word count removed matches the dropped span", removed == 6, removed)


def test_stray_panel_markers_without_figure_prefix_are_dropped():
    print("\n2. stray '(a) ... (b) ... (c) ...' panel markers with no 'Figure N.N' prefix are dropped")
    text = (
        "Mitosis occurs in four stages (a) Prophase (b) Metaphase (c) Anaphase "
        "followed by cytokinesis which divides the cytoplasm equally."
    )
    cleaned, removed = core.strip_figure_caption_noise(text)
    check(
        "clean sentence has no panel letters left",
        cleaned == "Mitosis occurs in four stages followed by cytokinesis which divides the "
        "cytoplasm equally.",
        cleaned,
    )
    check("6 words removed (3 panel letters + 3 one-word labels)", removed == 6, removed)


def test_lone_figure_reference_never_removed():
    print("\n3. a lone 'Figure N.N' inline cross-reference (no panel letters) is never touched")
    text = (
        "This structure is clearly visible as shown in Figure 9.9 of this chapter and helps "
        "students understand the process better overall."
    )
    cleaned, removed = core.strip_figure_caption_noise(text)
    check("text unchanged", cleaned == text, cleaned)
    check("nothing removed", removed == 0, removed)


def test_single_isolated_panel_letter_never_removed():
    print("\n4. a single isolated '(a)' with no second panel letter is left alone (ordinary prose)")
    text = "The root system can be of two main types (a) tap root which grows deep into the soil."
    cleaned, removed = core.strip_figure_caption_noise(text)
    check("text unchanged", cleaned == text, cleaned)
    check("nothing removed", removed == 0, removed)


def test_non_ascending_letters_never_removed():
    print("\n5. non-ascending panel letters ('(b) ... (a) ...') are not treated as a real panel run")
    text = "The diagram shows two things (b) wrong order first and (a) wrong order second here today."
    cleaned, removed = core.strip_figure_caption_noise(text)
    check("text unchanged", cleaned == text, cleaned)
    check("nothing removed", removed == 0, removed)


def test_panel_letters_too_far_apart_never_removed():
    print("\n6. panel letters separated by more than the gap cap are not treated as the same run")
    text = (
        "The process begins here (a) with one two three four five six seven eight words gap "
        "(b) ends here today."
    )
    cleaned, removed = core.strip_figure_caption_noise(text)
    check("text unchanged", cleaned == text, cleaned)
    check("nothing removed", removed == 0, removed)


def test_multi_word_title_case_label_absorbed_but_not_the_resuming_sentence():
    print("\n7. a multi-word Title Case label after the last panel letter is absorbed, the lower-case resuming sentence is not")
    text = (
        "The cell undergoes division as shown in the Figure 2.4 (a) Diagrammatic View "
        "(b) Actual Photograph before resuming normal metabolic activity afterward."
    )
    cleaned, removed = core.strip_figure_caption_noise(text)
    check(
        "resuming lower-case clause survives intact",
        cleaned == "The cell undergoes division as shown in the before resuming normal "
        "metabolic activity afterward.",
        cleaned,
    )
    check("all 8 caption words removed", removed == 8, removed)


def test_two_separate_panel_runs_in_one_block_both_dropped():
    print("\n8. two separate figure panel-label runs in the same block are each detected independently")
    text = (
        "The diagram has many stages (a) First (b) Second (c) Third and later a second figure "
        "shows (a) Fourth (b) Fifth stage before the chapter moves on to the next topic."
    )
    cleaned, removed = core.strip_figure_caption_noise(text)
    check("no panel letters survive", "(a)" not in cleaned and "(b)" not in cleaned and "(c)" not in cleaned, cleaned)
    check("the real resuming word 'stage' (lower-case) survives", "stage before the chapter" in cleaned, cleaned)
    check("10 words removed total (6 + 4)", removed == 10, removed)


def test_panel_run_at_end_of_text_no_crash():
    print("\n9. a panel-label run with nothing after it (end of the block's text) is handled cleanly")
    text = "The structure is divided into parts (a) Nucleus (b) Cytoplasm"
    cleaned, removed = core.strip_figure_caption_noise(text)
    check("clean text is exactly the sentence start, nothing trailing", cleaned == "The structure is divided into parts", cleaned)
    check("4 words removed", removed == 4, removed)


def test_strip_from_blocks_only_touches_para_and_list():
    print("\n10. strip_figure_caption_noise_from_blocks only cleans para/list blocks, never heading/table")
    blocks = [
        ("heading", "9.1 Cell Division (a) Mitosis (b) Meiosis"),
        ("para", "Mitosis occurs in four stages (a) Prophase (b) Metaphase followed by cytokinesis overall."),
        ("table", "Table 9.1 (a) Mitosis (b) Meiosis"),
    ]
    new_blocks, removed_total = core.strip_figure_caption_noise_from_blocks(blocks)
    check("heading block is untouched even though it contains panel letters", new_blocks[0] == blocks[0], new_blocks[0])
    check("table block is untouched even though it contains panel letters", new_blocks[2] == blocks[2], new_blocks[2])
    check("para block had its panel letters stripped", "(a)" not in new_blocks[1][1] and "(b)" not in new_blocks[1][1], new_blocks[1])
    check("removed word count is positive", removed_total > 0, removed_total)


def test_process_chapter_pdf_bytes_strips_wedged_caption_for_future_ingests():
    print("\n11. end-to-end: process_chapter_pdf_bytes strips a wedged figure-label run from a real synthetic PDF")
    pdf_bytes = make_chapter_pdf([
        "CHAPTER 9",
        "A CAPTION NOISE TEST CHAPTER",
        "9.1 Real Heading One",
        "The chromosomes condense and align at the equatorial plate",
        "Figure 9.9 (a) Prophase (b) Metaphase",
        "before being pulled apart towards opposite poles clearly and completely.",
        "9.2 Real Heading Two",
        "How wonderful is the living world and the wide range of living types that are found "
        "in the forests and the oceans of the earth.",
        "9.3 Real Heading Three",
        "All living organisms exhibit growth, reproduction and the ability to sense their "
        "environment and respond to it.",
    ])
    result = core.process_chapter_pdf_bytes(pdf_bytes)
    section_texts = [s["text"] for s in result["sections"]]
    check(
        "no section text still contains the wedged figure-caption marker",
        not any("Figure 9.9" in t for t in section_texts),
        section_texts,
    )
    check(
        "no section text still contains a bare panel letter marker",
        not any("(a)" in t or "(b)" in t for t in section_texts),
        section_texts,
    )
    check(
        "the real sentence survived whole, joined cleanly across the removed noise",
        any(
            "condense and align at the equatorial plate before being pulled apart" in t
            for t in section_texts
        ),
        section_texts,
    )
    check("chapter still validates clean (no new flags from this change)", result["flags"] == [], result["flags"])


if __name__ == "__main__":
    test_figure_prefix_with_two_panels_wedged_mid_sentence_is_dropped()
    test_stray_panel_markers_without_figure_prefix_are_dropped()
    test_lone_figure_reference_never_removed()
    test_single_isolated_panel_letter_never_removed()
    test_non_ascending_letters_never_removed()
    test_panel_letters_too_far_apart_never_removed()
    test_multi_word_title_case_label_absorbed_but_not_the_resuming_sentence()
    test_two_separate_panel_runs_in_one_block_both_dropped()
    test_panel_run_at_end_of_text_no_crash()
    test_strip_from_blocks_only_touches_para_and_list()
    test_process_chapter_pdf_bytes_strips_wedged_caption_for_future_ingests()
    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
