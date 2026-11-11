"""Unit tests for ncert_ingest.py: the garbled-font check, the skip logic,
and heading detection (bare numbered headings split across PDF text lines).
Fully offline, synthetic text only - no PDFs, no network, no model calls.

Run with: python test_ncert_ingest.py
Plain pass/fail harness like test_ncert_relevance.py.
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import ncert_ingest as ing

_passed = []
_failed = []


def check(label, condition, detail=""):
    if condition:
        _passed.append(label)
        print(f"  PASS: {label}")
    else:
        _failed.append(label)
        print(f"  FAIL: {label}" + (f" -- {detail}" if detail else ""))


CLEAN_PARA = (
    "How wonderful is the living world and the wide range of living types "
    "that are found in the forests and the oceans of the earth."
)


def section(heading, text, ordinal):
    return {"heading": heading, "text": text, "word_count": len(text.split()), "ordinal": ordinal}


def test_garbled_font_check():
    good_pages = [CLEAN_PARA] * 3
    flags = ing.validate_chapter("1", "Title", [("heading", "1.1 X"), ("heading", "1.2 Y"), ("heading", "1.3 Z")],
                                  [section("1.1 X", CLEAN_PARA, 0)], good_pages)
    check("clean English text has no garbled-font flag", not any("garbled" in f for f in flags), flags)

    pua_heavy = " " + CLEAN_PARA
    flags = ing.validate_chapter("1", "Title", [("heading", "1.1 X"), ("heading", "1.2 Y"), ("heading", "1.3 Z")],
                                  [section("1.1 X", pua_heavy, 0)], [pua_heavy])
    check("private-use-area heavy text is flagged as garbled", any("garbled" in f and "private-use" in f for f in flags), flags)

    non_english_tokens = "xqz vwk zzt plq mnb rst fgh jkl " * 20
    flags = ing.validate_chapter("1", "Title", [("heading", "1.1 X"), ("heading", "1.2 Y"), ("heading", "1.3 Z")],
                                  [section("1.1 X", non_english_tokens, 0)], [non_english_tokens])
    check("text with almost no common English words is flagged as garbled",
          any("garbled" in f and "common English words" in f for f in flags), flags)


def test_validate_chapter_gates():
    pages = [CLEAN_PARA]

    flags = ing.validate_chapter(None, None, [("heading", "1.1 X")], [section("1.1 X", CLEAN_PARA, 0)], pages)
    check("missing chapter title is flagged", any("chapter title not found" in f for f in flags), flags)

    flags = ing.validate_chapter("1", "Title", [("heading", "1.1 X")], [section("1.1 X", CLEAN_PARA, 0)], pages)
    check("fewer than 3 headings is flagged", any("heading(s) found" in f for f in flags), flags)

    blocks3 = [("heading", "1.1 X"), ("heading", "1.2 Y"), ("heading", "1.3 Z")]
    flags = ing.validate_chapter("1", "Title", blocks3,
                                  [section("1.1 X", CLEAN_PARA, 0), section("1.2 Y", "", 1)], pages)
    check("an empty section is flagged", any("empty section" in f for f in flags), flags)

    out_of_order = [("heading", "1.2 Y"), ("heading", "1.1 X"), ("heading", "1.3 Z")]
    flags = ing.validate_chapter("1", "Title", out_of_order,
                                  [section("1.2 Y", CLEAN_PARA, 0), section("1.1 X", CLEAN_PARA, 1)], pages)
    check("out-of-order numbered headings are flagged", any("section order" in f for f in flags), flags)

    flags = ing.validate_chapter("1", "Title", blocks3,
                                  [section("1.1 X", CLEAN_PARA, 0), section("1.2 Y", CLEAN_PARA, 1)], pages)
    check("a clean chapter has no flags", flags == [], flags)

    sparse_page = ["too short"]
    flags = ing.validate_chapter("1", "Title", blocks3,
                                  [section("1.1 X", CLEAN_PARA, 0), section("1.2 Y", CLEAN_PARA, 1)], sparse_page)
    check("a near-blank page is flagged", any("possibly blank/scanned-image page" in f for f in flags), flags)


def test_skip_logic():
    key = ("11", "Biology", "The Living World")
    check("not skipped when nothing exists and no checkpoint", not ing.should_skip(key, set(), {}, force=False))
    check("skipped when already in ncert_sections", ing.should_skip(key, {key}, {}, force=False))
    check("--force overrides an existing-in-DB skip", not ing.should_skip(key, {key}, {}, force=True))

    checkpoint = {"11|Biology|The Living World": {"status": "loaded"}}
    check("skipped when the checkpoint says loaded", ing.should_skip(key, set(), checkpoint, force=False))
    check("--force overrides a checkpoint skip", not ing.should_skip(key, set(), checkpoint, force=True))

    other_key = ("12", "Chemistry", "Some Other Chapter")
    check("a different chapter is not skipped", not ing.should_skip(other_key, {key}, checkpoint, force=False))


def test_merge_wrapped_headings():
    lines = [
        "The previous section ended here.",
        "1.2.2",
        "Genus",
        "Genus comprises a group of related species.",
    ]
    merged = ing.merge_wrapped_headings(lines)
    check("a bare heading number is merged with its title on the next line",
          merged == [
              "The previous section ended here.",
              "1.2.2 Genus",
              "Genus comprises a group of related species.",
          ], merged)

    sentence_after_number = [
        "See note 1.2.2",
        "1.2.2",
        "this sentence happens to start with a lowercase word and keeps going on and on.",
    ]
    merged = ing.merge_wrapped_headings(sentence_after_number)
    check("a long lowercase-starting continuation line is not merged as a heading title",
          merged[1] == "1.2.2", merged)

    no_bare_numbers = ["Just a normal paragraph.", "With two lines."]
    check("text with no bare heading numbers is unchanged",
          ing.merge_wrapped_headings(no_bare_numbers) == no_bare_numbers)


def test_recover_chapter_title():
    lines = [
        "...range and",
        "THE LIVING WORLD",
        "CHAPTER  1",
        "1.1 Diversity in the Living World",
    ]
    check("an all-caps title directly above 'CHAPTER n' is recovered",
          ing.recover_chapter_title(lines, "1") == "The Living World")

    check("no recovery when the chapter number isn't found", ing.recover_chapter_title(lines, "9") is None)

    lines_no_caps_above = [
        "some ordinary paragraph line.",
        "CHAPTER  1",
        "1.1 Diversity in the Living World",
    ]
    check("no recovery when the line above isn't a standalone caps phrase",
          ing.recover_chapter_title(lines_no_caps_above, "1") is None)


def test_resolve_class_subject():
    from pathlib import Path
    root = Path("ncert_pdfs")
    pdf = root / "Class11_Biology" / "01-The-Living-World.pdf"
    check("class/subject resolved from the folder name",
          ing.resolve_class_subject(pdf, root, None, None) == ("11", "Biology"))
    check("explicit overrides win over the folder name",
          ing.resolve_class_subject(pdf, root, "12", "Chemistry") == ("12", "Chemistry"))

    flat_pdf = root / "some_chapter.pdf"
    check("no folder match and no override leaves both unresolved",
          ing.resolve_class_subject(flat_pdf, root, None, None) == (None, None))
    check("a single override combines with the folder name for the other field",
          ing.resolve_class_subject(pdf, root, "12", None) == ("12", "Biology"))


if __name__ == "__main__":
    test_garbled_font_check()
    test_validate_chapter_gates()
    test_skip_logic()
    test_merge_wrapped_headings()
    test_recover_chapter_title()
    test_resolve_class_subject()
    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
