"""Unit tests for ncert_rechunk_v2.py's chunker and search_ncert_v2's response-shape parity.

Run with: python test_ncert_rechunk_v2.py
Plain pass/fail harness like test_cost_fixes.py. Fully offline -- synthetic sentence streams for
the chunker (no PDF parsing), mocked RPC responses for the parity check (no network/DB).

1. chunk_sentences: sentence boundaries respected (every chunk is a clean concatenation of whole
   sentences from the input), size stays in/near the 150-250 word target, consecutive chunks
   overlap by about OVERLAP_WORDS, and no chunk is ever empty.
2. _decode_stroke_run: the 5x-duplicated-glyph artifact decodes back to the original text.
3. The runaway-sentence safety valve: a punctuation-less run way past target_max is force-split
   at word boundaries without dropping or duplicating any words; normal prose is untouched.
4. search_ncert_v2 (server.py): same return shape as search_ncert for a mocked hybrid-RPC
   response, and respects NCERT_V2_ENABLED.
"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ncert_rechunk_v2 as rechunk

_passed = []
_failed = []


def check(label, condition, detail=""):
    if condition:
        _passed.append(label)
        print(f"  PASS: {label}")
    else:
        _failed.append(label)
        print(f"  FAIL: {label}" + (f" -- {detail}" if detail else ""))


def _words(s):
    return len(s.split())


def _make_section(heading, n_sentences, page=1, words_per_sentence=12):
    """A heading plus n_sentences short, distinguishable sentences, each ~words_per_sentence
    words -- e.g. heading='A', sentence 0 -> 'A sentence 0 word word word ...'."""
    out = []
    for i in range(n_sentences):
        filler = " ".join(["word"] * (words_per_sentence - 3))
        out.append((f"{heading} sentence {i} {filler}.", heading, page))
    return out


def test_sentence_boundaries_respected():
    print("\n1. chunk boundaries never cut a sentence")
    stream = _make_section("Intro", 40, page=1)
    chunks = rechunk.chunk_sentences(stream)
    all_sentences = {s for s, _, _ in stream}
    for c in chunks:
        # Splitting the chunk's content back on ". " (plus the trailing period) must reproduce
        # only whole sentences that were actually in the input.
        pieces = [p.strip() + "." for p in c["content"].split(". ")]
        pieces[-1] = pieces[-1].rstrip(".") + "."
        for p in pieces:
            check(f"chunk piece is a real whole sentence: {p[:40]!r}", p in all_sentences, p)


def test_chunk_size_range():
    print("\n2. chunk sizes stay near the 150-250 word target")
    stream = _make_section("Body", 60, page=1, words_per_sentence=12)  # 720 words total
    chunks = rechunk.chunk_sentences(stream)
    check("more than one chunk produced", len(chunks) > 1, len(chunks))
    for c in chunks[:-1]:  # the last chunk may legitimately be a short leftover
        check(f"chunk word_count in range (got {c['word_count']})",
              rechunk.TARGET_MIN_WORDS <= c["word_count"] <= rechunk.TARGET_MAX_WORDS + 15,
              c["word_count"])


def test_no_empty_chunks():
    print("\n3. no empty chunks, even for tiny input")
    chunks = rechunk.chunk_sentences([("Just one short sentence.", "H", 1)])
    check("exactly one chunk", len(chunks) == 1, chunks)
    check("chunk is non-empty", chunks and chunks[0]["content"].strip() != "", chunks)

    chunks = rechunk.chunk_sentences([])
    check("empty input produces no chunks (not a crash, not a [''] chunk)", chunks == [], chunks)


def test_overlap_between_consecutive_chunks():
    print("\n4. consecutive chunks overlap by about OVERLAP_WORDS")
    stream = _make_section("Body", 60, page=1, words_per_sentence=12)
    chunks = rechunk.chunk_sentences(stream)
    check("at least two chunks to compare", len(chunks) >= 2, len(chunks))
    first_tail = chunks[0]["content"].split(". ")[-3:]  # last few sentences of chunk 1
    second_head = chunks[1]["content"]
    overlap_found = any(piece.strip().rstrip(".") and piece.strip().rstrip(".") in second_head
                         for piece in first_tail)
    check("chunk 2 starts with trailing content from chunk 1 (real overlap)", overlap_found,
          (chunks[0]["content"][-80:], chunks[1]["content"][:80]))


def test_heading_changes_respected_once_big_enough():
    print("\n5. a heading change only flushes once the current chunk already hit target_min")
    stream = _make_section("A", 3, page=1, words_per_sentence=12) + _make_section("B", 20, page=2, words_per_sentence=12)
    chunks = rechunk.chunk_sentences(stream)
    # Section A alone (3 sentences, ~27 words) is well under target_min, so it must merge
    # forward into B's sentences rather than becoming its own undersized chunk.
    check("first chunk contains sentences from both A and B (A was too small to stand alone)",
          "A sentence 0" in chunks[0]["content"] and "B sentence 0" in chunks[0]["content"],
          chunks[0]["content"][:120])


def test_decode_stroke_run():
    print("\n6. _decode_stroke_run reverses the 5x-duplicated-glyph artifact")
    cases = {
        "GGGGGooooolllllgggggiiiii aaaaappppppppppaaaaarrrrraaaaatttttuuuuusssss": "Golgi apparatus",
        "88888.....55555.....33333.....11111 TTTTThhhhheeeee EEEEEnnnnndddddoooooppppplllllaaaaasssssmmmmmiiiiiccccc RRRRReeeeetttttiiiiicccccuuuuullllluuuuummmmm (((((EEEEERRRRR)))))":
            "8.5.3.1 The Endoplasmic Reticulum (ER)",
    }
    for raw, expected in cases.items():
        decoded = rechunk._decode_stroke_run(raw)
        check(f"decodes to {expected!r}", decoded == expected, decoded)


def test_runaway_sentence_safety_valve():
    print("\n7. a punctuation-less run way over target is force-split, not left as one giant chunk")
    normal = "This is a perfectly normal sentence with plenty of words in it for a real test."
    check("normal prose is untouched", rechunk.split_sentences(normal) == [normal], normal)

    runaway = " ".join(f"word{i}" for i in range(600))  # no punctuation at all
    pieces = rechunk.split_sentences(runaway)
    check("runaway text is split into multiple pieces", len(pieces) > 1, len(pieces))
    for p in pieces:
        check(f"each piece stays near target_max (got {len(p.split())})",
              len(p.split()) <= rechunk.TARGET_MAX_WORDS, p[:40])
    check("no words lost in the split", " ".join(pieces) == runaway, pieces)

    cid_text = "formation ((cid:28).(cid:23)(cid:21)(cid:12) which you will study"
    cleaned = rechunk._strip_cid_artifacts(cid_text)
    check("(cid:NN) glyph artifacts are stripped", "(cid:" not in cleaned, cleaned)


def test_progress_checkpoint_resume():
    print("\n8. progress checkpoint: round-trips, tolerates a truncated last line, resume skips done PDFs")
    import tempfile
    real_progress_file, real_samples_file = rechunk.PROGRESS_FILE, rechunk.SAMPLES_FILE
    tmp_dir = tempfile.mkdtemp()
    rechunk.PROGRESS_FILE = os.path.join(tmp_dir, "progress.jsonl")
    rechunk.SAMPLES_FILE = os.path.join(tmp_dir, "samples.json")
    try:
        check("no progress file yet -> empty dict", rechunk._load_progress() == {}, None)

        rec1 = {"pdf": "a.pdf", "subject": "Biology", "class": 11, "chapter_name": "X",
                "chunk_count": 2, "word_counts": [200, 50], "warnings": {}}
        rec2 = {"pdf": "b.pdf", "subject": "Physics", "class": 11, "chapter_name": "Y",
                "chunk_count": 1, "word_counts": [180], "warnings": {"cid_artifacts_stripped": 3}}
        rechunk._append_progress(rec1)
        rechunk._append_progress(rec2)
        loaded = rechunk._load_progress()
        check("both records round-trip", loaded == {"a.pdf": rec1, "b.pdf": rec2}, loaded)

        # Simulate a hard kill mid-write: a truncated, unparseable last line.
        with open(rechunk.PROGRESS_FILE, "a", encoding="utf-8") as f:
            f.write('{"pdf": "c.pdf", "chunk_count": 1, "word_cou')
        loaded_after_truncation = rechunk._load_progress()
        check("truncated last line is skipped, not fatal, earlier records still intact",
              loaded_after_truncation == {"a.pdf": rec1, "b.pdf": rec2}, loaded_after_truncation)

        samples = {"golgi apparatus": {"content": "..."}}
        rechunk._save_samples(samples)
        check("samples round-trip through an atomic write", rechunk._load_samples() == samples,
              rechunk._load_samples())
    finally:
        rechunk.PROGRESS_FILE, rechunk.SAMPLES_FILE = real_progress_file, real_samples_file


def test_chapter_groups_cover_known_counts():
    print("\n9. filename->chapter mapping covers the expected chapter counts")
    groups = rechunk._chapter_groups()
    counts = {}
    for subject, class_num, _, _ in groups:
        counts[(subject, class_num)] = counts.get((subject, class_num), 0) + 1
    expected = {
        ("Biology", 11): 19, ("Biology", 12): 13,
        ("Chemistry", 11): 9, ("Chemistry", 12): 10,
        ("Physics", 11): 14, ("Physics", 12): 14,
    }
    check("chapter counts match NEET_SYLLABUS lengths (Physics 11 excepted, no source PDF for "
          "'Mathematical Tools')", counts == expected, counts)


def test_search_ncert_v2_shape_parity():
    print("\n10. search_ncert_v2 response shape matches search_ncert, respects NCERT_V2_ENABLED")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import server

    row = {
        "id": "x1", "subject": "Biology", "class": 11, "chapter_name": "Cell: The Unit of Life",
        "chapter_name_en": None, "language": "en", "heading": "Golgi apparatus", "page": 96,
        "chunk_index": 3, "content": "The golgi apparatus performs the function of packaging.",
        "similarity": 0.81, "rrf_score": 0.03,
    }

    class FakeResp:
        def __init__(self, status, body):
            self.status_code = status
            self._body = body

        def json(self):
            return self._body

    class FakeClient:
        async def post(self, url, headers=None, json=None):
            assert url.endswith("/rpc/match_ncert_v2_hybrid"), url
            return FakeResp(200, [row])

    async def fake_embed(text, user_id=""):
        return [0.0] * 1536

    # NCERT_V2_ENABLED is read once at module import (same pattern as HINDI_UPLOADS_ENABLED etc.
    # elsewhere in server.py) -- flipping it for a test means patching the already-imported
    # module attribute directly, not os.environ after the fact.
    saved = (server.async_client, server.get_embedding, server.NCERT_V2_ENABLED)
    server.async_client = FakeClient()
    server.get_embedding = fake_embed
    try:
        server.NCERT_V2_ENABLED = True
        import asyncio
        results = asyncio.run(server.search_ncert_v2("function of golgi apparatus"))
        check("returns a list", isinstance(results, list), type(results))
        check("exactly one row", len(results) == 1, results)
        r = results[0]
        expected_keys = {"id", "subject", "class", "chapter_name", "chapter_name_en", "language",
                          "content", "similarity"}
        check("has the same keys search_ncert's callers rely on", expected_keys <= set(r.keys()),
              set(r.keys()))
        check("content round-trips", r["content"] == row["content"], r)

        server.NCERT_V2_ENABLED = False
        results_off = asyncio.run(server.search_ncert_v2("function of golgi apparatus"))
        check("disabled flag returns [] without calling the RPC", results_off == [], results_off)
    finally:
        server.async_client, server.get_embedding, server.NCERT_V2_ENABLED = saved


if __name__ == "__main__":
    test_sentence_boundaries_respected()
    test_chunk_size_range()
    test_no_empty_chunks()
    test_overlap_between_consecutive_chunks()
    test_heading_changes_respected_once_big_enough()
    test_decode_stroke_run()
    test_runaway_sentence_safety_valve()
    test_progress_checkpoint_resume()
    test_chapter_groups_cover_known_counts()
    test_search_ncert_v2_shape_parity()
    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
