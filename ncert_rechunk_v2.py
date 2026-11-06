"""Stage 1 of the English NCERT retrieval rework: rebuild ncert_content's English chunks from the
original source PDFs (ncert-pdfs/) into better-bounded chunks (heading/sentence-aware, 150-250
words, ~30 word overlap, never cutting mid-sentence) and load them into the new ncert_content_v2
table -- see ncert_chunks_v2.sql for the table/RPCs this writes to.

Default mode is a DRY RUN: extracts, chunks, prints stats and samples, makes no embedding calls
and writes nothing to Supabase. Pass --load to actually embed and insert (run ncert_chunks_v2.sql
in the Supabase SQL editor FIRST).

Run with:
    python ncert_rechunk_v2.py                 # dry run (default)
    python ncert_rechunk_v2.py --no-resume      # dry run, ignore any existing progress file
    python ncert_rechunk_v2.py --load           # embed + insert into ncert_content_v2

Keys are read from .env only (OPENAI_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY) -- never printed or
logged.

---- One PDF at a time, checkpointed, resumable ----
Every chapter PDF is opened, chunked and released (page objects dropped, gc.collect()) before the
next one opens -- never all 100 at once in one unbroken pass, which is what put enough sustained
memory pressure on a machine with other things open to hang it. Chapters are processed in groups
of GROUP_SIZE with a short pause between groups, printing one summary line per group rather than
one line per chapter or per page. The dry run checkpoints after every chapter to
ncert_rechunk_v2_progress.jsonl (one small JSON line per chapter: chunk count, chunk sizes, any
extraction-artifact warnings -- never the full chunk text) and to ncert_rechunk_v2_samples.json
(the keyword-sample matches found so far). If the run is interrupted for any reason, just re-run
the same command: it skips every chapter already in the progress file and continues from there
--no-resume starts over. Final stats/samples are built by reading those two files, not by holding
every chapter's chunks in memory at once.

---- Why PDFs instead of the old chunks ----
The source PDFs (ncert-pdfs/, 100 files) are still in the repo, so chunks are rebuilt from the
original text rather than by reconstituting the old 500-word chunks (which already damaged
paragraph/sentence boundaries -- reconstituting them would carry that damage forward). Each PDF
filename maps 1:1 to one NEET_SYLLABUS chapter by position -- see _chapter_groups() below; this
was verified against every filename in ncert-pdfs/ before writing this script (19/13/9/10/14/14
real chapter PDFs per subject+class, matching NEET_SYLLABUS's own list lengths exactly, except
Physics Class 11 which has 14 PDFs for a 15-entry list -- "Mathematical Tools" is a NEET-prep
addition with no NCERT source PDF, so it has no v2 rows yet).

---- The "5x stroke" heading artifact ----
NCERT renders numbered subsection headings (e.g. "8.5.3.1 The Endoplasmic Reticulum (ER)") with a
stroke/shadow effect that duplicates every glyph 5 times at a tiny offset. pdfplumber's text
extraction (correctly) pulls out all 5 overlapping copies of each character as separate
characters, so the raw text shows literal runs like "TTTTThhhhheeeee" instead of "The". This is
mechanically decodable: collapse every run of >=4 identical consecutive characters to
round(run_length / 5) copies (see _decode_stroke_run). Confirmed character-for-character against
kebo108.pdf's real "Endoplasmic Reticulum (ER)" and "Golgi apparatus" headings before relying on
it here. Top-level headings (8.1, 8.2, ...) are mostly NOT affected -- they extract as plain
capitalised text already and are matched directly by _PLAIN_HEADING_RE.

---- One pathological page (lebo101.pdf, "Sexual Reproduction in Flowering Plants") ----
One single page in the whole 100-PDF set has a 281 MB *decompressed* content stream (confirmed via
PyPDF2 -- its own embedded images are tiny, ~1275x203px, so this isn't an oversized-image problem;
it's an enormous number of drawing operators, almost certainly a very fine-detail vector
illustration). Both pdfplumber AND PyPDF2 independently balloon to multiple GB of RAM trying to
tokenize that page's operators into their object models -- this, not a leak in this script, is
what hung the machine when the original full-100-PDF dry run reached it. Fixed with a cheap
pre-check (see MAX_PAGE_CONTENT_STREAM_BYTES / _page_content_stream_size): a page's own
decompressed content-stream size is checked via PyPDF2 (fast -- it's one zlib.decompress() call,
nowhere near as expensive as what pdfminer does with the result) BEFORE pdfplumber ever touches
the page; a page over the threshold is skipped (that one physical page's body text is lost, logged
as a warning) rather than risking the same blowup again.
"""
import argparse
import gc
import itertools
import json
import os
import re
import sys
import time
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pdfplumber
import PyPDF2
from dotenv import load_dotenv

load_dotenv()

PDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ncert-pdfs")
EMBEDDING_MODEL = "text-embedding-3-small"  # same model as get_embedding() in server.py
TARGET_MIN_WORDS = 150
TARGET_MAX_WORDS = 250
OVERLAP_WORDS = 30

# Copied from NEET_SYLLABUS in server.py (kept in sync manually -- importing server.py here would
# also pull in its FastAPI app, DB clients and Sentry init just to read one dict).
NEET_SYLLABUS = {
    "Biology": {
        11: ['The Living World', 'Biological Classification', 'Plant Kingdom', 'Animal Kingdom', 'Morphology of Flowering Plants', 'Anatomy of Flowering Plants', 'Structural Organisation in Animals', 'Cell: The Unit of Life', 'Biomolecules', 'Cell Cycle and Cell Division', 'Photosynthesis in Higher Plants', 'Respiration in Plants', 'Plant Growth and Development', 'Breathing and Exchange of Gases', 'Body Fluids and Circulation', 'Excretory Products and their Elimination', 'Locomotion and Movement', 'Neural Control and Coordination', 'Chemical Coordination and Integration'],
        12: ['Sexual Reproduction in Flowering Plants', 'Human Reproduction', 'Reproductive Health', 'Principles of Inheritance and Variation', 'Molecular Basis of Inheritance', 'Evolution', 'Human Health and Disease', 'Microbes in Human Welfare', 'Biotechnology: Principles and Processes', 'Biotechnology and its Applications', 'Organisms and Populations', 'Ecosystem', 'Biodiversity and Conservation']
    },
    "Physics": {
        11: ['Units and Measurements', 'Motion in a Straight Line', 'Motion in a Plane', 'Laws of Motion', 'Work, Energy and Power', 'System of Particles and Rotational Motion', 'Gravitation', 'Mechanical Properties of Solids', 'Mechanical Properties of Fluids', 'Thermal Properties of Matter', 'Thermodynamics', 'Kinetic Theory', 'Oscillations', 'Waves', 'Mathematical Tools'],
        12: ['Electric Charges and Fields', 'Electrostatic Potential and Capacitance', 'Current Electricity', 'Moving Charges and Magnetism', 'Magnetism and Matter', 'Electromagnetic Induction', 'Alternating Current', 'Electromagnetic Waves', 'Ray Optics and Optical Instruments', 'Wave Optics', 'Dual Nature of Radiation and Matter', 'Atoms', 'Nuclei', 'Semiconductor Electronics: Materials, Devices and Simple Circuits']
    },
    "Chemistry": {
        11: ['Some Basic Concepts of Chemistry', 'Structure of Atom', 'Classification of Elements and Periodicity in Properties of Elements', 'Chemical Bonding and Molecular Structure', 'Thermodynamics', 'Equilibrium', 'Redox Reactions', 'Organic Chemistry: Some Basic Principles and Techniques', 'Hydrocarbons'],
        12: ['Solutions', 'Electrochemistry', 'Chemical Kinetics', 'The d and f Block Elements', 'Coordination Compounds', 'Haloalkanes and Haloarenes', 'Alcohols, Phenols and Ethers', 'Aldehydes, Ketones and Carboxylic Acids', 'Amines', 'Biomolecules']
    }
}

# ---------------- 1. Filename -> chapter mapping ----------------
_FILENAME_RE = re.compile(r'^(ke|le)(bo|ch|ph)(\d)(\d{2})\.pdf$', re.IGNORECASE)
_SUBJECT_BY_CODE = {"bo": "Biology", "ch": "Chemistry", "ph": "Physics"}
_CLASS_BY_PREFIX = {"ke": 11, "le": 12}


def _chapter_groups(pdf_dir: str = PDF_DIR):
    """[(subject, class, chapter_name, pdf_path), ...] -- one row per real chapter PDF, in
    syllabus order. Files that don't match a real numbered chapter (prelims "ps", answers "an",
    appendix "a1") are skipped -- they're supplementary, not chapter content."""
    by_key = defaultdict(list)
    for fname in sorted(os.listdir(pdf_dir)):
        m = _FILENAME_RE.match(fname)
        if not m:
            continue
        cls_pfx, subj_code, part, num = m.groups()
        key = (_SUBJECT_BY_CODE[subj_code], _CLASS_BY_PREFIX[cls_pfx])
        by_key[key].append((int(part), int(num), os.path.join(pdf_dir, fname)))

    groups = []
    for (subject, class_num), files in by_key.items():
        files.sort()
        chapters = NEET_SYLLABUS.get(subject, {}).get(class_num, [])
        for chapter_name, (_, _, path) in zip(chapters, files):
            groups.append((subject, class_num, chapter_name, path))
    return groups


# ---------------- 2. Page text cleanup + heading detection ----------------
_BOILERPLATE_RES = [
    re.compile(r'^\s*Reprint\s+\d{4}-\d{2}\s*$', re.IGNORECASE),
    re.compile(r'^\s*\d{1,4}\s+[A-Z][A-Z :]{2,60}$'),   # "88 BIOLOGY"
    re.compile(r'^\s*[A-Z][A-Z :]{2,60}\s+\d{1,4}\s*$'),  # "CELL: THE UNIT OF LIFE 95"
]

# A real heading line, already extracted cleanly (most top-level X.Y headings): "8.3 AN OVERVIEW
# OF CELL", "8.3 An Overview of Cell". Deliberately requires the number prefix so body sentences
# that happen to start with a capitalised word are never mistaken for a heading.
_PLAIN_HEADING_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){1,3})\s+([A-Za-z][A-Za-z0-9 ,:()'‘’\-]{2,85})$")

# The "5x stroke" artifact: a line is treated as a duplicated-heading line if it has at least 2
# separate runs of >=4 identical consecutive characters -- normal English prose essentially never
# does this, so it's a safe detector without needing font-size metadata.
_RUN_RE = re.compile(r'(.)\1{3,}')


def _decode_stroke_run(line: str) -> str:
    """Collapses the 5x-duplicated-glyph artifact: every run of >=4 identical characters becomes
    round(run_length / 5) copies of that character (minimum 1); shorter runs (real single
    separators, e.g. the space between a section number and its title, which is NOT duplicated)
    are left untouched."""
    out = []
    for ch, run in itertools.groupby(line):
        run_len = len(list(run))
        if run_len >= 4:
            out.append(ch * max(1, round(run_len / 5)))
        else:
            out.append(ch * run_len)
    return "".join(out)


def _classify_line(line: str):
    """Returns ("heading", number, text), ("text", None, line), or None (dropped boilerplate /
    an undecodable stroke-artifact line -- dropped rather than kept as body text, since leftover
    duplicated fragments would otherwise pollute the embedding with junk tokens)."""
    stripped = line.strip()
    if not stripped:
        return None
    if any(rx.match(stripped) for rx in _BOILERPLATE_RES):
        return None
    if len(_RUN_RE.findall(stripped)) >= 2:
        decoded = _decode_stroke_run(stripped)
        m = _PLAIN_HEADING_RE.match(decoded)
        if m:
            return ("heading", m.group(1), m.group(2).strip())
        return None  # undecodable stroke artifact -- drop, don't let it leak into body text
    m = _PLAIN_HEADING_RE.match(stripped)
    if m:
        text = m.group(2).strip()
        if text.isupper():
            text = text.title()
        return ("heading", m.group(1), text)
    return ("text", None, stripped)


# Custom/symbol fonts used for equation numbers and chemical-structure labels have no Unicode
# mapping, so pdfplumber emits the raw glyph id instead, e.g. "(cid:28)" for one digit of an
# equation number like "(9.38)" -- confirmed on kech203.pdf's Markovnikov section. Pure noise:
# stripped before any other processing so it never pollutes a heading or an embedding.
_CID_RE = re.compile(r'\(cid:\d+\)')


def _strip_cid_artifacts(text: str, warnings: dict = None) -> str:
    if warnings is not None:
        n = len(_CID_RE.findall(text))
        if n:
            warnings["cid_artifacts_stripped"] = warnings.get("cid_artifacts_stripped", 0) + n
    return re.sub(r'\s{2,}', ' ', _CID_RE.sub('', text))


# A page whose own decompressed content stream is bigger than this is skipped entirely (text
# lost, logged as a warning) rather than handed to pdfplumber. Normal NCERT pages measured in the
# tens-to-low-hundreds of KB; lebo101.pdf's page 14 measured 281 MB -- this threshold sits
# generously above real pages (10-30x the largest normal ones seen) while safely excluding that
# one. See the module docstring's "One pathological page" note.
MAX_PAGE_CONTENT_STREAM_BYTES = 3_000_000


def _page_content_stream_size(pypdf_page) -> int:
    """Decompressed size of one page's own content stream, in bytes. Cheap relative to what
    pdfminer/pdfplumber would do next (tokenize every operator into Python objects) -- this is a
    single zlib.decompress() bounded by the stream's own size. Returns 0 on any failure (no
    content stream, unreadable, etc.) -- a page that can't even be sized this way is treated as
    safe-sized and left to the normal extraction path, which will then either succeed or fail on
    its own terms."""
    try:
        contents = pypdf_page.get_contents()
        return len(contents.get_data()) if contents is not None else 0
    except Exception:
        return 0


def _extract_tagged_lines(pdf_path: str, warnings: dict = None):
    """[(page_num, kind, text), ...] for a whole chapter PDF -- kind is "heading" or "text",
    boilerplate (footers, running headers) and undecodable artifacts already dropped.

    Processes the PDF one page at a time and explicitly drops each page object as soon as its
    text is extracted (`del page`), rather than holding pdfplumber.open()'s whole page list alive
    for the duration -- a diagram-heavy page can hold a large internal character/object cache, and
    with 100 PDFs processed back to back that adds up to the kind of memory pressure that hung the
    machine when the old build_all_chunks() ran every PDF in one unbroken pass (see the module
    docstring and run_chapters() below). Each page is also size-checked (see
    MAX_PAGE_CONTENT_STREAM_BYTES) before pdfplumber touches it, for the one page found so far
    whose content stream alone is large enough to blow past that same memory budget by itself."""
    tagged = []
    pypdf_reader = PyPDF2.PdfReader(pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            size = _page_content_stream_size(pypdf_reader.pages[page_num - 1])
            if size > MAX_PAGE_CONTENT_STREAM_BYTES:
                if warnings is not None:
                    warnings["oversized_page_skipped"] = warnings.get("oversized_page_skipped", 0) + 1
                page.flush_cache()
                del page
                continue
            raw = _strip_cid_artifacts(page.extract_text() or "", warnings)
            for line in raw.split("\n"):
                classified = _classify_line(line)
                if classified is None:
                    continue
                kind = classified[0]
                text = classified[2] if kind == "heading" else classified[2]
                tagged.append((page_num, kind, text))
            page.flush_cache()
            del page
    del pypdf_reader
    return tagged


# ---------------- 3. Sentence splitting ----------------
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"‘“(])')

# A diagram-heavy page (reaction schemes, structure labels interleaved with body text -- see
# kech203.pdf's Markovnikov section) can produce a long run of text with no sentence-ending
# punctuation pdfplumber's reading order preserves correctly. Rather than let one such run become
# one giant "sentence" and blow the chunk target way past target_max, anything longer than this is
# force-split at word boundaries -- a last-resort safety valve for unparseable extraction
# artifacts, never triggered by normal prose (no real NCERT sentence runs this long).
_RUNAWAY_SENTENCE_WORDS = 2 * TARGET_MAX_WORDS


def _split_runaway(sentence: str, max_words: int = TARGET_MAX_WORDS):
    words = sentence.split()
    if len(words) <= _RUNAWAY_SENTENCE_WORDS:
        return [sentence]
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def split_sentences(text: str, warnings: dict = None):
    """Best-effort sentence splitter -- doesn't special-case abbreviations, but NCERT prose is
    plain enough that this holds up well in practice. Never returns an empty string."""
    text = text.strip()
    if not text:
        return []
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    out = []
    for s in sentences:
        pieces = _split_runaway(s)
        if len(pieces) > 1 and warnings is not None:
            warnings["runaway_sentences_split"] = warnings.get("runaway_sentences_split", 0) + 1
        out.extend(pieces)
    return out


def _build_sentence_stream(tagged_lines, warnings: dict = None):
    """[(sentence, heading_or_None, page), ...] -- heading is whichever heading is currently
    "active" (the most recent heading line seen), None for text before the chapter's first
    heading. Consecutive text lines on the same page are joined into one blob before sentence
    splitting, since pdfplumber gives one line per visual line, not per paragraph."""
    stream = []
    current_heading = None
    buf_lines = []
    buf_page = None

    def flush_buf():
        if not buf_lines:
            return
        blob = " ".join(buf_lines)
        for sent in split_sentences(blob, warnings):
            stream.append((sent, current_heading, buf_page))
        buf_lines.clear()

    for page_num, kind, text in tagged_lines:
        if kind == "heading":
            flush_buf()
            current_heading = text
            continue
        if buf_page is None:
            buf_page = page_num
        buf_lines.append(text)
    flush_buf()
    return stream


# ---------------- 4. Chunker (pure function -- the testable core) ----------------
def chunk_sentences(stream, target_min: int = TARGET_MIN_WORDS, target_max: int = TARGET_MAX_WORDS,
                     overlap_words: int = OVERLAP_WORDS):
    """stream: [(sentence, heading_or_None, page), ...] in reading order.
    Returns [{"heading":, "page":, "content":, "word_count":}, ...]. Never returns an empty chunk.
    Flushes a chunk when it's already >= target_min words AND either the next sentence would push
    it over target_max, or a new heading starts -- so a chunk only ever grows past target_max for
    a single sentence that's longer than the whole target range (never splits mid-sentence).
    Small headed sections merge into the next one rather than forcing an undersized chunk."""
    def word_count(s):
        return len(s.split())

    chunks = []
    buf = []  # [(sentence, heading, page), ...]
    buf_words = 0

    def finalize():
        content = " ".join(s for s, _, _ in buf)
        heading = next((h for _, h, _ in buf if h is not None), None)
        page = buf[0][2]
        chunks.append({"heading": heading, "page": page, "content": content, "word_count": buf_words})

    def carry_overlap():
        nonlocal buf_words
        tail, tail_words = [], 0
        for item in reversed(buf):
            if tail_words >= overlap_words:
                break
            tail.insert(0, item)
            tail_words += word_count(item[0])
        buf_words = tail_words
        return tail

    prev_heading = None
    for sentence, heading, page in stream:
        w = word_count(sentence)
        heading_changed = buf and heading is not None and heading != prev_heading
        if buf and buf_words >= target_min and (buf_words + w > target_max or heading_changed):
            finalize()
            buf = carry_overlap()
        buf.append((sentence, heading, page))
        buf_words += w
        prev_heading = heading if heading is not None else prev_heading

    if buf:
        finalize()
    return chunks


# ---------------- 5. Per-chapter processing (one PDF at a time, memory-bounded) ----------------
def build_chapter_chunks(subject, class_num, chapter_name, pdf_path, warnings: dict = None):
    tagged = _extract_tagged_lines(pdf_path, warnings)
    stream = _build_sentence_stream(tagged, warnings)
    chunks = chunk_sentences(stream)
    for idx, c in enumerate(chunks):
        c.update(subject=subject, class_=class_num, chapter_name=chapter_name, chunk_index=idx)
    return chunks


# Real NCERT PDFs use a typographic right-single-quote (U+2019), not an ASCII apostrophe --
# confirmed via "newton’s second law of motion" / "mendel’s laws of inheritance" in the
# actual extracted text. Matching with the ASCII apostrophe silently finds nothing.
SAMPLE_KEYWORDS = ["golgi apparatus", "markovnikov", "newton’s second law", "mendel’s law", "photosynthesis"]

# The laptop hung when build_all_chunks() used to open all 100 PDFs back to back in one unbroken
# pass -- pdfplumber's per-page caches (worse on diagram-heavy pages, see the "5x stroke" and
# "(cid:NN)" notes above) accumulate enough across 100 files to put real memory pressure on a
# machine that's doing other things too. Fixed by: one PDF fully processed and released (page
# objects dropped, gc.collect()) before the next one opens; a short pause every GROUP_SIZE PDFs to
# let the OS actually reclaim what was freed; and a JSONL checkpoint after every PDF so a slow or
# interrupted run resumes instead of starting over.
GROUP_SIZE = 10
GROUP_PAUSE_SECONDS = 3

PROGRESS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ncert_rechunk_v2_progress.jsonl")
SAMPLES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ncert_rechunk_v2_samples.json")


def _load_progress() -> dict:
    """pdf basename -> progress record, read from PROGRESS_FILE. Tolerant of one truncated last
    line -- a kill/hang mid-write leaves at most one partial JSON line, which is just skipped
    rather than treated as a fatal corruption of the whole file."""
    progress = {}
    if not os.path.exists(PROGRESS_FILE):
        return progress
    with open(PROGRESS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            progress[record["pdf"]] = record
    return progress


def _append_progress(record: dict):
    """fsync's immediately -- this is the checkpoint a resume relies on, so it must actually be on
    disk before this returns, not sitting in an OS write buffer a hang could lose."""
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _load_samples() -> dict:
    if not os.path.exists(SAMPLES_FILE):
        return {}
    try:
        with open(SAMPLES_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_samples(samples: dict):
    tmp_path = SAMPLES_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, SAMPLES_FILE)  # atomic -- never leaves a half-written samples file behind


def _process_one_chapter(subject, class_num, chapter_name, pdf_path, samples: dict) -> dict:
    """Extracts + chunks exactly one PDF. Records any new SAMPLE_KEYWORDS match into `samples`
    (mutated in place) and returns a small JSON-serialisable summary -- chunk content otherwise
    never leaves this function, and nothing it builds is kept alive past the return (the caller's
    gc.collect() then reclaims pdfplumber's internal caches for this file)."""
    warnings = {}
    chunks = build_chapter_chunks(subject, class_num, chapter_name, pdf_path, warnings)

    sizes = [c["word_count"] for c in chunks]
    for kw in SAMPLE_KEYWORDS:
        if kw in samples:
            continue
        match = next((c for c in chunks if kw in c["content"].lower()), None)
        if match:
            samples[kw] = {
                "subject": match["subject"], "class": match["class_"], "chapter_name": match["chapter_name"],
                "heading": match["heading"], "page": match["page"], "word_count": match["word_count"],
                "content": match["content"][:400],
            }

    record = {
        "pdf": os.path.basename(pdf_path),
        "subject": subject,
        "class": class_num,
        "chapter_name": chapter_name,
        "chunk_count": len(chunks),
        "word_counts": sizes,
        "warnings": warnings,
    }
    del chunks
    return record


# ---------------- 6. Dry-run reporting (built from the progress file, not from in-memory chunks) ----------------
def _print_stats_from_progress(progress: dict):
    by_subject = defaultdict(list)
    oversized = tiny = 0
    total_warnings = defaultdict(int)
    for record in progress.values():
        by_subject[record["subject"]].extend(record["word_counts"])
        sizes = record["word_counts"]
        oversized += sum(1 for w in sizes if w > TARGET_MAX_WORDS + 50)
        # Each chapter's own trailing chunk is allowed to be short (unavoidable leftover) --
        # only count a short chunk as "tiny" when it's NOT the last one in its chapter.
        tiny += sum(1 for w in sizes[:-1] if w < TARGET_MIN_WORDS)
        for k, v in record.get("warnings", {}).items():
            total_warnings[k] += v

    all_sizes = [w for sizes in by_subject.values() for w in sizes]
    print(f"\nChapters processed: {len(progress)}")
    print(f"Total chunks: {len(all_sizes)}")
    print("\nPer subject:")
    for subject, sizes in sorted(by_subject.items()):
        if not sizes:
            continue
        print(f"  {subject:10s} chunks={len(sizes):4d}  avg={sum(sizes)/len(sizes):6.1f}  "
              f"min={min(sizes):4d}  max={max(sizes):4d}")
    if all_sizes:
        print(f"\nOverall: avg={sum(all_sizes)/len(all_sizes):.1f}  min={min(all_sizes)}  max={max(all_sizes)}")
    print(f"\nOversized chunks (>{TARGET_MAX_WORDS + 50} words): {oversized}")
    print(f"Tiny chunks (<{TARGET_MIN_WORDS} words, excluding each chapter's own trailing chunk): {tiny}")
    if total_warnings:
        print("\nExtraction-artifact warnings:")
        for k, v in sorted(total_warnings.items()):
            print(f"  {k}: {v}")
    else:
        print("\nNo extraction-artifact warnings.")


def _print_samples(samples: dict):
    print("\nSample chunks:")
    for kw in SAMPLE_KEYWORDS:
        print(f"\n--- keyword: {kw!r} ---")
        match = samples.get(kw)
        if not match:
            print("  NOT FOUND (so far -- rerun once the remaining chapters finish; it may also "
                  "genuinely be absent)")
            continue
        print(f"  {match['subject']} class {match['class']} | {match['chapter_name']} | "
              f"heading={match['heading']!r} | page={match['page']} | words={match['word_count']}")
        print(f"  {match['content']}")


# ---------------- 7. --load: embed + insert into ncert_content_v2 ----------------
def _env_or_die(name: str) -> str:
    value = os.getenv(name)
    if not value:
        print(f"Missing required env var {name} (set it in .env)", file=sys.stderr)
        sys.exit(1)
    return value


def _load_chunks(chunks, client, headers, supabase_url) -> int:
    """Embeds + upserts one chapter's chunks (merge-duplicates on the unique constraint, so
    re-running --load after an interruption just harmlessly re-embeds/re-upserts whatever chapter
    was in flight, not anything already committed in an earlier batch within this chapter).
    Returns how many rows were actually inserted/updated."""
    import requests

    BATCH_SIZE = 20
    inserted = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        rows = []
        for c in batch:
            embedding = None
            for attempt in range(3):
                try:
                    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=c["content"])
                    embedding = resp.data[0].embedding
                    break
                except Exception as e:
                    print(f"  embed retry {attempt + 1}/3: {e}")
                    time.sleep(3)
            if embedding is None:
                print(f"  SKIPPED (embedding failed 3x): {c['chapter_name']} chunk {c['chunk_index']}")
                continue
            rows.append({
                "subject": c["subject"],
                "class": c["class_"],
                "chapter_name": c["chapter_name"],
                "chapter_name_en": None,
                "language": "en",
                "heading": c["heading"],
                "page": c["page"],
                "chunk_index": c["chunk_index"],
                "content": c["content"],
                "embedding": embedding,
            })
        if not rows:
            continue
        for attempt in range(3):
            try:
                resp = requests.post(f"{supabase_url}/rest/v1/ncert_content_v2", headers=headers,
                                      json=rows, timeout=30)
                if resp.status_code < 300:
                    inserted += len(rows)
                    break
                print(f"  insert failed (status {resp.status_code}): {resp.text[:300]}")
            except Exception as e:
                print(f"  insert retry {attempt + 1}/3: {e}")
                time.sleep(3)
    return inserted


def _process_and_load_one_chapter(subject, class_num, chapter_name, pdf_path, client, headers, supabase_url):
    chunks = build_chapter_chunks(subject, class_num, chapter_name, pdf_path)
    inserted = _load_chunks(chunks, client, headers, supabase_url)
    chunk_count = len(chunks)
    del chunks
    return chunk_count, inserted


# ---------------- 8. Main driver: strictly sequential, grouped, resumable ----------------
def run_chapters(load: bool, resume: bool = True):
    groups = _chapter_groups()
    progress = _load_progress() if (not load and resume) else {}
    samples = _load_samples() if not load else {}
    todo = [g for g in groups if os.path.basename(g[3]) not in progress]

    if progress:
        print(f"Resuming: {len(progress)}/{len(groups)} chapters already done, {len(todo)} remaining.")
    else:
        print(f"Starting: {len(todo)} chapters to process, one at a time, in groups of {GROUP_SIZE}.")

    client = headers = supabase_url = None
    if load:
        import openai
        openai_key = _env_or_die("OPENAI_KEY")
        supabase_url = _env_or_die("SUPABASE_URL")
        supabase_service_key = _env_or_die("SUPABASE_SERVICE_KEY")
        client = openai.OpenAI(api_key=openai_key)
        headers = {
            "apikey": supabase_service_key,
            "Authorization": f"Bearer {supabase_service_key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates",
        }

    total_inserted = 0
    num_groups = (len(todo) + GROUP_SIZE - 1) // GROUP_SIZE or 1
    for g_idx in range(0, len(todo), GROUP_SIZE):
        batch = todo[g_idx:g_idx + GROUP_SIZE]
        batch_chunk_counts = []
        for subject, class_num, chapter_name, path in batch:
            if load:
                chunk_count, inserted = _process_and_load_one_chapter(
                    subject, class_num, chapter_name, path, client, headers, supabase_url)
                total_inserted += inserted
            else:
                record = _process_one_chapter(subject, class_num, chapter_name, path, samples)
                _append_progress(record)
                progress[record["pdf"]] = record
                chunk_count = record["chunk_count"]
            batch_chunk_counts.append(chunk_count)
            gc.collect()  # release this PDF's pdfplumber/pdfminer caches before the next one opens

        if not load:
            _save_samples(samples)
        done_so_far = len(progress) if not load else (g_idx + len(batch))
        print(f"Group {g_idx // GROUP_SIZE + 1}/{num_groups}: {len(batch)} chapters done "
              f"(chunks this group: {sum(batch_chunk_counts)}) -- {done_so_far}/{len(groups)} total so far")
        if g_idx + GROUP_SIZE < len(todo):
            time.sleep(GROUP_PAUSE_SECONDS)

    if load:
        print(f"\nDone. {total_inserted} chunks loaded into ncert_content_v2.")
    else:
        _print_stats_from_progress(progress)
        _print_samples(samples)
        print("\n(dry run -- no embedding calls, no DB writes. Re-run with --load after running "
              "ncert_chunks_v2.sql in the Supabase SQL editor; --load reuses this same "
              "one-PDF-at-a-time loop.)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", action="store_true",
                         help="Embed and insert into ncert_content_v2 (run ncert_chunks_v2.sql first). "
                              "Default is a dry run: no embedding calls, no DB writes.")
    parser.add_argument("--no-resume", action="store_true",
                         help="Ignore any existing dry-run progress file (ncert_rechunk_v2_progress.jsonl) "
                              "and start over from scratch. Only affects the dry run.")
    args = parser.parse_args()
    run_chapters(load=args.load, resume=not args.no_resume)


if __name__ == "__main__":
    main()
