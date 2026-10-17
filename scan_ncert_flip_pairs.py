"""List Hindi NCERT chunks with flip-prone word pairs (permeable/impermeable, ascending/descending,
...) so a person can check them against the English book. Report only: reads, never writes, no AI.

Run with: python scan_ncert_flip_pairs.py [--max 20]

The Hindi Class 11 Biology loop-of-Henle chunk had "permeable" and "impermeable" swapped (now in
server.py's NCERT_ERRATA). A swap like that keeps the sentence fluent, so it only shows up when a
sentence names both sides of a pair. Ranked: sentences naming both words of one pair first, the
permeability pair first among those, then by how many pair words a chunk has. Chunks an errata
entry already fixes are skipped.
"""
import argparse
import io
import os
import re
import sys

import httpx
from dotenv import load_dotenv

# (pair name, word A, word B). Whole Devanagari words, matched by prefix so inflections count
# ("पारगम्यता"); the longest matching term wins, so "अपारगम्य" is never read as "पारगम्य" and
# "अपचयन" (reduction) never as "अपचय" (catabolism).
FLIP_PAIRS = [
    ("permeable/impermeable", "पारगम्य", "अपारगम्य"),
    ("ascending/descending", "आरोही", "अवरोही"),
    ("afferent/efferent", "अभिवाही", "अपवाही"),
    ("active/passive", "सक्रिय", "निष्क्रिय"),
    ("oxidation/reduction", "ऑक्सीकरण", "अपचयन"),
    ("anabolism/catabolism", "उपचय", "अपचय"),
    ("positive/negative", "धनात्मक", "ऋणात्मक"),
    ("acidic/basic", "अम्लीय", "क्षारीय"),
    ("contraction/relaxation", "संकुचन", "शिथिलन"),
    ("dominant/recessive", "प्रभावी", "अप्रभावी"),
]
_TERMS = sorted(((w, name) for name, a, b in FLIP_PAIRS for w in (a, b)), key=lambda t: -len(t[0]))
_WORD = re.compile(r"[ऀ-ॿ]+")
_SENTENCE = re.compile(r"[^।?!\n]+[।?!]?")


def flip_terms(sentence: str) -> list:
    """(pair name, term) for every flip-prone word in the sentence, in order."""
    found = []
    for m in _WORD.finditer(sentence):
        for term, name in _TERMS:
            if m.group().startswith(term):
                found.append((name, term))
                break
    return found


def scan_chunk(content: str) -> dict:
    """{"both": [(pair, sentence)], "terms": count} -- sentences naming both words of a pair."""
    both, terms = [], 0
    for m in _SENTENCE.finditer(content or ""):
        sentence = m.group().strip()
        found = flip_terms(sentence)
        terms += len(found)
        for name, a, b in FLIP_PAIRS:
            words = {t for n, t in found if n == name}
            if a in words and b in words:
                both.append((name, sentence))
    return {"both": both, "terms": terms}


def rank(rows: list, errata: list, limit: int) -> list:
    hits = []
    for r in rows:
        content = r.get("content") or ""
        if any(lang == "hi" and wrong in content for lang, wrong, _ in errata):
            continue
        result = scan_chunk(content)
        if not result["terms"]:
            continue
        permeability = any(name == FLIP_PAIRS[0][0] for name, _ in result["both"])
        hits.append(((not result["both"], not permeability, -len(result["both"]), -result["terms"]), r, result))
    hits.sort(key=lambda h: h[0])
    return [(r, result) for _, r, result in hits[:limit]]


def fetch_hindi_chunks() -> list:
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    url, key = os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"]
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    rows, page = [], 1000
    with httpx.Client(timeout=60) as client:
        while True:
            resp = client.get(f"{url}/rest/v1/ncert_content", headers=headers, params={
                "select": "id,subject,class,chapter_name_en,content", "language": "eq.hi",
                "order": "id", "offset": len(rows), "limit": page})
            resp.raise_for_status()
            batch = resp.json()
            rows += batch
            if len(batch) < page:
                return rows


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=20)
    args = parser.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from server import NCERT_ERRATA

    rows = fetch_hindi_chunks()
    hits = rank(rows, NCERT_ERRATA, args.max)
    print(f"{len(rows)} Hindi chunks scanned, showing {len(hits)}\n")
    for r, result in hits:
        print(f"{r['id']}  Class {r['class']} {r['subject']} -- {r.get('chapter_name_en')}  ({result['terms']} pair words)")
        for name, sentence in result["both"][:3]:
            print(f"    [{name}] {sentence}")
        print()
