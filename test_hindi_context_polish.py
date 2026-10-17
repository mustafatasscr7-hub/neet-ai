"""More Hindi textbook text per answer, Devanagari-only mnemonics, Hindi citation titles for Hinglish.

Run with: python test_hindi_context_polish.py
Plain pass/fail harness, fully offline -- reuses test_ncert_relevance.py's fakes.

1. Hindi search returns up to 6 chunks, capped at 5,000 chars; English still returns 3.
2. HINDI_LANG_CONTEXT keeps the full-Hindi rule and forbids Latin-letter memory tricks.
3. A Hinglish doubt answered in Hindi cites the Hindi chapter title (same chapter number);
   English questions, and Hinglish answered in English, keep the English title.
"""
import asyncio

import server
from test_ncert_relevance import Harness, FakeClient, drain, row, check, _passed, _failed, ANSWER_WITH_CITATION


def hi_row(id_, chars, sim=0.7):
    return {"id": id_, "subject": "Biology", "class": 11, "chapter_name": "प्रकाश-संश्लेषण", "chapter_name_en": "Photosynthesis in Higher Plants",
            "language": "hi", "content": ("प्रकाश " * chars)[:chars], "similarity": sim}


def _search_ids(rows, query):
    async def fake_embed(text):
        return [0.0] * 8

    saved = server.async_client, server.get_embedding, server._get_gemini_query_embedding
    server.async_client = FakeClient(rpc_rows=rows)
    server.get_embedding = server._get_gemini_query_embedding = fake_embed
    try:
        return [r["id"] for r in asyncio.run(server.search_ncert(query))]
    finally:
        server.async_client, server.get_embedding, server._get_gemini_query_embedding = saved


def test_hindi_chunk_count():
    print("\n1. Hindi chunk count and char cap")
    q = "प्रकाश-संश्लेषण क्या है?"
    ids = _search_ids([hi_row(f"h{i}", 700) for i in range(8)], q)
    check("700-char chunks: 6 returned (4,200 chars)", ids == [f"h{i}" for i in range(6)], ids)
    ids = _search_ids([hi_row(f"h{i}", 1200) for i in range(8)], q)
    check("1,200-char chunks: stops at 4 (a 5th passes 5,000)", ids == [f"h{i}" for i in range(4)], ids)
    ids = _search_ids([hi_row("big", 6000)] + [hi_row(f"h{i}", 700) for i in range(3)], q)
    check("an oversized top chunk is still kept, alone", ids == ["big"], ids)
    ids = _search_ids([hi_row("h0", 700), hi_row("h1", 700, sim=0.6)], q)
    check("Hindi threshold unchanged (0.6 < 0.62 dropped)", ids == ["h0"], ids)
    en = [row(f"e{i}", "Biology", 11, "Photosynthesis in Higher Plants", 0.6, "Light reaction " * 50) for i in range(8)]
    ids = _search_ids(en, "what happens in the light reaction of photosynthesis")
    check("English still returns 3", ids == ["e0", "e1", "e2"], ids)


def test_hindi_prompt_rules():
    print("\n2. Hindi prompt rules")
    check("full-Hindi rule kept", server.HINDI_ONLY_CORE in server.HINDI_LANG_CONTEXT)
    check("full-Hindi rule text unchanged", server.HINDI_ONLY_CORE.startswith("Respond ONLY in Hindi (Devanagari script) -- every word in Hindi"))
    check("mnemonics must be Devanagari", "memory trick or mnemonic must itself be in Devanagari" in server.HINDI_LANG_CONTEXT)
    check("Latin-letter acronyms named as forbidden", "never an acronym or phrase made of English/Latin letters" in server.HINDI_LANG_CONTEXT)


PHOTO = row("b1", "Biology", 11, "Photosynthesis in Higher Plants", 0.7, "Light reaction ...")
PHOTO_HI = server.NCERT_HINDI_TITLES["Biology|11|Photosynthesis in Higher Plants"]


def _retrieved_from(text, language, rows):
    with Harness(ANSWER_WITH_CITATION, rows) as h:
        asyncio.run(drain(server.stream_response(text, language=language, personalize=False, skip_cache=True)))
        return h.user_message().split("Retrieved from", 1)[1].split("Student Question:", 1)[0]


def test_hinglish_hindi_citation():
    print("\n3. Hindi chapter title for Hinglish doubts")
    english = server._ncert_chapter_citation("Biology", 11, "Photosynthesis in Higher Plants")
    hindi = english.replace("Photosynthesis in Higher Plants", PHOTO_HI)
    check("English citation has a chapter number", "Chapter " in english, english)
    got = _retrieved_from("photosynthesis kaise hota hai", "hi", [PHOTO])
    check("Hinglish + Hindi answer: Hindi title, same chapter number", f"- {hindi}\n" in got, got)
    got = _retrieved_from("how does photosynthesis happen", "hi", [PHOTO])
    check("English question + Hindi answer: English title", f"- {english}\n" in got, got)
    got = _retrieved_from("photosynthesis kaise hota hai", "en", [PHOTO])
    check("Hinglish + English answer: English title", f"- {english}\n" in got, got)
    units = row("p1", "Physics", 11, "Units and Measurement", 0.6)
    got = server._ncert_citations([units], hindi_titles=True)
    check("names that differ between books still match (Measurement/Measurements)", got and got[0].endswith("— मात्राक एवं मापन"), got)
    appendix = row("x1", "Physics", 11, "Some Chapter Not In Either Book", 0.6)
    got = server._ncert_citations([appendix], hindi_titles=True)
    check("no Hindi title found: English name kept", got and got[0].endswith("— Some Chapter Not In Either Book"), got)


if __name__ == "__main__":
    test_hindi_chunk_count()
    test_hindi_prompt_rules()
    test_hinglish_hindi_citation()
    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    raise SystemExit(1 if _failed else 0)
