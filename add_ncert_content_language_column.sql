-- Adds language support to ncert_content ahead of Hindi NCERT ingestion.
--
-- language: 'en' (default, all 1,501 existing rows) or 'hi' -- lets search_ncert() filter
-- vector search to the student's query language instead of mixing both.
--
-- chapter_name_en: nullable, only ever set on 'hi' rows. Hindi and English NCERT editions are
-- literal chapter-for-chapter translations, but chapter_name itself is stored in real Hindi
-- (matching the same "extract the real title, not a filename" approach already used for
-- English) -- which means it can't be looked up directly against NEET_SYLLABUS (the canonical
-- chapter-number list server.py already ports from the frontend, which is English-only) the way
-- English chapter_name already is. chapter_name_en carries the matching English chapter name
-- purely so _ncert_chapter_citation() can resolve the real chapter NUMBER for Hindi content too
-- -- the citation text shown to a Hindi-answered student still uses the real chapter_name (Hindi).
--
-- Purely additive: no existing column, constraint, or row is touched. Safe to run once in the
-- Supabase SQL editor.

alter table public.ncert_content add column if not exists language text not null default 'en';
alter table public.ncert_content add column if not exists chapter_name_en text;

-- Explicit backfill (the column default already covers this for any row that existed before
-- the ALTER ran, but doing it explicitly leaves nothing implicit).
update public.ncert_content set language = 'en' where language is null or language = 'en';

create index if not exists idx_ncert_content_language on public.ncert_content (language);
