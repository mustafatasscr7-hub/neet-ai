-- Excludes Appendix rows from match_ncert's candidate pool at the SQL level, instead of only
-- filtering them out of the top-N results after the fact in search_ncert() (server.py). With
-- match_count typically 1-3, an Appendix chunk sitting in the raw top-N can crowd out a
-- genuinely relevant chapter that would otherwise have ranked in that window -- confirmed live:
-- Hindi queries about Mendel's law of segregation had Appendix chunks occupying 2 of the top 3
-- raw slots (a coincidental lexical collision on the shared root word "पृथक्"/separate), while
-- the actually-correct Genetics chapter ranked #6.
--
-- Safe to do universally (not language-conditional): server.py's NCERT_NON_CHAPTER_LABELS
-- filter already strips Appendix from every citation today, for both languages -- Appendix was
-- never actually reachable as an answer source before this change either. This only changes
-- which REAL chapters get a chance to compete for the top-N slots; it removes no working
-- behavior a student could previously get an answer from.
--
-- Preliminary Pages/Answer Key are left alone here -- Appendix is the only one of the three
-- NCERT_NON_CHAPTER_LABELS entries confirmed (via live testing) to actually win real-topic
-- matches; add the same WHERE clause for the others if they're ever observed doing the same.
--
-- Run this in the Supabase SQL editor. This repo has no DDL/DATABASE_URL access to inspect the
-- live function definition directly, so this is a best-effort reconstruction based on its known
-- call signature and return shape as used in server.py -- if any column name/type here doesn't
-- match the live schema, this will fail loudly with a clear Postgres error rather than silently
-- doing the wrong thing, so it's safe to attempt.

create or replace function match_ncert(
  query_embedding vector(1536),
  match_threshold float,
  match_count int,
  filter_language text default null
)
returns table (
  id uuid,
  subject text,
  class int,
  chapter_name text,
  chapter_name_en text,
  language text,
  content text,
  similarity float
)
language sql stable
as $$
  select
    ncert_content.id,
    ncert_content.subject,
    ncert_content.class,
    ncert_content.chapter_name,
    ncert_content.chapter_name_en,
    ncert_content.language,
    ncert_content.content,
    1 - (ncert_content.embedding <=> query_embedding) as similarity
  from ncert_content
  where 1 - (ncert_content.embedding <=> query_embedding) > match_threshold
    and (filter_language is null or ncert_content.language = filter_language)
    and coalesce(ncert_content.chapter_name_en, ncert_content.chapter_name) != 'Appendix'
  order by ncert_content.embedding <=> query_embedding
  limit match_count;
$$;
