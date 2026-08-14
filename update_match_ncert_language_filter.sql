-- Recreates match_ncert to (a) return the two new columns added by
-- add_ncert_content_language_column.sql (language, chapter_name_en -- needed so server.py can
-- build correct chapter citations for Hindi rows) and (b) accept an optional filter_language
-- parameter, so search_ncert() can restrict vector search to the student's query language
-- instead of mixing 'en' and 'hi' rows in the same similarity ranking.
--
-- Mirrors match_diagrams' existing filter_chapter pattern (NULL = no filter, matching everything;
-- a value = only rows where that column matches).
--
-- Run this in the Supabase SQL editor. This repo has no DDL/DATABASE_URL access to inspect the
-- live function definition directly, so this is a best-effort reconstruction based on its known
-- call signature and return shape as used in server.py (query_embedding, match_threshold,
-- match_count -> subject/class/chapter_name/content/similarity) -- if any column name/type here
-- doesn't match the live schema, this will fail loudly with a clear Postgres error rather than
-- silently doing the wrong thing, so it's safe to attempt.

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
  order by ncert_content.embedding <=> query_embedding
  limit match_count;
$$;
