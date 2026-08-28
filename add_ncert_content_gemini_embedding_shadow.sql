-- SHADOW migration for the Hindi-only Gemini embedding trial (gemini-embedding-001, 3072 dims).
-- Purely additive: does not touch the existing `embedding` vector(1536) column (OpenAI
-- text-embedding-3-small) or any English row. The new column is populated for Hindi rows only
-- (language='hi') by a separate Python backfill script -- this file only creates the column and
-- the parallel RPC function, it does not write any data itself.
--
-- This repo has no DDL/DATABASE_URL access (same situation as update_match_ncert_language_filter.sql
-- and add_ncert_content_extraction_method_column.sql before it) -- run this once in the Supabase
-- SQL editor.

-- 1. New column, nullable, no default -- stays NULL for every English row and for every Hindi
--    row until the backfill script actually embeds it. Requires the pgvector extension, already
--    in use by the existing `embedding` column on this same table.
alter table public.ncert_content add column if not exists embedding_gemini vector(3072);

-- 2. New RPC function, parallel to (not replacing) the existing match_ncert. Hardcoded to
--    language='hi' rather than taking filter_language as a parameter, since this function only
--    exists to serve the Hindi-only shadow trial -- English keeps using match_ncert against the
--    original `embedding` column, untouched, for the entire duration of this trial.
create or replace function match_ncert_hi_gemini(
  query_embedding vector(3072),
  match_threshold float,
  match_count int
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
    1 - (ncert_content.embedding_gemini <=> query_embedding) as similarity
  from ncert_content
  where ncert_content.language = 'hi'
    and ncert_content.embedding_gemini is not null
    and 1 - (ncert_content.embedding_gemini <=> query_embedding) > match_threshold
  order by ncert_content.embedding_gemini <=> query_embedding
  limit match_count;
$$;
