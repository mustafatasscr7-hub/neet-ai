-- Adds diagram_url and has_diagram to match_pyq's returned columns. Currently, diagram-bearing
-- PYQ rows (has_diagram=true) are found by search correctly (the WHERE/ORDER BY only touch the
-- embedding column) but diagram_url isn't in the SELECT list, so chat.html never receives it and
-- students never see the diagram even when the matched question needs it.
--
-- Run this in the Supabase SQL editor. This repo has no DDL/DATABASE_URL access to inspect the
-- live function definition directly, so this is a best-effort reconstruction based on its known
-- call signature and return shape as used in server.py's search_pyq() (query_embedding,
-- match_threshold, match_count -> id/year/subject/question/option_a-d/correct_answer/similarity,
-- confirmed empirically this session) -- if any column name/type here doesn't match the live
-- schema, this will fail loudly with a clear Postgres error rather than silently doing the wrong
-- thing, so it's safe to attempt. Mirrors the same reconstruction approach used for match_ncert's
-- prior fixes (update_match_ncert_language_filter.sql, exclude_appendix_from_match_ncert.sql).

drop function if exists match_pyq(vector, double precision, integer);

create or replace function match_pyq(
  query_embedding vector(1536),
  match_threshold float,
  match_count int
)
returns table (
  id uuid,
  year int,
  subject text,
  question text,
  option_a text,
  option_b text,
  option_c text,
  option_d text,
  correct_answer text,
  diagram_url text,
  has_diagram boolean,
  similarity float
)
language sql stable
as $$
  select
    pyq.id,
    pyq.year,
    pyq.subject,
    pyq.question,
    pyq.option_a,
    pyq.option_b,
    pyq.option_c,
    pyq.option_d,
    pyq.correct_answer,
    pyq.diagram_url,
    pyq.has_diagram,
    1 - (pyq.embedding <=> query_embedding) as similarity
  from pyq
  where 1 - (pyq.embedding <=> query_embedding) > match_threshold
  order by pyq.embedding <=> query_embedding
  limit match_count;
$$;
