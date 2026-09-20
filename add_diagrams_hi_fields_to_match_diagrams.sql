-- Diagram language-awareness fix (2026-09-20): match_diagrams never returned name_hi/
-- description_hi even though the diagrams table has carried both columns since the bilingual
-- diagram-upload work (diagram_upload_tool memory). /diagram-match's response was built straight
-- from this RPC's own return row, so even a diagram WITH real Hindi content filled in could never
-- reach chat.html's Show Diagram flow -- confirmed live: zero of the current 14 diagrams have
-- name_hi set yet (pyq_diagram_hindi_content_gap memory), so this was undetectable until now, but
-- the gap is real and would silently misfire the moment Hindi diagram content is added.
--
-- Purely additive to the returned columns -- matching/ordering/threshold logic (the embedding
-- comparison itself) is untouched. The embedding column itself is still built from English
-- name/description/chapter only (build_diagram_embedding_text, server.py) -- this migration does
-- NOT add Hindi-aware matching, only Hindi-aware display of whatever already matched.
--
-- Postgres won't let CREATE OR REPLACE change an existing function's return type (same issue
-- add_diagrams_importance_rating.sql already hit), so the old signature is dropped first -- this
-- is still non-destructive, it just recreates the function with importance_rating (added by that
-- migration) plus the two new hi columns.
drop function if exists match_diagrams(vector(1536), float, int, text);

create or replace function match_diagrams (
  query_embedding vector(1536),
  match_threshold float,
  match_count int,
  filter_chapter text default null
)
returns table (
  id bigint,
  subject text,
  chapter text,
  name text,
  description text,
  name_hi text,
  description_hi text,
  image_url text,
  importance_rating smallint,
  similarity float
)
language sql stable
as $$
  select
    diagrams.id,
    diagrams.subject,
    diagrams.chapter,
    diagrams.name,
    diagrams.description,
    diagrams.name_hi,
    diagrams.description_hi,
    diagrams.image_url,
    diagrams.importance_rating,
    1 - (diagrams.embedding <=> query_embedding) as similarity
  from diagrams
  where diagrams.reviewed = true
    and diagrams.embedding is not null
    and (filter_chapter is null or diagrams.chapter = filter_chapter)
    and 1 - (diagrams.embedding <=> query_embedding) > match_threshold
  order by diagrams.embedding <=> query_embedding
  limit match_count;
$$;
