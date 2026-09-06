-- Manual "importance rating" for reference diagrams (1-5 stars), set by an admin in Diagram
-- Review / at upload time and shown to students wherever a diagram appears. Null/unset means
-- "not rated yet" -- deliberately distinct from 0, since an unrated diagram must show no stars
-- at all, not an empty 0-star row.
alter table public.diagrams add column if not exists importance_rating smallint;
alter table public.diagrams drop constraint if exists diagrams_importance_rating_check;
alter table public.diagrams add constraint diagrams_importance_rating_check
  check (importance_rating is null or importance_rating between 1 and 5);

-- match_diagrams needs to also return importance_rating so chat.html's diagram-match result (and
-- therefore the in-chat auto-embedded diagram / lightbox) can show the same star rating the
-- Diagram Library and admin review screen show. Postgres won't let CREATE OR REPLACE change an
-- existing function's return type, so the old signature is dropped first -- this is still
-- non-destructive, it just recreates the function.
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
