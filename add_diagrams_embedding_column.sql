-- Adds the embedding column chat.html's diagram-matching (doubt-solving) needs to run a
-- pgvector similarity search against uploaded reference diagrams. Same 1536-dim
-- text-embedding-3-small model already used for pyq/ncert_content embeddings elsewhere in this
-- project, so the vector comparison is apples-to-apples with everything else -- just a
-- different table.
alter table public.diagrams add column if not exists embedding vector(1536);

-- Mirrors the existing match_ncert RPC's shape/threshold convention (similarity = 1 - cosine
-- distance, match_threshold filters on similarity not distance) so the same mental model
-- applies elsewhere in the codebase. filter_chapter narrows the search to one chapter when the
-- doubt's chapter could be determined server-side; pass null to search all reviewed diagrams.
-- Only ever matches admin-CONFIRMED diagrams (reviewed = true) -- a newly-uploaded,
-- not-yet-checked diagram must not be shown to a student as if it were vetted.
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
    1 - (diagrams.embedding <=> query_embedding) as similarity
  from diagrams
  where diagrams.reviewed = true
    and diagrams.embedding is not null
    and (filter_chapter is null or diagrams.chapter = filter_chapter)
    and 1 - (diagrams.embedding <=> query_embedding) > match_threshold
  order by diagrams.embedding <=> query_embedding
  limit match_count;
$$;
