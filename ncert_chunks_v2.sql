-- Stage 1 of the English NCERT retrieval rework: a NEW table (ncert_content_v2) built beside the
-- existing ncert_content, with better chunk boundaries (heading/paragraph/sentence-aware instead
-- of the old fixed 500-word split -- see ncert_rechunk_v2.py) plus hybrid vector+keyword search.
--
-- Purely additive. Does NOT touch ncert_content, match_ncert, match_ncert_hi_gemini, or anything
-- the live /chat path reads. search_ncert() in server.py is unchanged; a new search_ncert_v2()
-- reads this table instead, gated behind NCERT_V2_ENABLED (default false, see server.py).
--
-- This repo has no DDL/DATABASE_URL access -- run this once in the Supabase SQL editor, in order,
-- one statement group at a time if the editor balks at running it all together. Requires the
-- pgvector extension, already enabled for the existing `embedding` column on ncert_content.
--
-- Run this file FIRST. Only after it succeeds, run:
--   python ncert_rechunk_v2.py --load
-- (the dry run -- no --load -- needs none of this; it writes nothing to Supabase.)

-- ---------- 1. Table ----------
-- embedding is vector(1536) to match the current English embedding model
-- (OpenAI text-embedding-3-small, same as the old `embedding` column on ncert_content and
-- get_embedding() in server.py). chapter_name_en mirrors ncert_content's own convention: null
-- for English rows (chapter_name IS the English name already), reserved for a future Hindi v2
-- pass the same way the old table uses it today.
create table if not exists public.ncert_content_v2 (
  id uuid primary key default gen_random_uuid(),
  subject text not null,
  class int not null,
  chapter_name text not null,
  chapter_name_en text,
  language text not null default 'en',
  heading text,
  page int,
  chunk_index int not null,
  content text not null,
  content_tsv tsvector generated always as (to_tsvector('english', content)) stored,
  embedding vector(1536),
  created_at timestamptz not null default now(),
  -- Lets ncert_rechunk_v2.py --load re-run safely (upsert on conflict) instead of duplicating
  -- rows if it's ever run twice against the same chapter.
  unique (subject, class, chapter_name, chunk_index)
);

comment on table public.ncert_content_v2 is
  'Stage 1 English NCERT rebuild (heading/paragraph/sentence-aware chunks). Parallel to '
  'ncert_content, not a replacement -- see ncert_chunks_v2.sql and ncert_rechunk_v2.py.';

-- ---------- 2. Keyword search index ----------
create index if not exists ncert_content_v2_tsv_idx
  on public.ncert_content_v2
  using gin (content_tsv);

-- ---------- 3. Vector search index ----------
-- vector(1536) fits pgvector's HNSW limit (2000 dims) directly -- no halfvec cast needed, unlike
-- the Hindi embedding_gemini column (3072 dims, see add_ncert_hi_gemini_hnsw_index.sql).
-- Safe to build now even though the table is empty right after step 1 (ncert_rechunk_v2.py
-- --load's inserts just grow the graph incrementally) -- matches the run order in the task:
-- this SQL first, the --load command after.
set statement_timeout = '15min';
set max_parallel_maintenance_workers = 0;
set maintenance_work_mem = '64MB';
create index if not exists ncert_content_v2_embedding_hnsw
  on public.ncert_content_v2
  using hnsw (embedding vector_cosine_ops);

-- ---------- 4. RPC: plain vector search ----------
-- Same shape as match_ncert, plus heading/page/chunk_index (new columns) and optional
-- filter_subject/filter_class/filter_chapter (match_diagrams' NULL-means-no-filter pattern).
create or replace function match_ncert_v2(
  query_embedding vector(1536),
  match_threshold float,
  match_count int,
  filter_subject text default null,
  filter_class int default null,
  filter_chapter text default null
)
returns table (
  id uuid,
  subject text,
  class int,
  chapter_name text,
  chapter_name_en text,
  language text,
  heading text,
  page int,
  chunk_index int,
  content text,
  similarity float
)
language sql stable
as $$
  select
    ncert_content_v2.id,
    ncert_content_v2.subject,
    ncert_content_v2.class,
    ncert_content_v2.chapter_name,
    ncert_content_v2.chapter_name_en,
    ncert_content_v2.language,
    ncert_content_v2.heading,
    ncert_content_v2.page,
    ncert_content_v2.chunk_index,
    ncert_content_v2.content,
    1 - (ncert_content_v2.embedding <=> query_embedding) as similarity
  from ncert_content_v2
  where ncert_content_v2.embedding is not null
    and 1 - (ncert_content_v2.embedding <=> query_embedding) > match_threshold
    and (filter_subject is null or ncert_content_v2.subject = filter_subject)
    and (filter_class is null or ncert_content_v2.class = filter_class)
    and (filter_chapter is null or ncert_content_v2.chapter_name = filter_chapter)
  order by ncert_content_v2.embedding <=> query_embedding
  limit match_count;
$$;

-- ---------- 5. RPC: hybrid vector + keyword search (reciprocal rank fusion) ----------
-- Standard RRF: each candidate set (vector-similarity order, keyword ts_rank order) contributes
-- 1/(rrf_k + rank) to a fused score; rows found by both lists score higher than a row found by
-- only one. rrf_k = 60 is the commonly-used RRF constant (flattens the gap between rank 1 and
-- rank 2 so one list doesn't dominate just because it ranks its top hit slightly higher).
-- websearch_to_tsquery tolerates arbitrary/garbage query_text (empty tsquery, matches nothing)
-- rather than raising, so a weird query never 500s this function.
create or replace function match_ncert_v2_hybrid(
  query_embedding vector(1536),
  query_text text,
  match_count int default 5,
  filter_subject text default null,
  filter_class int default null,
  filter_chapter text default null,
  rrf_k int default 60
)
returns table (
  id uuid,
  subject text,
  class int,
  chapter_name text,
  chapter_name_en text,
  language text,
  heading text,
  page int,
  chunk_index int,
  content text,
  similarity float,
  rrf_score float
)
language sql stable
as $$
  with vector_ranked as (
    select
      ncert_content_v2.id,
      row_number() over (order by ncert_content_v2.embedding <=> query_embedding) as rnk
    from ncert_content_v2
    where ncert_content_v2.embedding is not null
      and (filter_subject is null or ncert_content_v2.subject = filter_subject)
      and (filter_class is null or ncert_content_v2.class = filter_class)
      and (filter_chapter is null or ncert_content_v2.chapter_name = filter_chapter)
    order by ncert_content_v2.embedding <=> query_embedding
    limit greatest(match_count * 10, 50)
  ),
  keyword_ranked as (
    select
      ncert_content_v2.id,
      row_number() over (
        order by ts_rank(ncert_content_v2.content_tsv, websearch_to_tsquery('english', query_text)) desc
      ) as rnk
    from ncert_content_v2
    where query_text is not null
      and ncert_content_v2.content_tsv @@ websearch_to_tsquery('english', query_text)
      and (filter_subject is null or ncert_content_v2.subject = filter_subject)
      and (filter_class is null or ncert_content_v2.class = filter_class)
      and (filter_chapter is null or ncert_content_v2.chapter_name = filter_chapter)
    order by ts_rank(ncert_content_v2.content_tsv, websearch_to_tsquery('english', query_text)) desc
    limit greatest(match_count * 10, 50)
  ),
  fused as (
    select
      coalesce(v.id, k.id) as id,
      coalesce(1.0 / (rrf_k + v.rnk), 0) + coalesce(1.0 / (rrf_k + k.rnk), 0) as rrf_score
    from vector_ranked v
    full outer join keyword_ranked k on v.id = k.id
  )
  select
    nc.id,
    nc.subject,
    nc.class,
    nc.chapter_name,
    nc.chapter_name_en,
    nc.language,
    nc.heading,
    nc.page,
    nc.chunk_index,
    nc.content,
    coalesce(1 - (nc.embedding <=> query_embedding), 0) as similarity,
    f.rrf_score
  from fused f
  join ncert_content_v2 nc on nc.id = f.id
  order by f.rrf_score desc
  limit match_count;
$$;

-- ROLLBACK (instant, no data loss elsewhere): this table/these functions are purely additive --
--   drop function if exists match_ncert_v2_hybrid;
--   drop function if exists match_ncert_v2;
--   drop table if exists public.ncert_content_v2;
-- ncert_content, match_ncert and match_ncert_hi_gemini are completely untouched either way.
