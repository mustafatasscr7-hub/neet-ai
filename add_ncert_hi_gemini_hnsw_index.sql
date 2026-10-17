-- Fast index for the Hindi NCERT retrieval path (match_ncert_hi_gemini, embedding_gemini vector(3072)).
--
-- Why: embedding_gemini had no index, so every Hindi doubt read all 6,758 Hindi vectors (12 KB
-- each, ~80 MB of TOAST) in full. Warm that's ~0.6 s; once evicted from Supabase Free's small
-- cache it measured 8.4 s, enough to hit the statement timeout (HTTP 500).
--
-- pgvector can't index vector(3072) directly (HNSW/IVFFlat max 2,000 dims), but it can index it
-- cast to halfvec (max 4,000 dims, pgvector >= 0.7). The index only PICKS 40 candidates; the RPC
-- then scores them with the full-precision column, the same `1 - (embedding_gemini <=> q)` as
-- before. So similarity values (and the 0.62 threshold) are unchanged, and the top 10 match the old
-- exact scan as long as they're among the index's top 40 (checked on 10 Hindi questions).
--
-- No DDL access from this repo -- run in the Supabase SQL editor, one step at a time.

-- STEP 1. Check pgvector is >= 0.7.0 (needed for halfvec). If it's older, update the extension
--         first: Dashboard -> Database -> Extensions -> vector, or `alter extension vector update;`.
select extversion from pg_extension where extname = 'vector';

-- STEP 2. Build the index (~50 MB; check Dashboard -> Database size stays under the Free 500 MB).
--         Partial index on Hindi rows only, matching the RPC's WHERE clause exactly. Not
--         CONCURRENTLY: the SQL editor runs in a transaction, and nothing writes this table at
--         runtime. If the editor times out, the build keeps running server-side -- check with
--         `select * from pg_stat_progress_create_index;` and wait for STEP 3 to show the index.
--         No parallel workers: a parallel HNSW build puts maintenance_work_mem in a shared memory
--         segment, and at 128MB Supabase Free failed with "53200: could not resize shared memory
--         segment ... Cannot allocate memory". If the graph outgrows 64MB, pgvector logs a NOTICE
--         and keeps building, only slower.
set statement_timeout = '15min';
set max_parallel_maintenance_workers = 0;
set maintenance_work_mem = '64MB';
create index if not exists ncert_content_embedding_gemini_hi_hnsw
  on public.ncert_content
  using hnsw ((embedding_gemini::halfvec(3072)) halfvec_cosine_ops)
  where language = 'hi' and embedding_gemini is not null;

-- STEP 3. Confirm it exists.
select indexname, pg_size_pretty(pg_relation_size(indexname::regclass)) as size
from pg_indexes where indexname = 'ncert_content_embedding_gemini_hi_hnsw';

-- STEP 4. Swap the RPC to use it. Same signature, same returned columns, same similarity formula,
--         so server.py needs no change. ef_search 100 (default 40) so the index reliably returns
--         all 40 candidates. Function-level `set` only affects this function's queries.
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
set hnsw.ef_search = 100
as $$
  with candidates as (
    select ncert_content.id
    from ncert_content
    where ncert_content.language = 'hi'
      and ncert_content.embedding_gemini is not null
    order by ncert_content.embedding_gemini::halfvec(3072) <=> query_embedding::halfvec(3072)
    limit greatest(match_count * 4, 40)
  )
  select
    ncert_content.id,
    ncert_content.subject,
    ncert_content.class,
    ncert_content.chapter_name,
    ncert_content.chapter_name_en,
    ncert_content.language,
    ncert_content.content,
    1 - (ncert_content.embedding_gemini <=> query_embedding) as similarity
  from candidates
  join ncert_content on ncert_content.id = candidates.id
  where 1 - (ncert_content.embedding_gemini <=> query_embedding) > match_threshold
  order by ncert_content.embedding_gemini <=> query_embedding
  limit match_count;
$$;

-- STEP 5 (optional). Check the planner uses the index: the plan should show
--         "Index Scan using ncert_content_embedding_gemini_hi_hnsw", not "Seq Scan".
explain
select id from ncert_content
where language = 'hi' and embedding_gemini is not null
order by embedding_gemini::halfvec(3072) <=> (
  select embedding_gemini::halfvec(3072) from ncert_content
  where language = 'hi' and embedding_gemini is not null limit 1)
limit 40;

-- ROLLBACK (instant, no data change): re-run the original function in
-- add_ncert_content_gemini_embedding_shadow.sql (step 2 there), then
-- drop index if exists ncert_content_embedding_gemini_hi_hnsw;
