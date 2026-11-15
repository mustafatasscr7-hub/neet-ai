-- embedding_cache duplicate cleanup -- 2026-11-15.
--
-- Re-audit of the 2026-11-07 finding (see project memory "embedding_cache audit"): the table has
-- exactly 4 columns (id uuid pk, question_hash text, embedding vector(1536), created_at timestamp
-- without time zone) and, as of this re-check, 10,149 rows with 0 duplicate question_hash values
-- (grown from the 9,610 rows / 0 duplicates seen 8 days ago -- same "no duplicates today" result,
-- re-verified against live data rather than assumed from memory).
--
-- IMPORTANT, confirmed via PostgREST's OpenAPI schema (the only introspection available here --
-- no direct Postgres access): question_hash does NOT carry a documented unique constraint. The
-- OpenAPI doc only annotates primary keys ("Note:\nThis is a Primary Key.<pk/>", present on `id`
-- only) and foreign keys -- a plain UNIQUE constraint on question_hash, if one exists, is invisible
-- to this introspection method. So "0 duplicates today" is NOT proof duplicates are impossible;
-- without a confirmed constraint, a future write path (a cache-write race, a code change that
-- drops an upsert's on_conflict clause, etc.) could still insert one. This script exists for that
-- future case -- it is a no-op today (every SELECT below should return 0 rows / groups) but is
-- kept ready rather than re-written from scratch if duplicates ever do appear.
--
-- Near-duplicate detection (same question after lowercase/trim/whitespace-collapse) is
-- structurally impossible here: there is no column holding the original query text, only its
-- hash (question_hash). This script therefore only ever targets EXACT question_hash duplicates.
--
-- Run each step manually, in order. Nothing here is executed automatically.

-- ============================================================================================
-- STEP (a): PREVIEW ONLY -- counts duplicate groups, rows that would be removed, and an estimated
-- MB reclaimed. Safe to run any time; changes nothing.
-- ============================================================================================
with dup_groups as (
    select question_hash, count(*) as row_count
    from public.embedding_cache
    group by question_hash
    having count(*) > 1
)
select
    count(*)                                   as duplicate_groups,
    coalesce(sum(row_count) - count(*), 0)      as rows_that_would_be_removed,
    -- 8.4 KB/row is the user's own prior estimate (not independently re-measured here) for a
    -- vector(1536) embedding row's on-disk footprint -- used only to size the expected VACUUM
    -- FULL benefit, not as an exact figure.
    round((coalesce(sum(row_count) - count(*), 0) * 8.4) / 1024.0, 2) as estimated_mb_reclaimed
from dup_groups;

-- ============================================================================================
-- STEP (b): DELETE duplicates, keeping the newest row per question_hash group. Wrapped in an
-- explicit transaction -- review the row count DELETE reports before committing. Swap which of
-- the two trailing statements is commented out depending on what you see.
-- ============================================================================================
begin;

with ranked as (
    select
        id,
        question_hash,
        row_number() over (partition by question_hash order by created_at desc) as rn
    from public.embedding_cache
)
delete from public.embedding_cache
where id in (select id from ranked where rn > 1);

-- Review the "DELETE n" row count above, then uncomment exactly ONE of the next two lines:
-- rollback;
commit;

-- ============================================================================================
-- STEP (c): reclaim disk space. Run as its OWN statement, outside any transaction block (VACUUM
-- cannot run inside one). This takes an exclusive lock on embedding_cache for its duration -- run
-- off-peak, not immediately after step (b) during live traffic.
-- ============================================================================================
vacuum full public.embedding_cache;
