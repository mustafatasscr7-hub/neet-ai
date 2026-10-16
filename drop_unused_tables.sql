-- Drops two tables nothing uses any more (checked 2026-10-16: no reference in neet-ai or
-- neet-ai-mobile):
--   public.usage_limits      -- 13 rows (user_id, message_count, reset_date), last written 2026-07-21.
--                               Per-student data that /account/delete never knew about; replaced by
--                               usage_log long ago.
--   public."ncert-daigrams"  -- 0 rows (id, created_at). A table created by mistake with the storage
--                               bucket's name; the bucket itself (diagrams) is NOT touched.
-- The empty public storage bucket "Q-Images" (0 files) is removed from the dashboard -- see the end.
--
-- Run in the Supabase SQL editor, STEP 1 first, and only run STEP 2 if step 1 looks as described.


-- STEP 1 (read-only) ------------------------------------------------------------------------------
-- Expected: usage_limits 13 (or close), "ncert-daigrams" 0.
select 'usage_limits' as table_name, count(*) as row_count from public.usage_limits
union all
select 'ncert-daigrams', count(*) from public."ncert-daigrams";

-- Expected: NO rows. Any database function that still mentions either table would break once it's
-- dropped -- if one shows up, stop and send it to me.
select n.nspname as schema, p.proname as function_name
from pg_proc p join pg_namespace n on n.oid = p.pronamespace
where n.nspname not in ('pg_catalog', 'information_schema')
  and (p.prosrc ilike '%usage_limits%' or p.prosrc ilike '%"ncert-daigrams"%');

-- Expected: NO rows (no view built on either table).
select distinct v.relname as view_name
from pg_depend d
join pg_rewrite r on r.oid = d.objid
join pg_class v on v.oid = r.ev_class
where d.refobjid in ('public.usage_limits'::regclass, 'public."ncert-daigrams"'::regclass)
  and v.oid not in ('public.usage_limits'::regclass, 'public."ncert-daigrams"'::regclass);


-- STEP 2 (drop) -----------------------------------------------------------------------------------
-- No CASCADE on purpose: if something unexpected depends on a table, this fails instead of
-- silently dropping that too.
drop table if exists public.usage_limits;
drop table if exists public."ncert-daigrams";


-- STEP 3 (dashboard) ------------------------------------------------------------------------------
-- Storage > Q-Images: confirm it shows no files, then "..." next to the bucket > Delete bucket.
-- (Deleting buckets or files with SQL isn't supported by Supabase; the dashboard uses the API.)
