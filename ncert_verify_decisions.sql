-- Adds ncert_verify_decisions (2026-11-14): one row per difference an admin has marked "OK" on
-- the redesigned "Verify against PDF" result (admin-ncert-content.html). A "real" difference the
-- admin has decided is fine no longer counts against the PASS/FAIL verdict the next time the same
-- chapter is checked -- until the underlying text changes, which changes diff_key and makes the
-- decision stop matching (it is never auto-cleaned; a stale decision for a difference that no
-- longer occurs is simply never looked up again).
--
--   class/subject/chapter   which chapter this decision is scoped to.
--   diff_key                sha256(type + "|" + normalized pdf excerpt + "|" + normalized stored
--                            excerpt)[:24], computed in ncert_verify.py -- a hash, never the
--                            chapter's actual text. See ncert_verify._diff_key.
--   decided_at              when the admin marked this one "OK".
--
-- unique(class, subject, chapter, diff_key) -- marking the same difference "OK" twice is a no-op
-- (upsert on that key), not a second row.
--
-- RLS enabled, no policies at all -- same pattern as provider_usage_log/media_doubt_log: only the
-- service-role key (server.py's own Supabase client) can ever read or write this table, no anon/
-- authenticated access, nothing served to a client directly.
--
-- Purely additive, no backfill. Safe to run once in the Supabase SQL editor.
--
-- HOW TO RUN (Supabase dashboard -> SQL Editor):
--   1. Paste and run everything below. It's one transaction: all of it applies or none of it does.
--   2. Run the checks at the bottom and compare with the expected results written there.
--   3. Say "sql done"; only then does the redesigned verify-result feature continue.

begin;

create table if not exists public.ncert_verify_decisions (
  id bigserial primary key,
  class text not null,
  subject text not null,
  chapter text not null,
  diff_key text not null,
  decided_at timestamptz not null default now(),
  unique (class, subject, chapter, diff_key)
);

create index if not exists ncert_verify_decisions_chapter_idx
  on public.ncert_verify_decisions (class, subject, chapter);

alter table public.ncert_verify_decisions enable row level security;
-- No policies -- locked to the service-role key only, same as provider_usage_log/media_doubt_log.
-- This table never needs to be read by anon/authenticated clients.

commit;

-- ============================== checks ==============================
-- Expect the table with exactly these 6 columns.
select column_name, data_type, is_nullable from information_schema.columns
where table_schema = 'public' and table_name = 'ncert_verify_decisions'
order by ordinal_position;

-- Expect the unique constraint on (class, subject, chapter, diff_key).
select conname from pg_constraint
where conrelid = 'public.ncert_verify_decisions'::regclass and contype = 'u';

-- Expect row level security to be enabled and zero policies.
select relrowsecurity from pg_class where oid = 'public.ncert_verify_decisions'::regclass;
select count(*) as policy_count from pg_policies where tablename = 'ncert_verify_decisions';

-- Expect 0 (brand new table).
select count(*) from public.ncert_verify_decisions;
