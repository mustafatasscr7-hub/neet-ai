-- Adds ncert_chapter_log (2026-11-13): one row per NCERT chapter apply/edit/keyword-regeneration,
-- so admin-ncert-content.html can show cost and load details per chapter without recomputing them
-- from ncert_sections (which has no cost/model/timing columns of its own).
--
--   class / subject / chapter   same identity triple as ncert_sections.
--   source                      'pdf' (chapter applied from the PDF route) or 'manual' (added/edited
--                                by hand via the manual section form, including keyword regeneration
--                                on an existing manual section).
--   keywords_cost_usd           DeepSeek spend for this apply/edit/regeneration event only (not a
--                                running chapter total -- the page sums the chapter's rows itself).
--   model_calls                 number of DeepSeek keyword calls made in this event.
--   model                       DeepSeek model name used for those calls (NULL if model_calls = 0).
--   pages                       PDF page count (NULL for manual rows -- there is no PDF).
--   extract_seconds             wall-clock seconds the PDF extract/clean/validate pipeline took
--                                (NULL for manual rows).
--   sections / words            counts written by this event.
--   deleted_at                  set (not row-deleted) when the chapter is deleted from
--                                ncert_sections, so the chapter's cost history survives the delete.
--
-- Deliberately has no foreign key into ncert_sections: a chapter's log rows must survive that
-- chapter being deleted and reloaded under the same (class, subject, chapter) identity.
--
-- RLS enabled, no policies at all -- service-role key only (ADMIN_HEADERS in server.py), same
-- lockdown as provider_usage_log (create_provider_usage_log_table.sql) and pyq_solution_cache.
-- This is internal cost telemetry, never returned to a contractor response.
--
-- Purely additive. Safe to run once in the Supabase SQL editor.
--
-- HOW TO RUN (Supabase dashboard -> SQL Editor):
--   1. Paste and run everything below. It's one transaction: all of it applies or none of it does.
--   2. Run the checks at the bottom and compare with the expected results written there.
--   3. Say "sql done"; only then does the cost/load-details feature continue.

begin;

create table if not exists public.ncert_chapter_log (
  id bigserial primary key,
  class text not null,
  subject text not null,
  chapter text not null,
  source text not null check (source in ('pdf', 'manual')),
  keywords_cost_usd numeric not null default 0,
  model_calls int not null default 0,
  model text,
  pages int,
  extract_seconds numeric,
  sections int,
  words int,
  created_at timestamptz not null default now(),
  deleted_at timestamptz
);

create index if not exists idx_ncert_chapter_log_class_subject_chapter
  on public.ncert_chapter_log (class, subject, chapter);

alter table public.ncert_chapter_log enable row level security;

commit;

-- ============================== checks ==============================
-- Expect the table to exist with the columns above.
select column_name, data_type, is_nullable from information_schema.columns
where table_schema = 'public' and table_name = 'ncert_chapter_log'
order by ordinal_position;

-- Expect rowsecurity = true and zero policies (service role bypasses RLS; no anon/authenticated access).
select relrowsecurity from pg_class where relname = 'ncert_chapter_log';
select count(*) as policy_count from pg_policies where tablename = 'ncert_chapter_log';

-- Expect the new index to exist.
select indexname from pg_indexes where tablename = 'ncert_chapter_log';
