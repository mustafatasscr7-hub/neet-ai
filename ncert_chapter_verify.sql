-- Adds text cross-check columns to ncert_chapter_log (2026-11-13), for the upload-time PDF-vs-
-- cleaned-sections comparison (admin-ncert-content.html's Preview/Apply) and the "Verify against
-- PDF" action for chapters already loaded. See ncert_chapter_log.sql for the table itself.
--
--   source_words         total word count of the PDF's text layer (all pages, before cleaning).
--   kept_words           word count actually stored across the chapter's ncert_sections rows.
--   coverage_pct         kept_words / source_words * 100, written by the PDF-route Apply only --
--                         NOT touched by "Verify against PDF" (that action re-checks whatever is
--                         currently stored against a PDF; it doesn't redo the original load).
--   dropped_lines_count  how many PDF lines were classified as dropped (running header, page
--                         number, figure caption/label, box/sidebar, unit intro, repeated
--                         contents, other) during that same Apply.
--   verified_at          when a cross-check last ran for this chapter -- either the Apply that
--                         loaded it, or a later "Verify against PDF" run.
--   verify_status        'PASS' or 'FAIL' (null = never checked, shown as "never" on the page).
--   verify_summary        short plain-text summary (e.g. "FAIL - 94.1% coverage, 2 sentence
--                         mismatch(es), 1 missing paragraph(s)"); if an admin applied anyway on a
--                         FAIL, that is appended here (there is no separate override column).
--
-- source_words/kept_words/coverage_pct/dropped_lines_count are only ever written by the PDF-route
-- Apply (ncert_ingest.py's --apply does the same, see write_chapter_log). verified_at/verify_status/
-- verify_summary are written by that SAME Apply, and are later overwritten in place -- on every
-- non-deleted log row for the chapter -- by "Verify against PDF", so the page always reads the
-- chapter's current verify state off whichever row it looks at.
--
-- The dropped lines themselves and the mismatch/missing-paragraph excerpts are NEVER stored here
-- (or anywhere) -- only the counts and the short summary string. Nullable, additive, no backfill:
-- every chapter loaded before this migration simply has nulls here, same as it already does for
-- source/cost/model/pages (see ncert_chapter_log.sql's own note on that).
--
-- Purely additive. Safe to run once in the Supabase SQL editor.
--
-- HOW TO RUN (Supabase dashboard -> SQL Editor):
--   1. Paste and run everything below. It's one transaction: all of it applies or none of it does.
--   2. Run the checks at the bottom and compare with the expected results written there.
--   3. Say "sql done"; only then does the text cross-check feature continue.

begin;

alter table public.ncert_chapter_log
  add column if not exists source_words int,
  add column if not exists kept_words int,
  add column if not exists coverage_pct numeric,
  add column if not exists dropped_lines_count int,
  add column if not exists verified_at timestamptz,
  add column if not exists verify_status text,
  add column if not exists verify_summary text;

alter table public.ncert_chapter_log drop constraint if exists ncert_chapter_log_verify_status_check;
alter table public.ncert_chapter_log
  add constraint ncert_chapter_log_verify_status_check
  check (verify_status is null or verify_status in ('PASS', 'FAIL'));

commit;

-- ============================== checks ==============================
-- Expect the 7 new columns, all nullable.
select column_name, data_type, is_nullable from information_schema.columns
where table_schema = 'public' and table_name = 'ncert_chapter_log'
  and column_name in ('source_words', 'kept_words', 'coverage_pct', 'dropped_lines_count',
                       'verified_at', 'verify_status', 'verify_summary')
order by column_name;

-- Expect the check constraint to exist.
select conname from pg_constraint where conname = 'ncert_chapter_log_verify_status_check';

-- Expect every existing row's new columns to be null (nothing backfilled).
select count(*) as rows_with_any_verify_data from public.ncert_chapter_log
where source_words is not null or kept_words is not null or coverage_pct is not null
   or dropped_lines_count is not null or verified_at is not null or verify_status is not null
   or verify_summary is not null;
