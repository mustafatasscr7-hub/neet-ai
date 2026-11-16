-- Adds ncert_rename_chapter() (2026-11-16): renames a chapter across every table keyed by
-- (class, subject, chapter) -- ncert_sections, ncert_chapter_log (including soft-deleted rows, so
-- a chapter's cost/load history still matches after it's renamed) and ncert_verify_decisions (so
-- an already-approved verify difference keeps matching too) -- in one transaction, for
-- admin-ncert-content.html's chapter rename feature.
--
-- The caller (server.py's /admin/ncert-content/chapter/rename) checks the new name isn't already
-- used by another chapter of the same class/subject before calling this function; this function
-- itself only performs the renames and reports how many rows it touched in each table.
--
-- Safe to deploy the server/frontend change before running this -- the endpoint already handles
-- a missing function with a plain error message instead of a crash.
--
-- HOW TO RUN (Supabase dashboard -> SQL Editor):
--   1. Paste and run everything below.
--   2. Run the checks at the bottom and compare with the expected results written there.
--   3. Say "sql done"; only then does chapter rename work end to end.

create or replace function public.ncert_rename_chapter(
  p_class text, p_subject text, p_old_chapter text, p_new_chapter text
)
returns table (sections_updated bigint, log_updated bigint, decisions_updated bigint)
language plpgsql
as $$
declare
  v_sections bigint;
  v_log bigint;
  v_decisions bigint;
begin
  update public.ncert_sections set chapter = p_new_chapter
    where class = p_class and subject = p_subject and chapter = p_old_chapter;
  get diagnostics v_sections = row_count;

  update public.ncert_chapter_log set chapter = p_new_chapter
    where class = p_class and subject = p_subject and chapter = p_old_chapter;
  get diagnostics v_log = row_count;

  update public.ncert_verify_decisions set chapter = p_new_chapter
    where class = p_class and subject = p_subject and chapter = p_old_chapter;
  get diagnostics v_decisions = row_count;

  return query select v_sections, v_log, v_decisions;
end;
$$;

-- ============================== checks ==============================
-- Expect one row: this function, taking 4 arguments.
select proname, pronargs from pg_proc where proname = 'ncert_rename_chapter';
