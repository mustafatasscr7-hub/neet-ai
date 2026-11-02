-- Adds Hindi scanned-PDF support to the PYQ pipeline (2026-11-02): a single `language` column,
-- same pattern as add_ncert_content_language_column.sql.
--
--   language   'en' (default, every existing row) or 'hi'. Student-facing reads, auto-approve,
--              and the duplicate-check corpus are all scoped to a single language by server.py --
--              see PYQ_STUDENT_VISIBLE, auto_check(), _pyq_dup_corpus()/_auto_dup_corpus() --
--              Hindi rows stay invisible to students and out of auto-approve until a human
--              reviews them, same as any other 'pending' row.
--
-- Purely additive: no existing column, constraint, or row is touched. Safe to run once in the
-- Supabase SQL editor.
--
-- HOW TO RUN (Supabase dashboard -> SQL Editor):
--   1. Paste and run PART 1 on its own. It's one transaction: all of it applies or none of it does.
--   2. Paste and run PART 2 on its own (match_pyq). If it errors with "cannot change return type
--      of existing function", skip it and send the error over -- server.py's PYQ_STUDENT_VISIBLE
--      filter already drops non-English rows from search results itself, so students stay safe
--      either way.
--   3. Run PART 3 (checks) and compare with the expected results written there.
--   4. Say it's done; only then is the server change (language-aware scan/save/review) safe to push.

-- ============================== PART 1: column + constraint + index ==============================
begin;

alter table public.pyq add column if not exists language text not null default 'en';

update public.pyq set language = 'en' where language is null;

alter table public.pyq drop constraint if exists pyq_language_check;
alter table public.pyq add constraint pyq_language_check check (language in ('en', 'hi'));

create index if not exists idx_pyq_language on public.pyq (language);
-- The duplicate-check corpus and the review queue are both read scoped by (subject, language) --
-- see _pyq_dup_corpus/_auto_dup_corpus in server.py.
create index if not exists idx_pyq_subject_language on public.pyq (subject, language);

commit;

-- ============================== PART 2: match_pyq ==============================
-- Same signature and columns as add_pyq_review_status.sql's own match_pyq (review_status =
-- 'approved' and is_active), plus one more filter: English only, for now -- see requirement 5
-- ("Student-facing practice, tests and search: show only language = 'en'"). CREATE OR REPLACE (no
-- DROP): if the live function's return columns differ from these, Postgres refuses and nothing
-- changes.
create or replace function match_pyq(
  query_embedding vector(1536),
  match_threshold float,
  match_count int
)
returns table (
  id uuid,
  year int,
  subject text,
  question text,
  option_a text,
  option_b text,
  option_c text,
  option_d text,
  correct_answer text,
  diagram_url text,
  has_diagram boolean,
  similarity float
)
language sql stable
as $$
  select
    pyq.id,
    pyq.year,
    pyq.subject,
    pyq.question,
    pyq.option_a,
    pyq.option_b,
    pyq.option_c,
    pyq.option_d,
    pyq.correct_answer,
    pyq.diagram_url,
    pyq.has_diagram,
    1 - (pyq.embedding <=> query_embedding) as similarity
  from pyq
  where pyq.review_status = 'approved'
    and pyq.is_active
    and pyq.language = 'en'
    and 1 - (pyq.embedding <=> query_embedding) > match_threshold
  order by pyq.embedding <=> query_embedding
  limit match_count;
$$;

-- ============================== PART 3: checks ==============================
-- Expect exactly one row: 'en' | <your total PYQ count>.
select language, count(*) from public.pyq group by language;

-- Expect the constraint definition to mention 'en' and 'hi'.
select pg_get_constraintdef(oid) from pg_constraint where conname = 'pyq_language_check';

-- Expect 0 (no row missed the backfill).
select count(*) as missing_language from public.pyq where language is null;

-- Expect the new definition to contain "language = 'en'" (only if PART 2 ran).
select pg_get_functiondef('public.match_pyq(vector, double precision, integer)'::regprocedure) ilike '%language = ''en''%' as match_pyq_filters_language;
