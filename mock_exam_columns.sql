-- Adds Mock exam support to the pyq table (2026-11-11): lets the same table hold full NEET-pattern
-- mock tests (Physics 45 / Chemistry 45 / Biology 90, positions 1-180) alongside real PYQs,
-- without mock rows ever surfacing where only real PYQs are expected.
--
--   source         'pyq' (default, every existing row) or 'mock'. Same pattern as
--                  pyq_language.sql's `language` column -- purely additive, existing rows untouched.
--   mock_name      e.g. "Mock Test 1". NULL for source = 'pyq' rows.
--   mock_position  1-180, the question's position within that mock (Physics 1-45, Chemistry
--                  46-90, Biology 91-180). NULL for source = 'pyq' rows.
--
-- Every student-facing read, the duplicate-check corpus, and chat/search retrieval (match_pyq)
-- must stay scoped to source = 'pyq' unless the caller is explicitly mock-aware -- see
-- PYQ_STUDENT_VISIBLE and _pyq_dup_corpus()/_auto_dup_corpus() in server.py (updated separately,
-- after this migration runs). Mock rows are never auto-approved and always need manual review,
-- same gate as pyq_review_status.sql's review_status column.
--
-- Purely additive: no existing column, constraint, or row is touched. Safe to run once in the
-- Supabase SQL editor.
--
-- HOW TO RUN (Supabase dashboard -> SQL Editor):
--   1. Paste and run PART 1 on its own. It's one transaction: all of it applies or none of it does.
--   2. Paste and run PART 2 on its own (match_pyq). If it errors with "cannot change return type
--      of existing function", skip it and send the error over -- server.py's PYQ_STUDENT_VISIBLE
--      filter already drops mock rows from search results itself, so students stay safe either way.
--   3. Run PART 3 (checks) and compare with the expected results written there.
--   4. Say "sql done"; only then does the mock-exam review-page work continue.

-- ============================== PART 1: columns + constraints + index ==============================
begin;

alter table public.pyq add column if not exists source text not null default 'pyq';
alter table public.pyq add column if not exists mock_name text;
alter table public.pyq add column if not exists mock_position int;

update public.pyq set source = 'pyq' where source is null;

alter table public.pyq drop constraint if exists pyq_source_check;
alter table public.pyq add constraint pyq_source_check check (source in ('pyq', 'mock'));

alter table public.pyq drop constraint if exists pyq_mock_position_check;
alter table public.pyq add constraint pyq_mock_position_check
  check (mock_position is null or (mock_position between 1 and 180));

alter table public.pyq drop constraint if exists pyq_mock_name_check;
alter table public.pyq add constraint pyq_mock_name_check
  check ((source = 'mock') = (mock_name is not null));

create index if not exists idx_pyq_source on public.pyq (source);
create index if not exists idx_pyq_mock_name_position on public.pyq (mock_name, mock_position)
  where source = 'mock';

commit;

-- ============================== PART 2: match_pyq ==============================
-- Same signature and columns as pyq_language.sql's own match_pyq (review_status = 'approved',
-- is_active, language = 'en'), plus one more filter: real PYQs only. CREATE OR REPLACE (no DROP):
-- if the live function's return columns differ from these, Postgres refuses and nothing changes.
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
    and pyq.source = 'pyq'
    and 1 - (pyq.embedding <=> query_embedding) > match_threshold
  order by pyq.embedding <=> query_embedding
  limit match_count;
$$;

-- ============================== PART 3: checks ==============================
-- Expect exactly one row: 'pyq' | <your total PYQ count>.
select source, count(*) from public.pyq group by source;

-- Expect the constraint definitions to mention 'pyq'/'mock' and the 1-180 range.
select conname, pg_get_constraintdef(oid) from pg_constraint
where conname in ('pyq_source_check', 'pyq_mock_position_check', 'pyq_mock_name_check');

-- Expect 0 (no row missed the backfill).
select count(*) as missing_source from public.pyq where source is null;

-- Expect the new definition to contain "source = 'pyq'" (only if PART 2 ran).
select pg_get_functiondef('public.match_pyq(vector, double precision, integer)'::regprocedure) ilike '%source = ''pyq''%' as match_pyq_filters_source;
