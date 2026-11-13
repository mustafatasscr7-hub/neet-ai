-- Fixed, saved mock exams for the student mock test page (2026-11-13): lets an admin publish one
-- mock_name (pyq.source = 'mock', see mock_exam_columns.sql) as a real test students can take,
-- graded through the EXISTING server-side mock-tests pipeline (POST /mock-tests/attempts +
-- /submit) instead of a parallel one.
--
--   mock_exams          one row per mock_name, exactly as specified: the publish switch. false
--                       until an admin flips it on from admin-pyq-preview.html's new Mocks panel,
--                       which only allows that once the mock is "ready" (exactly 180 approved
--                       questions, positions 1-180, no gaps/duplicates, every one has an answer --
--                       checked server-side at publish time, not just in the UI). RLS enabled, no
--                       policies: service-role (server.py/ADMIN_HEADERS) only, same convention as
--                       mock_test_attempts below.
--
-- The two ALTERs below are the minimal additive wiring needed to serve a fixed mock through the
-- existing grading code without copying it (mock_test_submit's _grade_attempt etc. stay
-- untouched):
--   mock_test_attempts  gets a new allowed `kind` value, 'fixed_mock' (widening the existing CHECK,
--                       nothing removed), and a nullable `mock_name` column (FK to mock_exams) so
--                       a fixed-mock attempt can be told apart from a random/official/personalised
--                       one and matched back to its mock.
--   mock_results        gets the same nullable `mock_name` column, written at grading time, so
--                       GET /mock-exams can list a student's past attempts per mock the same way
--                       /mock-tests/available already does via mock_test_id. Unpublishing a mock
--                       later never touches these rows (ON DELETE SET NULL, and nothing here ever
--                       deletes a mock_exams row anyway) -- a student's existing attempts/results
--                       stay exactly as they were.
--
-- Purely additive: no existing column, constraint, or row is touched other than widening the
-- `kind` check on mock_test_attempts. Safe to run once in the Supabase SQL editor.
--
-- HOW TO RUN (Supabase dashboard -> SQL Editor):
--   1. Paste and run PART 1 on its own. It's one transaction: all of it applies or none of it does.
--   2. Run PART 2 (checks) and compare with the expected results written there.
--   3. Say "sql done"; only then does the rest of this feature continue.

-- ============================== PART 1: table + columns + constraints ==============================
begin;

create table if not exists public.mock_exams (
  mock_name text primary key,
  title text,
  published boolean not null default false,
  published_at timestamptz,
  created_at timestamptz default now()
);

alter table public.mock_exams enable row level security;
-- No policies at all, for either role -- this is a global admin-config table, not a per-user one,
-- so there's no "read your own row" case the way mock_test_attempts has.
revoke all on public.mock_exams from anon, authenticated;

alter table public.mock_test_attempts drop constraint if exists mock_test_attempts_kind_check;
alter table public.mock_test_attempts add constraint mock_test_attempts_kind_check
  check (kind in ('random', 'official', 'personalised', 'personalised_catalog', 'fixed_mock'));

alter table public.mock_test_attempts add column if not exists mock_name text
  references public.mock_exams(mock_name) on delete set null;

alter table public.mock_results add column if not exists mock_name text
  references public.mock_exams(mock_name) on delete set null;

create index if not exists idx_mock_test_attempts_mock_name on public.mock_test_attempts (mock_name) where mock_name is not null;
create index if not exists idx_mock_results_user_mock_name on public.mock_results (user_id, mock_name) where mock_name is not null;

commit;

-- ============================== PART 2: checks ==============================
-- Expect this to insert and read back cleanly, then disappear.
insert into public.mock_exams (mock_name, title) values ('__mock_exams_sql_check__', 'check') returning mock_name, published, created_at;
delete from public.mock_exams where mock_name = '__mock_exams_sql_check__';

-- Expect the definition to now list all five kinds, 'fixed_mock' included.
select pg_get_constraintdef(oid) from pg_constraint where conname = 'mock_test_attempts_kind_check';

-- Expect one row each.
select table_name, column_name from information_schema.columns
where table_name in ('mock_test_attempts', 'mock_results') and column_name = 'mock_name';

-- Expect 0 -- RLS on, no policies for mock_exams (this editor session still reads/writes it above
-- since it runs as the postgres superuser, which bypasses RLS regardless).
select count(*) from pg_policies where tablename = 'mock_exams';
