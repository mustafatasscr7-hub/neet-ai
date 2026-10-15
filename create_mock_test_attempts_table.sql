-- Server-side mock-test attempts (secure grading path, 2026-10-15). See MOCKTEST_GRADING_NOTES.md.
--
-- POST /mock-tests/attempts records one row per started test: who, which questions (in the order
-- served), when it started (server clock) and how long it may run. POST
-- /mock-tests/attempts/{id}/submit claims the row once (submitted_at, the student's answers,
-- is_late, and a pre-assigned result_id), grades on the server and writes the normal mock_results
-- row under that result_id -- which is what makes a repeat submit return the same result instead
-- of a duplicate. Only server.py (service-role key) ever writes here.
create table if not exists public.mock_test_attempts (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  kind text not null check (kind in ('random', 'official', 'personalised', 'personalised_catalog')),
  mock_test_id bigint references public.mock_tests(id) on delete set null,
  -- pyq ids (uuid text) or mock_test_questions ids (bigint as text), in the order served.
  question_ids text[] not null check (cardinality(question_ids) between 1 and 300),
  duration_seconds integer not null check (duration_seconds > 0),
  started_at timestamptz not null default now(),
  submitted_at timestamptz,
  answers jsonb,          -- {question_id: 'a'..'d'} as submitted; fixed once submitted
  is_late boolean,        -- submitted after started_at + duration + MOCK_TEST_SUBMIT_GRACE_SECONDS
  result_id uuid unique,  -- mock_results.id for this attempt (no FK: assigned just before the row exists)
  created_at timestamptz not null default now()
);

create index if not exists idx_mock_test_attempts_user_started on public.mock_test_attempts (user_id, started_at desc);

-- Students may read their own attempts and nothing else; all writes go through server.py.
alter table public.mock_test_attempts enable row level security;

drop policy if exists "Students can view their own mock test attempts" on public.mock_test_attempts;
create policy "Students can view their own mock test attempts" on public.mock_test_attempts
  for select to authenticated using (auth.uid() = user_id);

revoke all on public.mock_test_attempts from anon;
revoke insert, update, delete, truncate on public.mock_test_attempts from authenticated;

-- Close the answer-key hole on official mock tests: this policy let anyone holding the page's
-- anon key read every column of a published test's questions, correct_answer included.
-- server.py now reads mock_test_questions only with the service-role key
-- (/mock-tests/{id}/questions, /mock-tests/attempts, /classify-difficulty, admin tools), and no web
-- page or app screen queries this table directly, so nothing else depends on the policy.
drop policy if exists "mock_test_questions_public_select" on public.mock_test_questions;
