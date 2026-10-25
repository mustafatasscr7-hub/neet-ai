-- Lets students flag a problem with a question inside an official mock test. Deliberately a
-- SEPARATE table from question_reports rather than reusing it: question_reports.pyq_id is a uuid
-- tied to the `pyq` table, while mock_test_questions.id is a bigint identity column from a
-- completely different table -- same reasoning as create_diagram_reports_table.sql, there's no
-- clean way to overload one column for both without a nullable-either-way redesign of an
-- already-working table. Unlike question_reports, there's no deactivate-on-report step for these:
-- official mock tests are fixed papers, not a pool /mock-test-questions filters live, so there's
-- nothing to pull out of circulation the way pyq.is_active does.
create table if not exists public.mock_test_question_reports (
  id uuid primary key default gen_random_uuid(),
  question_id bigint not null references public.mock_test_questions(id) on delete cascade,
  user_id uuid not null,
  reason text not null check (reason in ('wrong_answer', 'unclear', 'diagram_issue', 'duplicate', 'other', 'solution_wrong')),
  optional_note text,
  solution_open_at_report boolean not null default false,
  created_at timestamptz not null default now(),
  resolved boolean not null default false
);

create index if not exists idx_mock_test_question_reports_question_id on public.mock_test_question_reports (question_id);
create index if not exists idx_mock_test_question_reports_resolved on public.mock_test_question_reports (resolved);
create index if not exists mock_test_question_reports_created_at_idx on public.mock_test_question_reports (created_at);

-- Same lock-down as question_reports/diagram_reports -- all reads/writes go through server.py's
-- service-role key (ADMIN_HEADERS), never the anon key directly. No public-select policy: report
-- rows carry a student's user_id and freeform note text.
alter table public.mock_test_question_reports enable row level security;