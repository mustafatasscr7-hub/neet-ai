-- Records whether the student had the solution panel open at the moment they filed a report --
-- lets admin-pyq-preview.html's Reports view tell "reported the explanation itself" (new
-- reason='solution_wrong', only offered client-side while Hide Solution is showing) apart from a
-- report filed before ever opening the solution, without needing a second table or endpoint.
-- Nullable/defaulted false rather than required: existing rows predate this column and never
-- captured the panel's state, so they default to false (a question reported with the panel never
-- opened) which is the correct, truthful value for all of them, not an unknown placeholder.
-- Safe to re-run -- ADD COLUMN IF NOT EXISTS avoids duplicate-column errors.

alter table public.question_reports
  add column if not exists solution_open_at_report boolean not null default false;
