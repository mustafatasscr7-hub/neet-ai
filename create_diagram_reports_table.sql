-- Lets students flag a problem with a diagram (wrong image, mislabeled, low quality, incorrect
-- content) from wherever it's shown -- chat.html's auto-embedded/button diagrams and the Diagram
-- Library, the two places a standalone `diagrams` row is displayed. Deliberately a SEPARATE table
-- from question_reports rather than reusing it: question_reports.pyq_id is a uuid tied to the
-- `pyq` table, while diagrams.id is a bigint identity column from a completely different table --
-- there's no clean way to overload one column for both without a nullable-either-way redesign of
-- an already-working table. A question's own attached diagram_url (PYQ Bank, mock tests, saved
-- questions, etc.) is a different case entirely -- that's already reportable today via
-- question_reports' existing reason='diagram_issue', since that diagram IS part of the question
-- row, not a standalone diagrams entry.
create table if not exists public.diagram_reports (
  id uuid primary key default gen_random_uuid(),
  diagram_id bigint not null references public.diagrams(id) on delete cascade,
  user_id uuid not null,
  source text not null check (source in ('chat', 'library')),
  reason text not null check (reason in ('wrong_image', 'mislabeled', 'low_quality', 'incorrect_content', 'other')),
  optional_note text,
  created_at timestamptz not null default now(),
  resolved boolean not null default false
);

create index if not exists idx_diagram_reports_diagram_id on public.diagram_reports (diagram_id);
create index if not exists idx_diagram_reports_resolved on public.diagram_reports (resolved);

-- Same lock-down as diagrams itself (create_diagrams_table.sql) -- all reads/writes go through
-- server.py's service-role key (ADMIN_HEADERS), never the anon key directly. Unlike diagrams,
-- there's no public-select policy here either: report rows carry a student's user_id and
-- freeform note text, which has no reason to be anon-readable the way the diagram catalog itself
-- does.
alter table public.diagram_reports enable row level security;
