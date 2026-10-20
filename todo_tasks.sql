-- Study To-do list (Pro/Max feature). To apply: run this whole file once in the Supabase SQL
-- editor. Not run automatically -- nothing in server.py depends on it being applied yet; the
-- /todos endpoints will simply fail with a 502 ("Could not load/create/update/delete your task")
-- until it is.
--
-- RLS is enabled with NO policies, same lockdown as the other student tables (chats,
-- saved_questions, etc.): every real read/write goes through server.py using the service-role
-- key, which bypasses RLS entirely, so the student's own anon-key session can never read or
-- write this table directly.

create table if not exists todo_tasks (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null,
  title text not null check (char_length(title) <= 200),
  due_date date null,
  done boolean not null default false,
  created_at timestamptz not null default now(),
  done_at timestamptz null
);

create index if not exists todo_tasks_user_done_due_idx
  on todo_tasks (user_id, done, due_date);

alter table todo_tasks enable row level security;
