-- Cross-device copy of two Settings controls (Personalize my AI tutor, Answer Style). To apply:
-- run this whole file once in the Supabase SQL editor; safe to run twice (create table if not
-- exists). Not run automatically -- until then, server.py's /me/settings endpoints simply report
-- {"available": false} and the website falls back to browser-only behaviour (see settings.js/
-- chat.html's sync code), so nothing breaks before this is applied.
--
-- RLS is enabled with NO policies, same lockdown as todo_tasks/chats/saved_questions: every real
-- read/write goes through server.py using the service-role key, which bypasses RLS entirely, so a
-- student's own anon-key session can never read or write this table directly.

create table if not exists public.user_settings (
  user_id uuid primary key references auth.users(id) on delete cascade,
  personalize boolean null,
  answer_style text null check (answer_style is null or answer_style in ('concise', 'medium', 'detailed')),
  updated_at timestamptz not null default now()
);

alter table public.user_settings enable row level security;
