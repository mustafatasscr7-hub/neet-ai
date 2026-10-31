-- Chapter-level signal of where students raise doubts from the Question Bank, saved for later use
-- by the personalised test (nothing reads this table yet). To apply: run this whole file once in
-- the Supabase SQL editor. Not run automatically -- until then, POST /me/doubt-topic still returns
-- 204 but the insert inside it just fails silently, same convention as the other /me endpoints.
--
-- No doubt text or answer text is ever stored here, only subject/chapter/class. RLS is enabled with
-- NO policies, same lockdown as the other student tables (chats, saved_questions, todo_tasks,
-- etc.): every real read/write goes through server.py using the service-role key, which bypasses
-- RLS entirely, so the student's own anon-key session can never read or write this table directly.

create table if not exists doubt_topics (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null,
  subject text not null,
  chapter text not null,
  class text null,
  created_at timestamptz not null default now()
);

create index if not exists doubt_topics_user_created_idx
  on doubt_topics (user_id, created_at desc);

alter table doubt_topics enable row level security;
