-- Session-level feedback: the one-line "How is NEET-AI doing this session?" strip above chat.html's
-- input (Bad / Fine / Good, optional "What went wrong?" comment on Bad). Reviewed in
-- admin-dashboard.html's Feedback tab under "Session ratings".
--
-- Deliberately no conversation text: only the rating, the optional comment, which chat was open
-- (reference-only uuid, same no-FK reasoning as response_feedback.chat_id) and how many AI replies
-- the student had received in that page session.
--
-- chat.html writes through server.py's POST /session-feedback (service role; user taken from the
-- verified JWT, one write per user per hour). It reads the student's own latest row directly with
-- the anon key to respect the 14-day "already rated" cooldown across devices.
create table if not exists public.session_feedback (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  rating text not null check (rating in ('bad', 'fine', 'good')),
  comment text check (comment is null or char_length(comment) <= 500),
  chat_id uuid,
  ai_reply_count int not null default 0 check (ai_reply_count >= 0),
  created_at timestamptz not null default now()
);

-- "Latest rating for this student" (cooldown check + server rate limit) and the admin list
-- ("newest first, optionally filtered by rating").
create index if not exists idx_session_feedback_user_created on public.session_feedback (user_id, created_at desc);
create index if not exists idx_session_feedback_rating_created on public.session_feedback (rating, created_at desc);

-- Students can insert and read only their own rows. No update/delete policy exists, so both are
-- denied for client roles; the privileges are revoked too so that holds even if a permissive
-- policy is added by mistake later. The service-role key used by server.py bypasses RLS.
alter table public.session_feedback enable row level security;

drop policy if exists "Students can insert their own session feedback" on public.session_feedback;
create policy "Students can insert their own session feedback" on public.session_feedback
  for insert to authenticated with check (auth.uid() = user_id);

drop policy if exists "Students can view their own session feedback" on public.session_feedback;
create policy "Students can view their own session feedback" on public.session_feedback
  for select to authenticated using (auth.uid() = user_id);

revoke all on public.session_feedback from anon;
revoke update, delete, truncate on public.session_feedback from authenticated;

-- A direct anon-key insert could otherwise backdate or future-date created_at, which would skew
-- that student's own cooldown and the admin ordering. Pin it for client roles (same
-- current_user check as response_feedback_guard_fields).
create or replace function public.session_feedback_guard_fields()
returns trigger
language plpgsql
as $$
begin
  if current_user in ('authenticated', 'anon') then
    new.created_at := now();
  end if;
  return new;
end;
$$;

drop trigger if exists session_feedback_guard_fields on public.session_feedback;
create trigger session_feedback_guard_fields
  before insert on public.session_feedback
  for each row execute function public.session_feedback_guard_fields();
