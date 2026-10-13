-- Admin review fields for response_feedback (Feedback tab in admin-dashboard.html, 2026-10-13).
--
-- Extends the existing table rather than creating a new one, so every vote already collected
-- stays reviewable and the mobile app (which still writes this table directly with the anon key,
-- see neet-ai-mobile/src/lib/chatStore.tsx) keeps working unchanged.
--
--   reviewed / reviewed_at / admin_note  -- set only by POST /admin/feedback/{id}/review
--   model        -- provider that generated the rated reply, inferred server-side from
--                   provider_usage_log at vote time (null when it couldn't be matched)
--   text_source  -- 'server': question/answer copied by POST /feedback from the stored chats row;
--                   'client': text as sent by the app (direct anon-key insert, or the chat row
--                   wasn't saved yet). null = rows from before this migration (all client-sent).
alter table public.response_feedback
  add column if not exists reviewed boolean not null default false,
  add column if not exists reviewed_at timestamptz,
  add column if not exists admin_note text,
  add column if not exists model text,
  add column if not exists text_source text check (text_source in ('server', 'client'));

-- Admin list is always "newest first, optionally filtered by rating/reviewed".
create index if not exists idx_response_feedback_rating_reviewed_created
  on public.response_feedback (rating, reviewed, created_at desc);

-- The existing per-user RLS policies stay (the mobile app still inserts/updates/deletes its own
-- rows directly), which means a student could otherwise write the admin/server-owned columns on
-- their own rows through the anon-key API -- e.g. insert with reviewed=true, or edit the
-- question/answer snapshot after the fact. This trigger pins those columns whenever the write
-- comes from a student session (role authenticated/anon); the service-role key used by
-- server.py is unaffected.
create or replace function public.response_feedback_guard_fields()
returns trigger
language plpgsql
as $$
begin
  if current_user in ('authenticated', 'anon') then
    if tg_op = 'INSERT' then
      new.reviewed := false;
      new.reviewed_at := null;
      new.admin_note := null;
      new.model := null;
      new.text_source := 'client';
    else
      new.reviewed := old.reviewed;
      new.reviewed_at := old.reviewed_at;
      new.admin_note := old.admin_note;
      new.model := old.model;
      new.text_source := old.text_source;
      new.user_id := old.user_id;
      new.chat_id := old.chat_id;
      new.msg_index := old.msg_index;
      new.question_text := old.question_text;
      new.answer_text := old.answer_text;
      new.created_at := old.created_at;
    end if;
  end if;
  return new;
end;
$$;

drop trigger if exists response_feedback_guard_fields on public.response_feedback;
create trigger response_feedback_guard_fields
  before insert or update on public.response_feedback
  for each row execute function public.response_feedback_guard_fields();
