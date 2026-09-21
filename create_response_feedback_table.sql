-- Thumbs up/down on any AI chat response (chat.html's message toolbar, next to Copy). Purely a
-- review log for Mustafa to spot response-quality patterns later -- no feature reads this table
-- back today, so it's kept deliberately small: a snapshot of the question/answer text at the time
-- of the vote (not a live join to `chats`, since retry/edit can change that message's content
-- later), the rating, and an optional "what was wrong?" comment for a thumbs-down.
--
-- Deliberately no FK to chats.id -- that table predates this repo's create-table-via-SQL-file
-- convention and its schema isn't declared anywhere in SQL, so chat_id is stored as a plain
-- reference-only uuid (nullable: best-effort, in the rare case a vote is cast before the chat's
-- own Supabase row exists yet).
create table if not exists public.response_feedback (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  chat_id uuid,
  msg_index int not null,
  question_text text,
  answer_text text,
  rating text not null check (rating in ('up', 'down')),
  comment text,
  created_at timestamptz not null default now()
);

create index if not exists idx_response_feedback_user_id on public.response_feedback (user_id);
create index if not exists idx_response_feedback_created_at on public.response_feedback (created_at);

-- chat.html writes this directly with the anon-key client (matching saved_questions' existing
-- client-direct pattern, rather than a new FastAPI endpoint for something this small) -- RLS scopes
-- every student to their own rows for insert/select/update (switching up<->down, adding a comment
-- after the fact) and delete (toggling a vote off removes the row).
alter table public.response_feedback enable row level security;

create policy "Users can insert their own feedback" on public.response_feedback
  for insert to authenticated with check (auth.uid() = user_id);

create policy "Users can view their own feedback" on public.response_feedback
  for select to authenticated using (auth.uid() = user_id);

create policy "Users can update their own feedback" on public.response_feedback
  for update to authenticated using (auth.uid() = user_id);

create policy "Users can delete their own feedback" on public.response_feedback
  for delete to authenticated using (auth.uid() = user_id);
