-- Contractor upload review (2026-10-18). Contractor PYQ uploads stay hidden from students until
-- an admin approves them in admin-pyq-preview.html.
--
--   uploaded_by    'admin' | 'contractor' -- set by the server from the key used, never the browser.
--                  NULL for rows from before this migration (who uploaded them isn't recorded).
--   uploaded_at    when the row was uploaded (backfilled from created_at for existing rows).
--   batch_id       one id per upload session (a PDF's review). NULL for existing rows.
--   review_status  'pending' | 'approved' | 'needs_fix'. Existing rows -> approved. New rows
--                  default to pending, so an insert that forgets to set it stays hidden.
--   review_note    the admin's "what to fix" note.
--   reviewed_at / reviewed_by   when (and by whom) the last Approve / Needs fix happened.
--
-- server.py filters every student read on review_status = 'approved' (PYQ_STUDENT_VISIBLE), and
-- would 400 on every student PYQ query if that column didn't exist -- so this runs BEFORE the
-- server change is deployed.
--
-- HOW TO RUN (Supabase dashboard -> SQL Editor):
--   1. Paste and run PART 1 on its own. It's one transaction: all of it applies or none of it does.
--   2. Paste and run PART 2 on its own (match_pyq). If it errors with "cannot change return type
--      of existing function", skip it and send the error over -- the server already drops
--      non-approved rows from search results itself, so students stay safe either way.
--   3. Run PART 3 (checks) and compare with the expected results written there.
--   4. Say it's done; only then is the server change pushed.

-- ============================== PART 1: columns ==============================
begin;

alter table public.pyq add column if not exists uploaded_by text;
alter table public.pyq add column if not exists uploaded_at timestamptz;
alter table public.pyq add column if not exists batch_id text;
-- Constant default: every existing row reads as 'approved' instantly (no table rewrite).
alter table public.pyq add column if not exists review_status text not null default 'approved';
alter table public.pyq add column if not exists review_note text;
alter table public.pyq add column if not exists reviewed_at timestamptz;
alter table public.pyq add column if not exists reviewed_by text;

update public.pyq set uploaded_at = coalesce(created_at, now()) where uploaded_at is null;

-- From here on, new rows start hidden unless the insert says otherwise.
alter table public.pyq alter column review_status set default 'pending';
alter table public.pyq alter column uploaded_at set default now();
alter table public.pyq alter column uploaded_at set not null;

alter table public.pyq drop constraint if exists pyq_review_status_check;
alter table public.pyq add constraint pyq_review_status_check
  check (review_status in ('pending', 'approved', 'needs_fix'));
alter table public.pyq drop constraint if exists pyq_uploaded_by_check;
alter table public.pyq add constraint pyq_uploaded_by_check
  check (uploaded_by is null or uploaded_by in ('admin', 'contractor'));

-- The review queue (small: only non-approved rows), today's counts, and the batch filter.
create index if not exists pyq_review_queue_idx on public.pyq (review_status, uploaded_at desc)
  where review_status <> 'approved';
create index if not exists pyq_uploaded_at_idx on public.pyq (uploaded_at desc);
create index if not exists pyq_reviewed_at_idx on public.pyq (reviewed_at) where reviewed_at is not null;
create index if not exists pyq_batch_id_idx on public.pyq (batch_id) where batch_id is not null;

commit;

-- ============================== PART 2: match_pyq ==============================
-- Same signature and columns as add_diagram_url_to_match_pyq.sql, plus two filters: approved
-- rows only, and active rows only (the chat search never checked is_active before; the server
-- now does too). CREATE OR REPLACE (no DROP): if the live function's return columns differ from
-- these, Postgres refuses and nothing changes.
create or replace function match_pyq(
  query_embedding vector(1536),
  match_threshold float,
  match_count int
)
returns table (
  id uuid,
  year int,
  subject text,
  question text,
  option_a text,
  option_b text,
  option_c text,
  option_d text,
  correct_answer text,
  diagram_url text,
  has_diagram boolean,
  similarity float
)
language sql stable
as $$
  select
    pyq.id,
    pyq.year,
    pyq.subject,
    pyq.question,
    pyq.option_a,
    pyq.option_b,
    pyq.option_c,
    pyq.option_d,
    pyq.correct_answer,
    pyq.diagram_url,
    pyq.has_diagram,
    1 - (pyq.embedding <=> query_embedding) as similarity
  from pyq
  where pyq.review_status = 'approved'
    and pyq.is_active
    and 1 - (pyq.embedding <=> query_embedding) > match_threshold
  order by pyq.embedding <=> query_embedding
  limit match_count;
$$;

-- ============================== PART 3: checks ==============================
-- Expect exactly one row: approved | <your total PYQ count>.
select review_status, count(*) from public.pyq group by review_status;

-- Expect: review_status -> 'pending'::text, uploaded_at -> now().
select column_name, column_default
from information_schema.columns
where table_schema = 'public' and table_name = 'pyq'
  and column_name in ('review_status', 'uploaded_at');

-- Expect 0 (every existing row got an upload time).
select count(*) as missing_uploaded_at from public.pyq where uploaded_at is null;

-- Expect the new definition to contain "review_status = 'approved'" (only if PART 2 ran).
select pg_get_functiondef('public.match_pyq(vector, double precision, integer)'::regprocedure) ilike '%review_status = ''approved''%' as match_pyq_filters_review;
