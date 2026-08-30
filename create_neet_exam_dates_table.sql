-- Backs the pricing page's NEET-year selector (date-based annual pricing, pre-Razorpay). One
-- row per NEET exam year; exam_date starts NULL ("TBD") for every year until Mustafa fills in
-- the real, NTA-confirmed date -- deliberately never fabricated/guessed here, since NTA can (and
-- has) shifted announced dates. Rows can be added/edited freely later for new years; nothing in
-- the app hardcodes which years exist, the frontend always reads this table directly.
create table if not exists public.neet_exam_dates (
  year integer primary key,
  exam_date date,  -- NULL = not yet announced ("TBD"); real date only, never a placeholder guess
  updated_at timestamptz not null default now()
);

-- Public read (anon) so the pricing page's year selector works for a logged-out visitor same as
-- everything else on that page -- writes go through server.py's service-role key only (see
-- /admin/set-neet-exam-date), same pattern as every other admin-owned table in this project.
alter table public.neet_exam_dates enable row level security;
drop policy if exists "neet_exam_dates_public_select" on public.neet_exam_dates;
create policy "neet_exam_dates_public_select" on public.neet_exam_dates
  for select to anon, authenticated using (true);

-- Seed placeholder rows, both TBD -- update the real exam_date for these (or add further years)
-- via POST /admin/set-neet-exam-date once NTA announces it. Picked 2027/2028 as forward-looking
-- starting rows purely because this migration is being run in mid/late-2026; delete or ignore
-- whichever doesn't end up mattering.
insert into public.neet_exam_dates (year, exam_date) values (2027, null), (2028, null)
  on conflict (year) do nothing;
