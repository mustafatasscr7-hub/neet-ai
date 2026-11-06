-- Auto-approve trial/real mode for PYQ contractor review (2026-11-02). Trial mode (the default --
-- AUTO_APPROVE_ENABLED unset/false in Railway) only ever flags a pending row with what
-- server.py's auto_check found; nothing leaves pending and students see no change at all.
-- Setting AUTO_APPROVE_ENABLED=true turns on the real switch: a row that passes at check time
-- approves itself, except a random 10% kept back (auto_sampled) for a human spot-check.
--
--   auto_pass        auto_check's verdict the last time this row was swept. NULL until the first
--                     sweep (/admin/pyq-auto-check) or contractor upload reaches it.
--   auto_reasons      comma-separated reason codes auto_check failed on (NULL when auto_pass is
--                     true). Codes: options, broken_text, latex, subject_chapter, duplicate,
--                     diagram, match_following, math_formula, chemical_formula, rejected.
--   auto_checked_at   when auto_check last ran on this row. NULL = never checked -- the sweep
--                     only ever picks up rows where this is still NULL.
--   auto_approved     true only for a row this feature itself approved (reviewed_by = 'auto' or
--                     'auto-bulk'), never set by a human Approve click.
--   auto_sampled      true for the random 10% that passed auto_check while AUTO_APPROVE_ENABLED
--                     was on, but were deliberately left pending for a human spot-check instead
--                     of being auto-approved.
--
-- HOW TO RUN (Supabase dashboard -> SQL Editor):
--   1. Paste and run PART 1 on its own. It's one transaction: all of it applies or none of it does.
--   2. Run PART 2 (checks) and compare with the expected results written there.
--   3. Say it's done; only then do /admin/pyq-auto-check, -auto-approve-all, -unapprove and
--      -auto-stats stop returning their "run pyq_auto_approve.sql first" error (upload and
--      review already work fine either way -- they never depended on this migration).

-- ============================== PART 1: columns + index ==============================
begin;

alter table public.pyq add column if not exists auto_pass boolean;
alter table public.pyq add column if not exists auto_reasons text;
alter table public.pyq add column if not exists auto_checked_at timestamptz;
alter table public.pyq add column if not exists auto_approved boolean default false;
alter table public.pyq add column if not exists auto_sampled boolean default false;

create index if not exists idx_pyq_review_status_auto_pass on public.pyq (review_status, auto_pass);

commit;

-- ============================== PART 2: checks ==============================
-- Expect 0 for both -- nothing has been swept yet.
select count(*) as already_checked from public.pyq where auto_checked_at is not null;
select count(*) as already_auto_approved from public.pyq where auto_approved is true;

-- Expect one row back, confirming the index exists.
select indexname from pg_indexes where tablename = 'pyq' and indexname = 'idx_pyq_review_status_auto_pass';
