-- Reject (error) flow for PYQ admin review (2026-11-01). A rejected question stays in the table
-- as proof of what the contractor submitted -- never deleted -- but is hidden from every student
-- read the same way 'pending'/'needs_fix' already are (review_status <> 'approved'), and is
-- deducted from the contractor's payable total.
--
--   reject_reason   preset label (optionally ": <admin's note>"), e.g. "Garbled text: OCR ate half
--                   the options". NULL unless review_status = 'rejected'.
--   rejected_at     when the Reject action was taken. NULL unless review_status = 'rejected'.
--
-- server.py's PYQ_STUDENT_VISIBLE filter (review_status = 'approved') already excludes a rejected
-- row from every student-facing read without any further change -- 'rejected' is just a third
-- value alongside 'pending'/'needs_fix' that isn't 'approved'.
--
-- HOW TO RUN (Supabase dashboard -> SQL Editor):
--   1. Paste and run PART 1 on its own. It's one transaction: all of it applies or none of it does.
--   2. Run PART 2 (checks) and compare with the expected results written there.
--   3. Say it's done; only then are the admin/contractor pages' Reject buttons safe to click
--      (the server already returns a clear error instead of crashing if this hasn't run yet).

-- ============================== PART 1: columns + constraint ==============================
begin;

alter table public.pyq add column if not exists reject_reason text;
alter table public.pyq add column if not exists rejected_at timestamptz;

alter table public.pyq drop constraint if exists pyq_review_status_check;
alter table public.pyq add constraint pyq_review_status_check
  check (review_status in ('pending', 'approved', 'needs_fix', 'rejected'));

commit;

-- ============================== PART 2: checks ==============================
-- Expect every existing row's status still in the original three values (none are 'rejected' yet).
select review_status, count(*) from public.pyq group by review_status;

-- Expect the constraint definition to mention 'rejected'.
select pg_get_constraintdef(oid) from pg_constraint where conname = 'pyq_review_status_check';

-- Expect 0 (no rejected rows exist until the first Reject click).
select count(*) as rejected_count from public.pyq where review_status = 'rejected';
