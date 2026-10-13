-- Lets POST /account/delete (server.py) ANONYMISE a deleted student's rows in these tables
-- instead of deleting them: their user id is set to NULL, the row itself stays.
--   referrals.referrer_id / referred_id -- keeps the OTHER student's referral history and
--     completed-referral count intact when one side of a referral deletes their account.
--   diagram_reports.user_id / question_reports.user_id -- keeps the content-quality signal
--     (which diagrams/questions were reported, and why) after the reporter is gone.
-- All four columns were NOT NULL, so the anonymise step would fail without this.
--
-- Safe to run more than once. No data changes -- only drops the NOT NULL constraints.
-- referrals.referred_id's UNIQUE constraint is unaffected: Postgres allows any number of NULLs
-- in a unique column, so several anonymised rows can coexist.
--
-- Run BEFORE deploying the /account/delete endpoint -- until then every deletion aborts at its
-- first step (anonymising), before any of the student's data has been deleted.

alter table public.referrals alter column referrer_id drop not null;
alter table public.referrals alter column referred_id drop not null;
alter table public.diagram_reports alter column user_id drop not null;
alter table public.question_reports alter column user_id drop not null;
