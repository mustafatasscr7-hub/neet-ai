-- Adds the "Show key points" setting (default ON) to the existing user_settings table (see
-- user_settings.sql). ADD COLUMN IF NOT EXISTS -- safe to run anytime, including more than once.
--
-- Not required before deploying: until this is run, server.py's GET /me/settings falls back to
-- a select without key_points (reporting it as null, same as an unset row) and PUT drops it from
-- the upsert and retries with whatever other fields were sent -- so personalize/answer_style sync
-- keeps working unaffected, and the student's key_points choice just stays local-only
-- (localStorage) on each device until this is applied.

alter table public.user_settings add column if not exists key_points boolean null;
