-- Adds the column needed for a rolling 24h cooldown, anchored to the moment a student's daily
-- token budget is actually crossed -- instead of the old behaviour of resetting at a fixed IST
-- midnight boundary (which meant the wait could be anywhere from 1 minute to ~24 hours depending
-- on what time of day the limit was hit).
--
-- Purely additive: no existing column, constraint, or row is touched. Safe to run once in the
-- Supabase SQL editor.

ALTER TABLE usage_log ADD COLUMN IF NOT EXISTS limit_reached_at timestamptz;
ALTER TABLE guest_usage_log ADD COLUMN IF NOT EXISTS limit_reached_at timestamptz;
