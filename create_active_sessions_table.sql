-- Backs device-limit enforcement (Pro = 1 device, Max = 2, Free = unenforced): one row per
-- (user, device) the student has ever logged into chat.html from. "Active" is determined at
-- query time in application code (last_active_at within a rolling window), not a boolean column
-- here, so a session naturally drops out of the count once the device goes idle/closes the tab --
-- no explicit logout tracking needed.
--
-- kick_grace_deadline/kicked_at implement the warning-then-kick flow: when a new login would
-- exceed the plan's device limit, the OLDEST other active session gets a kick_grace_deadline
-- (now + grace period) instead of being kicked immediately. Any subsequent heartbeat from that
-- device (or a fresh check from elsewhere) that finds `now() > kick_grace_deadline` sets
-- kicked_at and the device is told to log out, one time, on its next check-in -- this is
-- deliberately lazy/on-check-in rather than a background job, matching this app's existing
-- pattern of not running scheduled jobs.
--
-- Purely additive: no existing table/column/row is touched. Safe to run once in the Supabase
-- SQL editor.

CREATE TABLE IF NOT EXISTS active_sessions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  device_id text NOT NULL,
  device_label text,
  created_at timestamptz NOT NULL DEFAULT now(),
  last_active_at timestamptz NOT NULL DEFAULT now(),
  kick_grace_deadline timestamptz,
  kicked_at timestamptz,
  UNIQUE (user_id, device_id)
);

CREATE INDEX IF NOT EXISTS active_sessions_user_id_idx ON active_sessions (user_id);

-- No policies added -- locked down for anon/authenticated by default. The backend only ever
-- reads/writes this table with the service-role key, which bypasses RLS entirely.
ALTER TABLE active_sessions ENABLE ROW LEVEL SECURITY;
