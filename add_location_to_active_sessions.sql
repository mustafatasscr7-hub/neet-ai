-- Adds a `location` column to the active_sessions table (built for the device-limit feature)
-- to back the new "Active Sessions" list in chat.html's Account settings tab.
--
-- Location is a plain text label (e.g. "Mumbai, Maharashtra" or "Unknown"), resolved once via a
-- free IP-geolocation lookup at session-creation/relogin time and cached here -- never looked up
-- on every page load, since it never needs to change while a session stays alive.
--
-- Purely additive: no existing row/column is touched. Safe to run once in the Supabase SQL editor.

ALTER TABLE active_sessions ADD COLUMN IF NOT EXISTS location text;
