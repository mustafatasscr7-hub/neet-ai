-- Permanent cache for /solve: a PYQ's correct solution never changes once generated, so it's
-- keyed on the question's own row id (not a text hash) plus language, and never expires.
--
-- Purely additive: no existing table/column/row is touched. Safe to run once in the Supabase
-- SQL editor.

CREATE TABLE IF NOT EXISTS pyq_solution_cache (
  pyq_id text NOT NULL,
  language text NOT NULL DEFAULT 'en',
  solution text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (pyq_id, language)
);

-- No policies added -- locked down for anon/authenticated by default. The backend only ever
-- reads/writes this table with the service-role key, which bypasses RLS entirely.
ALTER TABLE pyq_solution_cache ENABLE ROW LEVEL SECURITY;
