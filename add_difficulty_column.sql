-- Adds a permanent difficulty classification (Easy/Moderate/Difficult) to pyq and
-- mock_test_questions, computed once by DeepSeek-V4-Flash and cached forever --
-- server.py only classifies a row when this column reads null, never re-classifies
-- an already-set row. Nullable so existing rows start unclassified and get filled in
-- lazily (on first display) or via the one-time /admin/backfill-difficulty sweep.
-- Safe to re-run -- ADD COLUMN IF NOT EXISTS avoids duplicate-column errors.

alter table public.pyq
  add column if not exists difficulty text
  check (difficulty is null or difficulty in ('Easy', 'Moderate', 'Difficult'));

alter table public.mock_test_questions
  add column if not exists difficulty text
  check (difficulty is null or difficulty in ('Easy', 'Moderate', 'Difficult'));
