-- Widens provider_usage_log's endpoint CHECK constraint to also allow '/ensure-correct-answer' --
-- same gap, same fix shape as alter_provider_usage_log_add_classify_difficulty_endpoint.sql did
-- for '/classify-difficulty'. solve_correct_answer() has always made a real, unlogged DeepSeek
-- call from both /ensure-correct-answer (live, triggered by a student clicking a blank-answer
-- PYQ's option) and /admin/pyq-backfill-correct-answer (the one-time sweep) -- this migration is
-- what lets the code fix that now logs both of those actually succeed at inserting.
--
-- A distinct tag from '/classify-difficulty' (not reused) -- unlike that fix's own live+admin
-- pair, which share a tag because they're the exact same operation triggered two ways, resolving
-- a correct_answer is a genuinely different operation from rating difficulty and deserves its own
-- bucket in cost analytics.
--
-- Safe to run once in the Supabase SQL editor -- purely widens an existing constraint, touches no
-- existing rows.
alter table public.provider_usage_log drop constraint provider_usage_log_endpoint_check;
alter table public.provider_usage_log add constraint provider_usage_log_endpoint_check
  check (endpoint in ('/chat', '/solve', '/classify-difficulty', '/ensure-correct-answer'));
