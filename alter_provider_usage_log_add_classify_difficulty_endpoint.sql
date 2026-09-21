-- Widens provider_usage_log's endpoint CHECK constraint to also allow '/classify-difficulty' --
-- create_provider_usage_log_table.sql originally scoped this to just ('/chat', '/solve'), the
-- only two endpoints that spent real provider tokens at the time. classify_difficulty() has
-- always made a real, unlogged DeepSeek call from both /classify-difficulty (live, on-demand,
-- triggered by ordinary student page loads) and /admin/backfill-difficulty (the one-time sweep) --
-- this migration is what lets the code fix that now logs both of those actually succeed at
-- inserting, instead of failing the CHECK constraint and being silently swallowed by the
-- existing best-effort try/except around every log_provider_usage() call.
--
-- Both routes share the same '/classify-difficulty' endpoint tag rather than getting two separate
-- values -- they're the same underlying operation (classify one question's difficulty), just
-- triggered two different ways; splitting the tag would only fragment cost analytics for no
-- benefit.
--
-- Safe to run once in the Supabase SQL editor -- purely widens an existing constraint, touches no
-- existing rows.
alter table public.provider_usage_log drop constraint provider_usage_log_endpoint_check;
alter table public.provider_usage_log add constraint provider_usage_log_endpoint_check
  check (endpoint in ('/chat', '/solve', '/classify-difficulty'));
