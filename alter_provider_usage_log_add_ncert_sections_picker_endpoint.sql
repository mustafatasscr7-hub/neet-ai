-- Widens provider_usage_log's endpoint CHECK constraint to also allow
-- '/chat-ncert-sections-picker' -- the "From NCERT" exact-quote test feature's cheap DeepSeek
-- picker call (see _pick_ncert_sections in server.py, NCERT_SECTIONS_ENABLED). Without this,
-- every log_provider_usage() call for that endpoint silently fails the insert (fails open --
-- the picker call itself still works fine, only its cost/analytics row is lost) -- confirmed
-- live during testing: the picker ran correctly end to end, but 0 of its rows ever reached this
-- table, all rejected by this exact constraint (23514/"violates check constraint
-- provider_usage_log_endpoint_check").
--
-- Based on the current list from alter_provider_usage_log_cache_hits_and_endpoints.sql (the
-- latest prior migration to touch this constraint) -- re-verify that list is still current
-- before running this if more endpoint tags have been added since.
--
-- Safe to run once in the Supabase SQL editor -- purely widens an existing constraint, touches
-- no existing rows.
alter table public.provider_usage_log drop constraint provider_usage_log_endpoint_check;
alter table public.provider_usage_log add constraint provider_usage_log_endpoint_check
  check (endpoint in ('/chat', '/solve', '/classify-difficulty', '/ensure-correct-answer', '/chat-summary',
                      '/chat-summary-truncated', '/chat-verify', '/gemini-cache-storage', '/title',
                      '/summarize-answer', '/chat-alternate-method', '/solve-alternate-method',
                      '/chat-ncert-sections-picker'));
