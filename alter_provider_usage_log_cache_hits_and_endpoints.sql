-- provider_usage_log: cache-hit tokens, an output-cap flag, and every call that was being dropped.
-- Run this BEFORE deploying the matching server.py (2026-10-17 cost batch). Safe to re-run.
--
-- 1. cache_hit_tokens: the part of input_tokens the provider billed at its cache-hit rate. Until
--    now this only existed inside the cost figure (it had to be worked backwards from `cost`).
-- 2. output_capped: the reply stopped at its max_tokens (cut off). server.py also prints
--    "OUTPUT CAP HIT" for these.
-- 3. provider: 'deepseek-flash' is DeepSeek's current name for the Flash model ('deepseek-v4-flash'
--    is retired and kept only for old rows), and 'gemini-3.5-flash-lite-cache-storage' is the
--    hourly storage cost of the Gemini system-prompt cache -- every one of those rows was being
--    rejected by the old constraint, silently.
-- 4. endpoint: '/chat-verify' (the Qwen verification passes) and '/gemini-cache-storage' were also
--    being rejected; '/title', '/summarize-answer', '/chat-alternate-method' and
--    '/solve-alternate-method' were never logged at all and now are.
--
-- Existing rows are untouched (new columns default to 0 / false).

alter table public.provider_usage_log add column if not exists cache_hit_tokens integer not null default 0;
alter table public.provider_usage_log add column if not exists output_capped boolean not null default false;

alter table public.provider_usage_log drop constraint if exists provider_usage_log_provider_check;
alter table public.provider_usage_log add constraint provider_usage_log_provider_check
  check (provider in ('deepseek-flash', 'deepseek-v4-flash', 'qwen-flash', 'gemini-3.5-flash-lite',
                      'gemini-3.5-flash-lite-cache-storage'));

alter table public.provider_usage_log drop constraint if exists provider_usage_log_endpoint_check;
alter table public.provider_usage_log add constraint provider_usage_log_endpoint_check
  check (endpoint in ('/chat', '/solve', '/classify-difficulty', '/ensure-correct-answer', '/chat-summary',
                      '/chat-summary-truncated', '/chat-verify', '/gemini-cache-storage', '/title',
                      '/summarize-answer', '/chat-alternate-method', '/solve-alternate-method'));

-- Check: expect exactly two CHECK constraints, provider_usage_log_provider_check and
-- provider_usage_log_endpoint_check, both listing the values above. If a third, older provider or
-- endpoint check shows up under a different name, drop it -- it would still reject the new values.
select conname, pg_get_constraintdef(oid)
from pg_constraint
where conrelid = 'public.provider_usage_log'::regclass and contype = 'c';
