-- Widens provider_usage_log's endpoint CHECK constraint to also allow
-- '/chat-summary-truncated' -- a distinct tag from '/chat-summary' used ONLY when the running-
-- summary background call (_update_running_summary) hit its max_tokens cap and got cut off
-- mid-generation (stop_reason == "max_tokens") instead of stopping naturally. Visibility only --
-- a truncated summary is still used exactly as before (no retry, no behavior change), this just
-- makes the event queryable (by endpoint + the existing user_id/created_at columns) instead of
-- invisible, so a recurrence of the original replace-instead-of-merge bug's mechanical cousin
-- (facts silently dropped because they didn't fit in the cap) can actually be noticed.
--
-- Re-lists '/chat-summary' too (not just appends) so this migration is self-sufficient whether
-- or not alter_provider_usage_log_add_chat_summary_endpoint.sql has already been run -- re-adding
-- an already-allowed literal to a CHECK constraint is a no-op, not an error.
--
-- Safe to run once in the Supabase SQL editor -- purely widens an existing constraint, touches no
-- existing rows.
alter table public.provider_usage_log drop constraint provider_usage_log_endpoint_check;
alter table public.provider_usage_log add constraint provider_usage_log_endpoint_check
  check (endpoint in ('/chat', '/solve', '/classify-difficulty', '/ensure-correct-answer', '/chat-summary', '/chat-summary-truncated'));
