-- Widens provider_usage_log's endpoint CHECK constraint to also allow '/chat-summary' -- the new
-- running-summary layer on top of /chat's sliding-2 window makes a real, fire-and-forget DeepSeek
-- call (_update_running_summary) every time a /chat text doubt's exchange falls out of the
-- sliding-2 window. A distinct tag from '/chat' itself (not reused) -- this is a genuinely
-- separate backend operation (background summarization) from the student-facing answer call, and
-- deserves its own bucket in cost analytics, same reasoning as '/ensure-correct-answer' getting
-- its own tag instead of reusing '/classify-difficulty''s.
--
-- Safe to run once in the Supabase SQL editor -- purely widens an existing constraint, touches no
-- existing rows.
alter table public.provider_usage_log drop constraint provider_usage_log_endpoint_check;
alter table public.provider_usage_log add constraint provider_usage_log_endpoint_check
  check (endpoint in ('/chat', '/solve', '/classify-difficulty', '/ensure-correct-answer', '/chat-summary'));
