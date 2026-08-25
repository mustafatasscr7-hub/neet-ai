-- Adds question_details to mock_results so a PAST attempt can show a real per-question review
-- (question text, selected answer, correct answer, right/wrong/skipped) on scoreboard.html.
--
-- This data already exists at test-completion time (mocktest.html/personalised-test.html both
-- build a full `results` array -- question + userAnswer + status -- right before submitting) but
-- was NEVER persisted anywhere: only aggregate stats (score/correct/wrong/skipped/subject
-- scores) get written to mock_results. The per-question array only ever lived in sessionStorage,
-- which is why mockresults.html's own "Question Review" section works (it reads that same-session
-- sessionStorage value) but scoreboard.html's history view -- loaded fresh from the DB, often much
-- later -- has never had this data to show.
--
-- Nullable, no default: existing rows (attempts taken before this column existed) will have
-- question_details = null. scoreboard.html handles that case with an honest "not available for
-- this older attempt" message rather than a broken/empty question list.
--
-- Run this in the Supabase SQL editor.

alter table mock_results add column if not exists question_details jsonb;
