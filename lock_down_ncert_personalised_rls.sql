-- Closes the anon/authenticated SELECT policies on ncert_content and personalised_test_sets
-- (scraping audit, 2026-09-19). Both were opened for public SELECT in fix_rls_open_tables.sql
-- (2026-08-07) with the stated reasoning "the app reads this with the anon key directly" -- that
-- reasoning no longer holds for either table:
--
--   - ncert_content: only ever read by server.py's search_ncert(), via the match_ncert /
--     match_ncert_hi_gemini RPCs -- no frontend page queries this table directly (confirmed via a
--     full grep across every .html file, zero matches). server.py has been switched to the
--     service-role key for this, which bypasses RLS entirely regardless of policy.
--   - personalised_test_sets: only ever read by server.py's /personalised-catalog and
--     /personalised-catalog-start -- same confirmation, same fix already applied.
--
-- RLS stays ENABLED on both (already was, from fix_rls_open_tables.sql) -- this just drops the
-- permissive policy, which makes both tables default-deny for anon/authenticated (no matching
-- policy = denied), same "no policy = no access" pattern already used for INSERT/UPDATE/DELETE
-- on every table this repo's RLS migrations touch. Safe to re-run -- DROP POLICY IF EXISTS avoids
-- an error if this has already been applied.
--
-- Deliberately does NOT touch pyq's own RLS (still public SELECT, on purpose) -- pyqbank.html
-- still reads that table directly from the browser, a separate, parked architecture decision.

drop policy if exists "ncert_content_public_select" on public.ncert_content;
drop policy if exists "personalised_test_sets_public_select" on public.personalised_test_sets;
