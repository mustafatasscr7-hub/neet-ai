-- Fixes the 3 tables found with RLS not enforced at all during the RLS audit
-- (2026-08-07): ncert_content, personalised_test_sets, user_streaks. Each was
-- confirmed wide open to the anon key -- full SELECT/INSERT/UPDATE/DELETE, no
-- restriction -- via live empirical testing (real seed row + real anon
-- request + service-role verification), not just by reading policy text.
-- Safe to re-run -- DROP POLICY IF EXISTS avoids duplicate-policy errors.

-- ncert_content: same public-read, service-role-write pattern as pyq.
-- RAG search (search_ncert() in server.py) reads this table with the anon
-- key from a plain user-facing request, so SELECT must stay open. Nothing
-- in the app ever writes to it from the frontend -- ingestion is the
-- standalone process_ncert.py script, which already uses the service key.
alter table public.ncert_content enable row level security;

drop policy if exists "ncert_content_public_select" on public.ncert_content;
create policy "ncert_content_public_select" on public.ncert_content
  for select to anon, authenticated using (true);

-- No INSERT/UPDATE/DELETE policy for anon/authenticated on purpose -- once RLS
-- is enabled, any operation without a matching policy is denied by default.

-- personalised_test_sets: same pattern. The test picker UI reads this list
-- with the anon key before a student even logs in, so SELECT must stay open.
-- Sets are only ever created by an admin/seed process, never from the
-- frontend -- same write-lockdown reasoning as ncert_content above.
alter table public.personalised_test_sets enable row level security;

drop policy if exists "personalised_test_sets_public_select" on public.personalised_test_sets;
create policy "personalised_test_sets_public_select" on public.personalised_test_sets
  for select to anon, authenticated using (true);

-- No INSERT/UPDATE/DELETE policy for anon/authenticated on purpose -- same as above.

-- user_streaks: different shape from the two above -- this is per-user data,
-- not shared public content, so a blanket USING (true) would leak every
-- student's activity streak to anyone holding the anon key (confirmed live in
-- the audit -- a real user_id + streak numbers came back on the very first
-- anon SELECT probe). Locked to the row's own owner instead.
--
-- user_id really does resolve against auth.uid(): the column carries a
-- foreign key into auth.users (confirmed via a live FK-violation error while
-- probing this table), and chat.html/pyqbank.html's updateStreak() both call
-- client.from('user_streaks') through the same Supabase JS client instance
-- that client.auth.verifyOtp() (login.html) already populated with a real
-- session -- so these requests carry a genuine authenticated JWT, not the
-- bare anon key, and auth.uid() resolves correctly.
--
-- Deviates from "SELECT and UPDATE only" by one policy: INSERT is included
-- too. updateStreak() does a plain .insert(...) the first time a user's
-- streak row doesn't exist yet (chat.html:657, pyqbank.html) -- SELECT+UPDATE
-- alone would RLS-block that insert and silently break streak tracking for
-- every new user. WITH CHECK (auth.uid() = user_id) keeps it exactly as
-- narrow as the other two: a user can only ever create their own row.
--
-- No policy at all for the anon role, deliberately -- both call sites only
-- invoke updateStreak() after confirming a real session exists and skip it
-- entirely for guests (chat.html:651-653), so guests never need or get any
-- access here, matching current real-world behavior exactly.
--
-- No DELETE policy -- neither call site ever deletes a streak row.
alter table public.user_streaks enable row level security;

drop policy if exists "user_streaks_own_select" on public.user_streaks;
create policy "user_streaks_own_select" on public.user_streaks
  for select to authenticated
  using (auth.uid() = user_id);

drop policy if exists "user_streaks_own_insert" on public.user_streaks;
create policy "user_streaks_own_insert" on public.user_streaks
  for insert to authenticated
  with check (auth.uid() = user_id);

drop policy if exists "user_streaks_own_update" on public.user_streaks;
create policy "user_streaks_own_update" on public.user_streaks
  for update to authenticated
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);
