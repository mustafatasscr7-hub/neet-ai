-- Diagnostic: shows whether RLS is actually enabled on pyq, and lists every
-- policy currently attached to it (there may be an old permissive policy
-- still granting broad access alongside the new SELECT-only one).

SELECT relrowsecurity AS rls_enabled, relforcerowsecurity AS rls_forced
FROM pg_class
WHERE oid = 'public.pyq'::regclass;

SELECT policyname, permissive, roles, cmd, qual, with_check
FROM pg_policies
WHERE schemaname = 'public' AND tablename = 'pyq';
