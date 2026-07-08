-- Locks down the pyq table so the public anon key (embedded in every page's HTML)
-- can only READ questions, never write/edit/delete them. Only the service_role key
-- (used server-side by the admin dashboard) bypasses RLS and can still write.
-- Safe to re-run — DROP POLICY IF EXISTS avoids duplicate-policy errors.

ALTER TABLE public.pyq ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "pyq_public_select" ON public.pyq;
CREATE POLICY "pyq_public_select"
  ON public.pyq
  FOR SELECT
  TO anon, authenticated
  USING (true);

-- No INSERT/UPDATE/DELETE policy is created for anon/authenticated on purpose —
-- once RLS is enabled, any operation without a matching policy is denied by default.
