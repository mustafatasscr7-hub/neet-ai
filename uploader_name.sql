-- Adds a nullable "who uploaded this" name tag to pyq, separate from the existing uploaded_by
-- role column (which only ever holds "contractor" or "admin"). uploader_name holds one of two
-- actual people's names: "Mr. Md Moniruzzaman" or "Admin" (enforced server-side in server.py,
-- not by a DB constraint, so the allowed-names list can change without a migration).
--
-- Existing rows stay NULL -- that means "earlier upload", i.e. uploaded before this column
-- existed. Does not touch uploaded_by or any existing value on any row.
--
-- Apply manually (Supabase SQL editor or psql) against the real database. Not run automatically
-- and not required for the app to keep working -- every write that sets this column omits it
-- entirely when the column is missing, so nothing breaks before this is applied.

ALTER TABLE pyq ADD COLUMN IF NOT EXISTS uploader_name text;
CREATE INDEX IF NOT EXISTS idx_pyq_uploader_name ON pyq (uploader_name);
