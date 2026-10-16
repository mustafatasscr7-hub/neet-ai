-- Private bucket for students' chat images and PDFs (server.py CHAT_UPLOADS_BUCKET).
--
-- Until now they sat in the PUBLIC ncert-daigrams bucket under chat-uploads/, where anyone with
-- the URL could open them. From now on the backend writes them here, keyed
-- <user_id>/<hex>.<ext> (PDFs under <user_id>/pdfs/), and students only ever get 1-hour signed
-- URLs from POST /chat-uploads/sign. Diagrams stay in ncert-daigrams, public, unchanged.
--
-- Deliberately NO storage.objects policies for this bucket: only the service key (backend) can
-- read, write, list or sign anything in it.
--
-- Run in the Supabase SQL editor, ONE STEP AT A TIME, in order. Safe to re-run.


-- STEP 1 (read-only check) ----------------------------------------------------------------------
-- A storage.objects policy that doesn't check bucket_id applies to EVERY bucket, including this
-- new private one -- it would let students list/download/sign each other's files with their own
-- login. Expected: no row with applies_to_every_bucket = true. If there is one, stop and send it
-- to me before step 2.
select policyname, cmd, roles,
       coalesce(qual, '') as using_expr,
       coalesce(with_check, '') as check_expr,
       (coalesce(qual, '') || coalesce(with_check, '')) not like '%bucket_id%' as applies_to_every_bucket
from pg_policies
where schemaname = 'storage' and tablename = 'objects';


-- STEP 2 (create the bucket) --------------------------------------------------------------------
-- 20 MB = the largest PDF any plan allows (server.py MAX_CHAT_PDF_BYTES); images are capped at
-- 8 MB by the backend. Only the types the upload endpoints actually write.
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('chat-uploads', 'chat-uploads', false, 20971520,
        array['image/jpeg', 'image/png', 'image/webp', 'image/gif', 'application/pdf'])
on conflict (id) do update
  set public = false,
      file_size_limit = excluded.file_size_limit,
      allowed_mime_types = excluded.allowed_mime_types;


-- STEP 3 (verify) -------------------------------------------------------------------------------
-- Expected: one row, public = false, file_size_limit = 20971520, the five types above.
select id, public, file_size_limit, allowed_mime_types
from storage.buckets
where id = 'chat-uploads';
