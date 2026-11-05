-- Supports the Processed PDFs popup's Remove / Add file / Undo actions (admin-pyq-preview.html).
--
-- deleted_at: soft-delete marker set by "Remove" and cleared by "Undo". A non-null value means
-- the row never shows up in the list, never counts toward a PDF's counts, and never counts
-- toward the chapter Found/Missing gap summary again -- but the row itself, and every question
-- ever extracted from it, is left completely untouched. Run against the existing processed_pdfs
-- table (see create_processed_pdfs_table.sql).
alter table processed_pdfs add column if not exists deleted_at timestamptz;

-- chapter_number: set directly by the "Add file" form's optional chapter-number field, so a
-- manually-added entry can fill a chapter gap without needing a "Chapter N" pattern baked into
-- its filename. Falls back to the existing filename regex when null, so every pre-existing row
-- behaves exactly as before.
alter table processed_pdfs add column if not exists chapter_number integer;

create index if not exists idx_processed_pdfs_deleted_at on processed_pdfs (deleted_at);
