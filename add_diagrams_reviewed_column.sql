-- Adds an admin-review flag to the diagrams table, mirroring pyq.reviewed's existing
-- "newly uploaded, needs a first check" queue pattern used by admin-pyq-preview.html's
-- Review Queue. Defaults false so every diagram uploaded before this migration (and any
-- new upload from admin-diagram-upload.html, which never sets this column) starts
-- "Pending Review" until an admin explicitly confirms it in the new Diagram Review section.
alter table public.diagrams add column if not exists reviewed boolean not null default false;
