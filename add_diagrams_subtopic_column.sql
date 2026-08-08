-- Lets diagrams be tagged more precisely than Subject -> Class -> Chapter (e.g. "2.3.1
-- Phycomycetes"). Free text, no fixed subtopic list exists yet -- admin types it manually
-- in admin-diagram-upload.html / admin-pyq-preview.html's Diagram Review section.
alter table public.diagrams add column if not exists subtopic text;
