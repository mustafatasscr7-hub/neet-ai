-- Optional Hindi-language display fields for diagrams, filled in side-by-side with the
-- existing English name/description at upload time (admin-diagram-upload.html) or added later
-- in admin-pyq-preview.html's Diagram Review section. Both nullable -- a diagram with no Hindi
-- text is exactly as valid as one with it, same as description already is. Deliberately not fed
-- into the diagrams.embedding column (build_diagram_embedding_text stays English-only) -- that's
-- a separate concern (matching a Hindi-phrased student doubt) this migration doesn't address.
alter table public.diagrams add column if not exists name_hi text;
alter table public.diagrams add column if not exists description_hi text;
