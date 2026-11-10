-- Run manually in Supabase SQL editor.
-- Adds a nullable keywords column to ncert_sections so the picker can see
-- more than just the heading for sections that cover many topics under one
-- title (e.g. "1.1 Diversity in the Living World", 1,100+ words).
-- Populated later by a one-time script (load_ncert_section_keywords.py); NULL
-- until then, and the picker falls back to headings-only when it is NULL.

alter table ncert_sections add column if not exists keywords text;
