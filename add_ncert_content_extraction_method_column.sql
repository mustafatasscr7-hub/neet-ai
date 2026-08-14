-- Tracks how each Hindi ncert_content chunk's source text was actually recovered, for the
-- Hindi NCERT ingestion pipeline (legacy Chanakya-font PDFs requiring vision-based OCR instead
-- of plain text extraction -- see the RECITATION-block testing this session).
--
-- 'gemini_direct'  -- whole page transcribed cleanly, no blocking, highest confidence.
-- 'gemini_split'   -- the page (or a sub-region of it) was blocked on the first attempt, but a
--                     smaller crop of the SAME region succeeded via Gemini -- still Gemini
--                     output, still high confidence.
-- 'qwen_fallback'  -- even the smallest Gemini crop of this fragment stayed blocked, so this
--                     specific fragment was recovered via Qwen vision instead. Confirmed working
--                     and mostly accurate, but measured with a real (if small) error rate on
--                     isolated fragments in testing, including one case of a dropped formula
--                     term -- treat as lower-confidence than the two Gemini-sourced tiers,
--                     without blocking on it.
--
-- NULL for all existing (English) rows -- not applicable, they were never part of this pipeline.
--
-- Purely additive: no existing column, constraint, or row is touched. Safe to run once in the
-- Supabase SQL editor.

alter table public.ncert_content add column if not exists extraction_method text
  check (extraction_method is null or extraction_method in ('gemini_direct', 'gemini_split', 'qwen_fallback'));

create index if not exists idx_ncert_content_extraction_method on public.ncert_content (extraction_method)
  where extraction_method is not null;
