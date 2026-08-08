-- Multiple Solution Methods feature: cache the alternate-method verdict alongside each
-- existing cached answer, in the same two tables that already cache the primary answer.
--
-- Tri-state per row, same defensive pattern as pyq.correct_answer's null-vs-blank handling:
--   NULL           -> never checked yet, ask DeepSeek
--   '' (empty)     -> checked once, genuinely no alternate method exists -- never re-check
--   non-empty text -> checked once, this is the alternate method content

alter table public.answer_cache
  add column if not exists alternate_method text;

alter table public.pyq_solution_cache
  add column if not exists alternate_method text;
