-- Speeds up the PYQ bank's "open a chapter" read (/pyq-browse, and the same PYQ_STUDENT_VISIBLE
-- shape used by /pyq-chapters, personalised tests, etc). That query filters on
-- subject + chapter + is_active + review_status + language (PYQ_STUDENT_VISIBLE in server.py),
-- but no existing index covers chapter at all -- idx_pyq_subject_language (pyq_language.sql)
-- only narrows by subject+language, leaving chapter and the review/active flags to be filtered
-- row-by-row. This partial index matches the exact student-visible predicate, so a chapter open
-- becomes an index lookup instead of a filtered scan over the subject's full row set.
--
-- Purely additive: no existing column, constraint, or row is touched. Safe to run once in the
-- Supabase SQL editor.
create index if not exists idx_pyq_student_browse
  on public.pyq (subject, chapter, language)
  where is_active and review_status = 'approved';
