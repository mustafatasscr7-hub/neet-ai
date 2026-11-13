-- Adds scan-cost tracking to processed_pdfs (2026-11-13), shown as a muted line per record in
-- the Processed PDFs popup (admin-pyq-preview.html): model cost, call count, model name(s),
-- pages scanned, and scan duration, for every scan mode. The three q_* columns are mock-exam-only
-- (process_mock_exam_text.scan_mock_exam_pdf + process_mock_exam_vision.apply_vision_to_flagged_questions):
-- questions found, still flagged, and fixed via the Gemini Vision re-read pass.
--
-- Purely additive, all nullable: no existing column, constraint, or row is touched. A scan mode
-- with no cost tracking of its own ("standard", "3col_book") or a mock scan with the vision
-- flag off simply leaves scan_cost_usd/model_calls/models null -- the popup shows that as
-- "no model calls" rather than guessing. Rows written before this migration stay entirely null
-- too, shown as "-". RLS is untouched (server.py only ever writes/reads this table with the
-- service-role key).
--
-- Safe to run once in the Supabase SQL editor.

alter table public.processed_pdfs add column if not exists scan_cost_usd numeric;
alter table public.processed_pdfs add column if not exists model_calls int;
alter table public.processed_pdfs add column if not exists models text;
alter table public.processed_pdfs add column if not exists pages int;
alter table public.processed_pdfs add column if not exists scan_seconds numeric;
alter table public.processed_pdfs add column if not exists q_found int;
alter table public.processed_pdfs add column if not exists q_flagged int;
alter table public.processed_pdfs add column if not exists q_model_read int;

-- Expect every new column to show up, all nullable.
select column_name, is_nullable, data_type from information_schema.columns
where table_schema = 'public' and table_name = 'processed_pdfs'
  and column_name in ('scan_cost_usd', 'model_calls', 'models', 'pages', 'scan_seconds',
                       'q_found', 'q_flagged', 'q_model_read')
order by column_name;
