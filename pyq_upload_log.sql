-- pyq_upload_log: one row per PYQ question ever uploaded by the contractor (uploader_name =
-- 'Mr. Md Moniruzzaman'), used for the Contractor Queue's permanent "Uploaded so far" totals.
-- Those totals only ever go up -- approving, deleting or editing a question later never removes
-- its log row. pyq_id + uploader_name + has_diagram + when -- no question text, no student data.
--
-- Safe to run before or after deploying the server change that writes to this table: until it
-- exists, those inserts just fail silently (logged server-side only) and the save still goes
-- ahead, same as pyq_delete_log. Safe to re-run -- `create table if not exists` and the indexes
-- below are no-ops the second time, and the backfill's ON CONFLICT DO NOTHING means re-running
-- it after the server has already started logging new uploads will not double count anything.

create table if not exists public.pyq_upload_log (
  id uuid primary key default gen_random_uuid(),
  pyq_id uuid null,
  uploader_name text null,
  has_diagram boolean not null default false,
  created_at timestamptz not null default now()
);

-- One question can never be counted twice.
create unique index if not exists pyq_upload_log_pyq_id_idx
  on public.pyq_upload_log (pyq_id);

create index if not exists pyq_upload_log_uploader_diagram_idx
  on public.pyq_upload_log (uploader_name, has_diagram);

-- Internal telemetry: service-role only, same as pyq_delete_log / ncert_retrieval_log -- no
-- anon/authenticated policy.
alter table public.pyq_upload_log enable row level security;

-- Backfill: one log row for every existing contractor-uploaded PYQ row, so the "Uploaded so far"
-- totals reflect history, not just uploads from today onward. Same "has an uploaded diagram"
-- definition as the existing With diagram / Without diagram counts in
-- /admin/pyq-contractor-queue-counts (any of the 5 diagram-url columns non-empty), and the row's
-- own created_at as the log's created_at. ON CONFLICT DO NOTHING on pyq_id means running this
-- backfill again later (e.g. after re-running this whole file) never creates a second row for a
-- question it already logged.
insert into public.pyq_upload_log (pyq_id, uploader_name, has_diagram, created_at)
select
  p.id,
  p.uploader_name,
  (coalesce(p.diagram_url, '') <> ''
    or coalesce(p.option_a_diagram_url, '') <> ''
    or coalesce(p.option_b_diagram_url, '') <> ''
    or coalesce(p.option_c_diagram_url, '') <> ''
    or coalesce(p.option_d_diagram_url, '') <> ''),
  p.created_at
from public.pyq p
where p.uploader_name = 'Mr. Md Moniruzzaman'
on conflict (pyq_id) do nothing;

-- Handy afterwards: all-time uploads per uploader/diagram split.
-- select uploader_name, has_diagram, count(*) from public.pyq_upload_log group by 1, 2 order by 1, 2;

-- How many contractor rows the backfill above did NOT cover (uploaded_by='contractor' but no
-- uploader_name -- an old cached contractor-portal page from before the uploader-name picker
-- existed, or any other pre-feature row):
-- select count(*) from public.pyq where uploaded_by = 'contractor' and uploader_name is null;
