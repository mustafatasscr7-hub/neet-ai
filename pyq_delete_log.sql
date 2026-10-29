-- pyq_delete_log: one row per PYQ question permanently deleted (see every pyq delete path in
-- server.py). pyq_id + uploader_name + when -- no question text, no student data. Safe to run
-- before or after deploying; until it exists the server's inserts just fail silently and the
-- delete still goes ahead. Safe to re-run.

create table if not exists public.pyq_delete_log (
  id uuid primary key default gen_random_uuid(),
  pyq_id uuid null,
  uploader_name text null,
  deleted_at timestamptz not null default now()
);
create index if not exists pyq_delete_log_uploader_deleted_idx
  on public.pyq_delete_log (uploader_name, deleted_at);

-- Internal telemetry: service-role only, same as ncert_retrieval_log -- no anon/authenticated policy.
alter table public.pyq_delete_log enable row level security;

-- Handy afterwards: all-time deletes per uploader.
-- select uploader_name, count(*) from public.pyq_delete_log group by 1 order by 2 desc;
