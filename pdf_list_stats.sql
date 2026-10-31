-- Processed PDFs popup speed-up (admin-pdf-review.html, contractor-pdf-review.html,
-- admin-pyq-preview.html's Contractor Queue "PDFs" button all share this list).
--
-- The old /admin/processed-pdfs implementation fired ~6 Supabase count queries PER PDF row
-- (total/approved/pending/needs_fix/with_diagram + a sampled uploader_name), fanned out with a
-- concurrency-15 semaphore. This function computes the same aggregate, for every
-- source_pdf_filename at once, in a single round trip. The backend calls it in parallel with its
-- one `processed_pdfs` list read (asyncio.gather) and falls back to the old per-PDF queries if
-- this function is missing or errors -- so it is always safe to defer running this migration.
--
-- Read-only (SELECT only, no writes), SECURITY DEFINER so a contractor key (which only ever has
-- the service-role key server-side, never direct DB access) can still get the aggregate without
-- needing broader table grants. search_path is fixed to avoid the usual SECURITY DEFINER
-- search_path hijack. Safe to run twice -- CREATE OR REPLACE + GRANT/REVOKE are all idempotent.

create or replace function public.pdf_list_stats()
returns table (
  source_pdf_filename text,
  uploader_name text,
  total bigint,
  approved bigint,
  pending bigint,
  needs_fix bigint,
  with_diagram bigint,
  without_diagram bigint
)
language sql
stable
security definer
set search_path = public
as $$
  with active as (
    select *
    from pyq
    where is_active = true
      and source_pdf_filename is not null
      and source_pdf_filename <> ''
  ),
  agg as (
    select
      a.source_pdf_filename,
      count(*) as total,
      count(*) filter (where a.review_status = 'approved') as approved,
      count(*) filter (where a.review_status = 'pending') as pending,
      count(*) filter (where a.review_status = 'needs_fix') as needs_fix,
      -- Same OR-group _PYQ_HAS_DIAGRAM_OR encodes as a PostgREST filter: "has at least one
      -- non-empty diagram url across the 5 diagram columns".
      count(*) filter (
        where coalesce(a.diagram_url, '') <> ''
           or coalesce(a.option_a_diagram_url, '') <> ''
           or coalesce(a.option_b_diagram_url, '') <> ''
           or coalesce(a.option_c_diagram_url, '') <> ''
           or coalesce(a.option_d_diagram_url, '') <> ''
      ) as with_diagram
    from active a
    group by a.source_pdf_filename
  ),
  -- Same "uploader_name" definition as the old _processed_pdf_counts.sampled_uploader(): the
  -- most-recently-created active row for that filename, whatever its uploader_name happens to be
  -- (including null) -- not a vote or a distinct list.
  latest_uploader as (
    select distinct on (a.source_pdf_filename)
      a.source_pdf_filename,
      a.uploader_name
    from active a
    order by a.source_pdf_filename, a.created_at desc
  )
  select
    agg.source_pdf_filename,
    latest_uploader.uploader_name,
    agg.total,
    agg.approved,
    agg.pending,
    agg.needs_fix,
    agg.with_diagram,
    (agg.total - agg.with_diagram) as without_diagram
  from agg
  left join latest_uploader using (source_pdf_filename);
$$;

revoke all on function public.pdf_list_stats() from public;
revoke all on function public.pdf_list_stats() from anon;
revoke all on function public.pdf_list_stats() from authenticated;
grant execute on function public.pdf_list_stats() to service_role;
