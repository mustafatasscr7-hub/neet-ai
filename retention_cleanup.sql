-- Data retention: a daily cleanup job (Supabase pg_cron), 2026-10-16.
--
-- The periods (LEGAL_DATA_AUDIT.md section 1 and privacy.html section 6 say the same):
--   chats, saved_questions, mock_results, mock_test_attempts, user_streaks .... until the student deletes them / the account
--   chat uploads (private chat-uploads bucket) .................................. while linked from one of the owner's chats;
--                                                                                 unlinked files are deleted once 7 days old
--   guest_usage_log (IP + daily tokens) ......................................... 30 days
--   guest_device_usage_log (guest device ID + daily tokens) ..................... 30 days
--   active_sessions (devices) ................................................... 90 days after last activity
--   media_doubt_log (image/PDF doubt records) ................................... 90 days
--   usage_log, max_usage_alerts (per-account usage) ............................. 12 months
--   provider_usage_log (AI cost records) ........................................ user_id cleared at 12 months, rows deleted at 24 months
--   question_reports, diagram_reports, response_feedback, session_feedback ...... user_id (and chat_id) cleared at 12 months, content kept
--   auth.audit_log_entries (Supabase sign-in log, holds IP addresses) ........... 12 months
--   answer_cache (shared answers, not personal) ................................. 30 days (server.py stops serving them at 30 days too)
--   answer_cache_log (cache hit/miss counts, no user or text) ................... 90 days
--   embedding_cache, pyq_solution_cache ......................................... no limit (shared, not personal)
-- plus housekeeping: cron.job_run_details older than 30 days, retention_runs older than 90 days.
--
-- How it runs: two pg_cron jobs, 03:00 and 03:15 IST.
--   run_retention_cleanup()     -- deletes / anonymises rows, in batches (default 10,000 rows x up to
--                                  20 batches per table per run; a backlog bigger than that drains
--                                  over the following days and the summary says "capped").
--   sweep_orphan_chat_uploads() -- deletes unlinked chat files. Supabase only allows deleting files
--                                  through the Storage API (deleting storage.objects rows in SQL leaves
--                                  the file behind), so it calls that API with pg_net, using the
--                                  service-role key kept in Vault. "Linked" = the file's 32-char name
--                                  appears in one of ITS OWNER's chats -- every stored form of a link
--                                  (chat-upload:// reference, old public URL, signed URL) contains it.
--                                  chat_upload_refs (kept up to date by a trigger on chats) finds the
--                                  candidates fast; each candidate is re-checked against the owner's
--                                  chats themselves before anything is deleted.
-- Every run (and every dry run) writes its counts to public.retention_runs.
--
-- Run ONE PART AT A TIME in the Supabase SQL editor, in order. Every part is safe to re-run.


-- PART 0 (dashboard, once) ------------------------------------------------------------------------
-- 1. Database > Extensions: enable "pg_cron" and "pg_net".
-- 2. Vault (Integrations > Vault, or Project Settings > Vault) > "Add new secret", twice:
--      name: retention_project_url        value: https://hvhnfttrfouajlyhvunq.supabase.co
--      name: retention_service_role_key   value: the service_role key (Project Settings > API Keys)
--    Use the Vault screen, not an SQL insert, so the key never lands in the SQL editor's history.


-- PART 1 (schema: run once) -----------------------------------------------------------------------
-- Anonymising feedback means clearing user_id, which was NOT NULL on these two.
alter table public.response_feedback alter column user_id drop not null;
alter table public.session_feedback alter column user_id drop not null;

-- The cleanup filters by date; without these indexes every daily run would scan whole tables.
create index if not exists guest_usage_log_usage_date_idx on public.guest_usage_log (usage_date);
-- guest_device_usage_log and its index come from create_guest_device_usage_log.sql (run that first).
create index if not exists usage_log_usage_date_idx on public.usage_log (usage_date);
create index if not exists active_sessions_last_active_at_idx on public.active_sessions (last_active_at);
create index if not exists media_doubt_log_created_at_idx on public.media_doubt_log (created_at);
create index if not exists max_usage_alerts_created_at_idx on public.max_usage_alerts (created_at);
create index if not exists provider_usage_log_created_at_idx on public.provider_usage_log (created_at);
create index if not exists question_reports_created_at_idx on public.question_reports (created_at);
create index if not exists diagram_reports_created_at_idx on public.diagram_reports (created_at);
create index if not exists response_feedback_created_at_idx on public.response_feedback (created_at);
create index if not exists session_feedback_created_at_idx on public.session_feedback (created_at);
create index if not exists chats_user_id_idx on public.chats (user_id);

-- Counts from every run. No personal data except, for the file sweep, the deleted files' names
-- (<user_id>/<hex>.<ext>), kept 90 days so a wrong deletion could be traced.
create table if not exists public.retention_runs (
  id bigint generated always as identity primary key,
  job text not null,
  dry_run boolean not null,
  ran_at timestamptz not null default now(),
  summary jsonb not null
);
alter table public.retention_runs enable row level security;
revoke all on public.retention_runs from anon, authenticated;

-- Which chat files each chat links to, maintained by the trigger below. Only a candidate filter for
-- the sweep (which re-checks the chats themselves), so a missed trigger can never delete a file.
create table if not exists public.chat_upload_refs (
  chat_id uuid not null references public.chats (id) on delete cascade,
  user_id uuid not null,
  file_hex text not null,
  primary key (chat_id, file_hex)
);
create index if not exists chat_upload_refs_user_hex_idx on public.chat_upload_refs (user_id, file_hex);
alter table public.chat_upload_refs enable row level security;
revoke all on public.chat_upload_refs from anon, authenticated;

-- Never lets a chat save fail: any error is swallowed (logged as a warning) and the sweep's re-check
-- covers for it. Skips the regex entirely for the common case of a chat with no attachments.
create or replace function public.sync_chat_upload_refs() returns trigger
language plpgsql security definer set search_path = public, pg_temp as $$
begin
  begin
    delete from public.chat_upload_refs where chat_id = new.id;
    if new.user_id is not null and new.messages is not null and strpos(new.messages::text, 'chat-upload') > 0 then
      insert into public.chat_upload_refs (chat_id, user_id, file_hex)
      select distinct new.id, new.user_id, m[1]
      from regexp_matches(new.messages::text, 'chat-upload(?:s/|://)(?:[0-9a-f-]{36}/)?(?:pdfs/)?([0-9a-f]{32})\.', 'g') as m
      on conflict do nothing;
    end if;
  exception when others then
    raise warning 'sync_chat_upload_refs: chat %: %', new.id, sqlerrm;
  end;
  return null;
end $$;

drop trigger if exists chats_sync_upload_refs on public.chats;
create trigger chats_sync_upload_refs
  after insert or update of messages, user_id on public.chats
  for each row execute function public.sync_chat_upload_refs();

-- Backfill for the chats that already exist.
insert into public.chat_upload_refs (chat_id, user_id, file_hex)
select distinct c.id, c.user_id, m[1]
from public.chats c,
     regexp_matches(c.messages::text, 'chat-upload(?:s/|://)(?:[0-9a-f-]{36}/)?(?:pdfs/)?([0-9a-f]{32})\.', 'g') as m
where c.user_id is not null
on conflict do nothing;


-- PART 2 (the cleanup functions) ------------------------------------------------------------------
-- One rule: delete (p_set null) or update (p_set = 'col = null, ...') rows matching p_where, in
-- batches of p_batch rows by ctid, at most p_max_batches batches. Dry run = count only.
create or replace function public._retention_apply(p_table text, p_where text, p_set text, p_dry_run boolean, p_batch int, p_max_batches int)
returns jsonb language plpgsql security definer set search_path = public, pg_temp as $$
declare
  v_n bigint;
  v_total bigint := 0;
  v_batches int := 0;
begin
  if p_dry_run then
    execute format('select count(*) from %s where %s', p_table, p_where) into v_n;
    return jsonb_build_object('eligible', v_n);
  end if;
  loop
    if p_set is null then
      execute format('delete from %1$s where ctid = any (array(select ctid from %1$s where %2$s limit %3$s))', p_table, p_where, p_batch);
    else
      execute format('update %1$s set %4$s where ctid = any (array(select ctid from %1$s where %2$s limit %3$s))', p_table, p_where, p_batch, p_set);
    end if;
    get diagnostics v_n = row_count;
    v_total := v_total + v_n;
    v_batches := v_batches + 1;
    exit when v_n < p_batch or v_batches >= p_max_batches;
  end loop;
  return jsonb_build_object(case when p_set is null then 'deleted' else 'anonymised' end, v_total, 'capped', v_n >= p_batch);
end $$;

create or replace function public.run_retention_cleanup(p_dry_run boolean default true, p_batch int default 10000, p_max_batches int default 20)
returns jsonb language plpgsql security definer set search_path = public, pg_temp as $$
declare
  -- guest_usage_log / usage_log store the IST calendar day (server.py), so cut off on IST days too.
  v_ist_today date := (now() at time zone 'Asia/Kolkata')::date;
  v_rules text[][] := array[
    -- label,                         table,                       where,                                                                                    set
    array['guest_usage_log',          'public.guest_usage_log',    format('usage_date < %L', v_ist_today - 30),                                              null],
    array['guest_device_usage_log',   'public.guest_device_usage_log', format('usage_date < %L', v_ist_today - 30),                                          null],
    array['active_sessions',          'public.active_sessions',    $w$last_active_at < now() - interval '90 days'$w$,                                       null],
    array['media_doubt_log',          'public.media_doubt_log',    $w$created_at < now() - interval '90 days'$w$,                                           null],
    array['answer_cache',             'public.answer_cache',       $w$created_at < now() - interval '30 days'$w$,                                           null],
    array['answer_cache_log',         'public.answer_cache_log',   $w$created_at < now() - interval '90 days'$w$,                                           null],
    array['usage_log',                'public.usage_log',          format('usage_date < %L', (v_ist_today - interval '12 months')::date),                   null],
    array['max_usage_alerts',         'public.max_usage_alerts',   $w$created_at < now() - interval '12 months'$w$,                                         null],
    array['provider_usage_log.user',  'public.provider_usage_log', $w$user_id is not null and created_at < now() - interval '12 months'$w$,                 'user_id = null'],
    array['provider_usage_log',       'public.provider_usage_log', $w$created_at < now() - interval '24 months'$w$,                                         null],
    array['question_reports',         'public.question_reports',   $w$user_id is not null and created_at < now() - interval '12 months'$w$,                 'user_id = null'],
    array['diagram_reports',          'public.diagram_reports',    $w$user_id is not null and created_at < now() - interval '12 months'$w$,                 'user_id = null'],
    array['response_feedback',        'public.response_feedback',  $w$(user_id is not null or chat_id is not null) and created_at < now() - interval '12 months'$w$, 'user_id = null, chat_id = null'],
    array['session_feedback',         'public.session_feedback',   $w$(user_id is not null or chat_id is not null) and created_at < now() - interval '12 months'$w$, 'user_id = null, chat_id = null'],
    array['auth.audit_log_entries',   'auth.audit_log_entries',    $w$created_at < now() - interval '12 months'$w$,                                         null],
    array['cron.job_run_details',     'cron.job_run_details',      $w$end_time < now() - interval '30 days'$w$,                                             null],
    array['retention_runs',           'public.retention_runs',     $w$ran_at < now() - interval '90 days'$w$,                                               null]
  ];
  v_summary jsonb := '{}'::jsonb;
  v_result jsonb;
  v_total bigint;
  i int;
begin
  for i in 1 .. array_length(v_rules, 1) loop
    begin
      v_result := public._retention_apply(v_rules[i][2], v_rules[i][3], v_rules[i][4], p_dry_run, p_batch, p_max_batches);
      if p_dry_run then
        execute format('select count(*) from %s', v_rules[i][2]) into v_total;
        v_result := v_result || jsonb_build_object('total_rows', v_total);
      end if;
    exception when others then
      -- One table failing (e.g. no permission on auth.audit_log_entries) never stops the others.
      v_result := jsonb_build_object('error', sqlerrm);
    end;
    v_summary := v_summary || jsonb_build_object(v_rules[i][1], v_result);
  end loop;
  insert into public.retention_runs (job, dry_run, summary) values ('rows', p_dry_run, v_summary);
  return v_summary;
end $$;

create or replace function public.sweep_orphan_chat_uploads(p_dry_run boolean default true, p_min_age interval default '7 days', p_max_files int default 1000)
returns jsonb language plpgsql security definer set search_path = public, pg_temp as $$
declare
  v_names text[];
  v_unexpected bigint;
  v_url text;
  v_key text;
  v_name text;
  v_summary jsonb;
begin
  with candidates as materialized (
    select o.name, o.created_at,
           split_part(o.name, '/', 1)::uuid as owner,
           substring(o.name from '([0-9a-f]{32})\.[a-z]+$') as file_hex
    from storage.objects o
    where o.bucket_id = 'chat-uploads'
      and o.created_at < now() - p_min_age
      and o.name ~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/(pdfs/)?[0-9a-f]{32}\.(jpg|jpeg|png|webp|gif|pdf)$'
  )
  select array_agg(name order by created_at) into v_names
  from (
    select c.name, c.created_at
    from candidates c
    where not exists (select 1 from public.chat_upload_refs r where r.user_id = c.owner and r.file_hex = c.file_hex)
      and not exists (select 1 from public.chats ch where ch.user_id = c.owner and strpos(ch.messages::text, c.file_hex) > 0)
    order by c.created_at
    limit p_max_files
  ) s;

  -- Anything the backend wouldn't have written; never deleted automatically, just reported.
  select count(*) into v_unexpected
  from storage.objects o
  where o.bucket_id = 'chat-uploads'
    and o.name !~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/(pdfs/)?[0-9a-f]{32}\.(jpg|jpeg|png|webp|gif|pdf)$';

  if not p_dry_run and coalesce(array_length(v_names, 1), 0) > 0 then
    select decrypted_secret into v_url from vault.decrypted_secrets where name = 'retention_project_url';
    select decrypted_secret into v_key from vault.decrypted_secrets where name = 'retention_service_role_key';
    if v_url is null or v_key is null then
      raise exception 'Vault secrets retention_project_url / retention_service_role_key are missing (PART 0)';
    end if;
    -- Queued now, sent by pg_net's worker after this transaction commits (about 200 per second).
    -- A delete that fails leaves the file in place, so tomorrow's run simply finds it again.
    foreach v_name in array v_names loop
      perform net.http_delete(
        url := rtrim(v_url, '/') || '/storage/v1/object/chat-uploads/' || v_name,
        headers := jsonb_build_object('apikey', v_key, 'Authorization', 'Bearer ' || v_key),
        timeout_milliseconds := 10000
      );
    end loop;
  end if;

  v_summary := jsonb_build_object(
    case when p_dry_run then 'orphans_found' else 'delete_requests_sent' end, coalesce(array_length(v_names, 1), 0),
    'capped', coalesce(array_length(v_names, 1), 0) >= p_max_files,
    'unexpected_names', v_unexpected,
    'min_age', p_min_age::text,
    'files', coalesce(to_jsonb(v_names), '[]'::jsonb)
  );
  insert into public.retention_runs (job, dry_run, summary) values ('chat_uploads', p_dry_run, v_summary);
  return v_summary;
end $$;

-- Supabase grants EXECUTE on new public functions to anon/authenticated by default -- without these
-- revokes any student could call them through the API.
revoke all on function public._retention_apply(text, text, text, boolean, int, int) from public, anon, authenticated;
revoke all on function public.run_retention_cleanup(boolean, int, int) from public, anon, authenticated;
revoke all on function public.sweep_orphan_chat_uploads(boolean, interval, int) from public, anon, authenticated;
revoke all on function public.sync_chat_upload_refs() from public, anon, authenticated;


-- PART 3 (dry runs: change nothing except adding a retention_runs row) ---------------------------
-- Row rules: per table, "eligible" = rows today's run would delete/anonymise, "total_rows" = all rows.
-- The auth.audit_log_entries line is the count you asked to see first (total and older than 12 months).
-- Any "error" here (most likely on auth.audit_log_entries or cron.job_run_details) -> send it to me.
select jsonb_pretty(public.run_retention_cleanup(true));

-- File sweep: run AFTER create_chat_uploads_bucket.sql and migrate_chat_uploads_private.py --apply.
-- "orphans_found" = unlinked files older than 7 days that a real run would delete.
select jsonb_pretty(public.sweep_orphan_chat_uploads(true));


-- PART 4 (first real run, then the schedule) ------------------------------------------------------
-- Run the row cleanup once by hand and look at the result before scheduling it.
select jsonb_pretty(public.run_retention_cleanup(false));

-- 21:30 / 21:45 UTC = 03:00 / 03:15 IST. Re-running cron.schedule with the same name updates the job.
select cron.schedule('retention-rows-daily', '30 21 * * *',
  $$set statement_timeout = '15min'; select public.run_retention_cleanup(false)$$);
select cron.schedule('retention-chat-uploads-daily', '45 21 * * *',
  $$set statement_timeout = '15min'; select public.sweep_orphan_chat_uploads(false)$$);


-- PART 5 (checks, any time) -----------------------------------------------------------------------
select jobname, schedule, active from cron.job where jobname like 'retention-%';
select j.jobname, d.status, d.return_message, d.start_time, d.end_time
from cron.job_run_details d join cron.job j on j.jobid = d.jobid
where j.jobname like 'retention-%' order by d.start_time desc limit 10;
select job, dry_run, ran_at, summary from public.retention_runs order by ran_at desc limit 10;
-- The file deletes' HTTP results (pg_net keeps them about 6 hours). Expected status_code 200.
select status_code, left(content::text, 120) as body, created from net._http_response order by created desc limit 20;
