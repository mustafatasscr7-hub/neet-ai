-- Guest daily AI budget per device (rate-limit redesign, 2026-10-17).
--
-- Carrier NAT puts many students behind one public IP, so the guest budget (14,000 tokens a day)
-- moves from the IP to the device: the random X-Device-Id the website already generates. The
-- existing guest_usage_log stays, now as the per-IP total of every guest's tokens: server.py stops
-- new guest answers from an IP once that total reaches 280,000 in an IST day (20 guests' worth),
-- so making up device ids gets one IP no further than that.
--
-- Written only by server.py with the service-role key (log_token_usage, /guest-usage-status,
-- /merge-guest-usage). Rows older than 30 days are deleted by run_retention_cleanup()
-- (retention_cleanup.sql).
--
-- Run once in the Supabase SQL editor BEFORE deploying the server.py that uses it -- until then
-- that server.py fails open on the device budget (guests limited only by the per-IP total).
-- Safe to re-run.

create table if not exists public.guest_device_usage_log (
  device_id text not null check (char_length(device_id) between 1 and 64),
  usage_date date not null,           -- IST calendar day, same as guest_usage_log / usage_log
  tokens_used integer not null default 0,
  limit_reached_at timestamptz,       -- start of the 24h block once the day's budget is crossed
  primary key (device_id, usage_date)
);

-- The retention job deletes by date.
create index if not exists guest_device_usage_log_usage_date_idx on public.guest_device_usage_log (usage_date);

-- Service role only: RLS on with no policies, and no client privileges at all.
alter table public.guest_device_usage_log enable row level security;
revoke all on public.guest_device_usage_log from anon, authenticated;

-- Same job as increment_guest_usage for the IP table: add tokens to today's row, creating it.
create or replace function public.increment_guest_device_usage(p_device_id text, p_date date, p_tokens integer)
returns void
language sql
set search_path = public
as $$
  insert into public.guest_device_usage_log (device_id, usage_date, tokens_used)
  values (p_device_id, p_date, p_tokens)
  on conflict (device_id, usage_date)
  do update set tokens_used = public.guest_device_usage_log.tokens_used + excluded.tokens_used;
$$;

revoke execute on function public.increment_guest_device_usage(text, date, integer) from public, anon, authenticated;
grant execute on function public.increment_guest_device_usage(text, date, integer) to service_role;

-- Make the new table and function visible to the API straight away.
notify pgrst, 'reload schema';
