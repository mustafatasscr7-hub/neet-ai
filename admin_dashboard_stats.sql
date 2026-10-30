-- Read-only aggregate functions for admin-dashboard.html's "Usage & Cost" tab. Every function here
-- only SELECTs (provider_usage_log, usage_log, user_plan, auth.users) -- none of them write, and
-- none of them change any existing table, rate constant, cap or cost-logging behavior. Cost/token
-- caps themselves stay Python-side constants (DAILY_TOKEN_BUDGET_FREE etc. in server.py);
-- admin_near_cap_counts takes them as parameters instead of duplicating them here, so this file
-- never goes stale if those constants change.
--
-- All functions: SECURITY DEFINER with a fixed search_path (so they can read auth.users like the
-- existing retention_cleanup.sql functions already do), EXECUTE revoked from public/anon/
-- authenticated and granted only to service_role -- the backend calls these via
-- /rest/v1/rpc/<name> using the service-role key (ADMIN_HEADERS in server.py), same pattern as
-- increment_guest_device_usage etc. No row of personal data leaves any of them except
-- admin_top_users_by_cost's masked phone + user_id.
--
-- Safe to run multiple times in the Supabase SQL editor (every CREATE is OR REPLACE / preceded by
-- a matching DROP IF EXISTS for the table-returning ones, whose column list could otherwise not be
-- changed in place by a future edit).


-- ---------- Spend summary: today / 7d / 30d / this month + a zero-filled 30-day daily series ----------
create or replace function public.admin_spend_summary()
returns jsonb
language plpgsql security definer set search_path = public, pg_temp as $$
declare
  v_ist_today date := (now() at time zone 'Asia/Kolkata')::date;
  v_month_start date := date_trunc('month', v_ist_today)::date;
  v_daily jsonb;
  v_today jsonb;
  v_7d jsonb;
  v_30d jsonb;
  v_month jsonb;
begin
  with days as (
    select generate_series(v_ist_today - 29, v_ist_today, interval '1 day')::date as d
  ),
  agg as (
    select (created_at at time zone 'Asia/Kolkata')::date as d,
           sum(cost) as cost, sum(input_tokens) as input_tokens, sum(output_tokens) as output_tokens,
           sum(cache_hit_tokens) as cache_hit_tokens, count(*) as calls
    from public.provider_usage_log
    where (created_at at time zone 'Asia/Kolkata')::date >= v_ist_today - 29
    group by 1
  )
  select jsonb_agg(jsonb_build_object(
           'date', days.d, 'cost', coalesce(agg.cost, 0), 'input_tokens', coalesce(agg.input_tokens, 0),
           'output_tokens', coalesce(agg.output_tokens, 0), 'cache_hit_tokens', coalesce(agg.cache_hit_tokens, 0),
           'calls', coalesce(agg.calls, 0)
         ) order by days.d)
  into v_daily
  from days left join agg on agg.d = days.d;

  select jsonb_build_object('cost', coalesce(sum(cost), 0), 'input_tokens', coalesce(sum(input_tokens), 0),
           'output_tokens', coalesce(sum(output_tokens), 0), 'cache_hit_tokens', coalesce(sum(cache_hit_tokens), 0),
           'calls', coalesce(count(*), 0))
  into v_today from public.provider_usage_log where (created_at at time zone 'Asia/Kolkata')::date = v_ist_today;

  select jsonb_build_object('cost', coalesce(sum(cost), 0), 'input_tokens', coalesce(sum(input_tokens), 0),
           'output_tokens', coalesce(sum(output_tokens), 0), 'cache_hit_tokens', coalesce(sum(cache_hit_tokens), 0),
           'calls', coalesce(count(*), 0))
  into v_7d from public.provider_usage_log where (created_at at time zone 'Asia/Kolkata')::date >= v_ist_today - 6;

  select jsonb_build_object('cost', coalesce(sum(cost), 0), 'input_tokens', coalesce(sum(input_tokens), 0),
           'output_tokens', coalesce(sum(output_tokens), 0), 'cache_hit_tokens', coalesce(sum(cache_hit_tokens), 0),
           'calls', coalesce(count(*), 0))
  into v_30d from public.provider_usage_log where (created_at at time zone 'Asia/Kolkata')::date >= v_ist_today - 29;

  select jsonb_build_object('cost', coalesce(sum(cost), 0), 'input_tokens', coalesce(sum(input_tokens), 0),
           'output_tokens', coalesce(sum(output_tokens), 0), 'cache_hit_tokens', coalesce(sum(cache_hit_tokens), 0),
           'calls', coalesce(count(*), 0))
  into v_month from public.provider_usage_log where (created_at at time zone 'Asia/Kolkata')::date >= v_month_start;

  return jsonb_build_object('today', v_today, 'last_7d', v_7d, 'last_30d', v_30d, 'month', v_month, 'daily', coalesce(v_daily, '[]'::jsonb));
end $$;

revoke all on function public.admin_spend_summary() from public, anon, authenticated;
grant execute on function public.admin_spend_summary() to service_role;


-- ---------- Spend by provider (the provider column already encodes the model, e.g.
-- 'deepseek-flash', 'qwen-flash', 'gemini-3.5-flash-lite') since p_since, grouped ----------
drop function if exists public.admin_spend_by_provider(timestamptz);
create function public.admin_spend_by_provider(p_since timestamptz)
returns table(provider text, calls bigint, input_tokens bigint, output_tokens bigint, cache_hit_tokens bigint,
              cost numeric, peak_count bigint, output_capped_count bigint)
language sql security definer set search_path = public, pg_temp as $$
  select provider, count(*)::bigint, coalesce(sum(input_tokens), 0)::bigint, coalesce(sum(output_tokens), 0)::bigint,
         coalesce(sum(cache_hit_tokens), 0)::bigint, coalesce(sum(cost), 0)::numeric,
         coalesce(sum(peak_window::int), 0)::bigint, coalesce(sum(output_capped::int), 0)::bigint
  from public.provider_usage_log
  where created_at >= p_since
  group by provider
  order by sum(cost) desc;
$$;

revoke all on function public.admin_spend_by_provider(timestamptz) from public, anon, authenticated;
grant execute on function public.admin_spend_by_provider(timestamptz) to service_role;


-- ---------- Spend by endpoint (/chat, /solve, /title, ...) since p_since, grouped ----------
drop function if exists public.admin_spend_by_endpoint(timestamptz);
create function public.admin_spend_by_endpoint(p_since timestamptz)
returns table(endpoint text, calls bigint, input_tokens bigint, output_tokens bigint, cost numeric)
language sql security definer set search_path = public, pg_temp as $$
  select endpoint, count(*)::bigint, coalesce(sum(input_tokens), 0)::bigint, coalesce(sum(output_tokens), 0)::bigint,
         coalesce(sum(cost), 0)::numeric
  from public.provider_usage_log
  where created_at >= p_since
  group by endpoint
  order by sum(cost) desc;
$$;

revoke all on function public.admin_spend_by_endpoint(timestamptz) from public, anon, authenticated;
grant execute on function public.admin_spend_by_endpoint(timestamptz) to service_role;


-- ---------- User counts: total (auth.users), new per day (last 30 days), plan split ----------
-- No row in user_plan means Free (same fallback get_user_plan() uses), so free = total - pro - max.
create or replace function public.admin_user_stats()
returns jsonb
language plpgsql security definer set search_path = public, auth, pg_temp as $$
declare
  v_ist_today date := (now() at time zone 'Asia/Kolkata')::date;
  v_total bigint;
  v_pro bigint;
  v_max bigint;
  v_new_today bigint;
  v_new_7d bigint;
  v_new_per_day jsonb;
begin
  select count(*) into v_total from auth.users;
  select count(*) into v_pro from public.user_plan where plan = 'pro';
  select count(*) into v_max from public.user_plan where plan = 'max';
  select count(*) into v_new_today from auth.users where (created_at at time zone 'Asia/Kolkata')::date = v_ist_today;
  select count(*) into v_new_7d from auth.users where (created_at at time zone 'Asia/Kolkata')::date >= v_ist_today - 6;

  with days as (
    select generate_series(v_ist_today - 29, v_ist_today, interval '1 day')::date as d
  ),
  agg as (
    select (created_at at time zone 'Asia/Kolkata')::date as d, count(*) as c
    from auth.users
    where (created_at at time zone 'Asia/Kolkata')::date >= v_ist_today - 29
    group by 1
  )
  select jsonb_agg(jsonb_build_object('date', days.d, 'count', coalesce(agg.c, 0)) order by days.d)
  into v_new_per_day
  from days left join agg on agg.d = days.d;

  return jsonb_build_object(
    'total_users', v_total,
    'plan_free', greatest(v_total - v_pro - v_max, 0),
    'plan_pro', v_pro,
    'plan_max', v_max,
    'new_today', v_new_today,
    'new_7d', v_new_7d,
    'new_per_day', coalesce(v_new_per_day, '[]'::jsonb)
  );
end $$;

revoke all on function public.admin_user_stats() from public, anon, authenticated;
grant execute on function public.admin_user_stats() to service_role;


-- ---------- Active users: distinct user_id in provider_usage_log, last 24h / 7d ----------
create or replace function public.admin_active_user_counts()
returns jsonb
language sql security definer set search_path = public, pg_temp as $$
  select jsonb_build_object(
    'last_24h', (select count(distinct user_id) from public.provider_usage_log
                 where user_id is not null and created_at >= now() - interval '24 hours'),
    'last_7d', (select count(distinct user_id) from public.provider_usage_log
                where user_id is not null and created_at >= now() - interval '7 days')
  );
$$;

revoke all on function public.admin_active_user_counts() from public, anon, authenticated;
grant execute on function public.admin_active_user_counts() to service_role;


-- ---------- Near-cap users today (daily) and this IST week (weekly), against thresholds the
-- caller computes from its own cap constants (e.g. 0.8 * DAILY_TOKEN_BUDGET_FREE) -- this file
-- never hardcodes a cap number, so it can't drift from server.py's real ones. ----------
create or replace function public.admin_near_cap_counts(
  p_daily_free numeric, p_daily_pro numeric, p_daily_max numeric,
  p_weekly_free numeric, p_weekly_pro numeric, p_weekly_max numeric
)
returns jsonb
language plpgsql security definer set search_path = public, pg_temp as $$
declare
  v_ist_today date := (now() at time zone 'Asia/Kolkata')::date;
  v_daily_free bigint; v_daily_pro bigint; v_daily_max bigint;
  v_weekly_free bigint; v_weekly_pro bigint; v_weekly_max bigint;
begin
  with today_usage as (
    select user_id, tokens_used from public.usage_log where usage_date = v_ist_today
  ),
  joined as (
    select tu.tokens_used, coalesce(up.plan, 'free') as plan
    from today_usage tu left join public.user_plan up on up.user_id = tu.user_id
  )
  select count(*) filter (where plan = 'free' and tokens_used >= p_daily_free),
         count(*) filter (where plan = 'pro' and tokens_used >= p_daily_pro),
         count(*) filter (where plan = 'max' and tokens_used >= p_daily_max)
  into v_daily_free, v_daily_pro, v_daily_max
  from joined;

  with week_usage as (
    select user_id, sum(tokens_used) as tok from public.usage_log
    where usage_date >= v_ist_today - 6 group by user_id
  ),
  joined as (
    select wu.tok, coalesce(up.plan, 'free') as plan
    from week_usage wu left join public.user_plan up on up.user_id = wu.user_id
  )
  select count(*) filter (where plan = 'free' and tok >= p_weekly_free),
         count(*) filter (where plan = 'pro' and tok >= p_weekly_pro),
         count(*) filter (where plan = 'max' and tok >= p_weekly_max)
  into v_weekly_free, v_weekly_pro, v_weekly_max
  from joined;

  return jsonb_build_object(
    'daily', jsonb_build_object('free', v_daily_free, 'pro', v_daily_pro, 'max', v_daily_max),
    'weekly', jsonb_build_object('free', v_weekly_free, 'pro', v_weekly_pro, 'max', v_weekly_max)
  );
end $$;

revoke all on function public.admin_near_cap_counts(numeric, numeric, numeric, numeric, numeric, numeric) from public, anon, authenticated;
grant execute on function public.admin_near_cap_counts(numeric, numeric, numeric, numeric, numeric, numeric) to service_role;


-- ---------- Top N users by cost since p_since -- user_id + plan + a MASKED phone only. Never the
-- full phone, never the email, never the name. India-shaped numbers (+91 and a 10-digit number)
-- mask to e.g. "+91 98xxxxxx79"; anything shorter is masked entirely. ----------
create or replace function public._admin_mask_phone(p_phone text)
returns text
language plpgsql immutable as $$
declare
  v_digits text;
  v_cc text;
  v_rest text;
begin
  if p_phone is null or length(trim(p_phone)) = 0 then
    return null;
  end if;
  v_digits := regexp_replace(p_phone, '\D', '', 'g');
  if length(v_digits) <= 6 then
    return repeat('x', length(v_digits));
  end if;
  -- Assume a 2-digit country code (India = 91) followed by the subscriber number, matching how
  -- Supabase phone auth stores it (digits only, no leading +). Falls back gracefully if that
  -- assumption is wrong -- still masks the middle, just without a clean country-code split.
  v_cc := left(v_digits, 2);
  v_rest := substring(v_digits from 3);
  if length(v_rest) <= 4 then
    return '+' || v_cc || ' ' || repeat('x', length(v_rest));
  end if;
  return '+' || v_cc || ' ' || left(v_rest, 2) || repeat('x', length(v_rest) - 4) || right(v_rest, 2);
end $$;

revoke all on function public._admin_mask_phone(text) from public, anon, authenticated;
grant execute on function public._admin_mask_phone(text) to service_role;

drop function if exists public.admin_top_users_by_cost(timestamptz, int);
create function public.admin_top_users_by_cost(p_since timestamptz, p_limit int default 10)
returns table(user_id uuid, plan text, masked_phone text, cost numeric, calls bigint)
language plpgsql security definer set search_path = public, auth, pg_temp as $$
begin
  return query
  with agg as (
    select pul.user_id as uid, sum(pul.cost) as total_cost, count(*) as total_calls
    from public.provider_usage_log pul
    where pul.created_at >= p_since and pul.user_id is not null
    group by pul.user_id
    order by sum(pul.cost) desc
    limit p_limit
  )
  select a.uid, coalesce(up.plan, 'free'), public._admin_mask_phone(u.phone), a.total_cost, a.total_calls
  from agg a
  left join public.user_plan up on up.user_id = a.uid
  left join auth.users u on u.id = a.uid
  order by a.total_cost desc;
end $$;

revoke all on function public.admin_top_users_by_cost(timestamptz, int) from public, anon, authenticated;
grant execute on function public.admin_top_users_by_cost(timestamptz, int) to service_role;

-- Make the new functions visible to the API straight away.
notify pgrst, 'reload schema';
