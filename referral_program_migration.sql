-- Referral program migration. Run this in the Supabase SQL editor before the new
-- /referral/* endpoints or the bonus-token billing path will work.

-- Referral tracking: one row per referred student, ever. referred_id is UNIQUE so a student
-- can only ever be tied to one referral code for the lifetime of their account -- enforced at
-- the database level, not just in application code.
create table if not exists referrals (
  id uuid primary key default gen_random_uuid(),
  referrer_id uuid not null,
  referred_id uuid not null unique,
  referral_code text not null,
  status text not null default 'pending' check (status in ('pending', 'completed')),
  created_at timestamptz not null default now(),
  reward_granted_at timestamptz,
  subscription_purchase_id text
);
create index if not exists idx_referrals_referrer_id on referrals(referrer_id);
create index if not exists idx_referrals_status on referrals(status);

-- Bonus token balance -- separate from the daily plan cap (usage_log/guest_usage_log), never
-- resets at the normal daily boundary. Only ever moves via a referral credit or being spent by
-- a text-based doubt.
create table if not exists bonus_tokens (
  user_id uuid primary key,
  balance bigint not null default 0,
  updated_at timestamptz not null default now()
);

-- First-ever paid-plan timestamp, set once and never overwritten. Needed to tell a genuine
-- first purchase apart from a later re-subscription (a student set back to 'free' and then
-- 'pro' again later must NOT re-trigger a referral reward).
alter table user_plan add column if not exists first_paid_at timestamptz;

-- Atomic credit (same RPC-based pattern as the existing increment_daily_usage/
-- increment_guest_usage functions this codebase already uses for the daily cap).
create or replace function increment_bonus_tokens(p_user_id uuid, p_amount bigint)
returns bigint
language plpgsql
as $$
declare
  new_balance bigint;
begin
  insert into bonus_tokens (user_id, balance, updated_at)
  values (p_user_id, p_amount, now())
  on conflict (user_id) do update
    set balance = bonus_tokens.balance + excluded.balance,
        updated_at = now()
  returning balance into new_balance;
  return new_balance;
end;
$$;

-- Atomic debit, capped at the available balance -- returns the amount ACTUALLY deducted (may be
-- less than p_amount once the balance runs out mid-doubt), so the caller knows how much of the
-- doubt's real cost still needs to be charged to the student's normal daily plan allowance.
create or replace function decrement_bonus_tokens(p_user_id uuid, p_amount bigint)
returns bigint
language plpgsql
as $$
declare
  current_balance bigint;
  actually_deducted bigint;
begin
  select balance into current_balance from bonus_tokens where user_id = p_user_id for update;
  if current_balance is null then
    return 0;
  end if;
  actually_deducted := least(current_balance, p_amount);
  update bonus_tokens set balance = balance - actually_deducted, updated_at = now()
    where user_id = p_user_id;
  return actually_deducted;
end;
$$;
