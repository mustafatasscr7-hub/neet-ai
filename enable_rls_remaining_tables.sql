-- Enables RLS on the 4 tables found with no RLS at all during the RLS audit:
-- processed_pdfs, referrals, bonus_tokens, user_referral_codes. All reads/writes to these
-- tables go through server.py's service-role key, never the anon key directly, so no
-- policies are added -- the service role bypasses RLS entirely, and there is no
-- frontend/anon-key access path to preserve.
alter table processed_pdfs enable row level security;
alter table referrals enable row level security;
alter table bonus_tokens enable row level security;
alter table user_referral_codes enable row level security;
