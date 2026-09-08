-- Referral short-code redesign. Run this in the Supabase SQL editor before the updated
-- /referral/* endpoints in server.py will work.
--
-- Replaces the old "code" -- which was never actually stored, just the student's own user_id
-- with hyphens stripped (32 raw hex characters, e.g. "a1b2c3d4e5f6...") -- with a real, stored,
-- short, human-readable code (7 characters, e.g. "7K4XPQR"). The old scheme was mechanically a
-- referral code but never designed as one: unreadable, untypeable by hand, and it exposed the
-- referrer's real user_id to anyone who received a link.
--
-- This is a CLEAN CUTOVER, not a dual-format transition -- old ?ref=<32-hex-uuid> links stop
-- resolving once this ships (see server.py's _user_id_from_referral_code, which now only
-- recognizes the new short-code format). Chosen deliberately: this app has 7 real registered
-- users total right now, so there is essentially no real-world link circulation to preserve
-- compatibility for, and keeping two code formats alive indefinitely is real permanent
-- complexity for approximately zero real benefit at this stage. If that calculus changes later
-- (many more users, links already shared widely), the old derivation function is preserved
-- verbatim in git history and can be restored as a fallback branch in
-- _user_id_from_referral_code.

create table if not exists user_referral_codes (
  user_id uuid primary key,
  code text not null unique,
  created_at timestamptz not null default now()
);

-- No separate index on `code` -- the `unique` constraint above already creates one implicitly.
