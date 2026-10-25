-- Adds a status column to provider_usage_log so failed primary-provider attempts (before the
-- peak-hour fallback kicks in, see _stream_with_peak_fallback in server.py) can be logged
-- alongside successful calls, instead of only ever appearing in the "PROVIDER USAGE:" print line.
--
-- No check constraint/enum -- status is a free-form tag ('ok', 'failed', more may follow) and
-- existing rows all get 'ok' automatically via the default. Safe to run once in the Supabase SQL
-- editor; purely additive, touches no existing row's meaning.

alter table public.provider_usage_log add column if not exists status text not null default 'ok';
