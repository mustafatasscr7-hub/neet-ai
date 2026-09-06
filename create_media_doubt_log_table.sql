-- Server-side audit trail for image/PDF doubts (Gemini 3.5 Flash-Lite path in /chat).
--
-- Before this table existed, the only record of an image/PDF doubt was provider_usage_log
-- (token counts/cost, no content) plus whatever the *client* chose to save to `chats` after
-- receiving the full stream -- which turned out to be nothing for every real image/PDF call ever
-- logged (confirmed via a live audit: 6 real Gemini calls total, zero matching `chats` rows for
-- any of them). This table is written server-side, unconditionally, the moment generation
-- completes -- independent of whether the client is still connected or ever saves anything.
--
-- Raw images/PDFs are never stored anywhere (sent as base64 straight to Gemini and discarded) --
-- `files` stores only a hash + mime type + byte size per submitted file, enough for a future
-- audit to confirm what kind of input triggered a given response without paying real storage
-- cost or retaining student-submitted file content.
create table if not exists media_doubt_log (
    id bigint generated always as identity primary key,
    user_id uuid null,
    ip text null,
    doubt_type text not null check (doubt_type in ('image', 'pdf')),
    files jsonb not null default '[]'::jsonb,
    response_text text not null,
    created_at timestamptz not null default now()
);

create index if not exists media_doubt_log_user_id_idx on media_doubt_log (user_id);
create index if not exists media_doubt_log_created_at_idx on media_doubt_log (created_at);

alter table media_doubt_log enable row level security;
-- Locked to the service-role key only -- no anon/authenticated policy at all, same as
-- provider_usage_log and pyq_solution_cache. This is internal audit telemetry, never served to
-- a client.
