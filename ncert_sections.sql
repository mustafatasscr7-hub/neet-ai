-- Run manually in Supabase SQL editor.
-- Holds manually-cleaned NCERT chapter sections, keyed by heading.
-- No embeddings column: retrieval/search is a separate, later feature.

create table if not exists ncert_sections (
    id bigserial primary key,
    class text not null,
    subject text not null,
    chapter text not null,
    heading text not null,
    ordinal int not null,
    text text not null,
    word_count int not null,
    created_at timestamptz not null default now(),
    unique (class, subject, chapter, ordinal)
);

alter table ncert_sections enable row level security;
-- No policies: only the service role (which bypasses RLS) can read or write.
