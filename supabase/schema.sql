-- Staff dashboard storage for the SSL chatbot.
--
-- Run once in the Supabase SQL editor (Dashboard -> SQL Editor -> New query).
--
-- Design: content and metrics are kept in separate tables.
--   * chat_metrics  - one numbers-only row per answer. No question or answer
--                     text is ever written here, so ordinary visitor chats
--                     leave no transcript behind.
--   * flagged_chats - the full transcript, but only for answers the pipeline
--                     flagged as bad and that staff need to review.
-- Employee accounts live in Supabase Auth, so there is no users table: invite
-- staff under Authentication -> Users and they can sign in immediately, with no
-- Space restart and no redeploy.

-- ---------------------------------------------------------------------------
-- Numbers for every answer. Content-free.
-- ---------------------------------------------------------------------------
create table if not exists public.chat_metrics (
    id                  text primary key,
    created_at          timestamptz not null default now(),
    status              text,
    response_mode       text,
    path_label          text,
    blocked             boolean not null default false,
    needs_clarification boolean not null default false,

    latency_ms          double precision,
    retrieval_ms        double precision,
    llm_ms              double precision,

    total_tokens        integer,
    input_tokens        integer,
    output_tokens       integer,
    cost_usd            numeric(12, 8),
    llm_call_count      integer,

    confidence_score    double precision,
    is_low_confidence   boolean,
    top_score           double precision,
    score_gap           double precision,
    source_count        integer,
    retrieved_count     integer,

    flagged             boolean not null default false,
    flag_reasons        text[] not null default '{}'
);

create index if not exists chat_metrics_created_at_idx on public.chat_metrics (created_at desc);
create index if not exists chat_metrics_flagged_idx on public.chat_metrics (flagged) where flagged;

-- ---------------------------------------------------------------------------
-- Transcripts, only for flagged answers.
-- ---------------------------------------------------------------------------
create table if not exists public.flagged_chats (
    id              text primary key references public.chat_metrics (id) on delete cascade,
    created_at      timestamptz not null default now(),
    conversation_id text,
    question        text,
    answer          text,
    flag_reasons    text[] not null default '{}',
    sources         jsonb not null default '[]'::jsonb,
    trace           jsonb not null default '{}'::jsonb,
    reviewed_by     text,
    reviewed_at     timestamptz,
    review_note     text
);

create index if not exists flagged_chats_created_at_idx on public.flagged_chats (created_at desc);
create index if not exists flagged_chats_unreviewed_idx on public.flagged_chats (created_at desc)
    where reviewed_at is null;

-- ---------------------------------------------------------------------------
-- Who signed in and which interaction they opened. The dashboard exposes real
-- visitor questions, so views are attributable to a named employee.
-- ---------------------------------------------------------------------------
create table if not exists public.admin_audit_events (
    id         bigserial primary key,
    created_at timestamptz not null default now(),
    username   text not null,
    action     text not null,
    detail     text
);

create index if not exists admin_audit_created_at_idx on public.admin_audit_events (created_at desc);

-- ---------------------------------------------------------------------------
-- Daily rollup. A view, so averages and percentiles are always exact rather
-- than depending on counters staying in sync.
-- ---------------------------------------------------------------------------
create or replace view public.daily_metrics as
select
    (created_at at time zone 'UTC')::date            as day,
    count(*)                                          as chat_count,
    count(*) filter (where flagged)                   as flagged_count,
    count(*) filter (where blocked)                   as blocked_count,
    count(*) filter (where status = 'error')          as error_count,
    count(*) filter (where needs_clarification)       as clarification_count,
    count(*) filter (where is_low_confidence)         as low_confidence_count,

    round(avg(latency_ms)::numeric, 1)                            as avg_latency_ms,
    round((percentile_cont(0.95) within group (order by latency_ms))::numeric, 1) as p95_latency_ms,
    round(avg(retrieval_ms)::numeric, 1)              as avg_retrieval_ms,
    round(avg(llm_ms)::numeric, 1)                    as avg_llm_ms,

    sum(total_tokens)                                 as total_tokens,
    round(avg(total_tokens)::numeric, 0)              as avg_tokens,
    sum(cost_usd)                                     as total_cost_usd,
    round(avg(cost_usd), 8)                           as avg_cost_usd,

    -- Quality proxies computed by the pipeline itself, no judge model needed.
    round(avg(confidence_score)::numeric, 3)          as avg_confidence_score,
    round(avg(top_score)::numeric, 4)                 as avg_top_score,
    round(avg(score_gap)::numeric, 4)                 as avg_score_gap,
    round(avg(source_count)::numeric, 2)              as avg_source_count
from public.chat_metrics
group by 1
order by 1 desc;

-- ---------------------------------------------------------------------------
-- Lock the tables down. The backend uses the service role key, which bypasses
-- RLS; enabling RLS with no permissive policy means a leaked anon key cannot
-- read flagged transcripts or metrics.
-- ---------------------------------------------------------------------------
alter table public.chat_metrics enable row level security;
alter table public.flagged_chats enable row level security;
alter table public.admin_audit_events enable row level security;

-- ---------------------------------------------------------------------------
-- Visitor chat history.
--
-- Entirely separate from the staff tables above. Signing in is optional: an
-- anonymous visitor is stored nowhere, exactly as before. A signed-in visitor
-- gets their own history and can only ever see their own.
--
-- Isolation is enforced by Postgres, not by application code. The backend uses
-- the visitor's own access token for these tables, so the policies below decide
-- what they can see. The service role key is never used against them.
-- ---------------------------------------------------------------------------
create table if not exists public.visitor_conversations (
    id         uuid primary key default gen_random_uuid(),
    user_id    uuid not null references auth.users (id) on delete cascade,
    title      text not null default 'New chat',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists visitor_conversations_user_idx
    on public.visitor_conversations (user_id, updated_at desc);

create table if not exists public.visitor_messages (
    id              bigserial primary key,
    conversation_id uuid not null references public.visitor_conversations (id) on delete cascade,
    user_id         uuid not null references auth.users (id) on delete cascade,
    role            text not null check (role in ('user', 'assistant')),
    content         text not null,
    sources         jsonb not null default '[]'::jsonb,
    created_at      timestamptz not null default now()
);

create index if not exists visitor_messages_conversation_idx
    on public.visitor_messages (conversation_id, created_at);

alter table public.visitor_conversations enable row level security;
alter table public.visitor_messages enable row level security;

-- A visitor may read, write, and delete their own rows and nobody else's.
-- auth.uid() comes from the caller's JWT, so these cannot be bypassed from the
-- client or by a mistake in the backend.
drop policy if exists "own conversations" on public.visitor_conversations;
create policy "own conversations" on public.visitor_conversations
    for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

drop policy if exists "own messages" on public.visitor_messages;
create policy "own messages" on public.visitor_messages
    for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

-- Staff tables carry no visitor identity. Reviewing a flagged answer shows the
-- question and the retrieval trace, never who asked it, so quality review
-- cannot become a way to read one named person's chat history.
