-- Run this in the Supabase SQL editor for your existing project (same project as your other bots).

create table if not exists touch_bot_trades (
    id bigserial primary key,
    symbol text not null,
    regime_label text,
    distance_sigma double precision,
    duration_minutes double precision,
    stake double precision,
    payout double precision,
    p_no_touch_est double precision,
    won boolean,
    profit double precision,
    contract_id bigint,
    created_at double precision
);

create table if not exists touch_bot_bayesian_state (
    id int primary key,
    state_json text not null,
    updated_at double precision
);

create table if not exists touch_bot_calibration_runs (
    id bigserial primary key,
    picks_json text not null,
    created_at double precision
);

create table if not exists touch_bot_active_symbols (
    id int primary key,
    slots_json text not null,
    updated_at double precision
);

-- Self-Improvement Engine: versioned Bayesian-tracker snapshots (rollback protection)
create table if not exists touch_bot_model_versions (
    id bigserial primary key,
    tracker_state_json text not null,
    validation_brier double precision,
    created_at double precision
);

create table if not exists touch_bot_self_improvement_runs (
    id bigserial primary key,
    summary_json text not null,
    created_at double precision
);

create table if not exists touch_bot_self_improvement_state (
    id int primary key,
    last_run_at double precision
);

create index if not exists idx_touch_bot_model_versions_created_at on touch_bot_model_versions (created_at);

create index if not exists idx_touch_bot_trades_symbol on touch_bot_trades (symbol);
create index if not exists idx_touch_bot_trades_created_at on touch_bot_trades (created_at);
