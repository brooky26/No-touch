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

create index if not exists idx_touch_bot_trades_symbol on touch_bot_trades (symbol);
create index if not exists idx_touch_bot_trades_created_at on touch_bot_trades (created_at);
