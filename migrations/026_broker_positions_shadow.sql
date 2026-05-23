-- Migration 026: Historical bootstrap for the broker-authoritative positions cutover
--
-- This migration created the temporary `broker_positions_shadow` table used by
-- the original compare-and-cutover rollout. Migration 028 renames that table
-- to `account_positions` and makes it canonical. Keep this file only so a
-- sequential fresh install can still replay the historical migration chain.

BEGIN;

CREATE TABLE IF NOT EXISTS broker_positions_shadow (
    id SERIAL PRIMARY KEY,
    account_id INTEGER NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    connection_id INTEGER NULL,
    broker_account_id VARCHAR(64) NOT NULL,
    broker_type VARCHAR(32) NOT NULL,
    position_key VARCHAR(255) NOT NULL,
    instrument_key VARCHAR(255) NOT NULL,
    symbol VARCHAR(64) NOT NULL,
    asset_type VARCHAR(32) NULL,
    position_bucket VARCHAR(32) NOT NULL DEFAULT 'net',
    side VARCHAR(10) NOT NULL,
    quantity DOUBLE PRECISION NOT NULL DEFAULT 0,
    avg_price DOUBLE PRECISION NULL,
    market_value DOUBLE PRECISION NULL,
    currency VARCHAR(16) NULL,
    snapshot_id VARCHAR(255) NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    extra JSONB NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_broker_positions_shadow_account_position_key
    ON broker_positions_shadow(account_id, position_key);

CREATE INDEX IF NOT EXISTS ix_broker_positions_shadow_account_id
    ON broker_positions_shadow(account_id);

CREATE INDEX IF NOT EXISTS ix_broker_positions_shadow_connection_id
    ON broker_positions_shadow(connection_id);

CREATE INDEX IF NOT EXISTS ix_broker_positions_shadow_broker_account_id
    ON broker_positions_shadow(broker_account_id);

CREATE INDEX IF NOT EXISTS ix_broker_positions_shadow_symbol
    ON broker_positions_shadow(symbol);

CREATE INDEX IF NOT EXISTS ix_broker_positions_shadow_observed_at
    ON broker_positions_shadow(observed_at DESC);

CREATE INDEX IF NOT EXISTS ix_broker_positions_shadow_snapshot_id
    ON broker_positions_shadow(snapshot_id);

COMMIT;