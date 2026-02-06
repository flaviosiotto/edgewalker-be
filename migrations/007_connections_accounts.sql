-- Migration 007: Add connections and accounts tables
-- Connections hold broker/exchange connection settings (previously in data_sources.config)
-- Accounts represent trading accounts exposed by a connection

BEGIN;

-- ─── CONNECTIONS TABLE ───
CREATE TABLE IF NOT EXISTS connections (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(100) NOT NULL UNIQUE,
    broker_type     VARCHAR(30)  NOT NULL,             -- 'ibkr', 'binance', etc.
    
    -- Connection configuration (JSON for flexibility per broker)
    config          JSONB NOT NULL DEFAULT '{}',       -- host, port, client_id, api_key, etc.
    
    -- Connection status
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    status          VARCHAR(20) NOT NULL DEFAULT 'disconnected',  -- 'connected', 'disconnected', 'error'
    status_message  TEXT,
    last_connected_at TIMESTAMPTZ,
    
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_connections_broker_type ON connections (broker_type);
CREATE INDEX IF NOT EXISTS idx_connections_is_active   ON connections (is_active);

-- ─── ACCOUNTS TABLE ───
CREATE TABLE IF NOT EXISTS accounts (
    id              SERIAL PRIMARY KEY,
    connection_id   INT NOT NULL REFERENCES connections(id) ON DELETE CASCADE,
    
    account_id      VARCHAR(50)  NOT NULL,             -- Broker-specific account code (e.g. "DU1234567")
    display_name    VARCHAR(100),                       -- User-friendly label
    account_type    VARCHAR(30),                        -- 'paper', 'live', 'margin', etc.
    currency        VARCHAR(10)  NOT NULL DEFAULT 'USD',
    
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    
    -- Broker-specific extra info
    extra           JSONB DEFAULT '{}',
    
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    UNIQUE (connection_id, account_id)
);

CREATE INDEX IF NOT EXISTS idx_accounts_connection_id ON accounts (connection_id);

-- ─── ADD connection_id FK TO data_sources (nullable) ───
ALTER TABLE data_sources
    ADD COLUMN IF NOT EXISTS connection_id INT REFERENCES connections(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_data_sources_connection_id ON data_sources (connection_id);

-- ─── ADD account_id FK TO strategies (nullable, for live trading) ───
ALTER TABLE strategies
    ADD COLUMN IF NOT EXISTS live_account_id INT REFERENCES accounts(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_strategies_live_account_id ON strategies (live_account_id);

COMMIT;
