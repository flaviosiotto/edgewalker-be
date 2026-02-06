-- Migration 008: Live Trading Tables (Orders, Trades, Positions)
--
-- Persist live trading state for strategies connected to broker accounts.
-- Enables startup reconciliation and full audit trail.

BEGIN;

-- ── Live Orders ──────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS live_orders (
    id              SERIAL PRIMARY KEY,
    strategy_id     INTEGER NOT NULL REFERENCES strategies(id) ON DELETE CASCADE,
    account_id      INTEGER REFERENCES accounts(id) ON DELETE SET NULL,

    broker_order_id VARCHAR(100),

    symbol          VARCHAR(32) NOT NULL,
    side            VARCHAR(10) NOT NULL,      -- buy / sell
    order_type      VARCHAR(20) NOT NULL,      -- market / limit / stop / stop_limit
    quantity        DOUBLE PRECISION NOT NULL,

    limit_price     DOUBLE PRECISION,
    stop_price      DOUBLE PRECISION,

    filled_quantity DOUBLE PRECISION NOT NULL DEFAULT 0,
    avg_fill_price  DOUBLE PRECISION,
    commission      DOUBLE PRECISION,

    status          VARCHAR(30) NOT NULL DEFAULT 'pending',
    status_message  TEXT,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    submitted_at    TIMESTAMPTZ,
    filled_at       TIMESTAMPTZ,
    cancelled_at    TIMESTAMPTZ,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    extra           JSONB
);

CREATE INDEX IF NOT EXISTS idx_live_orders_strategy ON live_orders(strategy_id);
CREATE INDEX IF NOT EXISTS idx_live_orders_account  ON live_orders(account_id);
CREATE INDEX IF NOT EXISTS idx_live_orders_status   ON live_orders(status);
CREATE INDEX IF NOT EXISTS idx_live_orders_broker   ON live_orders(broker_order_id);

-- ── Live Trades (Fills) ──────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS live_trades (
    id              SERIAL PRIMARY KEY,
    strategy_id     INTEGER NOT NULL REFERENCES strategies(id) ON DELETE CASCADE,
    account_id      INTEGER REFERENCES accounts(id) ON DELETE SET NULL,
    order_id        INTEGER REFERENCES live_orders(id) ON DELETE SET NULL,

    broker_trade_id VARCHAR(100),

    symbol          VARCHAR(32) NOT NULL,
    side            VARCHAR(10) NOT NULL,
    quantity        DOUBLE PRECISION NOT NULL,
    price           DOUBLE PRECISION NOT NULL,
    commission      DOUBLE PRECISION DEFAULT 0,

    realized_pnl    DOUBLE PRECISION,

    trade_time      TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    extra           JSONB
);

CREATE INDEX IF NOT EXISTS idx_live_trades_strategy   ON live_trades(strategy_id);
CREATE INDEX IF NOT EXISTS idx_live_trades_account    ON live_trades(account_id);
CREATE INDEX IF NOT EXISTS idx_live_trades_order      ON live_trades(order_id);
CREATE INDEX IF NOT EXISTS idx_live_trades_broker     ON live_trades(broker_trade_id);
CREATE INDEX IF NOT EXISTS idx_live_trades_trade_time ON live_trades(trade_time);

-- ── Live Positions ───────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS live_positions (
    id              SERIAL PRIMARY KEY,
    strategy_id     INTEGER NOT NULL REFERENCES strategies(id) ON DELETE CASCADE,
    account_id      INTEGER REFERENCES accounts(id) ON DELETE SET NULL,

    symbol          VARCHAR(32) NOT NULL,
    side            VARCHAR(10) NOT NULL DEFAULT 'flat',  -- long / short / flat
    quantity        DOUBLE PRECISION NOT NULL DEFAULT 0,
    avg_price       DOUBLE PRECISION,
    cost_basis      DOUBLE PRECISION,

    unrealized_pnl  DOUBLE PRECISION,
    realized_pnl    DOUBLE PRECISION DEFAULT 0,
    market_value    DOUBLE PRECISION,

    status          VARCHAR(10) NOT NULL DEFAULT 'open',  -- open / closed

    opened_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at       TIMESTAMPTZ,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    extra           JSONB
);

CREATE INDEX IF NOT EXISTS idx_live_positions_strategy ON live_positions(strategy_id);
CREATE INDEX IF NOT EXISTS idx_live_positions_account  ON live_positions(account_id);
CREATE INDEX IF NOT EXISTS idx_live_positions_symbol   ON live_positions(symbol);
CREATE INDEX IF NOT EXISTS idx_live_positions_status   ON live_positions(status);

-- Unique constraint: one open position per strategy+account+symbol
CREATE UNIQUE INDEX IF NOT EXISTS ux_live_positions_open
    ON live_positions(strategy_id, account_id, symbol)
    WHERE status = 'open';

COMMIT;
