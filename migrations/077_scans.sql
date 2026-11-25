-- 077: symbol scans (docs/valutazione-scanner-simboli.md).
--
-- A scan reads one strategy's indicators and rule conditions on the last
-- closed bar of many symbols of a connection. It is a table: one row per
-- symbol, measures only.
--
-- scans      the request and its progress. The backend writes the row at
--            creation (request, config = snapshot of the strategy definition);
--            the coordinator (strategy-backtest) writes status, counters and
--            `plan` (rules as evaluated, window, notes).
-- scan_rows  one row per symbol, written by the coordinator only. Unique on
--            (scan_id, symbol): a scan that is refreshed at every bar close
--            overwrites its rows in place, so readers of an on-demand scan and
--            of a continuous one see the same tables. `mode` says which of the
--            two wrote them; only 'on_demand' exists today.
--
-- Run BEFORE deploying the backend and strategy-backtest (the coordinator's
-- ORM maps both tables; create_all would create them without the comments).
BEGIN;

CREATE TABLE IF NOT EXISTS scans (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    strategy_id     INTEGER REFERENCES strategies(id) ON DELETE SET NULL,
    strategy_name   VARCHAR(255),
    connection_id   INTEGER REFERENCES connections(id) ON DELETE SET NULL,
    broker_type     VARCHAR(32),
    market_type     VARCHAR(32),
    timeframe       VARCHAR(16) NOT NULL,
    mode            VARCHAR(16) NOT NULL DEFAULT 'on_demand',
    request         JSONB NOT NULL DEFAULT '{}'::jsonb,
    config          JSONB NOT NULL,
    plan            JSONB,
    status          VARCHAR(16) NOT NULL DEFAULT 'pending',
    symbols_total   INTEGER,
    symbols_done    INTEGER NOT NULL DEFAULT 0,
    symbols_failed  INTEGER NOT NULL DEFAULT 0,
    error_message   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS ix_scans_user_created ON scans (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_scans_strategy_id ON scans (strategy_id);

COMMENT ON TABLE scans IS 'Symbol scan: a strategy''s indicators and rule conditions read on the last closed bar of many symbols of a connection';
COMMENT ON COLUMN scans.mode IS 'How the rows are produced: on_demand = once, when asked';
COMMENT ON COLUMN scans.request IS 'What was asked: {"symbols": [...] | null, "min_turnover_24h": float | null, "max_symbols": int}';
COMMENT ON COLUMN scans.config IS 'Snapshot of the strategy definition the scan evaluates (indicator refs injected)';
COMMENT ON COLUMN scans.plan IS 'Written by the coordinator: {"rules": [...], "window": int, "warmup": int, "notes": [...]}';
COMMENT ON COLUMN scans.status IS 'pending | running | completed | failed';

CREATE TABLE IF NOT EXISTS scan_rows (
    id                BIGSERIAL PRIMARY KEY,
    scan_id           INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
    symbol            VARCHAR(64) NOT NULL,
    name              VARCHAR(128),
    bar_ts            BIGINT,
    market            JSONB,
    "values"          JSONB,
    rules             JSONB,
    matched_rules     INTEGER NOT NULL DEFAULT 0,
    conditions_true   INTEGER NOT NULL DEFAULT 0,
    conditions_total  INTEGER NOT NULL DEFAULT 0,
    error             TEXT,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_scan_rows_scan_symbol UNIQUE (scan_id, symbol)
);

CREATE INDEX IF NOT EXISTS ix_scan_rows_scan_id ON scan_rows (scan_id);

COMMENT ON TABLE scan_rows IS 'One symbol of a scan: market snapshot, indicator values and condition outcomes on its last closed bar';
COMMENT ON COLUMN scan_rows.bar_ts IS 'Open time (ms) of the closed bar the values refer to';
COMMENT ON COLUMN scan_rows.market IS 'Row of the venue''s bulk ticker: last, change_24h_pct, volume_24h (base), turnover_24h (quote; volume x last when the venue does not report it)';
COMMENT ON COLUMN scan_rows."values" IS 'Measures by field path: price.close, indicators.<key>';
COMMENT ON COLUMN scan_rows.rules IS 'Per rule: {"name", "matched": bool | null, "conditions": [{"passed": bool | null, "actual", "expected" | "lower"/"upper" | "previous"}]}; null = not evaluable in a scan';

COMMIT;
