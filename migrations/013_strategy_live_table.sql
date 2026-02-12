-- Migration 013: Separate design-time from runtime
--
-- 1. Create strategy_live table (runtime state for live strategies)
-- 2. Migrate existing live_* data from strategies into strategy_live
-- 3. Re-point live_orders, live_trades, live_positions FKs:
--    strategy_id -> strategy_live_id
-- 4. Add connection_id FK on strategies (for datafeed binding)
-- 5. Drop live_* columns from strategies
-- ============================================================

BEGIN;

-- ── 1. Create strategy_live table ──────────────────────────────────

CREATE TABLE IF NOT EXISTS strategy_live (
    id              SERIAL PRIMARY KEY,
    strategy_id     INTEGER NOT NULL REFERENCES strategies(id) ON DELETE CASCADE,
    status          VARCHAR(20) NOT NULL DEFAULT 'stopped',
    container_id    VARCHAR(100),
    symbol          VARCHAR(30),
    timeframe       VARCHAR(10),
    account_id      INTEGER REFERENCES accounts(id) ON DELETE SET NULL,
    started_at      TIMESTAMPTZ,
    stopped_at      TIMESTAMPTZ,
    error_message   TEXT,
    metrics         JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_strategy_live_strategy_id ON strategy_live(strategy_id);

-- ── 2. Migrate existing live data from strategies ──────────────────

INSERT INTO strategy_live (
    strategy_id, status, container_id, symbol, timeframe,
    account_id, started_at, stopped_at, error_message, metrics,
    created_at, updated_at
)
SELECT
    id,
    COALESCE(live_status, 'stopped'),
    live_container_id,
    live_symbol,
    live_timeframe,
    live_account_id,
    live_started_at,
    live_stopped_at,
    live_error_message,
    live_metrics,
    COALESCE(created_at, NOW()),
    COALESCE(updated_at, NOW())
FROM strategies
WHERE live_status IS NOT NULL
  AND live_status != 'stopped';

-- Also create a row for strategies that had any live history
-- (stopped_at IS NOT NULL means they ran at some point)
INSERT INTO strategy_live (
    strategy_id, status, container_id, symbol, timeframe,
    account_id, started_at, stopped_at, error_message, metrics,
    created_at, updated_at
)
SELECT
    id,
    COALESCE(live_status, 'stopped'),
    live_container_id,
    live_symbol,
    live_timeframe,
    live_account_id,
    live_started_at,
    live_stopped_at,
    live_error_message,
    live_metrics,
    COALESCE(created_at, NOW()),
    COALESCE(updated_at, NOW())
FROM strategies
WHERE (live_status IS NULL OR live_status = 'stopped')
  AND live_stopped_at IS NOT NULL
  AND id NOT IN (SELECT strategy_id FROM strategy_live);

-- ── 3. Re-point live_orders, live_trades, live_positions ───────────

-- 3a. Add strategy_live_id column to each table
ALTER TABLE live_orders    ADD COLUMN IF NOT EXISTS strategy_live_id INTEGER;
ALTER TABLE live_trades    ADD COLUMN IF NOT EXISTS strategy_live_id INTEGER;
ALTER TABLE live_positions ADD COLUMN IF NOT EXISTS strategy_live_id INTEGER;

-- 3b. Populate strategy_live_id from strategy_id
UPDATE live_orders lo
SET strategy_live_id = sl.id
FROM strategy_live sl
WHERE sl.strategy_id = lo.strategy_id;

UPDATE live_trades lt
SET strategy_live_id = sl.id
FROM strategy_live sl
WHERE sl.strategy_id = lt.strategy_id;

UPDATE live_positions lp
SET strategy_live_id = sl.id
FROM strategy_live sl
WHERE sl.strategy_id = lp.strategy_id;

-- 3c. For any orphan rows where no strategy_live exists yet, create one
INSERT INTO strategy_live (strategy_id, status, created_at, updated_at)
SELECT DISTINCT lo.strategy_id, 'stopped', NOW(), NOW()
FROM live_orders lo
WHERE lo.strategy_live_id IS NULL
  AND lo.strategy_id IS NOT NULL
  AND lo.strategy_id NOT IN (SELECT strategy_id FROM strategy_live)
ON CONFLICT DO NOTHING;

INSERT INTO strategy_live (strategy_id, status, created_at, updated_at)
SELECT DISTINCT lt.strategy_id, 'stopped', NOW(), NOW()
FROM live_trades lt
WHERE lt.strategy_live_id IS NULL
  AND lt.strategy_id IS NOT NULL
  AND lt.strategy_id NOT IN (SELECT strategy_id FROM strategy_live)
ON CONFLICT DO NOTHING;

INSERT INTO strategy_live (strategy_id, status, created_at, updated_at)
SELECT DISTINCT lp.strategy_id, 'stopped', NOW(), NOW()
FROM live_positions lp
WHERE lp.strategy_live_id IS NULL
  AND lp.strategy_id IS NOT NULL
  AND lp.strategy_id NOT IN (SELECT strategy_id FROM strategy_live)
ON CONFLICT DO NOTHING;

-- Re-run population for newly created strategy_live rows
UPDATE live_orders lo
SET strategy_live_id = sl.id
FROM strategy_live sl
WHERE sl.strategy_id = lo.strategy_id
  AND lo.strategy_live_id IS NULL;

UPDATE live_trades lt
SET strategy_live_id = sl.id
FROM strategy_live sl
WHERE sl.strategy_id = lt.strategy_id
  AND lt.strategy_live_id IS NULL;

UPDATE live_positions lp
SET strategy_live_id = sl.id
FROM strategy_live sl
WHERE sl.strategy_id = lp.strategy_id
  AND lp.strategy_live_id IS NULL;

-- 3d. Add FK constraints
ALTER TABLE live_orders
    ADD CONSTRAINT fk_live_orders_strategy_live
    FOREIGN KEY (strategy_live_id) REFERENCES strategy_live(id) ON DELETE CASCADE;

ALTER TABLE live_trades
    ADD CONSTRAINT fk_live_trades_strategy_live
    FOREIGN KEY (strategy_live_id) REFERENCES strategy_live(id) ON DELETE CASCADE;

ALTER TABLE live_positions
    ADD CONSTRAINT fk_live_positions_strategy_live
    FOREIGN KEY (strategy_live_id) REFERENCES strategy_live(id) ON DELETE CASCADE;

-- 3e. Drop old strategy_id columns from live tables
ALTER TABLE live_orders    DROP COLUMN IF EXISTS strategy_id;
ALTER TABLE live_trades    DROP COLUMN IF EXISTS strategy_id;
ALTER TABLE live_positions DROP COLUMN IF EXISTS strategy_id;

-- ── 4. Add connection_id to strategies (datafeed binding) ──────────

ALTER TABLE strategies
    ADD COLUMN IF NOT EXISTS connection_id INTEGER REFERENCES connections(id) ON DELETE SET NULL;

-- ── 5. Drop live_* columns from strategies ─────────────────────────

ALTER TABLE strategies DROP COLUMN IF EXISTS live_status;
ALTER TABLE strategies DROP COLUMN IF EXISTS live_container_id;
ALTER TABLE strategies DROP COLUMN IF EXISTS live_symbol;
ALTER TABLE strategies DROP COLUMN IF EXISTS live_timeframe;
ALTER TABLE strategies DROP COLUMN IF EXISTS live_started_at;
ALTER TABLE strategies DROP COLUMN IF EXISTS live_stopped_at;
ALTER TABLE strategies DROP COLUMN IF EXISTS live_error_message;
ALTER TABLE strategies DROP COLUMN IF EXISTS live_metrics;
ALTER TABLE strategies DROP COLUMN IF EXISTS live_account_id;

COMMIT;
