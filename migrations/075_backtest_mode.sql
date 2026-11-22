-- 075: execution mode of a backtest (docs/valutazione-backtest-massivo.md).
--
-- 'replay' is the mode every run had so far: the coordinator publishes the
-- bars on Redis at a wall-clock pace and a runner container evaluates the
-- rules as they arrive (forming bars included, agent turns block the clock).
-- 'bulk' applies the rules in-process over the whole dataset, on closed bars,
-- with no runner container and no agent: meant for long historical periods.
--
-- Existing rows are replay runs.
--
-- Run BEFORE deploying the backend and strategy-backtest (create_all adds
-- tables, never columns; the coordinator's ORM model maps this column too).
BEGIN;

ALTER TABLE strategy_backtests
    ADD COLUMN IF NOT EXISTS mode VARCHAR(16) NOT NULL DEFAULT 'replay';

COMMENT ON COLUMN strategy_backtests.mode IS 'How the run is executed: replay (runner container + streams, forming bars, agent) | bulk (in-process on closed bars, no agent)';

COMMIT;
