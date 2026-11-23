-- 076: parameter exploration of a bulk backtest
-- (docs/valutazione-esplorazione-parametri.md §3.5, §3.9).
--
-- A bulk backtest is an exploration with one trial. A backtest row may carry
-- an exploration request: the coordinator first runs the backtest on the
-- strategy's own values (the row's KPI columns, as today), then repeats it
-- over the same prepared data for every combination of the requested space
-- and stores only the scalar measures of each trial. No separate
-- "exploration" entity.
--
-- strategy_backtests.exploration = {"request": {...}, "state": {...}}:
--   request  written once by the backend at creation
--            ({"dimensions": [...], "oos_fraction": float, "max_trials": int});
--   state    written by the coordinator only (status running | completed |
--            stopped | failed | interrupted, done, failed, total, sampled,
--            space, split_ts, error, started_at, completed_at, elapsed_s).
--
-- strategy_backtest_trials: one row per trial, trial_index 0 = the backtest
-- itself (is_reference). No ledger, no chat, no dataset.
--
-- Run BEFORE deploying the backend and strategy-backtest (create_all adds
-- tables, never columns; the coordinator's ORM model maps the column and the
-- table too).
BEGIN;

ALTER TABLE strategy_backtests
    ADD COLUMN IF NOT EXISTS exploration JSONB;

COMMENT ON COLUMN strategy_backtests.exploration IS 'Parameter exploration of a bulk run: {"request": written by the backend at creation, "state": written by the coordinator}; NULL = no exploration';

CREATE TABLE IF NOT EXISTS strategy_backtest_trials (
    id            SERIAL PRIMARY KEY,
    backtest_id   INTEGER NOT NULL REFERENCES strategy_backtests(id) ON DELETE CASCADE,
    trial_index   INTEGER NOT NULL,
    is_reference  BOOLEAN NOT NULL DEFAULT FALSE,
    params        JSONB NOT NULL,
    metrics       JSONB,
    in_sample     JSONB,
    out_of_sample JSONB,
    error         TEXT,
    elapsed_s     DOUBLE PRECISION,
    CONSTRAINT uq_strategy_backtest_trials_backtest_index UNIQUE (backtest_id, trial_index)
);

CREATE INDEX IF NOT EXISTS ix_strategy_backtest_trials_backtest_id
    ON strategy_backtest_trials (backtest_id);

COMMENT ON TABLE strategy_backtest_trials IS 'One trial of a backtest parameter exploration: the values tried and the scalar measures (whole period, in-sample, out-of-sample); trial_index 0 = the backtest itself';
COMMENT ON COLUMN strategy_backtest_trials.params IS 'Values of the trial by dimension key (main.bb.timeperiod, params.sl_atr)';

COMMIT;
