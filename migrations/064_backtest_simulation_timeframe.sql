-- 064: simulation driver of a backtest.
--
-- The replay clock may run on bars finer than the primary chart (e.g. 1m
-- under a 4h strategy): fills, TP/SL, alerts and the forming bars of every
-- chart follow this timeframe. NULL = legacy, the primary chart is the clock.
-- See docs/valutazione-backtest-driver-e-dsl-multichart.md (F1).
--
-- Safe to re-run.

BEGIN;

ALTER TABLE strategy_backtests
    ADD COLUMN IF NOT EXISTS simulation_timeframe VARCHAR(10);

COMMENT ON COLUMN strategy_backtests.simulation_timeframe IS 'Replay clock timeframe when finer than the primary chart (1m, 5m, ...); NULL = primary chart bars';

COMMIT;
