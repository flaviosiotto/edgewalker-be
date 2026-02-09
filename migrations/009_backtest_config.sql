-- Migration: Add config JSONB column to strategy_backtests
-- Stores the full strategy configuration snapshot at backtest creation time

ALTER TABLE strategy_backtests
    ADD COLUMN IF NOT EXISTS config JSONB;

-- Backfill existing rows: copy strategy definition into config
UPDATE strategy_backtests bt
SET config = s.definition
FROM strategies s
WHERE bt.strategy_id = s.id
  AND bt.config IS NULL;

COMMENT ON COLUMN strategy_backtests.config IS 'Full strategy configuration snapshot at backtest creation time (immutable)';
