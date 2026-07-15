-- Migration: link a simulated Account to each backtest
-- Date: 2026-07-15
--
-- So the agent can query GET /accounts/{id} identically in backtest and live.
-- The account row holds identity + seed capital; the live balance/equity is
-- resolved on-read by proxying the backtest service (see accounts endpoint).
-- Parallels strategy_live.account_id for live sessions. ON DELETE SET NULL so
-- purging a simulated account never cascades into backtest history.

ALTER TABLE strategy_backtests
    ADD COLUMN IF NOT EXISTS account_id INT REFERENCES accounts(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_strategy_backtests_account_id ON strategy_backtests (account_id);

COMMENT ON COLUMN strategy_backtests.account_id IS 'FK to the simulated accounts row backing this backtest (parallels strategy_live.account_id).';
