-- Migration: Add status tracking, agent support, and detailed metrics to strategy_backtests
-- Run this migration on your PostgreSQL database

-- Add agent_id column (FK to agent table)
ALTER TABLE strategy_backtests 
ADD COLUMN IF NOT EXISTS agent_id INTEGER REFERENCES agent(id_agent) ON DELETE SET NULL;

-- Add new columns for status tracking
ALTER TABLE strategy_backtests 
ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'pending';

ALTER TABLE strategy_backtests 
ADD COLUMN IF NOT EXISTS started_at TIMESTAMP WITH TIME ZONE;

ALTER TABLE strategy_backtests 
ADD COLUMN IF NOT EXISTS completed_at TIMESTAMP WITH TIME ZONE;

ALTER TABLE strategy_backtests 
ADD COLUMN IF NOT EXISTS error_message TEXT;

ALTER TABLE strategy_backtests 
ADD COLUMN IF NOT EXISTS html_report_url VARCHAR(500);

-- Add typed metric columns (aligned with edgewalker BacktestResult)
ALTER TABLE strategy_backtests 
ADD COLUMN IF NOT EXISTS return_pct DOUBLE PRECISION;

ALTER TABLE strategy_backtests 
ADD COLUMN IF NOT EXISTS sharpe_ratio DOUBLE PRECISION;

ALTER TABLE strategy_backtests 
ADD COLUMN IF NOT EXISTS max_drawdown_pct DOUBLE PRECISION;

ALTER TABLE strategy_backtests 
ADD COLUMN IF NOT EXISTS win_rate_pct DOUBLE PRECISION;

ALTER TABLE strategy_backtests 
ADD COLUMN IF NOT EXISTS profit_factor DOUBLE PRECISION;

ALTER TABLE strategy_backtests 
ADD COLUMN IF NOT EXISTS total_trades INTEGER;

ALTER TABLE strategy_backtests 
ADD COLUMN IF NOT EXISTS equity_final DOUBLE PRECISION;

ALTER TABLE strategy_backtests 
ADD COLUMN IF NOT EXISTS equity_peak DOUBLE PRECISION;

-- Create indexes
CREATE INDEX IF NOT EXISTS ix_strategy_backtests_status ON strategy_backtests(status);
CREATE INDEX IF NOT EXISTS ix_strategy_backtests_agent_id ON strategy_backtests(agent_id);

-- Update existing backtests to 'completed' status if they have metrics
UPDATE strategy_backtests 
SET status = 'completed', completed_at = created_at 
WHERE metrics IS NOT NULL AND status = 'pending';

-- Comments
COMMENT ON COLUMN strategy_backtests.status IS 'Backtest status: pending, running, completed, failed, error';
COMMENT ON COLUMN strategy_backtests.agent_id IS 'Agent that executes this backtest via n8n webhook';
COMMENT ON COLUMN strategy_backtests.return_pct IS 'Total return percentage from edgewalker';
COMMENT ON COLUMN strategy_backtests.sharpe_ratio IS 'Sharpe ratio from edgewalker';
COMMENT ON COLUMN strategy_backtests.max_drawdown_pct IS 'Maximum drawdown percentage';
COMMENT ON COLUMN strategy_backtests.win_rate_pct IS 'Percentage of winning trades';
COMMENT ON COLUMN strategy_backtests.profit_factor IS 'Gross profit / Gross loss ratio';
COMMENT ON COLUMN strategy_backtests.total_trades IS 'Total number of trades executed';
COMMENT ON COLUMN strategy_backtests.equity_final IS 'Final equity value';
COMMENT ON COLUMN strategy_backtests.equity_peak IS 'Peak equity value reached';
