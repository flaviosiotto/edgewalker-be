-- Migration: Add status tracking and agent support to strategy_backtests
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

-- Create indexes
CREATE INDEX IF NOT EXISTS ix_strategy_backtests_status ON strategy_backtests(status);
CREATE INDEX IF NOT EXISTS ix_strategy_backtests_agent_id ON strategy_backtests(agent_id);

-- Update existing backtests to 'completed' status if they have metrics
UPDATE strategy_backtests 
SET status = 'completed', completed_at = created_at 
WHERE metrics IS NOT NULL AND status = 'pending';

-- Comments
COMMENT ON COLUMN strategy_backtests.status IS 'Backtest status: pending, running, completed, failed';
COMMENT ON COLUMN strategy_backtests.agent_id IS 'Agent that executes this backtest via n8n webhook';
