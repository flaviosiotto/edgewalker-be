-- Migration: Add live trading state columns to strategies table
-- Date: 2026-02-05

-- Add live trading state columns
ALTER TABLE strategies
ADD COLUMN IF NOT EXISTS live_status VARCHAR(20) NOT NULL DEFAULT 'stopped',
ADD COLUMN IF NOT EXISTS live_container_id VARCHAR(64),
ADD COLUMN IF NOT EXISTS live_symbol VARCHAR(32),
ADD COLUMN IF NOT EXISTS live_timeframe VARCHAR(10),
ADD COLUMN IF NOT EXISTS live_started_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS live_stopped_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS live_error_message TEXT,
ADD COLUMN IF NOT EXISTS live_metrics JSONB;

-- Add index for live_status queries
CREATE INDEX IF NOT EXISTS idx_strategies_live_status ON strategies(live_status);

-- Comment on columns
COMMENT ON COLUMN strategies.live_status IS 'Live trading status: stopped, starting, running, stopping, error';
COMMENT ON COLUMN strategies.live_container_id IS 'Docker container short ID when running';
COMMENT ON COLUMN strategies.live_symbol IS 'Symbol being traded in live mode';
COMMENT ON COLUMN strategies.live_timeframe IS 'Bar timeframe for live trading (e.g., 5s, 1m)';
COMMENT ON COLUMN strategies.live_started_at IS 'Timestamp when live trading was started';
COMMENT ON COLUMN strategies.live_stopped_at IS 'Timestamp when live trading was stopped';
COMMENT ON COLUMN strategies.live_error_message IS 'Error message if live_status is error';
COMMENT ON COLUMN strategies.live_metrics IS 'Runtime metrics snapshot from strategy runner';
