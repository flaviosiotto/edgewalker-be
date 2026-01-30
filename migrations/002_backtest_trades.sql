-- Migration: Update strategy_backtest_trades table to align with edgewalker TradeRecord
-- Run this migration on your PostgreSQL database
-- This migration is idempotent - safe to run multiple times

-- Rename columns to match edgewalker naming convention (only if old names exist)
DO $$ 
BEGIN
    -- ts_open -> entry_time
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='strategy_backtest_trades' AND column_name='ts_open') THEN
        ALTER TABLE strategy_backtest_trades RENAME COLUMN ts_open TO entry_time;
    END IF;
    
    -- ts_close -> exit_time
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='strategy_backtest_trades' AND column_name='ts_close') THEN
        ALTER TABLE strategy_backtest_trades RENAME COLUMN ts_close TO exit_time;
    END IF;
    
    -- side -> direction
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='strategy_backtest_trades' AND column_name='side') THEN
        ALTER TABLE strategy_backtest_trades RENAME COLUMN side TO direction;
    END IF;
    
    -- quantity -> size
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='strategy_backtest_trades' AND column_name='quantity') THEN
        ALTER TABLE strategy_backtest_trades RENAME COLUMN quantity TO size;
    END IF;
    
    -- meta -> extra
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='strategy_backtest_trades' AND column_name='meta') THEN
        ALTER TABLE strategy_backtest_trades RENAME COLUMN meta TO extra;
    END IF;
    
    -- Drop fees column if exists
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='strategy_backtest_trades' AND column_name='fees') THEN
        ALTER TABLE strategy_backtest_trades DROP COLUMN fees;
    END IF;
END $$;

-- Add new columns (IF NOT EXISTS handles idempotency)
ALTER TABLE strategy_backtest_trades 
ADD COLUMN IF NOT EXISTS pnl_pct DOUBLE PRECISION;

ALTER TABLE strategy_backtest_trades 
ADD COLUMN IF NOT EXISTS session_date DATE;

ALTER TABLE strategy_backtest_trades 
ADD COLUMN IF NOT EXISTS exit_reason VARCHAR(50);

ALTER TABLE strategy_backtest_trades 
ADD COLUMN IF NOT EXISTS extra JSONB;

-- Create index on session_date for faster queries
CREATE INDEX IF NOT EXISTS ix_strategy_backtest_trades_session_date ON strategy_backtest_trades(session_date);

-- Comments (only on columns that exist)
DO $$ 
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='strategy_backtest_trades' AND column_name='entry_time') THEN
        COMMENT ON COLUMN strategy_backtest_trades.entry_time IS 'Trade entry timestamp';
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='strategy_backtest_trades' AND column_name='exit_time') THEN
        COMMENT ON COLUMN strategy_backtest_trades.exit_time IS 'Trade exit timestamp';
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='strategy_backtest_trades' AND column_name='direction') THEN
        COMMENT ON COLUMN strategy_backtest_trades.direction IS 'Trade direction: long or short';
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='strategy_backtest_trades' AND column_name='size') THEN
        COMMENT ON COLUMN strategy_backtest_trades.size IS 'Position size';
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='strategy_backtest_trades' AND column_name='pnl_pct') THEN
        COMMENT ON COLUMN strategy_backtest_trades.pnl_pct IS 'Return percentage for this trade';
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='strategy_backtest_trades' AND column_name='session_date') THEN
        COMMENT ON COLUMN strategy_backtest_trades.session_date IS 'Trading session date (ET timezone)';
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='strategy_backtest_trades' AND column_name='exit_reason') THEN
        COMMENT ON COLUMN strategy_backtest_trades.exit_reason IS 'Reason for exit: stop_loss, take_profit, eod, etc.';
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='strategy_backtest_trades' AND column_name='extra') THEN
        COMMENT ON COLUMN strategy_backtest_trades.extra IS 'Additional trade data as JSONB';
    END IF;
END $$;
