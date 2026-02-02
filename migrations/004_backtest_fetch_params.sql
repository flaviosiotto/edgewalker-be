-- Migration: Add fetch and backtest parameters to strategy_backtests table
-- Date: 2026-02-02

-- Data source parameters (for fetch)
ALTER TABLE strategy_backtests ADD COLUMN IF NOT EXISTS source VARCHAR(20) DEFAULT 'ibkr';
ALTER TABLE strategy_backtests ADD COLUMN IF NOT EXISTS timeframe VARCHAR(10) DEFAULT '5m';
ALTER TABLE strategy_backtests ADD COLUMN IF NOT EXISTS asset VARCHAR(20) DEFAULT 'stock';
ALTER TABLE strategy_backtests ADD COLUMN IF NOT EXISTS rth VARCHAR(10) DEFAULT 'true';

-- IBKR-specific parameters
ALTER TABLE strategy_backtests ADD COLUMN IF NOT EXISTS ibkr_config VARCHAR(255);
ALTER TABLE strategy_backtests ADD COLUMN IF NOT EXISTS exchange VARCHAR(20) DEFAULT 'SMART';
ALTER TABLE strategy_backtests ADD COLUMN IF NOT EXISTS currency VARCHAR(10) DEFAULT 'USD';
ALTER TABLE strategy_backtests ADD COLUMN IF NOT EXISTS expiry VARCHAR(20);

-- Backtest execution parameters
ALTER TABLE strategy_backtests ADD COLUMN IF NOT EXISTS initial_capital FLOAT DEFAULT 100000.0;
ALTER TABLE strategy_backtests ADD COLUMN IF NOT EXISTS commission FLOAT DEFAULT 0.0;
