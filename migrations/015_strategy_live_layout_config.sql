-- Add layout_config JSONB column to strategy_live table
-- for persisting UI widget positions, chart settings, etc.
ALTER TABLE strategy_live ADD COLUMN IF NOT EXISTS layout_config JSONB;
