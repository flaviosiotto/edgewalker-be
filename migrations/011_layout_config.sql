-- Migration: Add layout_config JSONB column to strategies and strategy_backtests
--
-- Stores per-user layout preferences for the detail pages:
--   - grid: GridStack widget positions/sizes (array of {id, x, y, w, h})
--   - timeRange: selected time range / date range
--   - timezone: selected timezone string (e.g. "America/New_York")
--   - any other UI preferences
--
-- Example JSON:
-- {
--   "grid": [
--     {"id": "chart", "x": 0, "y": 0, "w": 9, "h": 9},
--     {"id": "rules-chat", "x": 9, "y": 0, "w": 3, "h": 9},
--     {"id": "backtests", "x": 0, "y": 9, "w": 12, "h": 5}
--   ],
--   "timeRange": {"start": "2024-01-01", "end": "2024-12-31"},
--   "timezone": "America/New_York"
-- }

-- Add to strategies table
ALTER TABLE strategies
ADD COLUMN IF NOT EXISTS layout_config JSONB DEFAULT NULL;

-- Add to strategy_backtests table
ALTER TABLE strategy_backtests
ADD COLUMN IF NOT EXISTS layout_config JSONB DEFAULT NULL;
