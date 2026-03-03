-- Migration 017: Rename live_trades → live_fills
--
-- Pattern: Order (command) → Fill (event) → Position (state)
-- The table previously called "live_trades" actually stores execution fills
-- (individual broker fills from orders).  Renaming to "live_fills" removes
-- the ambiguity between "trade" (colloquial for position/roundtrip) and
-- "fill" (a single execution event on a single order).
--
-- This is a backwards-compatible rename: existing FK references from
-- live_orders.trades → live_fills are preserved via the table rename.

-- 1. Rename the table
ALTER TABLE IF EXISTS live_trades RENAME TO live_fills;

-- 2. Rename indexes (Postgres auto-renames PK constraint, but custom
--    indexes keep their old name — rename for clarity)
ALTER INDEX IF EXISTS ix_live_trades_strategy_live_id RENAME TO ix_live_fills_strategy_live_id;
ALTER INDEX IF EXISTS ix_live_trades_account_id RENAME TO ix_live_fills_account_id;
ALTER INDEX IF EXISTS ix_live_trades_order_id RENAME TO ix_live_fills_order_id;
ALTER INDEX IF EXISTS ix_live_trades_broker_trade_id RENAME TO ix_live_fills_broker_trade_id;
ALTER INDEX IF EXISTS ix_live_trades_trade_time RENAME TO ix_live_fills_fill_time;

-- 3. Rename the trade_time column to fill_time for clarity
ALTER TABLE live_fills RENAME COLUMN trade_time TO fill_time;

-- 4. Rename broker_trade_id → broker_fill_id for consistency
ALTER TABLE live_fills RENAME COLUMN broker_trade_id TO broker_fill_id;
