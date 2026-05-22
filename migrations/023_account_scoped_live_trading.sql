-- Migration 023: Make live trading projections account-scoped
--
-- Orders, fills, and positions are now canonical broker-account projections.
-- strategy_live_id remains optional correlation metadata and must no longer own
-- the lifecycle of these rows.

BEGIN;

-- Backfill account ownership from strategy_live where possible.
UPDATE live_orders lo
SET account_id = sl.account_id
FROM strategy_live sl
WHERE lo.strategy_live_id = sl.id
  AND lo.account_id IS NULL
  AND sl.account_id IS NOT NULL;

UPDATE live_fills lf
SET account_id = COALESCE(
  lo.account_id,
  (
    SELECT sl.account_id
    FROM strategy_live sl
    WHERE sl.id = lf.strategy_live_id
  )
)
FROM live_orders lo
WHERE lf.order_id = lo.id
  AND lf.account_id IS NULL;

UPDATE live_fills lf
SET account_id = sl.account_id
FROM strategy_live sl
WHERE lf.strategy_live_id = sl.id
  AND lf.account_id IS NULL
  AND sl.account_id IS NOT NULL;

UPDATE live_positions lp
SET account_id = sl.account_id
FROM strategy_live sl
WHERE lp.strategy_live_id = sl.id
  AND lp.account_id IS NULL
  AND sl.account_id IS NOT NULL;

ALTER TABLE live_orders DROP CONSTRAINT IF EXISTS fk_live_orders_strategy_live;
ALTER TABLE live_fills DROP CONSTRAINT IF EXISTS fk_live_fills_strategy_live;
ALTER TABLE live_positions DROP CONSTRAINT IF EXISTS fk_live_positions_strategy_live;

ALTER TABLE live_orders DROP CONSTRAINT IF EXISTS live_orders_strategy_live_id_fkey;
ALTER TABLE live_fills DROP CONSTRAINT IF EXISTS live_fills_strategy_live_id_fkey;
ALTER TABLE live_positions DROP CONSTRAINT IF EXISTS live_positions_strategy_live_id_fkey;

ALTER TABLE live_orders DROP CONSTRAINT IF EXISTS live_orders_account_id_fkey;
ALTER TABLE live_fills DROP CONSTRAINT IF EXISTS live_fills_account_id_fkey;
ALTER TABLE live_positions DROP CONSTRAINT IF EXISTS live_positions_account_id_fkey;

ALTER TABLE live_orders ALTER COLUMN strategy_live_id DROP NOT NULL;
ALTER TABLE live_fills ALTER COLUMN strategy_live_id DROP NOT NULL;
ALTER TABLE live_positions ALTER COLUMN strategy_live_id DROP NOT NULL;

ALTER TABLE live_orders ALTER COLUMN account_id SET NOT NULL;
ALTER TABLE live_fills ALTER COLUMN account_id SET NOT NULL;
ALTER TABLE live_positions ALTER COLUMN account_id SET NOT NULL;

ALTER TABLE live_orders
    ADD CONSTRAINT fk_live_orders_strategy_live_nullable
    FOREIGN KEY (strategy_live_id) REFERENCES strategy_live(id) ON DELETE SET NULL;
ALTER TABLE live_fills
    ADD CONSTRAINT fk_live_fills_strategy_live_nullable
    FOREIGN KEY (strategy_live_id) REFERENCES strategy_live(id) ON DELETE SET NULL;
ALTER TABLE live_positions
    ADD CONSTRAINT fk_live_positions_strategy_live_nullable
    FOREIGN KEY (strategy_live_id) REFERENCES strategy_live(id) ON DELETE SET NULL;

ALTER TABLE live_orders
    ADD CONSTRAINT fk_live_orders_account_required
    FOREIGN KEY (account_id) REFERENCES accounts(id) ON DELETE CASCADE;
ALTER TABLE live_fills
    ADD CONSTRAINT fk_live_fills_account_required
    FOREIGN KEY (account_id) REFERENCES accounts(id) ON DELETE CASCADE;
ALTER TABLE live_positions
    ADD CONSTRAINT fk_live_positions_account_required
    FOREIGN KEY (account_id) REFERENCES accounts(id) ON DELETE CASCADE;

DROP INDEX IF EXISTS ux_live_positions_open;
CREATE UNIQUE INDEX IF NOT EXISTS ux_live_positions_open_account_symbol
    ON live_positions(account_id, symbol)
    WHERE status = 'open';

COMMIT;