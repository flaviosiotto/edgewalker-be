-- Migration 025: Reconcile projection tables after partial rename rollout
--
-- Problem:
-- - backend startup created empty canonical tables (orders/fills/positions)
--   before migration 024 could rename the populated legacy live_* tables.
-- - migration 024 then skipped the rename because both source and target
--   relations existed, leaving runtime code pointed at empty canonical tables
--   while historical data remained in live_*.
--
-- Safe automatic repair for the observed mixed state:
-- - if both legacy and canonical projection tables exist AND the canonical
--   tables are still empty, drop the empty canonical tables and finish the
--   legacy -> canonical rename.
-- - if canonical tables already contain data, stop and require manual merge.

BEGIN;

DO $$
DECLARE
    legacy_orders_exists boolean := to_regclass('public.live_orders') IS NOT NULL;
    legacy_fills_exists boolean := to_regclass('public.live_fills') IS NOT NULL;
    legacy_positions_exists boolean := to_regclass('public.live_positions') IS NOT NULL;
    canonical_orders_exists boolean := to_regclass('public.orders') IS NOT NULL;
    canonical_fills_exists boolean := to_regclass('public.fills') IS NOT NULL;
    canonical_positions_exists boolean := to_regclass('public.positions') IS NOT NULL;
    orders_count bigint := 0;
    fills_count bigint := 0;
    positions_count bigint := 0;
BEGIN
    IF canonical_orders_exists THEN
        EXECUTE 'SELECT count(*) FROM public.orders' INTO orders_count;
    END IF;
    IF canonical_fills_exists THEN
        EXECUTE 'SELECT count(*) FROM public.fills' INTO fills_count;
    END IF;
    IF canonical_positions_exists THEN
        EXECUTE 'SELECT count(*) FROM public.positions' INTO positions_count;
    END IF;

    IF (legacy_orders_exists OR legacy_fills_exists OR legacy_positions_exists)
       AND (canonical_orders_exists OR canonical_fills_exists OR canonical_positions_exists) THEN
        IF orders_count > 0 OR fills_count > 0 OR positions_count > 0 THEN
            RAISE EXCEPTION
                'Cannot auto-reconcile projection tables: canonical tables already contain data (orders=%, fills=%, positions=%)',
                orders_count,
                fills_count,
                positions_count;
        END IF;

        DROP TABLE IF EXISTS public.fills;
        DROP TABLE IF EXISTS public.positions;
        DROP TABLE IF EXISTS public.orders;
    END IF;
END
$$;

ALTER TABLE IF EXISTS live_orders RENAME TO orders;
ALTER TABLE IF EXISTS live_fills RENAME TO fills;
ALTER TABLE IF EXISTS live_positions RENAME TO positions;

ALTER SEQUENCE IF EXISTS live_orders_id_seq RENAME TO orders_id_seq;
ALTER SEQUENCE IF EXISTS live_fills_id_seq RENAME TO fills_id_seq;
ALTER SEQUENCE IF EXISTS live_positions_id_seq RENAME TO positions_id_seq;

ALTER INDEX IF EXISTS live_orders_pkey RENAME TO orders_pkey;
ALTER INDEX IF EXISTS live_fills_pkey RENAME TO fills_pkey;
ALTER INDEX IF EXISTS live_positions_pkey RENAME TO positions_pkey;

ALTER INDEX IF EXISTS idx_live_orders_strategy RENAME TO idx_orders_strategy_live;
ALTER INDEX IF EXISTS idx_live_orders_account RENAME TO idx_orders_account;
ALTER INDEX IF EXISTS idx_live_orders_status RENAME TO idx_orders_status;
ALTER INDEX IF EXISTS idx_live_orders_broker RENAME TO idx_orders_broker;
ALTER INDEX IF EXISTS ix_live_orders_strategy_live_id RENAME TO ix_orders_strategy_live_id;
ALTER INDEX IF EXISTS ix_live_orders_account_id RENAME TO ix_orders_account_id;
ALTER INDEX IF EXISTS ix_live_orders_broker_order_id RENAME TO ix_orders_broker_order_id;

ALTER INDEX IF EXISTS idx_live_fills_strategy RENAME TO idx_fills_strategy_live;
ALTER INDEX IF EXISTS idx_live_fills_account RENAME TO idx_fills_account;
ALTER INDEX IF EXISTS idx_live_fills_order RENAME TO idx_fills_order;
ALTER INDEX IF EXISTS idx_live_fills_broker RENAME TO idx_fills_broker;
ALTER INDEX IF EXISTS idx_live_fills_trade_time RENAME TO idx_fills_fill_time;
ALTER INDEX IF EXISTS ix_live_fills_strategy_live_id RENAME TO ix_fills_strategy_live_id;
ALTER INDEX IF EXISTS ix_live_fills_account_id RENAME TO ix_fills_account_id;
ALTER INDEX IF EXISTS ix_live_fills_order_id RENAME TO ix_fills_order_id;
ALTER INDEX IF EXISTS ix_live_fills_broker_fill_id RENAME TO ix_fills_broker_fill_id;
ALTER INDEX IF EXISTS ix_live_fills_fill_time RENAME TO ix_fills_fill_time;

ALTER INDEX IF EXISTS idx_live_positions_strategy RENAME TO idx_positions_strategy_live;
ALTER INDEX IF EXISTS idx_live_positions_account RENAME TO idx_positions_account;
ALTER INDEX IF EXISTS idx_live_positions_symbol RENAME TO idx_positions_symbol;
ALTER INDEX IF EXISTS idx_live_positions_status RENAME TO idx_positions_status;
ALTER INDEX IF EXISTS ux_live_positions_open RENAME TO ux_positions_open;
ALTER INDEX IF EXISTS ux_live_positions_open_account_symbol RENAME TO ux_positions_open_account_symbol;
ALTER INDEX IF EXISTS ix_live_positions_strategy_live_id RENAME TO ix_positions_strategy_live_id;
ALTER INDEX IF EXISTS ix_live_positions_account_id RENAME TO ix_positions_account_id;
ALTER INDEX IF EXISTS ix_live_positions_symbol RENAME TO ix_positions_symbol;
ALTER INDEX IF EXISTS ix_live_positions_status RENAME TO ix_positions_status;

COMMIT;