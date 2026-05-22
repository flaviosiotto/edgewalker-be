-- Migration 024: Rename live trading projection tables to canonical names
--
-- The order-aggregator now owns canonical broker-account projections.
-- Physical table names should match the domain model directly and avoid the
-- legacy live_* prefix.

BEGIN;

DO $$
BEGIN
	IF to_regclass('public.live_orders') IS NOT NULL AND to_regclass('public.orders') IS NULL THEN
		EXECUTE 'ALTER TABLE public.live_orders RENAME TO orders';
	ELSIF to_regclass('public.live_orders') IS NOT NULL AND to_regclass('public.orders') IS NOT NULL THEN
		RAISE NOTICE 'Skipping rename live_orders -> orders because both relations exist';
	END IF;

	IF to_regclass('public.live_fills') IS NOT NULL AND to_regclass('public.fills') IS NULL THEN
		EXECUTE 'ALTER TABLE public.live_fills RENAME TO fills';
	ELSIF to_regclass('public.live_fills') IS NOT NULL AND to_regclass('public.fills') IS NOT NULL THEN
		RAISE NOTICE 'Skipping rename live_fills -> fills because both relations exist';
	END IF;

	IF to_regclass('public.live_positions') IS NOT NULL AND to_regclass('public.positions') IS NULL THEN
		EXECUTE 'ALTER TABLE public.live_positions RENAME TO positions';
	ELSIF to_regclass('public.live_positions') IS NOT NULL AND to_regclass('public.positions') IS NOT NULL THEN
		RAISE NOTICE 'Skipping rename live_positions -> positions because both relations exist';
	END IF;
END
$$;

DO $$
BEGIN
	IF to_regclass('public.live_orders_id_seq') IS NOT NULL AND to_regclass('public.orders_id_seq') IS NULL THEN
		EXECUTE 'ALTER SEQUENCE public.live_orders_id_seq RENAME TO orders_id_seq';
	END IF;

	IF to_regclass('public.live_fills_id_seq') IS NOT NULL AND to_regclass('public.fills_id_seq') IS NULL THEN
		EXECUTE 'ALTER SEQUENCE public.live_fills_id_seq RENAME TO fills_id_seq';
	END IF;

	IF to_regclass('public.live_positions_id_seq') IS NOT NULL AND to_regclass('public.positions_id_seq') IS NULL THEN
		EXECUTE 'ALTER SEQUENCE public.live_positions_id_seq RENAME TO positions_id_seq';
	END IF;
END
$$;

DO $$
BEGIN
	IF to_regclass('public.live_orders_pkey') IS NOT NULL AND to_regclass('public.orders_pkey') IS NULL THEN
		EXECUTE 'ALTER INDEX public.live_orders_pkey RENAME TO orders_pkey';
	END IF;

	IF to_regclass('public.live_fills_pkey') IS NOT NULL AND to_regclass('public.fills_pkey') IS NULL THEN
		EXECUTE 'ALTER INDEX public.live_fills_pkey RENAME TO fills_pkey';
	END IF;

	IF to_regclass('public.live_positions_pkey') IS NOT NULL AND to_regclass('public.positions_pkey') IS NULL THEN
		EXECUTE 'ALTER INDEX public.live_positions_pkey RENAME TO positions_pkey';
	END IF;
END
$$;

DO $$
BEGIN
	IF to_regclass('public.idx_live_orders_strategy') IS NOT NULL AND to_regclass('public.idx_orders_strategy_live') IS NULL THEN
		EXECUTE 'ALTER INDEX public.idx_live_orders_strategy RENAME TO idx_orders_strategy_live';
	END IF;
	IF to_regclass('public.idx_live_orders_account') IS NOT NULL AND to_regclass('public.idx_orders_account') IS NULL THEN
		EXECUTE 'ALTER INDEX public.idx_live_orders_account RENAME TO idx_orders_account';
	END IF;
	IF to_regclass('public.idx_live_orders_status') IS NOT NULL AND to_regclass('public.idx_orders_status') IS NULL THEN
		EXECUTE 'ALTER INDEX public.idx_live_orders_status RENAME TO idx_orders_status';
	END IF;
	IF to_regclass('public.idx_live_orders_broker') IS NOT NULL AND to_regclass('public.idx_orders_broker') IS NULL THEN
		EXECUTE 'ALTER INDEX public.idx_live_orders_broker RENAME TO idx_orders_broker';
	END IF;
	IF to_regclass('public.ix_live_orders_strategy_live_id') IS NOT NULL AND to_regclass('public.ix_orders_strategy_live_id') IS NULL THEN
		EXECUTE 'ALTER INDEX public.ix_live_orders_strategy_live_id RENAME TO ix_orders_strategy_live_id';
	END IF;
	IF to_regclass('public.ix_live_orders_account_id') IS NOT NULL AND to_regclass('public.ix_orders_account_id') IS NULL THEN
		EXECUTE 'ALTER INDEX public.ix_live_orders_account_id RENAME TO ix_orders_account_id';
	END IF;
	IF to_regclass('public.ix_live_orders_broker_order_id') IS NOT NULL AND to_regclass('public.ix_orders_broker_order_id') IS NULL THEN
		EXECUTE 'ALTER INDEX public.ix_live_orders_broker_order_id RENAME TO ix_orders_broker_order_id';
	END IF;
END
$$;

DO $$
BEGIN
	IF to_regclass('public.idx_live_fills_strategy') IS NOT NULL AND to_regclass('public.idx_fills_strategy_live') IS NULL THEN
		EXECUTE 'ALTER INDEX public.idx_live_fills_strategy RENAME TO idx_fills_strategy_live';
	END IF;
	IF to_regclass('public.idx_live_fills_account') IS NOT NULL AND to_regclass('public.idx_fills_account') IS NULL THEN
		EXECUTE 'ALTER INDEX public.idx_live_fills_account RENAME TO idx_fills_account';
	END IF;
	IF to_regclass('public.idx_live_fills_order') IS NOT NULL AND to_regclass('public.idx_fills_order') IS NULL THEN
		EXECUTE 'ALTER INDEX public.idx_live_fills_order RENAME TO idx_fills_order';
	END IF;
	IF to_regclass('public.idx_live_fills_broker') IS NOT NULL AND to_regclass('public.idx_fills_broker') IS NULL THEN
		EXECUTE 'ALTER INDEX public.idx_live_fills_broker RENAME TO idx_fills_broker';
	END IF;
	IF to_regclass('public.idx_live_fills_trade_time') IS NOT NULL AND to_regclass('public.idx_fills_fill_time') IS NULL THEN
		EXECUTE 'ALTER INDEX public.idx_live_fills_trade_time RENAME TO idx_fills_fill_time';
	END IF;
	IF to_regclass('public.ix_live_fills_strategy_live_id') IS NOT NULL AND to_regclass('public.ix_fills_strategy_live_id') IS NULL THEN
		EXECUTE 'ALTER INDEX public.ix_live_fills_strategy_live_id RENAME TO ix_fills_strategy_live_id';
	END IF;
	IF to_regclass('public.ix_live_fills_account_id') IS NOT NULL AND to_regclass('public.ix_fills_account_id') IS NULL THEN
		EXECUTE 'ALTER INDEX public.ix_live_fills_account_id RENAME TO ix_fills_account_id';
	END IF;
	IF to_regclass('public.ix_live_fills_order_id') IS NOT NULL AND to_regclass('public.ix_fills_order_id') IS NULL THEN
		EXECUTE 'ALTER INDEX public.ix_live_fills_order_id RENAME TO ix_fills_order_id';
	END IF;
	IF to_regclass('public.ix_live_fills_broker_fill_id') IS NOT NULL AND to_regclass('public.ix_fills_broker_fill_id') IS NULL THEN
		EXECUTE 'ALTER INDEX public.ix_live_fills_broker_fill_id RENAME TO ix_fills_broker_fill_id';
	END IF;
	IF to_regclass('public.ix_live_fills_fill_time') IS NOT NULL AND to_regclass('public.ix_fills_fill_time') IS NULL THEN
		EXECUTE 'ALTER INDEX public.ix_live_fills_fill_time RENAME TO ix_fills_fill_time';
	END IF;
END
$$;

DO $$
BEGIN
	IF to_regclass('public.idx_live_positions_strategy') IS NOT NULL AND to_regclass('public.idx_positions_strategy_live') IS NULL THEN
		EXECUTE 'ALTER INDEX public.idx_live_positions_strategy RENAME TO idx_positions_strategy_live';
	END IF;
	IF to_regclass('public.idx_live_positions_account') IS NOT NULL AND to_regclass('public.idx_positions_account') IS NULL THEN
		EXECUTE 'ALTER INDEX public.idx_live_positions_account RENAME TO idx_positions_account';
	END IF;
	IF to_regclass('public.idx_live_positions_symbol') IS NOT NULL AND to_regclass('public.idx_positions_symbol') IS NULL THEN
		EXECUTE 'ALTER INDEX public.idx_live_positions_symbol RENAME TO idx_positions_symbol';
	END IF;
	IF to_regclass('public.idx_live_positions_status') IS NOT NULL AND to_regclass('public.idx_positions_status') IS NULL THEN
		EXECUTE 'ALTER INDEX public.idx_live_positions_status RENAME TO idx_positions_status';
	END IF;
	IF to_regclass('public.ux_live_positions_open') IS NOT NULL AND to_regclass('public.ux_positions_open') IS NULL THEN
		EXECUTE 'ALTER INDEX public.ux_live_positions_open RENAME TO ux_positions_open';
	END IF;
	IF to_regclass('public.ux_live_positions_open_account_symbol') IS NOT NULL AND to_regclass('public.ux_positions_open_account_symbol') IS NULL THEN
		EXECUTE 'ALTER INDEX public.ux_live_positions_open_account_symbol RENAME TO ux_positions_open_account_symbol';
	END IF;
	IF to_regclass('public.ix_live_positions_strategy_live_id') IS NOT NULL AND to_regclass('public.ix_positions_strategy_live_id') IS NULL THEN
		EXECUTE 'ALTER INDEX public.ix_live_positions_strategy_live_id RENAME TO ix_positions_strategy_live_id';
	END IF;
	IF to_regclass('public.ix_live_positions_account_id') IS NOT NULL AND to_regclass('public.ix_positions_account_id') IS NULL THEN
		EXECUTE 'ALTER INDEX public.ix_live_positions_account_id RENAME TO ix_positions_account_id';
	END IF;
	IF to_regclass('public.ix_live_positions_symbol') IS NOT NULL AND to_regclass('public.ix_positions_symbol') IS NULL THEN
		EXECUTE 'ALTER INDEX public.ix_live_positions_symbol RENAME TO ix_positions_symbol';
	END IF;
	IF to_regclass('public.ix_live_positions_status') IS NOT NULL AND to_regclass('public.ix_positions_status') IS NULL THEN
		EXECUTE 'ALTER INDEX public.ix_live_positions_status RENAME TO ix_positions_status';
	END IF;
END
$$;

COMMIT;