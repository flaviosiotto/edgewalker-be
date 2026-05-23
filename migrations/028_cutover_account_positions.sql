-- Migration 028: Make broker-authoritative current positions canonical as account_positions
--
-- Any `broker_positions_shadow` references below are historical compatibility
-- hooks for upgrading databases created before the direct cutover. After this
-- migration, the canonical table name is `account_positions`.

BEGIN;

DO $$
BEGIN
    IF to_regclass('public.account_positions') IS NULL
       AND to_regclass('public.broker_positions_shadow') IS NOT NULL THEN
        EXECUTE 'ALTER TABLE public.broker_positions_shadow RENAME TO account_positions';
    END IF;

    IF to_regclass('public.account_positions') IS NOT NULL
       AND to_regclass('public.positions') IS NOT NULL THEN
        EXECUTE 'DROP TABLE public.positions';
    END IF;
END
$$;

ALTER TABLE IF EXISTS public.broker_positions_shadow DROP COLUMN IF EXISTS unrealized_pnl;
ALTER TABLE IF EXISTS public.account_positions DROP COLUMN IF EXISTS unrealized_pnl;

DO $$
BEGIN
    IF to_regclass('public.account_positions') IS NULL THEN
        RAISE NOTICE 'Skipping account_positions cutover because source table is missing';
        RETURN;
    END IF;

    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'account_positions'
          AND column_name = 'first_seen_at'
    ) AND NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'account_positions'
          AND column_name = 'opened_at'
    ) THEN
        EXECUTE 'ALTER TABLE public.account_positions RENAME COLUMN first_seen_at TO opened_at';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'account_positions'
          AND column_name = 'last_seen_at'
    ) AND NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'account_positions'
          AND column_name = 'updated_at'
    ) THEN
        EXECUTE 'ALTER TABLE public.account_positions RENAME COLUMN last_seen_at TO updated_at';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'account_positions'
          AND column_name = 'strategy_live_id'
    ) THEN
        EXECUTE 'ALTER TABLE public.account_positions ADD COLUMN strategy_live_id INTEGER NULL';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'account_positions'
          AND column_name = 'status'
    ) THEN
        EXECUTE 'ALTER TABLE public.account_positions ADD COLUMN status VARCHAR(10) NOT NULL DEFAULT ''open''';
    END IF;

    EXECUTE 'UPDATE public.account_positions SET status = ''open'' WHERE status IS DISTINCT FROM ''open''';

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_account_positions_strategy_live_id'
    ) THEN
        EXECUTE 'ALTER TABLE public.account_positions ADD CONSTRAINT fk_account_positions_strategy_live_id FOREIGN KEY (strategy_live_id) REFERENCES public.strategy_live(id) ON DELETE SET NULL';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_indexes
        WHERE schemaname = 'public'
          AND indexname = 'ix_account_positions_strategy_live_id'
    ) THEN
        EXECUTE 'CREATE INDEX ix_account_positions_strategy_live_id ON public.account_positions(strategy_live_id)';
    END IF;
END
$$;

ALTER INDEX IF EXISTS public.broker_positions_shadow_account_id_idx RENAME TO account_positions_account_id_idx;
ALTER INDEX IF EXISTS public.broker_positions_shadow_connection_id_idx RENAME TO account_positions_connection_id_idx;
ALTER INDEX IF EXISTS public.broker_positions_shadow_broker_account_id_idx RENAME TO account_positions_broker_account_id_idx;
ALTER INDEX IF EXISTS public.broker_positions_shadow_broker_type_idx RENAME TO account_positions_broker_type_idx;
ALTER INDEX IF EXISTS public.broker_positions_shadow_position_key_idx RENAME TO account_positions_position_key_idx;
ALTER INDEX IF EXISTS public.broker_positions_shadow_symbol_idx RENAME TO account_positions_symbol_idx;
ALTER INDEX IF EXISTS public.broker_positions_shadow_asset_type_idx RENAME TO account_positions_asset_type_idx;
ALTER INDEX IF EXISTS public.broker_positions_shadow_snapshot_id_idx RENAME TO account_positions_snapshot_id_idx;
ALTER INDEX IF EXISTS public.broker_positions_shadow_observed_at_idx RENAME TO account_positions_observed_at_idx;
ALTER INDEX IF EXISTS public.broker_positions_shadow_last_seen_at_idx RENAME TO account_positions_updated_at_idx;
ALTER SEQUENCE IF EXISTS public.broker_positions_shadow_id_seq RENAME TO account_positions_id_seq;

COMMIT;