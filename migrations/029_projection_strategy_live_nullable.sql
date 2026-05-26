-- Migration 029: Ensure projection strategy_live_id stays nullable on canonical tables
--
-- Some environments still carry an older NOT NULL shape for strategy_live_id
-- on broker projection tables. The runtime projector now creates account-scoped
-- rows that may legitimately have no strategy_live correlation, so the column
-- must be nullable everywhere.

BEGIN;

ALTER TABLE IF EXISTS public.live_orders
    ALTER COLUMN strategy_live_id DROP NOT NULL;

ALTER TABLE IF EXISTS public.orders
    ALTER COLUMN strategy_live_id DROP NOT NULL;

ALTER TABLE IF EXISTS public.live_fills
    ALTER COLUMN strategy_live_id DROP NOT NULL;

ALTER TABLE IF EXISTS public.fills
    ALTER COLUMN strategy_live_id DROP NOT NULL;

ALTER TABLE IF EXISTS public.live_positions
    ALTER COLUMN strategy_live_id DROP NOT NULL;

ALTER TABLE IF EXISTS public.positions
    ALTER COLUMN strategy_live_id DROP NOT NULL;

ALTER TABLE IF EXISTS public.account_positions
    ALTER COLUMN strategy_live_id DROP NOT NULL;

COMMIT;