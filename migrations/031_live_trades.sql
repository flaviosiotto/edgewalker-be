-- Migration 031: Materialized FIFO trades (closed lots) with realized PnL
--
-- Each row is one FIFO close match (an opening lot closed by a reducing fill),
-- account-scoped and rebuilt deterministically by the order-aggregator from the
-- fill history. The dashboard and the Trade History view read realized PnL from
-- this table instead of recomputing it from fills on every request.

BEGIN;

CREATE TABLE IF NOT EXISTS public.trades (
    id                BIGSERIAL PRIMARY KEY,
    strategy_live_id  INTEGER REFERENCES public.strategy_live(id) ON DELETE SET NULL,
    account_id        INTEGER NOT NULL REFERENCES public.accounts(id) ON DELETE CASCADE,
    symbol            VARCHAR(32) NOT NULL,
    direction         VARCHAR(10) NOT NULL,          -- long / short (side of the closed lot)
    quantity          DOUBLE PRECISION NOT NULL,
    entry_price       DOUBLE PRECISION,              -- NULL for untrusted carry-in lots
    exit_price        DOUBLE PRECISION NOT NULL,
    multiplier        DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    realized_pnl      DOUBLE PRECISION,              -- NULL when closed against an untrusted lot
    commission        DOUBLE PRECISION DEFAULT 0.0,
    net_pnl           DOUBLE PRECISION,              -- realized_pnl - commission (NULL when realized is NULL)
    trusted           BOOLEAN NOT NULL DEFAULT TRUE,
    entry_fill_id     INTEGER,
    exit_fill_id      INTEGER,
    entry_time        TIMESTAMPTZ,
    exit_time         TIMESTAMPTZ NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    extra             JSONB
);

CREATE INDEX IF NOT EXISTS ix_trades_account_id ON public.trades (account_id);
CREATE INDEX IF NOT EXISTS ix_trades_symbol ON public.trades (symbol);
CREATE INDEX IF NOT EXISTS ix_trades_strategy_live_id ON public.trades (strategy_live_id);
CREATE INDEX IF NOT EXISTS ix_trades_exit_time ON public.trades (exit_time);
CREATE INDEX IF NOT EXISTS ix_trades_entry_time ON public.trades (entry_time);
CREATE INDEX IF NOT EXISTS ix_trades_account_exit_time ON public.trades (account_id, exit_time);

COMMIT;
