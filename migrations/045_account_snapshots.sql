-- Migration: account equity history
-- Date: 2026-08-25
--
-- accounts keeps only the latest broker snapshot. To draw a real equity curve
-- and to reconcile the trades ledger against the account we need the history:
-- one append-only row per observed change of the broker account state.
-- Written by the backend account-sync consumer (connection_manager), read by
-- performance_service (equity_start / equity_end / reconciliation gap).
--
-- Safe to re-run: every statement is guarded.

BEGIN;

CREATE TABLE IF NOT EXISTS account_snapshots (
    id                  BIGSERIAL PRIMARY KEY,
    account_id          INTEGER NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    observed_at         TIMESTAMPTZ NOT NULL,
    currency            VARCHAR(10) NOT NULL,
    cash_balance        DOUBLE PRECISION NULL,
    equity              DOUBLE PRECISION NULL,
    unrealized_pnl      DOUBLE PRECISION NULL,
    margin_used         DOUBLE PRECISION NULL,
    source              VARCHAR(32) NOT NULL DEFAULT 'broker_sync',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_account_snapshots_account_observed
    ON account_snapshots (account_id, observed_at);

-- Seed the history with the current snapshot of every account that has one,
-- so the reconciliation window has a starting point from day one.
INSERT INTO account_snapshots (account_id, observed_at, currency, cash_balance, equity, unrealized_pnl, margin_used, source)
SELECT a.id, COALESCE(a.snapshot_at, a.updated_at), a.currency, a.cash_balance, a.equity, a.unrealized_pnl, a.margin_used, 'migration_045'
FROM accounts a
WHERE a.equity IS NOT NULL
  AND NOT EXISTS (SELECT 1 FROM account_snapshots s WHERE s.account_id = a.id);

COMMIT;
