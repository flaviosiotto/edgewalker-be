-- Migration: persist normalized account-state snapshots on accounts
-- Date: 2026-05-11

ALTER TABLE accounts
ADD COLUMN IF NOT EXISTS cash_balance DOUBLE PRECISION NULL,
ADD COLUMN IF NOT EXISTS equity DOUBLE PRECISION NULL,
ADD COLUMN IF NOT EXISTS buying_power DOUBLE PRECISION NULL,
ADD COLUMN IF NOT EXISTS available_funds DOUBLE PRECISION NULL,
ADD COLUMN IF NOT EXISTS snapshot_at TIMESTAMPTZ NULL;

CREATE INDEX IF NOT EXISTS idx_accounts_snapshot_at
ON accounts(snapshot_at);

COMMENT ON COLUMN accounts.cash_balance IS 'Normalized current cash balance in account currency';
COMMENT ON COLUMN accounts.equity IS 'Normalized current equity or net liquidation in account currency';
COMMENT ON COLUMN accounts.buying_power IS 'Normalized current buying power when provided by the broker';
COMMENT ON COLUMN accounts.available_funds IS 'Normalized currently available funds when provided by the broker';
COMMENT ON COLUMN accounts.snapshot_at IS 'Timestamp of the last normalized account-state snapshot received from the gateway';