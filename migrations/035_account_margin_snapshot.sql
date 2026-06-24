-- Migration: persist normalized margin + unrealized PnL on account-state snapshots
-- Date: 2026-06-24

ALTER TABLE accounts
ADD COLUMN IF NOT EXISTS unrealized_pnl DOUBLE PRECISION NULL,
ADD COLUMN IF NOT EXISTS margin_used DOUBLE PRECISION NULL,
ADD COLUMN IF NOT EXISTS maintenance_margin DOUBLE PRECISION NULL,
ADD COLUMN IF NOT EXISTS init_margin DOUBLE PRECISION NULL;

COMMENT ON COLUMN accounts.unrealized_pnl IS 'Account-level unrealized PnL (mark-to-market) from the last normalized snapshot';
COMMENT ON COLUMN accounts.margin_used IS 'Normalized margin currently used by open positions when provided by the broker';
COMMENT ON COLUMN accounts.maintenance_margin IS 'Normalized maintenance margin requirement when provided by the broker';
COMMENT ON COLUMN accounts.init_margin IS 'Normalized initial margin requirement when provided by the broker';
