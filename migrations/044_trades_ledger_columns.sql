-- Migration: trades ledger semantics
-- Date: 2026-08-25
--
-- The trades projection becomes the single ledger every performance number is
-- derived from. Fixed invariant per row:
--   realized_pnl  = GROSS realized (before costs)
--   commission    = total cost (commissions + conversion fees), positive
--   swap          = financing / rollover (signed), NULL when the broker has none
--   net_pnl       = realized_pnl + COALESCE(swap, 0) - commission
--   currency      = currency the amounts are expressed in (contract currency)
--   fx_to_account = rate currency -> account currency at exit (1.0 when equal)
--   net_pnl_account_ccy = net_pnl * fx_to_account when the rate is known
--   gap_reason    = why realized_pnl is NULL (untrusted_carry_in,
--                   inventory_incomplete, unmatched) instead of a silent drop
--
-- The order-aggregator rebuilds trades per (account, symbol) from fills, so
-- history is repaired by re-materializing (scripts/rematerialize_trades.py),
-- not by SQL. The backfill below only fixes the IBKR double-counted commission
-- for rows that are not re-materialized: IB CommissionReport.realizedPNL is
-- already net of commissions, so realized_pnl was stored net and net_pnl
-- subtracted the commission a second time.
--
-- Safe to re-run: every statement is guarded.

BEGIN;

ALTER TABLE trades ADD COLUMN IF NOT EXISTS swap DOUBLE PRECISION NULL;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS currency VARCHAR(16) NULL;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS fx_to_account DOUBLE PRECISION NULL;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS net_pnl_account_ccy DOUBLE PRECISION NULL;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS gap_reason VARCHAR(64) NULL;

-- IBKR: gross the broker-net realized back up so net_pnl = realized - commission
-- holds. Idempotent thanks to the extra flag.
UPDATE trades t
SET realized_pnl = t.realized_pnl + COALESCE(t.commission, 0),
    net_pnl = t.realized_pnl,
    extra = COALESCE(t.extra, '{}'::jsonb) || '{"broker_realized_is_net": true, "ledger_backfill": "044"}'::jsonb
FROM fills f
WHERE f.id = t.exit_fill_id
  AND t.realized_pnl IS NOT NULL
  AND t.extra->>'realized_source' = 'broker'
  AND COALESCE(t.extra->>'broker_realized_is_net', 'false') <> 'true'
  AND lower(COALESCE(f.extra->'last_broker_fill_event'->>'broker_type', '')) = 'ibkr';

-- Explain the existing NULL realized rows.
UPDATE trades
SET gap_reason = CASE
        WHEN COALESCE((extra->>'inventory_complete')::boolean, true) = false THEN 'inventory_incomplete'
        WHEN COALESCE((extra->>'carry_in')::boolean, false) THEN 'untrusted_carry_in'
        ELSE 'unmatched'
    END
WHERE realized_pnl IS NULL AND gap_reason IS NULL;

CREATE INDEX IF NOT EXISTS idx_trades_account_exit_time ON trades (account_id, exit_time);
CREATE INDEX IF NOT EXISTS idx_trades_live_exit_time ON trades (strategy_live_id, exit_time);

COMMIT;
