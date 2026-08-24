-- Migration: remove the simulated backtest accounts (rollback of 036)
-- Date: 2026-08-24
--
-- Backtests no longer create a row in accounts: the coordinator ledger is the
-- single source and every consumer (FE, agent, MCP) addresses the run through
-- the flat /backtests/{id}/runtime/* API. This keeps accounts at real-broker
-- cardinality: one BT-* row per executed backtest accumulated forever, was
-- scanned by the live dashboard scope, the connection health tick and
-- portfolio-realtime, and leaked into every account selector.
--
-- Safe to re-run: every statement is guarded.

BEGIN;

-- Drop the FK column first so the account rows are no longer referenced.
DROP INDEX IF EXISTS idx_strategy_backtests_account_id;

ALTER TABLE strategy_backtests
    DROP COLUMN IF EXISTS account_id;

-- Purge the simulated account rows (identified by account_type; the BT- code
-- prefix is a redundant guard against deleting anything real).
DELETE FROM accounts
WHERE account_type = 'simulated'
  AND account_id LIKE 'BT-%';

COMMIT;
