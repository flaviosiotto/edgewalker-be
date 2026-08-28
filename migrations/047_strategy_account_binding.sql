-- 047: every strategy is bound to ONE trading account from birth.
--
-- The account is the single source of truth: strategies.connection_id is now
-- a denormalized copy of accounts.connection_id (kept because chart meta,
-- symbol precision, backtests and the MCP tools still read it) and is written
-- by the backend, never by the client.
--
-- Backfill order for existing rows:
--   1. account of the most recent strategy_live session that had one;
--   2. otherwise the ONLY account of strategies.connection_id (unambiguous);
--   3. anything left is an error: assign it by hand and re-run.
--      SELECT id, name, connection_id FROM strategies WHERE account_id IS NULL;
--
-- ON DELETE RESTRICT: an account with strategies attached cannot be deleted
-- (previously the connection -> account cascade would have silently detached
-- strategies from their datafeed).

BEGIN;

ALTER TABLE strategies
    ADD COLUMN IF NOT EXISTS account_id INTEGER
        REFERENCES accounts(id) ON DELETE RESTRICT;

-- 0. manual assignment (prod, 27/08/2026): GOLD_STRAT never went live, bound
--    to 113/17171553 like the other FTMO strategies. No-op elsewhere.
UPDATE strategies SET account_id = 113 WHERE id = 8 AND account_id IS NULL;

-- 1. last live session with an account
UPDATE strategies s
SET account_id = ll.account_id
FROM (
    SELECT DISTINCT ON (strategy_id) strategy_id, account_id
    FROM strategy_live
    WHERE account_id IS NOT NULL
    ORDER BY strategy_id, id DESC
) ll
WHERE ll.strategy_id = s.id
  AND s.account_id IS NULL;

-- 2. the only account of the datafeed connection
UPDATE strategies s
SET account_id = ca.account_id
FROM (
    SELECT connection_id, MIN(id) AS account_id
    FROM accounts
    GROUP BY connection_id
    HAVING COUNT(*) = 1
) ca
WHERE ca.connection_id = s.connection_id
  AND s.account_id IS NULL;

-- 3. fail loudly on leftovers
DO $$
DECLARE
    leftovers INTEGER;
BEGIN
    SELECT COUNT(*) INTO leftovers FROM strategies WHERE account_id IS NULL;
    IF leftovers > 0 THEN
        RAISE EXCEPTION
            '047: % strategies without an assignable account — assign them by hand (UPDATE strategies SET account_id = ... WHERE id = ...) and re-run',
            leftovers;
    END IF;
END $$;

-- Re-align the denormalized datafeed connection with the bound account.
UPDATE strategies s
SET connection_id = a.connection_id
FROM accounts a
WHERE a.id = s.account_id
  AND s.connection_id IS DISTINCT FROM a.connection_id;

ALTER TABLE strategies ALTER COLUMN account_id SET NOT NULL;
ALTER TABLE strategies ALTER COLUMN connection_id SET NOT NULL;

CREATE INDEX IF NOT EXISTS ix_strategies_account_id ON strategies(account_id);

COMMIT;
