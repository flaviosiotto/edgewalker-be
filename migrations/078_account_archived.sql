-- 078: archived accounts.
--
-- An account listed under a connection is one that can be traded through
-- that connection now. When the gateway stops listing it (closed at the
-- broker, or no longer the account a cTrader connection is connected on)
-- the row is kept, because lives, trades and performance point at it, and
-- marked with the moment it was last found missing.
--
-- archived_at  NULL = the gateway lists the account. Written by the backend
--              at connect (connection_manager._sync_accounts) and cleared
--              there or by the order-aggregator when the account is reported
--              again.
--
-- Run BEFORE deploying the backend and the order-aggregator (both ORMs map
-- the column). Nothing is archived by this file: the first connect of each
-- connection after the deploy does it.
BEGIN;

ALTER TABLE accounts ADD COLUMN IF NOT EXISTS archived_at TIMESTAMPTZ;

COMMENT ON COLUMN accounts.archived_at IS
    'When the gateway stopped listing the account; NULL while it is tradable through its connection';

COMMIT;
