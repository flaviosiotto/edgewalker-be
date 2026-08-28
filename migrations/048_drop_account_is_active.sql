-- 048: drop the active/inactive concept on broker accounts.
--
-- accounts.is_active was derived state ("seen in the last sync of a connected
-- connection") that duplicated connections.status and went stale on every
-- backend restart: the startup reset and the health-loop disconnect path
-- deactivated accounts, but reactivation only happened on an explicit
-- connect() — the designed self-heal (order-aggregator projection of
-- broker.account.sync) was dead code because the consumer's accept filter
-- dropped the event. Result: recurring "Account X is not active" on live
-- start (26-28/08/2026).
--
-- Liveness is now judged from connections.status alone; account existence is
-- enough. connections.is_active (the user-facing toggle) is untouched.
--
-- Apply ONLY AFTER deploying backend + order-aggregator builds that no longer
-- reference the column (the old backend selects it on every account read).

BEGIN;

ALTER TABLE accounts DROP COLUMN IF EXISTS is_active;

COMMIT;
