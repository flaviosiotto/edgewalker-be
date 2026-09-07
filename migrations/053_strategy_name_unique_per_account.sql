-- 053: strategy name unique per (user, account), no longer per user.
--
-- Copying a strategy to another account (typical case: a new Prop account
-- after a challenge) must keep the strategy's name, and the FE already groups
-- strategies by account, so the natural uniqueness scope is the account.
-- The partial index from migration 020 (user_id, name) is replaced by
-- (user_id, account_id, name). No data change: every existing (user, name)
-- pair is trivially unique inside its account too.
--
-- Deploy order: apply BEFORE the backend deploy (the new backend expects the
-- relaxed constraint; the old backend keeps working under it because its own
-- 409 check is stricter than the index).
--
-- Safe to re-run: every statement is guarded.

BEGIN;

DROP INDEX IF EXISTS uq_strategies_user_name;

CREATE UNIQUE INDEX IF NOT EXISTS uq_strategies_user_account_name
    ON strategies(user_id, account_id, name);

COMMIT;
