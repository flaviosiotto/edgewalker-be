-- 054: drop the leftover GLOBAL unique index on strategies.name.
--
-- `ix_strategies_name` was created as UNIQUE (name) by the original
-- create_all, long before user ownership. Migration 020 dropped the
-- `strategies_name_key` constraint but not this index, so the name has kept
-- being unique across ALL users and accounts. Copying a strategy onto another
-- account (migr. 053: uniqueness per user+account) hit it with a 500.
-- The index is recreated non-unique (the ORM declares index=True on name).
--
-- Safe to re-run: every statement is guarded.

BEGIN;

DROP INDEX IF EXISTS ix_strategies_name;

CREATE INDEX IF NOT EXISTS ix_strategies_name
    ON strategies(name);

COMMIT;
