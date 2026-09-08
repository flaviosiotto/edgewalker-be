BEGIN;

SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

CREATE UNIQUE INDEX IF NOT EXISTS uq_connections_user_name
    ON connections(user_id, name);

ALTER TABLE connections DROP CONSTRAINT IF EXISTS connections_name_key;
DROP INDEX IF EXISTS ix_connections_name;
CREATE INDEX ix_connections_name ON connections(name);

COMMIT;