-- 079: watchlist.
--
-- The symbols a user keeps an eye on. A symbol is the broker's own name for
-- an instrument, so an item belongs to a connection; deleting the connection
-- deletes its items.
--
-- info  the broker's description when the symbol was added (tick size,
--       contract, expiry…), so the row and its chart open without asking the
--       gateway again.
--
-- Run BEFORE deploying the backend (create_all would create the table
-- without the comments). No other service maps it.
BEGIN;

CREATE TABLE IF NOT EXISTS watchlist_items (
    id             SERIAL PRIMARY KEY,
    user_id        INTEGER NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    connection_id  INTEGER NOT NULL REFERENCES connections(id) ON DELETE CASCADE,
    symbol         VARCHAR(64) NOT NULL,
    name           VARCHAR(255),
    asset_type     VARCHAR(32),
    exchange       VARCHAR(64),
    currency       VARCHAR(16),
    info           JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_watchlist_items_user_conn_symbol UNIQUE (user_id, connection_id, symbol)
);

CREATE INDEX IF NOT EXISTS ix_watchlist_items_user_id ON watchlist_items (user_id);
CREATE INDEX IF NOT EXISTS ix_watchlist_items_connection_id ON watchlist_items (connection_id);

COMMENT ON TABLE watchlist_items IS 'Symbols a user keeps an eye on, one row per (connection, symbol)';
COMMENT ON COLUMN watchlist_items.symbol IS 'Written the way the broker spells it';
COMMENT ON COLUMN watchlist_items.info IS 'Broker description of the symbol when it was added (tick size, contract, expiry…)';

COMMIT;
