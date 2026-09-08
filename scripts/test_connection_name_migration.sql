\set ON_ERROR_STOP on

CREATE TEMP TABLE connections (
    user_id integer NOT NULL,
    name varchar(100) NOT NULL,
    CONSTRAINT connections_name_key UNIQUE (name)
);
CREATE UNIQUE INDEX ix_connections_name ON connections(name);
CREATE UNIQUE INDEX uq_connections_user_name ON connections(user_id, name);

INSERT INTO connections VALUES (1, 'cTrader - Primi passi');

DO $$
BEGIN
    BEGIN
        INSERT INTO connections VALUES (2, 'cTrader - Primi passi');
        RAISE EXCEPTION 'Legacy global uniqueness was not reproduced';
    EXCEPTION WHEN unique_violation THEN
        NULL;
    END;
END $$;

\ir ../migrations/056_drop_global_unique_connection_name.sql
\ir ../migrations/056_drop_global_unique_connection_name.sql

INSERT INTO connections VALUES
    (2, 'cTrader - Primi passi'),
    (1, 'Binance - Dati pubblici'),
    (2, 'Binance - Dati pubblici');

DO $$
BEGIN
    IF (SELECT count(*) FROM connections) <> 4 THEN
        RAISE EXCEPTION 'Expected both users to retain both connections';
    END IF;
    BEGIN
        INSERT INTO connections VALUES (1, 'cTrader - Primi passi');
        RAISE EXCEPTION 'Per-user uniqueness was lost';
    EXCEPTION WHEN unique_violation THEN
        NULL;
    END;
    IF EXISTS (
        SELECT 1 FROM pg_index
        WHERE indexrelid = 'pg_temp.ix_connections_name'::regclass AND indisunique
    ) THEN
        RAISE EXCEPTION 'Name lookup index must be non-unique';
    END IF;
END $$;

DROP TABLE pg_temp.connections;
SELECT 'Connection name migration: passed' AS result;