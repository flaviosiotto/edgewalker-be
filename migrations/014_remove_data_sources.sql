-- Migration 014: Remove data_sources table, migrate symbol_cache/symbol_sync_log to Connection
-- DataSource model is replaced by Connection (which already has broker_type, config, etc.)

BEGIN;

-- ─── ADD SYNC FIELDS TO CONNECTIONS ───
ALTER TABLE connections ADD COLUMN IF NOT EXISTS sync_enabled BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE connections ADD COLUMN IF NOT EXISTS sync_interval_minutes FLOAT NOT NULL DEFAULT 1440;
ALTER TABLE connections ADD COLUMN IF NOT EXISTS last_sync_at TIMESTAMPTZ;
ALTER TABLE connections ADD COLUMN IF NOT EXISTS last_sync_status VARCHAR(20);
ALTER TABLE connections ADD COLUMN IF NOT EXISTS last_sync_error TEXT;
ALTER TABLE connections ADD COLUMN IF NOT EXISTS symbols_count INTEGER NOT NULL DEFAULT 0;

-- ─── MIGRATE symbol_cache: source_id → connection_id ───
-- Only run if the old column exists (idempotent)
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'symbol_cache' AND column_name = 'source_id'
    ) THEN
        -- Add new columns
        ALTER TABLE symbol_cache ADD COLUMN IF NOT EXISTS connection_id INTEGER;
        ALTER TABLE symbol_cache ADD COLUMN IF NOT EXISTS broker_type VARCHAR(30);

        -- Migrate data: map source_id → connection via data_sources.connection_id
        UPDATE symbol_cache sc
        SET connection_id = ds.connection_id,
            broker_type = ds.source_type
        FROM data_sources ds
        WHERE sc.source_id = ds.id
          AND ds.connection_id IS NOT NULL;

        -- For rows without a matching connection, try to match by source_name
        UPDATE symbol_cache sc
        SET connection_id = c.id,
            broker_type = c.broker_type
        FROM connections c
        WHERE sc.connection_id IS NULL
          AND sc.source_name = c.broker_type;

        -- Drop rows that couldn't be migrated
        DELETE FROM symbol_cache WHERE connection_id IS NULL;

        -- Make connection_id NOT NULL and add FK
        ALTER TABLE symbol_cache ALTER COLUMN connection_id SET NOT NULL;
        ALTER TABLE symbol_cache ADD CONSTRAINT fk_symbol_cache_connection
            FOREIGN KEY (connection_id) REFERENCES connections(id) ON DELETE CASCADE;

        -- Drop old columns
        ALTER TABLE symbol_cache DROP COLUMN IF EXISTS source_id;
        ALTER TABLE symbol_cache DROP COLUMN IF EXISTS source_name;

        -- Add indexes
        CREATE INDEX IF NOT EXISTS idx_symbol_cache_connection_id ON symbol_cache (connection_id);
        CREATE INDEX IF NOT EXISTS idx_symbol_cache_broker_type ON symbol_cache (broker_type);
    END IF;
END $$;

-- ─── MIGRATE symbol_sync_log: source_id → connection_id ───
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'symbol_sync_log' AND column_name = 'source_id'
    ) THEN
        -- Add new columns
        ALTER TABLE symbol_sync_log ADD COLUMN IF NOT EXISTS connection_id INTEGER;
        ALTER TABLE symbol_sync_log ADD COLUMN IF NOT EXISTS connection_name VARCHAR(100);

        -- Migrate data
        UPDATE symbol_sync_log sl
        SET connection_id = ds.connection_id,
            connection_name = c.name
        FROM data_sources ds
        JOIN connections c ON c.id = ds.connection_id
        WHERE sl.source_id = ds.id
          AND ds.connection_id IS NOT NULL;

        -- Drop rows that couldn't be migrated
        DELETE FROM symbol_sync_log WHERE connection_id IS NULL;

        -- Make connection_id NOT NULL and add FK
        ALTER TABLE symbol_sync_log ALTER COLUMN connection_id SET NOT NULL;
        ALTER TABLE symbol_sync_log ADD CONSTRAINT fk_symbol_sync_log_connection
            FOREIGN KEY (connection_id) REFERENCES connections(id) ON DELETE CASCADE;

        -- Drop old columns
        ALTER TABLE symbol_sync_log DROP COLUMN IF EXISTS source_id;
        ALTER TABLE symbol_sync_log DROP COLUMN IF EXISTS source_name;

        -- Add indexes
        CREATE INDEX IF NOT EXISTS idx_symbol_sync_log_connection_id ON symbol_sync_log (connection_id);
        CREATE INDEX IF NOT EXISTS idx_symbol_sync_log_connection_name ON symbol_sync_log (connection_name);
    END IF;
END $$;

-- ─── DROP data_sources TABLE ───
DROP TABLE IF EXISTS data_sources CASCADE;

COMMIT;
