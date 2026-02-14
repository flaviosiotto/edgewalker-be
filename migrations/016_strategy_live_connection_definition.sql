-- Add connection_id and definition (snapshot) to strategy_live
-- connection_id: FK to connections table, set when live session is created
-- definition: JSONB snapshot of the strategy definition at creation time (decoupled from design)

ALTER TABLE strategy_live
    ADD COLUMN IF NOT EXISTS connection_id INTEGER REFERENCES connections(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS definition JSONB;

CREATE INDEX IF NOT EXISTS ix_strategy_live_connection_id ON strategy_live (connection_id);

-- Backfill existing records from parent strategy
UPDATE strategy_live sl
SET connection_id = s.connection_id,
    definition = s.definition
FROM strategies s
WHERE sl.strategy_id = s.id
  AND sl.connection_id IS NULL;
