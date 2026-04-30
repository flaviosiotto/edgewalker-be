-- Migration: add per-user ownership roots and base user role field
-- Date: 2026-04-29

DO $$
DECLARE
    owner_user_id INTEGER;
BEGIN
    SELECT id INTO owner_user_id
    FROM "user"
    ORDER BY id
    LIMIT 1;

    IF owner_user_id IS NULL THEN
        RAISE EXCEPTION 'Cannot backfill ownership without at least one user';
    END IF;

    ALTER TABLE "user"
    ADD COLUMN IF NOT EXISTS role VARCHAR(32) NOT NULL DEFAULT 'user';

    UPDATE "user"
    SET role = 'admin'
    WHERE id = owner_user_id
      AND role = 'user';

    ALTER TABLE strategies
    ADD COLUMN IF NOT EXISTS user_id INTEGER;

    UPDATE strategies
    SET user_id = owner_user_id
    WHERE user_id IS NULL;

    ALTER TABLE strategies
    ALTER COLUMN user_id SET NOT NULL;

    ALTER TABLE strategies
    DROP CONSTRAINT IF EXISTS strategies_user_id_fkey,
    ADD CONSTRAINT strategies_user_id_fkey
        FOREIGN KEY (user_id) REFERENCES "user"(id) ON DELETE CASCADE;

    ALTER TABLE strategies
    DROP CONSTRAINT IF EXISTS strategies_name_key;

    CREATE UNIQUE INDEX IF NOT EXISTS uq_strategies_user_name
    ON strategies(user_id, name);

    ALTER TABLE agent
    ADD COLUMN IF NOT EXISTS user_id INTEGER;

    UPDATE agent
    SET user_id = owner_user_id
    WHERE user_id IS NULL;

    ALTER TABLE agent
    ALTER COLUMN user_id SET NOT NULL;

    ALTER TABLE agent
    DROP CONSTRAINT IF EXISTS agent_user_id_fkey,
    ADD CONSTRAINT agent_user_id_fkey
        FOREIGN KEY (user_id) REFERENCES "user"(id) ON DELETE CASCADE;

    ALTER TABLE agent
    DROP CONSTRAINT IF EXISTS agent_agent_name_key;

    CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_user_name
    ON agent(user_id, agent_name);

    ALTER TABLE connections
    ADD COLUMN IF NOT EXISTS user_id INTEGER;

    UPDATE connections
    SET user_id = owner_user_id
    WHERE user_id IS NULL;

    ALTER TABLE connections
    ALTER COLUMN user_id SET NOT NULL;

    ALTER TABLE connections
    DROP CONSTRAINT IF EXISTS connections_user_id_fkey,
    ADD CONSTRAINT connections_user_id_fkey
        FOREIGN KEY (user_id) REFERENCES "user"(id) ON DELETE CASCADE;

    ALTER TABLE connections
    DROP CONSTRAINT IF EXISTS connections_name_key;

    CREATE UNIQUE INDEX IF NOT EXISTS uq_connections_user_name
    ON connections(user_id, name);

    UPDATE chat
    SET user_id = owner_user_id
    WHERE user_id IS NULL;

    ALTER TABLE chat
    ALTER COLUMN user_id SET NOT NULL;

    ALTER TABLE chat
    DROP CONSTRAINT IF EXISTS chat_user_id_fkey,
    ADD CONSTRAINT chat_user_id_fkey
        FOREIGN KEY (user_id) REFERENCES "user"(id) ON DELETE CASCADE;
END $$;

CREATE INDEX IF NOT EXISTS idx_user_role ON "user"(role);
CREATE INDEX IF NOT EXISTS idx_strategies_user_id ON strategies(user_id);
CREATE INDEX IF NOT EXISTS idx_agent_user_id ON agent(user_id);
CREATE INDEX IF NOT EXISTS idx_connections_user_id ON connections(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_user_id ON chat(user_id);

COMMENT ON COLUMN "user".role IS 'Base global role for future RBAC bootstrap: user or admin';
COMMENT ON COLUMN strategies.user_id IS 'Owning user for tenant isolation';
COMMENT ON COLUMN agent.user_id IS 'Owning user for tenant isolation';
COMMENT ON COLUMN connections.user_id IS 'Owning user for tenant isolation';
COMMENT ON COLUMN chat.user_id IS 'Owning user for tenant isolation';