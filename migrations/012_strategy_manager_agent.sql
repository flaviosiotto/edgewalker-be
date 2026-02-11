-- 012: Add AI Agent Manager to strategies
-- Each strategy can have a manager agent that oversees live execution.
-- A dedicated "live" chat type is added for real-time communication.

-- 1. Add manager_agent_id FK to strategies
ALTER TABLE strategies
    ADD COLUMN IF NOT EXISTS manager_agent_id INTEGER
        REFERENCES agent(id_agent) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS ix_strategies_manager_agent_id
    ON strategies(manager_agent_id);

-- 2. Extend chat_type enum to include 'live'
-- chat_type is stored as VARCHAR (native_enum=False), so no ALTER TYPE needed.
-- The SQLModel enum definition handles validation in Python.
