-- 018: Link strategy_live to the chat and manager agent actually used
--
-- Goals:
-- 1. Preserve the existing model of one live chat per strategy.
-- 2. Snapshot the chat and manager agent used by each live session.
-- 3. Enforce basic DB integrity for live chats.

BEGIN;

-- Add chat/agent snapshot columns to the runtime session table.
ALTER TABLE strategy_live
    ADD COLUMN IF NOT EXISTS chat_id INTEGER REFERENCES chat(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS manager_agent_id INTEGER REFERENCES agent(id_agent) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS ix_strategy_live_chat_id
    ON strategy_live(chat_id);

CREATE INDEX IF NOT EXISTS ix_strategy_live_manager_agent_id
    ON strategy_live(manager_agent_id);

-- Backfill the manager agent snapshot from the parent strategy.
UPDATE strategy_live sl
SET manager_agent_id = s.manager_agent_id
FROM strategies s
WHERE sl.strategy_id = s.id
  AND sl.manager_agent_id IS NULL;

-- Backfill chat_id from the strategy's live chat when present.
UPDATE strategy_live sl
SET chat_id = c.id
FROM chat c
WHERE c.strategy_id = sl.strategy_id
  AND c.chat_type = 'live'
  AND sl.chat_id IS NULL;

-- Prevent orphan live chats: a live chat must belong to a strategy.
ALTER TABLE chat
    DROP CONSTRAINT IF EXISTS ck_chat_live_requires_strategy;

ALTER TABLE chat
    ADD CONSTRAINT ck_chat_live_requires_strategy
    CHECK (chat_type <> 'live' OR strategy_id IS NOT NULL);

-- Ensure there is at most one live chat per strategy.
-- NOTE: this will fail if duplicates already exist and should therefore
-- be applied together with data cleanup if legacy data is inconsistent.
CREATE UNIQUE INDEX IF NOT EXISTS ux_chat_live_per_strategy
    ON chat(strategy_id)
    WHERE chat_type = 'live';

COMMIT;