-- 072: chat of a Studio (docs/valutazione-studi-pubblicazione-e-agente.md, phase 3).
--
-- A chat bound to a Studio (chat_type = "studio", scope "studio" in
-- agent-svc): the agent chosen for the Studio reads and edits its notebook
-- by cells, runs it and reads the outputs. One more owner FK next to
-- strategy_id / backtest_id / live_id, mutually exclusive by construction
-- (GET /chats/studio/{id} creates at most one chat per Studio). The `studio`
-- table belongs to studio-svc (same database, as the "user" FK of its 001).
-- The chat_type enum is not native (VARCHAR): the new value needs no DDL.
--
-- Run BEFORE deploying the backend (create_all adds tables, never columns).
BEGIN;

ALTER TABLE chat
    ADD COLUMN IF NOT EXISTS studio_id INTEGER REFERENCES studio(id) ON DELETE CASCADE;
CREATE INDEX IF NOT EXISTS ix_chat_studio_id ON chat (studio_id);

COMMENT ON COLUMN chat.studio_id IS 'Studio this chat belongs to (chat_type studio): the agent assists on its notebook';

COMMIT;
