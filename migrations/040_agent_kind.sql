-- Migration: agent.agent_kind — distingue gli agent di design (modificano la
--            strategia in design mode) dagli agent di running (gestiscono i
--            run live/backtest: ask_agent, alert, review).
-- Date: 2026-08-02
--
-- Safe to re-run: every statement is guarded.

BEGIN;

ALTER TABLE agent
ADD COLUMN IF NOT EXISTS agent_kind VARCHAR(16) NOT NULL DEFAULT 'running';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'ck_agent_kind'
    ) THEN
        ALTER TABLE agent
        ADD CONSTRAINT ck_agent_kind CHECK (agent_kind IN ('design', 'running'));
    END IF;
END $$;

COMMENT ON COLUMN agent.agent_kind IS
'design = assiste il design della strategia; running = gestisce i run live/backtest (ask_agent, alert, review)';

COMMIT;
