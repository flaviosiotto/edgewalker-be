-- Migration: agent_lessons — persistent, auditable playbook the manager agent
--            distills during backtest reviews (self-learning phase B).
-- Date: 2026-07-28
--
-- Safe to re-run: every statement is guarded.

BEGIN;

CREATE TABLE IF NOT EXISTS agent_lessons (
    id            SERIAL PRIMARY KEY,
    strategy_id   INTEGER NOT NULL REFERENCES strategies(id) ON DELETE CASCADE,
    user_id       INTEGER NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    lesson        TEXT NOT NULL,
    context       TEXT,
    status        VARCHAR(16) NOT NULL DEFAULT 'active',
    confidence    DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    source        VARCHAR(16) NOT NULL DEFAULT 'backtest',
    backtest_id   INTEGER REFERENCES strategy_backtests(id) ON DELETE SET NULL,
    evidence      JSONB,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_agent_lessons_strategy_id ON agent_lessons (strategy_id);
CREATE INDEX IF NOT EXISTS ix_agent_lessons_user_id ON agent_lessons (user_id);
CREATE INDEX IF NOT EXISTS ix_agent_lessons_strategy_status
    ON agent_lessons (strategy_id, status);

COMMIT;
