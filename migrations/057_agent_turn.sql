-- 057: registro di debug dei turni dell'agente (agent-svc), con le immagini
-- dei chart mostrate al modello. Sostituisce la vista esecuzioni di n8n:
-- la console /admin › Turni agente legge queste tabelle.
-- Retention: agent-svc cancella i turni piu vecchi di AGENT_TURN_RETENTION_DAYS.

BEGIN;

SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

CREATE TABLE IF NOT EXISTS agent_turn (
    turn_id           VARCHAR(32) PRIMARY KEY,
    session_id        VARCHAR(255) NOT NULL,
    user_id           INTEGER NULL REFERENCES "user"(id) ON DELETE CASCADE,
    agent_id          INTEGER NULL,
    strategy_id       INTEGER NULL,
    strategy_live_id  INTEGER NULL,
    backtest_id       INTEGER NULL,
    kind              VARCHAR(160) NULL,          -- riga TURNO del prompt (operativo/conversazionale)
    trigger_type      VARCHAR(64) NULL,           -- chat | runner_alert_triggered | agent_request | ...
    correlation_id    VARCHAR(100) NULL,          -- stesso id del ledger crediti / agent_call
    status            VARCHAR(16) NOT NULL,       -- ok | failed
    error             TEXT NULL,
    started_at        TIMESTAMPTZ NOT NULL,
    finished_at       TIMESTAMPTZ NULL,
    duration_ms       INTEGER NULL,
    model             VARCHAR(120) NULL,
    requests          INTEGER NULL,
    tool_calls        INTEGER NULL,
    tokens_input      INTEGER NULL,
    tokens_output     INTEGER NULL,
    tokens_reasoning  INTEGER NULL,
    tokens_cached     INTEGER NULL,
    prompt_chars      INTEGER NULL,
    response_chars    INTEGER NULL,
    user_message      TEXT NULL,
    system_prompt     TEXT NULL,
    response          TEXT NULL,
    steps             JSONB NOT NULL DEFAULT '[]'::jsonb,   -- [{kind: llm|tool, ...}]
    context           JSONB NOT NULL DEFAULT '{}'::jsonb    -- run, stream, immagini, gate
);

CREATE INDEX IF NOT EXISTS ix_agent_turn_session ON agent_turn (session_id, started_at DESC);
CREATE INDEX IF NOT EXISTS ix_agent_turn_user ON agent_turn (user_id, started_at DESC);
CREATE INDEX IF NOT EXISTS ix_agent_turn_started ON agent_turn (started_at);

CREATE TABLE IF NOT EXISTS agent_turn_image (
    id        SERIAL PRIMARY KEY,
    turn_id   VARCHAR(32) NOT NULL REFERENCES agent_turn(turn_id) ON DELETE CASCADE,
    name      VARCHAR(120) NOT NULL,
    mime      VARCHAR(64) NOT NULL DEFAULT 'image/png',
    bytes     INTEGER NOT NULL,
    data      BYTEA NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_agent_turn_image_turn ON agent_turn_image (turn_id);

COMMIT;
