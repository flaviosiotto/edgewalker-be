-- 069: agent bridge, phase F4 "EdgeWalker as an agent"
-- (docs/valutazione-agent-bridge.md §4 G5/G8, §6 F4, decisions D12-D14).
--
-- * agent_run: a task handed to a HOSTED agent from outside (REST
--   POST /agents/{id}/runs, the Paperclip http adapter, A2A message/send).
--   The run is executed as one turn in a chat of the agent (its own chat
--   for scope=agent, a design chat of the strategy for scope=strategy); the
--   answer, the usage and the credits of that turn are copied here and,
--   when a callback is configured, pushed back to the caller.
-- * agent.budget_credits_month: optional monthly cap on the AI credits the
--   runs of one agent may spend (NULL = only the wallet limits apply).
BEGIN;

ALTER TABLE agent ADD COLUMN IF NOT EXISTS budget_credits_month INTEGER NULL;
COMMENT ON COLUMN agent.budget_credits_month IS 'monthly cap on AI credits spent by external runs of this agent (NULL = no cap beyond the wallet)';

CREATE TABLE IF NOT EXISTS agent_run (
    id                 SERIAL PRIMARY KEY,
    agent_id           INTEGER NOT NULL REFERENCES agent(id_agent) ON DELETE CASCADE,
    user_id            INTEGER NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    source             VARCHAR(16) NOT NULL DEFAULT 'api',
    external_run_id    VARCHAR(128) NULL,
    scope              VARCHAR(16) NOT NULL DEFAULT 'agent',
    strategy_id        INTEGER NULL REFERENCES strategies(id) ON DELETE SET NULL,
    chat_id            INTEGER NULL REFERENCES chat(id) ON DELETE SET NULL,
    task               TEXT NOT NULL,
    context            JSONB NOT NULL DEFAULT '{}'::jsonb,
    status             VARCHAR(16) NOT NULL DEFAULT 'queued',
    request_id         VARCHAR(64) NULL,
    result             TEXT NULL,
    error              TEXT NULL,
    usage              JSONB NULL,
    cost_credits       NUMERIC(12, 4) NULL,
    callback_url       VARCHAR(1024) NULL,
    callback_auth      TEXT NULL,
    callback_status    VARCHAR(16) NULL,
    callback_error     TEXT NULL,
    created_by         JSONB NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at         TIMESTAMPTZ NULL,
    finished_at        TIMESTAMPTZ NULL,
    CONSTRAINT ck_agent_run_source CHECK (source IN ('api', 'paperclip', 'a2a', 'mcp')),
    CONSTRAINT ck_agent_run_scope CHECK (scope IN ('agent', 'strategy')),
    CONSTRAINT ck_agent_run_status CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled'))
);
CREATE INDEX IF NOT EXISTS ix_agent_run_agent_created ON agent_run (agent_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_agent_run_user_created ON agent_run (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_agent_run_external ON agent_run (agent_id, external_run_id) WHERE external_run_id IS NOT NULL;
COMMENT ON TABLE agent_run IS 'tasks handed to a hosted agent from outside (REST, Paperclip, A2A): one chat turn each, with usage, credits and optional callback';
COMMENT ON COLUMN agent_run.callback_auth IS 'Fernet-encrypted Authorization header value for the callback (e.g. the Paperclip agent API key)';

COMMIT;
