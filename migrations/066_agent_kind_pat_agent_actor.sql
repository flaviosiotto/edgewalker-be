-- 066: agent bridge, phase F1 (docs/valutazione-agent-bridge.md §5.3, decisions D1-D4).
--
-- * agent.kind: 'hosted' (runs in agent-svc, trades through strategies) or
--   'external' (an identity for an agent that runs in the user's own
--   orchestrator and reaches EdgeWalker through MCP / the API with a
--   personal access token bound to it). In v1 an external agent never
--   trades and is never the manager of a strategy, a live or a backtest;
--   those rules live in the services (422/400), NOT in schema constraints,
--   so the scenario can be opened later without a migration.
-- * personal_access_token.agent_id: a PAT may act *as* one of the user's
--   agents. What the token does is then attributed to that agent (actor
--   columns below) and, for an external agent, the 'trade' scope is refused.
-- * actor attribution: who last changed a strategy, who started a live
--   session. JSONB {via: ui|pat|agent|runner, user_id, agent_id?, pat_id?,
--   pat_name?, agent_name?}; NULL on rows written before this migration.
BEGIN;

ALTER TABLE agent
    ADD COLUMN IF NOT EXISTS kind VARCHAR(16) NOT NULL DEFAULT 'hosted';
ALTER TABLE agent DROP CONSTRAINT IF EXISTS ck_agent_kind_values;
ALTER TABLE agent
    ADD CONSTRAINT ck_agent_kind_values CHECK (kind IN ('hosted', 'external'));
COMMENT ON COLUMN agent.kind IS
    'hosted = runs in agent-svc; external = identity of an agent running in the user''s orchestrator (MCP/API via a bound PAT)';

ALTER TABLE personal_access_token
    ADD COLUMN IF NOT EXISTS agent_id INTEGER NULL
        REFERENCES agent(id_agent) ON DELETE CASCADE;
CREATE INDEX IF NOT EXISTS ix_personal_access_token_agent_id
    ON personal_access_token (agent_id);
COMMENT ON COLUMN personal_access_token.agent_id IS
    'The agent this token acts as (attribution); NULL = the user themselves';

ALTER TABLE strategies
    ADD COLUMN IF NOT EXISTS updated_by JSONB NULL;
COMMENT ON COLUMN strategies.updated_by IS
    'Actor of the last change: {via, user_id, agent_id?, pat_id?, pat_name?, agent_name?}';

ALTER TABLE strategy_live
    ADD COLUMN IF NOT EXISTS started_by JSONB NULL;
COMMENT ON COLUMN strategy_live.started_by IS
    'Actor that started the session: {via, user_id, agent_id?, pat_id?, pat_name?, agent_name?}';

COMMIT;
