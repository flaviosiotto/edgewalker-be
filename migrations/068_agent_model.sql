-- 068: agent bridge, phase F3 "internal Agent model"
-- (docs/valutazione-agent-bridge.md §5.3 / §6 F3, decisions D5-D10).
--
-- * agent.tool_policy: {"<tool|group>": "allow"|"ask"|"off"} validated by
--   edgewalker_platform.agent_tools (shared catalogue). Empty = defaults
--   (autonomy execute -> trading allow, propose -> trading ask; external
--   agents never trade whatever the document says).
-- * agent.skills: allowlist of the user's skill names the agent loads
--   (empty = all of the user's skills).
-- * agent.slug: stable public name (Agent Card, run endpoint of F4),
--   unique per user.
-- * agent_skill: the user's skills in SKILL.md form (agentskills.io):
--   procedural knowledge the hosted agent loads on demand (load_skill) and
--   an external agent reads through MCP. Never a declared tool (D9).
-- * agent_memory: curated, bounded text memory of a hosted agent, three
--   kinds, written by the agent (memory_update) or edited by the user (D8).
-- * agent_action_request: the "ask first" queue — a trading action the
--   hosted agent proposed under a tool policy of "ask", decided by the
--   user (UI) or by a trade-scoped PAT, executed by the backend on
--   approval, expired by a sweeper.
BEGIN;

ALTER TABLE agent ADD COLUMN IF NOT EXISTS tool_policy JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE agent ADD COLUMN IF NOT EXISTS skills JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE agent ADD COLUMN IF NOT EXISTS slug VARCHAR(64);
COMMENT ON COLUMN agent.tool_policy IS 'per tool or group: allow | ask | off (edgewalker_platform.agent_tools)';
COMMENT ON COLUMN agent.skills IS 'allowlist of agent_skill.name (empty = every skill of the user)';
COMMENT ON COLUMN agent.slug IS 'stable public name of the agent, unique per user';

-- Backfill the slug from the name: lowercase, non-alphanumerics -> '-',
-- the agent id appended when two agents of a user collide.
UPDATE agent SET slug = sub.slug
FROM (
    SELECT id_agent,
           CASE WHEN COUNT(*) OVER (PARTITION BY user_id, base) > 1 THEN base || '-' || id_agent ELSE base END AS slug
    FROM (
        SELECT id_agent, user_id,
               NULLIF(TRIM(BOTH '-' FROM LOWER(REGEXP_REPLACE(agent_name, '[^a-zA-Z0-9]+', '-', 'g'))), '') AS base
        FROM agent
    ) b
    WHERE base IS NOT NULL
) sub
WHERE agent.id_agent = sub.id_agent AND agent.slug IS NULL;
UPDATE agent SET slug = 'agent-' || id_agent WHERE slug IS NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_user_slug ON agent (user_id, slug);

CREATE TABLE IF NOT EXISTS agent_skill (
    id                 SERIAL PRIMARY KEY,
    user_id            INTEGER NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    name               VARCHAR(64) NOT NULL,
    description        VARCHAR(1024) NOT NULL DEFAULT '',
    body               TEXT NOT NULL,
    version            INTEGER NOT NULL DEFAULT 1,
    origin             VARCHAR(16) NOT NULL DEFAULT 'user',
    source_strategy_id INTEGER NULL REFERENCES strategies(id) ON DELETE SET NULL,
    source_backtest_id INTEGER NULL REFERENCES strategy_backtests(id) ON DELETE SET NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_agent_skill_user_name UNIQUE (user_id, name),
    CONSTRAINT ck_agent_skill_origin CHECK (origin IN ('user', 'playbook', 'import'))
);
COMMENT ON TABLE agent_skill IS 'user skills (SKILL.md, agentskills.io): loaded on demand by hosted agents, read via MCP by external ones';

CREATE TABLE IF NOT EXISTS agent_memory (
    id         SERIAL PRIMARY KEY,
    agent_id   INTEGER NOT NULL REFERENCES agent(id_agent) ON DELETE CASCADE,
    kind       VARCHAR(32) NOT NULL,
    content    TEXT NOT NULL DEFAULT '',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_by VARCHAR(16) NOT NULL DEFAULT 'user',
    CONSTRAINT uq_agent_memory_agent_kind UNIQUE (agent_id, kind),
    CONSTRAINT ck_agent_memory_kind CHECK (kind IN ('user_profile', 'market_notes', 'operating_rules')),
    CONSTRAINT ck_agent_memory_updated_by CHECK (updated_by IN ('agent', 'user'))
);
COMMENT ON TABLE agent_memory IS 'curated memory of a hosted agent: user_profile | market_notes | operating_rules, bounded text';

CREATE TABLE IF NOT EXISTS agent_action_request (
    id               SERIAL PRIMARY KEY,
    agent_id         INTEGER NOT NULL REFERENCES agent(id_agent) ON DELETE CASCADE,
    user_id          INTEGER NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    chat_id          INTEGER NULL REFERENCES chat(id) ON DELETE SET NULL,
    strategy_live_id INTEGER NULL REFERENCES strategy_live(id) ON DELETE SET NULL,
    account_id       INTEGER NULL REFERENCES accounts(id) ON DELETE SET NULL,
    tool_name        VARCHAR(64) NOT NULL,
    args             JSONB NOT NULL DEFAULT '{}'::jsonb,
    rationale        TEXT NULL,
    status           VARCHAR(16) NOT NULL DEFAULT 'pending',
    expires_at       TIMESTAMPTZ NOT NULL,
    decided_at       TIMESTAMPTZ NULL,
    decided_by       JSONB NULL,
    result           JSONB NULL,
    error            TEXT NULL,
    chat_row_id      INTEGER NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_agent_action_request_status CHECK (status IN ('pending', 'approved', 'rejected', 'expired', 'failed'))
);
CREATE INDEX IF NOT EXISTS ix_agent_action_request_user_created ON agent_action_request (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_agent_action_request_pending ON agent_action_request (expires_at) WHERE status = 'pending';
COMMENT ON TABLE agent_action_request IS 'ask-first queue: trading actions a hosted agent proposed under tool_policy=ask, executed by the backend when approved';
COMMENT ON COLUMN agent_action_request.chat_row_id IS 'id of the n8n_chat_histories system row that shows the request in the chat (status patched on decision)';

COMMIT;
