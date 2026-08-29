-- 049: the agent is a persona, not a kind.
--
-- Two changes, one intent:
--
-- 1. agent_kind (migr. 040, design|running) is DROPPED. It never carried any
--    authorization: it only fed `GET /agents/?kind=` and the split of the FE
--    selectors. The real design/running boundary lives in the tokens (the
--    consultative token now also gets strategies:read / strategies:write), so
--    every agent can both design a strategy and trade it.
--
-- 2. The agent gains an identity the user can recognise and the prompt can
--    render: avatar preset + accent colour (no upload, no storage on our
--    side — avatar_url is an optional external image), a free-text
--    description, a risk_profile and an extensible `persona` JSONB
--    ({horizon, max_risk_per_trade_pct, style, notes}). The whole block is
--    shipped to n8n as `metadata.agent` in the webhook payload.
--
-- Deploy order: apply this migration IMMEDIATELY BEFORE the backend deploy.
-- The new backend without 049 fails every agent SELECT; the old backend with
-- 049 breaks on the missing agent_kind. There is no safe window either way,
-- so keep the two adjacent.
--
-- Safe to re-run: every statement is guarded.

BEGIN;

ALTER TABLE agent
    ADD COLUMN IF NOT EXISTS avatar        VARCHAR(64)   NOT NULL DEFAULT 'robot',
    ADD COLUMN IF NOT EXISTS accent_color  VARCHAR(16)   NOT NULL DEFAULT '#6f42c1',
    ADD COLUMN IF NOT EXISTS avatar_url    VARCHAR(1024),
    ADD COLUMN IF NOT EXISTS description   TEXT,
    ADD COLUMN IF NOT EXISTS risk_profile  VARCHAR(16)   NOT NULL DEFAULT 'balanced',
    ADD COLUMN IF NOT EXISTS persona       JSONB         NOT NULL DEFAULT '{}'::jsonb;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'ck_agent_risk_profile'
    ) THEN
        ALTER TABLE agent
        ADD CONSTRAINT ck_agent_risk_profile
        CHECK (risk_profile IN ('conservative', 'balanced', 'aggressive'));
    END IF;
END $$;

COMMENT ON COLUMN agent.avatar IS
'Chiave del set di avatar SVG del FE (analyst, quant, veteran, executive, strategist, portfolio_manager, risk_officer, senior_partner, macro, chartist, associate, advisor)';
COMMENT ON COLUMN agent.accent_color IS
'Colore identitario dell''agent, hex #RRGGBB';
COMMENT ON COLUMN agent.avatar_url IS
'Immagine esterna facoltativa: nessuno storage lato piattaforma';
COMMENT ON COLUMN agent.risk_profile IS
'conservative | balanced | aggressive — reso nel blocco == CHI SEI == del prompt';
COMMENT ON COLUMN agent.persona IS
'Tratti estensibili: {horizon: intraday|swing|position, max_risk_per_trade_pct, style, notes}';

ALTER TABLE agent DROP CONSTRAINT IF EXISTS ck_agent_kind;
ALTER TABLE agent DROP COLUMN IF EXISTS agent_kind;

COMMIT;
