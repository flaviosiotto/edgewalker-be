-- 060: strategy templates.
--
-- A template is a strategy definition detached from any market: rules,
-- indicators, parameters, lessons and the chart timeframes (multi-chart
-- structure is strategy logic), but NO symbol, asset, contract data or
-- broker — those are chosen when the template is instantiated on an account.
--
-- user_id NULL = official EdgeWalker template, synced at backend startup from
-- edgewalker-be/system_templates/<key>.json (upsert by key, same pattern as
-- indicator-svc system_indicators/). User templates are private.
--
-- Deploy order: apply BEFORE the backend deploy (the new backend syncs the
-- official templates in its lifespan and needs the table; the old backend
-- ignores it).
--
-- Safe to re-run: every statement is guarded.

BEGIN;

CREATE TABLE IF NOT EXISTS strategy_templates (
    id            SERIAL PRIMARY KEY,
    user_id       INTEGER NULL REFERENCES "user"(id) ON DELETE CASCADE,
    key           VARCHAR(64) NULL,
    name          VARCHAR(80) NOT NULL,
    description   TEXT NULL,
    tags          JSONB NOT NULL DEFAULT '[]'::jsonb,
    definition    JSONB NOT NULL,
    lessons       JSONB NOT NULL DEFAULT '[]'::jsonb,
    charts_meta   JSONB NOT NULL DEFAULT '[]'::jsonb,
    origin        JSONB NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_strategy_templates_user_id
    ON strategy_templates(user_id);

-- One name per user (official templates are keyed, not named).
CREATE UNIQUE INDEX IF NOT EXISTS uq_strategy_templates_user_name
    ON strategy_templates(user_id, name) WHERE user_id IS NOT NULL;

-- One official template per file key.
CREATE UNIQUE INDEX IF NOT EXISTS uq_strategy_templates_official_key
    ON strategy_templates(key) WHERE user_id IS NULL;

COMMIT;
