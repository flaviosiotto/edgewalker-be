-- 019: Persistent live alerts for runner-managed agent recall

BEGIN;

CREATE TABLE IF NOT EXISTS live_alert (
    id SERIAL PRIMARY KEY,
    strategy_live_id INTEGER NOT NULL REFERENCES strategy_live(id) ON DELETE CASCADE,
    request_id VARCHAR(100),
    name VARCHAR(255) NOT NULL,
    trigger_type VARCHAR(32) NOT NULL,
    trigger JSONB NOT NULL,
    recipient VARCHAR(32) NOT NULL DEFAULT 'agent',
    fire_mode VARCHAR(16) NOT NULL DEFAULT 'once',
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    message TEXT,
    expires_at TIMESTAMPTZ,
    last_triggered_at TIMESTAMPTZ,
    trigger_count INTEGER NOT NULL DEFAULT 0,
    last_triggered_price DOUBLE PRECISION,
    last_error TEXT,
    target JSONB,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_live_alert_strategy_live_id
    ON live_alert(strategy_live_id);

CREATE INDEX IF NOT EXISTS ix_live_alert_request_id
    ON live_alert(request_id);

CREATE INDEX IF NOT EXISTS ix_live_alert_status
    ON live_alert(status);

CREATE INDEX IF NOT EXISTS ix_live_alert_trigger_type
    ON live_alert(trigger_type);

COMMIT;