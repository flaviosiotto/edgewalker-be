-- 067: outbound webhooks (agent bridge, phase F2: docs/valutazione-agent-bridge.md §G2).
--
-- A user subscribes a URL to platform events (live alert triggered, trade
-- closed, live status changed, backtest completed/failed, agent turn
-- completed, credits exhausted, connection stale). Each matching event
-- becomes one delivery row (outbox): a background dispatcher in the backend
-- POSTs it with an HMAC-SHA256 signature and retries with backoff until it
-- succeeds or the attempts run out. Nothing is lost on a restart: the queue
-- is the table.
BEGIN;

CREATE TABLE IF NOT EXISTS webhook_subscription (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    -- optional: the external agent this endpoint belongs to (attribution + UI)
    agent_id        INTEGER NULL REFERENCES agent(id_agent) ON DELETE SET NULL,
    name            VARCHAR(120) NOT NULL,
    url             VARCHAR(2048) NOT NULL,
    -- Fernet-encrypted signing secret (SECRETS_ENCRYPTION_KEY, same key as user_secret)
    secret_encrypted TEXT NOT NULL,
    -- event names, or ["*"] for everything
    events          JSONB NOT NULL DEFAULT '["*"]'::jsonb,
    active          BOOLEAN NOT NULL DEFAULT TRUE,
    -- consecutive failed deliveries; the dispatcher disables the endpoint past a threshold
    failure_streak  INTEGER NOT NULL DEFAULT 0,
    disabled_reason TEXT NULL,
    last_success_at TIMESTAMPTZ NULL,
    last_failure_at TIMESTAMPTZ NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_webhook_subscription_user_id ON webhook_subscription (user_id);
CREATE INDEX IF NOT EXISTS ix_webhook_subscription_agent_id ON webhook_subscription (agent_id);
COMMENT ON TABLE webhook_subscription IS 'Outbound webhook endpoint of a user: URL, signing secret, event filter';

CREATE TABLE IF NOT EXISTS webhook_delivery (
    id              BIGSERIAL PRIMARY KEY,
    subscription_id INTEGER NOT NULL REFERENCES webhook_subscription(id) ON DELETE CASCADE,
    event_id        UUID NOT NULL,
    event_type      VARCHAR(64) NOT NULL,
    -- the same event never enqueued twice for the same endpoint (state
    -- transitions reported by more than one path)
    dedupe_key      VARCHAR(200) NULL,
    payload         JSONB NOT NULL,
    status          VARCHAR(16) NOT NULL DEFAULT 'pending',   -- pending | delivering | succeeded | failed
    attempts        INTEGER NOT NULL DEFAULT 0,
    max_attempts    INTEGER NOT NULL DEFAULT 8,
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_status_code INTEGER NULL,
    last_error      TEXT NULL,
    last_attempt_at TIMESTAMPTZ NULL,
    delivered_at    TIMESTAMPTZ NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_webhook_delivery_due
    ON webhook_delivery (next_attempt_at) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS ix_webhook_delivery_subscription
    ON webhook_delivery (subscription_id, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS uq_webhook_delivery_dedupe
    ON webhook_delivery (subscription_id, dedupe_key) WHERE dedupe_key IS NOT NULL;
COMMENT ON TABLE webhook_delivery IS 'Outbox of outbound webhook deliveries with retry state';

-- ── Event sources: NOTIFY from the tables other services write ──────────────
--
-- Backtest completion is written by the backtest coordinator, alerts by the
-- strategy runner, trades by the order-aggregator, agent turns by agent-svc:
-- the backend never sees those writes. A single trigger function raises a
-- NOTIFY on the channel `ew_webhook_events` with {table, op, id}; the
-- backend's listener (services/webhook_sources.py) loads the row, decides
-- whether it is an event, and enqueues the deliveries (dedupe_key makes a
-- second backend replica harmless). Live status and connection status are
-- written by the backend itself but go through the same path for uniformity.
CREATE OR REPLACE FUNCTION ew_webhook_notify() RETURNS trigger AS $$
BEGIN
    PERFORM pg_notify(
        'ew_webhook_events',
        json_build_object('table', TG_TABLE_NAME, 'op', TG_OP, 'id', NEW.id)::text
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_ew_webhook_strategy_live ON strategy_live;
CREATE TRIGGER trg_ew_webhook_strategy_live
    AFTER UPDATE OF status ON strategy_live
    FOR EACH ROW WHEN (OLD.status IS DISTINCT FROM NEW.status)
    EXECUTE FUNCTION ew_webhook_notify();

DROP TRIGGER IF EXISTS trg_ew_webhook_strategy_backtests ON strategy_backtests;
CREATE TRIGGER trg_ew_webhook_strategy_backtests
    AFTER UPDATE OF status ON strategy_backtests
    FOR EACH ROW WHEN (OLD.status IS DISTINCT FROM NEW.status AND NEW.status IN ('completed', 'failed'))
    EXECUTE FUNCTION ew_webhook_notify();

DROP TRIGGER IF EXISTS trg_ew_webhook_live_alert ON live_alert;
CREATE TRIGGER trg_ew_webhook_live_alert
    AFTER UPDATE OF last_triggered_at ON live_alert
    FOR EACH ROW WHEN (NEW.last_triggered_at IS NOT NULL AND OLD.last_triggered_at IS DISTINCT FROM NEW.last_triggered_at)
    EXECUTE FUNCTION ew_webhook_notify();

DROP TRIGGER IF EXISTS trg_ew_webhook_trades ON trades;
CREATE TRIGGER trg_ew_webhook_trades
    AFTER INSERT ON trades
    FOR EACH ROW
    EXECUTE FUNCTION ew_webhook_notify();

DROP TRIGGER IF EXISTS trg_ew_webhook_connections ON connections;
CREATE TRIGGER trg_ew_webhook_connections
    AFTER UPDATE OF status ON connections
    FOR EACH ROW WHEN (OLD.status IS DISTINCT FROM NEW.status)
    EXECUTE FUNCTION ew_webhook_notify();

-- agent_turn has a text primary key (turn_id): NEW.id does not exist there.
CREATE OR REPLACE FUNCTION ew_webhook_notify_agent_turn() RETURNS trigger AS $$
BEGIN
    PERFORM pg_notify(
        'ew_webhook_events',
        json_build_object('table', TG_TABLE_NAME, 'op', TG_OP, 'id', NEW.turn_id)::text
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_ew_webhook_agent_turn ON agent_turn;
CREATE TRIGGER trg_ew_webhook_agent_turn
    AFTER INSERT OR UPDATE OF finished_at ON agent_turn
    FOR EACH ROW WHEN (NEW.finished_at IS NOT NULL)
    EXECUTE FUNCTION ew_webhook_notify_agent_turn();

COMMIT;
