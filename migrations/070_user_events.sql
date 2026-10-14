-- 070: outcomes of long-running tasks per user (docs/valutazione-notifiche-task.md).
--
-- One row per outcome the user may have missed: connection connected /
-- failed / lost, backtest completed / failed, live started / failed /
-- stopped, studio run succeeded / failed. Written by the webhook event
-- sources (services/webhook_sources.py → services/user_events.record) for
-- EVERY user, before the webhook subscription filter. Read by GET /events
-- (the Activity page). Retention: 90 days (webhook dispatcher loop).
--
-- Two touches to the 067 plumbing:
--   * ew_webhook_notify() also carries the previous status ("old_status"),
--     so the handler can tell "connecting → error" (connection.failed) from
--     "connected → error" (connection.lost) and "starting → error" from a
--     crash. to_jsonb(OLD)->>'status' is NULL on tables without a status
--     column (trades) and on INSERT: handlers tolerate both.
--   * a trigger on studio_run (written by studio-svc, same database) for
--     the fourth family. Guarded: the table may not exist on a fresh
--     database before studio-svc has started once.
BEGIN;

CREATE TABLE IF NOT EXISTS user_event (
    id          BIGSERIAL PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    type        VARCHAR(64) NOT NULL,      -- closed catalogue: services/user_events.UserEventType
    severity    VARCHAR(16) NOT NULL,      -- info | success | warning | error
    entity_type VARCHAR(32) NOT NULL,      -- connection | backtest | live | studio_run
    entity_id   INTEGER NOT NULL,
    entity_name VARCHAR(200) NULL,
    message     TEXT NULL,                 -- status_message / error_message / run.error, verbatim
    details     TEXT NULL,                 -- runner log tail; only GET /events/{id} serves it
    data        JSONB NOT NULL DEFAULT '{}'::jsonb,
    links       JSONB NOT NULL DEFAULT '{}'::jsonb,
    dedupe_key  VARCHAR(200) NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_user_event_user_id ON user_event (user_id);
CREATE INDEX IF NOT EXISTS ix_user_event_user_created ON user_event (user_id, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS uq_user_event_dedupe
    ON user_event (user_id, dedupe_key) WHERE dedupe_key IS NOT NULL;
COMMENT ON TABLE user_event IS 'Outcomes of long-running tasks shown on the Activity page (closed catalogue, 90-day retention)';

-- old_status in the NOTIFY payload (backward compatible: 067 handlers ignore it)
CREATE OR REPLACE FUNCTION ew_webhook_notify() RETURNS trigger AS $$
BEGIN
    PERFORM pg_notify(
        'ew_webhook_events',
        json_build_object(
            'table', TG_TABLE_NAME,
            'op', TG_OP,
            'id', NEW.id,
            'old_status', CASE WHEN TG_OP = 'UPDATE' THEN to_jsonb(OLD)->>'status' ELSE NULL END
        )::text
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- studio_run is created by studio-svc (same database): the trigger is the
-- backend's because the backend is the listener.
DO $$
BEGIN
    IF to_regclass('public.studio_run') IS NOT NULL THEN
        DROP TRIGGER IF EXISTS trg_ew_webhook_studio_run ON studio_run;
        CREATE TRIGGER trg_ew_webhook_studio_run
            AFTER UPDATE OF status ON studio_run
            FOR EACH ROW WHEN (OLD.status IS DISTINCT FROM NEW.status AND NEW.status IN ('succeeded', 'failed', 'cancelled'))
            EXECUTE FUNCTION ew_webhook_notify();
    ELSE
        RAISE NOTICE 'studio_run not present: re-run the studio_run block of 070 after studio-svc has created it';
    END IF;
END $$;

COMMIT;
