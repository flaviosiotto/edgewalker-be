-- 030: Notify backend whenever a new row is inserted into n8n_chat_histories
--      so the chat widget can render incoming messages without manual refresh.

BEGIN;

CREATE OR REPLACE FUNCTION notify_chat_history_insert() RETURNS TRIGGER AS $$
DECLARE
    payload TEXT;
BEGIN
    payload := json_build_object(
        'id', NEW.id,
        'session_id', NEW.session_id
    )::text;
    PERFORM pg_notify('chat_history_changes', payload);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_n8n_chat_histories_notify ON n8n_chat_histories;

CREATE TRIGGER trg_n8n_chat_histories_notify
    AFTER INSERT ON n8n_chat_histories
    FOR EACH ROW
    EXECUTE FUNCTION notify_chat_history_insert();

COMMIT;
