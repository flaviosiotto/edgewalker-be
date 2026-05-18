-- 019: Add timestamp support and paging index to n8n_chat_histories

BEGIN;

ALTER TABLE n8n_chat_histories
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ;

ALTER TABLE n8n_chat_histories
    ALTER COLUMN created_at SET DEFAULT NOW();

CREATE INDEX IF NOT EXISTS ix_n8n_chat_histories_session_id_id_desc
    ON n8n_chat_histories(session_id, id DESC);

COMMIT;