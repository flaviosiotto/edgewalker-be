-- 046: "Who is asking" in the agent chat.
--
-- ┌──────────────────────────────────────────────────────────────────────────┐
-- │ WORKAROUND — n8n-specific, to be REMOVED when n8n is dismissed.          │
-- │                                                                          │
-- │ The asker (human / external agent via MCP / ask_agent rule / alert) is  │
-- │ persisted by the backend or the strategy-runner BEFORE the agent turn,  │
-- │ as a ``human`` row carrying ``message->'metadata'->>'sender_kind'``.     │
-- │ n8n's Postgres Chat Memory node cannot be told "the input is already    │
-- │ stored": it always writes its own copy of the human text at turn end    │
-- │ (no metadata). This trigger exists ONLY to swallow that duplicate.       │
-- │                                                                          │
-- │ Removal criterion: once no agent runtime writes to n8n_chat_histories   │
-- │ except EdgeWalker code (backend/runner), run:                            │
-- │   DROP TRIGGER trg_n8n_chat_histories_dedup_asker ON n8n_chat_histories;│
-- │   DROP FUNCTION dedup_attributed_chat_history_insert();                 │
-- │ The attributed-row write path stays: it is the intended design.         │
-- └──────────────────────────────────────────────────────────────────────────┘
--
-- Rule: an unattributed human row whose content equals the most recent human
-- row of the same session is skipped when that row is attributed.

BEGIN;

CREATE OR REPLACE FUNCTION dedup_attributed_chat_history_insert() RETURNS TRIGGER AS $$
DECLARE
    prev_message JSONB;
BEGIN
    IF NEW.message->>'type' <> 'human' OR NEW.message ? 'metadata' THEN
        RETURN NEW;
    END IF;

    SELECT message INTO prev_message
    FROM n8n_chat_histories
    WHERE session_id = NEW.session_id
      AND message->>'type' = 'human'
    ORDER BY id DESC
    LIMIT 1;

    IF prev_message IS NOT NULL
       AND prev_message->'metadata'->>'sender_kind' IS NOT NULL
       AND prev_message->>'content' = NEW.message->>'content' THEN
        RETURN NULL;  -- skip the insert: already recorded with attribution
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_n8n_chat_histories_dedup_asker ON n8n_chat_histories;

CREATE TRIGGER trg_n8n_chat_histories_dedup_asker
    BEFORE INSERT ON n8n_chat_histories
    FOR EACH ROW
    EXECUTE FUNCTION dedup_attributed_chat_history_insert();

COMMIT;
