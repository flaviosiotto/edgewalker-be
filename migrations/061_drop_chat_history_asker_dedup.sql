-- 061: drop the n8n-only dedup trigger introduced by migration 046.
--
-- 046 swallowed the unattributed copy of the human text that n8n's Postgres
-- Chat Memory node wrote at turn end. n8n no longer runs any agent: every
-- turn goes to agent-svc (AGENT_SVC_WEBHOOK_URL), which checks by itself
-- whether the caller already recorded the prompt
-- (agent-svc app/turn.py::_prompt_already_recorded) before writing its own
-- copy. Nothing relies on the trigger any more.
--
-- Deploy order: agent-svc with that check FIRST, then this migration
-- (the other way round a turn whose attributed row is no longer the last
-- one would show the question twice).
--
-- The notify trigger of migration 030 (trg_n8n_chat_histories_notify) is
-- NOT touched: the chat SSE (new_message) depends on it.

BEGIN;

DROP TRIGGER IF EXISTS trg_n8n_chat_histories_dedup_asker ON n8n_chat_histories;
DROP FUNCTION IF EXISTS dedup_attributed_chat_history_insert();

COMMIT;
