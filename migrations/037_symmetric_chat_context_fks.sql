-- 037: Symmetric per-context chat ownership via chat-side FKs.
--
-- Before: every chat carried `chat.strategy_id`, so run chats (live/backtest)
-- appeared as design chats; the design/run split relied on `chat_type`.
--
-- After: a chat is owned by exactly ONE context, expressed as a chat-side FK:
--   * design   -> chat.strategy_id  (the ONLY chats that keep strategy_id)
--   * backtest -> chat.backtest_id  (1:1 with a backtest)
--   * live     -> chat.live_id      (1:1 with a live SESSION; a restart yields
--                                    a fresh, empty chat)
-- All three ON DELETE CASCADE, so deleting a context removes its chat(s).
-- `Strategy.chats` becomes `chat WHERE strategy_id` — pure FK, no chat_type.

BEGIN;

-- ── New chat-side ownership columns ──────────────────────────────────────────
ALTER TABLE chat
    ADD COLUMN IF NOT EXISTS backtest_id INTEGER
        REFERENCES strategy_backtests(id) ON DELETE CASCADE;
ALTER TABLE chat
    ADD COLUMN IF NOT EXISTS live_id INTEGER
        REFERENCES strategy_live(id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS ix_chat_backtest_id ON chat(backtest_id);
CREATE INDEX IF NOT EXISTS ix_chat_live_id ON chat(live_id);
-- A backtest owns exactly one chat.
CREATE UNIQUE INDEX IF NOT EXISTS ux_chat_backtest_id
    ON chat(backtest_id) WHERE backtest_id IS NOT NULL;

-- ── Backfill from the old reverse FKs ────────────────────────────────────────
UPDATE chat c
    SET backtest_id = b.id
    FROM strategy_backtests b
    WHERE b.chat_id = c.id;

-- Historical live chats were shared per-strategy across sessions; attribute the
-- shared chat to (any) one referencing session. Going forward each session gets
-- its own chat, so this only affects pre-migration history.
UPDATE chat c
    SET live_id = l.id
    FROM strategy_live l
    WHERE l.chat_id = c.id;

-- Drop the old invariant (run chats required strategy_id) BEFORE nulling it,
-- otherwise the UPDATE below would violate the still-present CHECK.
ALTER TABLE chat
    DROP CONSTRAINT IF EXISTS ck_chat_runtime_requires_strategy;

-- Run chats no longer belong to the strategy design; only design chats keep it.
UPDATE chat
    SET strategy_id = NULL
    WHERE backtest_id IS NOT NULL OR live_id IS NOT NULL;

-- ── Drop the old reverse FK columns ──────────────────────────────────────────
ALTER TABLE strategy_backtests DROP COLUMN IF EXISTS chat_id;
ALTER TABLE strategy_live       DROP COLUMN IF EXISTS chat_id;

COMMIT;
