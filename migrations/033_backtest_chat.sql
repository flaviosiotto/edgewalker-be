-- 033: Associate each backtest instance with a dedicated chat

BEGIN;

ALTER TABLE strategy_backtests
    ADD COLUMN IF NOT EXISTS chat_id INTEGER REFERENCES chat(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS ix_strategy_backtests_chat_id
    ON strategy_backtests(chat_id);

CREATE UNIQUE INDEX IF NOT EXISTS ux_strategy_backtests_chat_id
    ON strategy_backtests(chat_id)
    WHERE chat_id IS NOT NULL;

ALTER TABLE chat
    DROP CONSTRAINT IF EXISTS ck_chat_live_requires_strategy;

ALTER TABLE chat
    DROP CONSTRAINT IF EXISTS ck_chat_runtime_requires_strategy;

ALTER TABLE chat
    ADD CONSTRAINT ck_chat_runtime_requires_strategy
    CHECK (chat_type NOT IN ('live', 'backtest') OR strategy_id IS NOT NULL);

DO $$
DECLARE
    bt RECORD;
    new_chat_id INTEGER;
BEGIN
    FOR bt IN
        SELECT
            sb.id,
            sb.strategy_id,
            sb.agent_id,
            sb.symbol,
            sb.start_date,
            sb.end_date,
            s.user_id,
            s.manager_agent_id
        FROM strategy_backtests sb
        JOIN strategies s ON s.id = sb.strategy_id
        WHERE sb.chat_id IS NULL
        ORDER BY sb.id
    LOOP
        INSERT INTO chat (
            user_id,
            id_agent,
            strategy_id,
            nome,
            descrizione,
            chat_type,
            created_at
        ) VALUES (
            bt.user_id,
            COALESCE(bt.agent_id, bt.manager_agent_id),
            bt.strategy_id,
            'Backtest #' || bt.id::text,
            'Chat backtest ' || COALESCE(bt.symbol, '') || ' ' || bt.start_date::text || ' - ' || bt.end_date::text,
            'backtest',
            NOW()
        )
        RETURNING id INTO new_chat_id;

        UPDATE strategy_backtests
        SET
            chat_id = new_chat_id,
            agent_id = COALESCE(agent_id, bt.manager_agent_id)
        WHERE id = bt.id;
    END LOOP;
END $$;

COMMIT;