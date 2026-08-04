-- Migration: agent_call — audit log of every runner→manager-agent invocation
--            (alerts, ask_agent rules, reinvokes, trade reviews, notifications).
--            bar_ts is the SIMULATED replay/bar timestamp (epoch ms) so calls
--            can be placed on the chart at the right candle; called_at is the
--            wall-clock time of the dispatch. tokens_*/model are reserved for
--            future usage accounting.
-- Date: 2026-08-04
--
-- Safe to re-run: every statement is guarded.

BEGIN;

CREATE TABLE IF NOT EXISTS agent_call (
    id               SERIAL PRIMARY KEY,
    strategy_id      INTEGER REFERENCES strategies(id) ON DELETE CASCADE,
    backtest_id      INTEGER REFERENCES strategy_backtests(id) ON DELETE CASCADE,
    strategy_live_id INTEGER REFERENCES strategy_live(id) ON DELETE CASCADE,
    trigger_type     VARCHAR(64) NOT NULL,
    trigger_name     VARCHAR(255),
    correlation_id   VARCHAR(100),
    session_id       VARCHAR(100),
    bar_ts           BIGINT,
    called_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    duration_ms      INTEGER,
    status           VARCHAR(16) NOT NULL DEFAULT 'delivered',
    prompt_chars     INTEGER,
    response_chars   INTEGER,
    tokens_input     INTEGER,
    tokens_output    INTEGER,
    model            VARCHAR(64),
    extra            JSONB
);

CREATE INDEX IF NOT EXISTS ix_agent_call_backtest_id ON agent_call (backtest_id);
CREATE INDEX IF NOT EXISTS ix_agent_call_strategy_live_id ON agent_call (strategy_live_id);
CREATE INDEX IF NOT EXISTS ix_agent_call_strategy_called ON agent_call (strategy_id, called_at);

COMMIT;
