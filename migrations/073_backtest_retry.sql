-- 073: restart of a backtest (docs/valutazione-riavvio-backtest.md, F0 + F1).
--
-- F0 "Riprova": a failed/completed run is never re-run in place (chat,
-- agent calls, lessons, events and tokens are all keyed on the id); it is
-- cloned into a new row that points back to the attempt it repeats.
-- `failure_kind` classifies a terminal failure (user_stop | shutdown |
-- agent_timeout | runner_attach | data | start | internal): the backend sets
-- user_stop BEFORE asking the coordinator to cancel, the coordinator fills
-- the rest at finalize and never overwrites a kind already set.
--
-- F1: an agent turn that outlives its hold no longer fails the run; the
-- runner marks the agent_call row `timeout` and the trigger below turns it
-- into a user_event (backtest.agent_timeout) through the webhook bus of 067.
--
-- Run BEFORE deploying the backend (create_all adds tables, never columns).
BEGIN;

ALTER TABLE strategy_backtests
    ADD COLUMN IF NOT EXISTS retried_from_backtest_id INTEGER
        REFERENCES strategy_backtests(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS failure_kind VARCHAR(32);
CREATE INDEX IF NOT EXISTS ix_strategy_backtests_retried_from
    ON strategy_backtests (retried_from_backtest_id);

COMMENT ON COLUMN strategy_backtests.retried_from_backtest_id IS 'Backtest this row repeats (F0 Riprova): same inputs and config snapshot, fresh chat/playbook';
COMMENT ON COLUMN strategy_backtests.failure_kind IS 'Why a terminal failed/error row ended: user_stop | shutdown | agent_timeout | runner_attach | data | start | internal';

DROP TRIGGER IF EXISTS trg_ew_webhook_agent_call ON agent_call;
CREATE TRIGGER trg_ew_webhook_agent_call
    AFTER INSERT OR UPDATE OF status ON agent_call
    FOR EACH ROW WHEN (NEW.status = 'timeout')
    EXECUTE FUNCTION ew_webhook_notify();

COMMIT;
