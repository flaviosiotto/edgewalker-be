-- 062: lessons become per-run playbooks (docs/valutazione-playbook-lezioni.md).
--
-- Until now agent_lessons was one mutable pool per strategy: every backtest
-- read and rewrote it, live picked from it by confidence. Now a backtest
-- starts from a playbook chosen at launch (copied into its own rows), its
-- output freezes when the run ends, and live attaches to the output of ONE
-- backtest. The strategy only points at the backtest whose output is its
-- current playbook.
--
-- Existing rows keep scope='strategy': the "initial playbook" of the
-- strategy, still selectable as a run input until replaced.
--
-- Safe to re-run: every statement is guarded.

BEGIN;

ALTER TABLE agent_lessons
    ADD COLUMN IF NOT EXISTS scope VARCHAR(16) NOT NULL DEFAULT 'strategy';
ALTER TABLE agent_lessons
    ADD COLUMN IF NOT EXISTS run_backtest_id INTEGER
        REFERENCES strategy_backtests(id) ON DELETE CASCADE;
ALTER TABLE agent_lessons
    ADD COLUMN IF NOT EXISTS parent_id INTEGER
        REFERENCES agent_lessons(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS ix_agent_lessons_run_backtest
    ON agent_lessons (run_backtest_id, status);

COMMENT ON COLUMN agent_lessons.scope IS 'strategy = initial playbook of the strategy (pre-062 rows, manual rows); backtest = row of the playbook of run_backtest_id';
COMMENT ON COLUMN agent_lessons.run_backtest_id IS 'The run whose playbook this row belongs to (scope=backtest)';
COMMENT ON COLUMN agent_lessons.parent_id IS 'Input row this one was copied from at launch (lineage); NULL = born in the run';
COMMENT ON COLUMN agent_lessons.backtest_id IS 'The run the lesson was BORN in (kept across copies)';

ALTER TABLE strategies
    ADD COLUMN IF NOT EXISTS playbook_backtest_id INTEGER
        REFERENCES strategy_backtests(id) ON DELETE SET NULL;
COMMENT ON COLUMN strategies.playbook_backtest_id IS 'Backtest whose output playbook is the current one (default input of new runs and of live)';

ALTER TABLE strategy_live
    ADD COLUMN IF NOT EXISTS playbook_backtest_id INTEGER
        REFERENCES strategy_backtests(id) ON DELETE SET NULL;
COMMENT ON COLUMN strategy_live.playbook_backtest_id IS 'Playbook attached at launch (audit; live never writes lessons)';

COMMIT;
