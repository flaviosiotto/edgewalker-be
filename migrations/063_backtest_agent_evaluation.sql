-- 063: the agent's final evaluation of a backtest, structured.
--
-- At the end of a run the agent submits (tool submit_backtest_evaluation):
-- six axis scores 0..100 for the radar (edge, risk, consistency, discipline,
-- execution, robustness), a short diagnosis and its hints. The backend
-- derives the overall score_pct from the axis weights. Shown in the
-- Performance tab as "Agent Score" and "Agent Hint".
--
-- Safe to re-run.

BEGIN;

ALTER TABLE strategy_backtests
    ADD COLUMN IF NOT EXISTS agent_evaluation JSONB;

COMMENT ON COLUMN strategy_backtests.agent_evaluation IS '{scores: {axis: {score, rationale}}, score_pct, summary, hints: [{title, detail, category, priority}], playbook_recommended, agent_id, created_at}';

COMMIT;
