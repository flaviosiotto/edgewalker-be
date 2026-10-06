-- 065: tail of the runner container log for failed backtests.
-- Captured by the backend reaper before removing the exited runner container
-- (the container, and its log, disappear minutes after a failure).
BEGIN;

ALTER TABLE strategy_backtests
    ADD COLUMN IF NOT EXISTS runner_log_tail TEXT;

COMMENT ON COLUMN strategy_backtests.runner_log_tail IS
    'Last lines of the runner container log, saved when a failed run''s runner container is reaped';

COMMIT;
