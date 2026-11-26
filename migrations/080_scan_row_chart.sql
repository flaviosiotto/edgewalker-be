-- 080: scan_rows.chart — the bars behind a scan row's miniature.
--
-- The last closed bars the scan read for the symbol, kept so the row can be
-- drawn without asking the broker again:
--   {"ts": [ms…], "o": […], "h": […], "l": […], "c": […],
--    "series": {"<indicator key>": […]}}
-- Arrays are aligned, oldest first; a missing value is null. Written by the
-- coordinator (strategy-backtest) only; rows of earlier scans keep NULL.
--
-- Run BEFORE deploying the backend and strategy-backtest: both map the column.
BEGIN;

ALTER TABLE scan_rows ADD COLUMN IF NOT EXISTS chart JSONB;

COMMENT ON COLUMN scan_rows.chart IS 'Last closed bars read by the scan, for the row miniature: ts/o/h/l/c arrays and indicator series, oldest first';

COMMIT;
