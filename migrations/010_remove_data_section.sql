-- Migration: Remove deprecated "data" section from strategy definitions and backtest configs
--
-- The "data" top-level key in the YAML/JSON definition is redundant:
--   - data.source  → already in backtest.source
--   - data.symbol  → already in strategy.symbol
--   - data.*        → merge remaining keys into backtest section
--
-- This migration:
-- 1. Merges any keys from definition->'data' into definition->'backtest' (without overwriting existing)
-- 2. Removes the 'data' key from definition
-- 3. Does the same for strategy_backtests.config

-- Step 1: Strategies — merge data into backtest, then remove data key
UPDATE strategies
SET definition = (
    -- Remove 'data' key after merging into 'backtest'
    (
        CASE
            WHEN definition ? 'backtest' THEN
                -- Merge: data keys as base, backtest keys on top (backtest wins on conflict)
                jsonb_set(
                    definition,
                    '{backtest}',
                    (COALESCE(definition->'data', '{}'::jsonb) || COALESCE(definition->'backtest', '{}'::jsonb))
                    -- Remove 'symbol' from backtest since it belongs in strategy.symbol
                    - 'symbol'
                )
            ELSE
                -- No backtest section: rename data to backtest (minus symbol)
                jsonb_set(definition, '{backtest}', (COALESCE(definition->'data', '{}'::jsonb) - 'symbol'))
        END
    ) - 'data'
)
WHERE definition ? 'data';

-- Step 2: Backtest configs — same logic
UPDATE strategy_backtests
SET config = (
    (
        CASE
            WHEN config ? 'backtest' THEN
                jsonb_set(
                    config,
                    '{backtest}',
                    (COALESCE(config->'data', '{}'::jsonb) || COALESCE(config->'backtest', '{}'::jsonb))
                    - 'symbol'
                )
            ELSE
                jsonb_set(config, '{backtest}', (COALESCE(config->'data', '{}'::jsonb) - 'symbol'))
        END
    ) - 'data'
)
WHERE config ? 'data';
