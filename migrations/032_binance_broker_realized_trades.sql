-- Migration 032: Rebuild Binance live trades from broker-realized fills
--
-- Binance futures userTrades carries broker-authoritative realizedPnl per
-- execution. The dashboard-facing trades projection should use that value for
-- Binance instead of locally materialized FIFO matches.

BEGIN;

DELETE FROM public.trades t
USING public.accounts a
JOIN public.connections c ON c.id = a.connection_id
WHERE t.account_id = a.id
  AND lower(c.broker_type) = 'binance';

INSERT INTO public.trades (
    strategy_live_id,
    account_id,
    symbol,
    direction,
    quantity,
    entry_price,
    exit_price,
    multiplier,
    realized_pnl,
    commission,
    net_pnl,
    trusted,
    entry_fill_id,
    exit_fill_id,
    entry_time,
    exit_time,
    created_at,
    extra
)
SELECT
    f.strategy_live_id,
    f.account_id,
    f.symbol,
    CASE WHEN lower(f.side) LIKE 'buy%' THEN 'short' ELSE 'long' END AS direction,
    abs(f.quantity) AS quantity,
    NULL::double precision AS entry_price,
    f.price AS exit_price,
    1.0 AS multiplier,
    f.realized_pnl,
    coalesce(f.commission, 0.0) AS commission,
    f.realized_pnl - coalesce(f.commission, 0.0) AS net_pnl,
    TRUE AS trusted,
    NULL::integer AS entry_fill_id,
    f.id AS exit_fill_id,
    NULL::timestamptz AS entry_time,
    f.fill_time AS exit_time,
    now() AS created_at,
    jsonb_build_object(
        'source', 'broker_realized_fill',
        'broker_type', 'binance',
        'broker_fill_id', f.broker_fill_id
    ) AS extra
FROM public.fills f
JOIN public.accounts a ON a.id = f.account_id
JOIN public.connections c ON c.id = a.connection_id
WHERE lower(c.broker_type) = 'binance'
  AND f.realized_pnl IS NOT NULL
  AND abs(f.realized_pnl) > 1e-12
  AND f.extra->'last_broker_fill_event' ? 'realized_pnl';

COMMIT;