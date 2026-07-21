"""Single account-scoped write path for orders.

The mirror of ``position_command_service``: callers (frontend, manager agent)
get one endpoint and commands always reach the broker the same way.

* simulated account -> the strategy-backtest ledger
* everything else   -> the broker gateway

The strategy-runner is deliberately not in this path. Its HTTP order endpoints
only ever built an ``order_ref`` and forwarded to the gateway — they never took
the engine's ``_order_lock``, so routing through it bought no serialization
against the rule engine. What it did own, strategy attribution, is rebuilt here
from ``strategy_live_id`` so orders keep the same reference format.
"""
from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from sqlmodel import Session, select

from app.models.connection import Connection
from app.models.live_trading import LiveOrder
from app.models.strategy import BacktestResult, StrategyLive
from app.services.connection_service import SIMULATED_ACCOUNT_TYPE
from app.services.gateway_client import GatewayClient

logger = logging.getLogger(__name__)

COMMAND_TIMEOUT = float(os.getenv("ORDER_COMMAND_TIMEOUT", "30"))

# Kept at 33 so Binance child refs can append ":tp" / ":sl" / ":oco" and still
# fit the broker's 36-char client-order-id limit.
MAX_ORDER_REF_LEN = 33

VALID_SIDES = {"buy", "sell"}
VALID_ORDER_TYPES = {"market", "limit", "stop"}


def build_order_ref(strategy_id: Any, strategy_live_id: Any) -> str:
    """Reproduce the runner's reference format: ``strategy-{id}:live-{id}:{token}``.

    The projection scopes orders to a strategy by parsing this prefix, so the
    format is a contract shared with strategy-runner's ``_build_order_ref`` —
    changing one without the other silently unscopes orders in the UI.
    """
    safe_strategy = str(strategy_id or "unknown").replace(":", "_")
    safe_live = str(strategy_live_id or "unknown").replace(":", "_")
    prefix = f"strategy-{safe_strategy}:live-{safe_live}:"
    token_len = max(1, MAX_ORDER_REF_LEN - len(prefix))
    return f"{prefix}{uuid.uuid4().hex[:token_len]}"[:MAX_ORDER_REF_LEN]


def _resolve_order_ref(session: Session, strategy_live_id: int | None) -> str | None:
    """Attribution is opt-in: an order with no strategy context stays unscoped."""
    if strategy_live_id is None:
        return None
    live = session.get(StrategyLive, strategy_live_id)
    if live is None:
        raise ValueError(f"strategy_live {strategy_live_id} not found")
    return build_order_ref(live.strategy_id, live.id)


def _validate(side: str, order_type: str, quantity: float, limit_price: float | None,
              stop_price: float | None) -> tuple[str, str]:
    normalised_side = str(side or "").lower()
    normalised_type = str(order_type or "market").lower()
    if normalised_side not in VALID_SIDES:
        raise ValueError("side must be 'buy' or 'sell'")
    if normalised_type not in VALID_ORDER_TYPES:
        raise ValueError("order_type must be 'market', 'limit' or 'stop'")
    if quantity <= 0:
        raise ValueError("quantity must be > 0")
    if normalised_type == "limit" and limit_price is None:
        raise ValueError("limit_price required for limit orders")
    if normalised_type == "stop" and stop_price is None:
        raise ValueError("stop_price required for stop orders")
    return normalised_side, normalised_type


def _backtest_for_account(session: Session, account: Any) -> BacktestResult:
    backtest = session.exec(
        select(BacktestResult).where(BacktestResult.account_id == account.id)
    ).first()
    if backtest is None:
        raise ValueError("No backtest backs this simulated account")
    return backtest


async def place_account_order(
    session: Session,
    account: Any,
    *,
    symbol: str,
    side: str,
    order_type: str = "market",
    quantity: float,
    limit_price: float | None = None,
    stop_price: float | None = None,
    take_profit_price: float | None = None,
    stop_loss_price: float | None = None,
    strategy_live_id: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Submit an order for an account through whoever is authoritative."""
    normalised_side, normalised_type = _validate(
        side, order_type, quantity, limit_price, stop_price
    )
    for label, price in (
        ("take_profit_price", take_profit_price),
        ("stop_loss_price", stop_loss_price),
        ("limit_price", limit_price),
        ("stop_price", stop_price),
    ):
        if price is not None and price <= 0:
            raise ValueError(f"{label} must be > 0")

    command_extra = {"source": "backend_account_api", **(extra or {})}

    if account.account_type == SIMULATED_ACCOUNT_TYPE:
        backtest = _backtest_for_account(session, account)
        from app.services.backtest_runner_service import backtest_runner_service

        payload: dict[str, Any] = {
            "symbol": symbol,
            "side": normalised_side,
            "order_type": normalised_type,
            "quantity": quantity,
            "extra": command_extra,
        }
        for key, value in (
            ("limit_price", limit_price),
            ("stop_price", stop_price),
            ("take_profit_price", take_profit_price),
            ("stop_loss_price", stop_loss_price),
        ):
            if value is not None:
                payload[key] = value
        result = backtest_runner_service.place_backtest_order(backtest.id, payload)
        return {"venue": "backtest", "backtest_id": backtest.id, "result": result}

    connection = session.get(Connection, account.connection_id)
    if connection is None:
        raise ValueError("Account has no connection configured")

    order_ref = _resolve_order_ref(session, strategy_live_id)
    client = GatewayClient(
        connection.id, broker_type=connection.broker_type, timeout=COMMAND_TIMEOUT
    )
    try:
        result = await client.place_order(
            symbol=symbol,
            side=normalised_side,
            order_type=normalised_type,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            account=str(account.account_id) if account.account_id else None,
            order_ref=order_ref,
            extra=command_extra,
        )
    except Exception as exc:
        raise RuntimeError(f"Gateway order placement failed: {exc}") from exc
    return {
        "venue": "gateway",
        "connection_id": connection.id,
        "order_ref": order_ref,
        "result": result,
    }


def find_account_order(session: Session, account_id: int, order_id: str) -> LiveOrder | None:
    """Resolve an order of this account by row id or broker order id."""
    target = str(order_id).strip()
    if not target:
        return None
    if target.isdigit():
        row = session.get(LiveOrder, int(target))
        if row is not None and row.account_id == account_id:
            return row
    return session.exec(
        select(LiveOrder).where(
            LiveOrder.account_id == account_id,  # type: ignore[arg-type]
            LiveOrder.broker_order_id == target,  # type: ignore[arg-type]
        )
    ).first()


async def cancel_account_order(
    session: Session,
    account: Any,
    order_id: str,
) -> dict[str, Any]:
    """Cancel a working order of this account."""
    if account.account_type == SIMULATED_ACCOUNT_TYPE:
        backtest = _backtest_for_account(session, account)
        from app.services.backtest_runner_service import backtest_runner_service

        result = backtest_runner_service.cancel_backtest_order(
            backtest.id, str(order_id), status_message="cancelled_via_account_api"
        )
        return {"venue": "backtest", "backtest_id": backtest.id, "result": result}

    # The gateway speaks broker order ids; the UI and the agent both address
    # orders by the projection row id, so translate whenever the row is known.
    row = find_account_order(session, account.id, order_id)
    broker_order_id = str(order_id)
    symbol: str | None = None
    if row is not None:
        if not row.broker_order_id:
            raise ValueError("Order has no broker id yet: it cannot be cancelled")
        broker_order_id = row.broker_order_id
        symbol = row.symbol

    connection = session.get(Connection, account.connection_id)
    if connection is None:
        raise ValueError("Account has no connection configured")
    client = GatewayClient(
        connection.id, broker_type=connection.broker_type, timeout=COMMAND_TIMEOUT
    )
    try:
        result = await client.cancel_order(broker_order_id, symbol=symbol)
    except Exception as exc:
        raise RuntimeError(f"Gateway order cancel failed: {exc}") from exc
    return {"venue": "gateway", "connection_id": connection.id, "result": result}
