"""Single account-scoped write path for positions.

Callers (frontend, manager agent) get one endpoint; this module routes the
command to whoever is authoritative for that position, so there is never more
than one commander at a time:

* simulated account  -> the strategy-backtest ledger
* a live runner is running for the position -> that runner, whose ``_order_lock``
  serializes the close against the rule engine's own decisions
* nobody is running  -> straight to the broker gateway

Sending an offsetting order is never a close: on hedging / ticket-based brokers
(e.g. cTrader) it opens a new contrary position. Every branch below closes by
broker position id through a real close path.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from sqlmodel import Session, select

from app.models.connection import Connection
from app.models.live_trading import LivePosition
from app.models.strategy import BacktestResult, LiveStatus, StrategyLive
from app.services.connection_service import SIMULATED_ACCOUNT_TYPE
from app.services.gateway_client import GatewayClient

logger = logging.getLogger(__name__)

BACKTEST_SERVICE_URL = os.getenv("BACKTEST_SERVICE_URL", "http://strategy-backtest:8080")
RUNNER_CONTAINER_PREFIX = os.getenv("LIVE_RUNNER_CONTAINER_PREFIX", "edgewalker-live-")
RUNNER_PORT = int(os.getenv("LIVE_RUNNER_PORT", "8080"))
COMMAND_TIMEOUT = float(os.getenv("POSITION_COMMAND_TIMEOUT", "20"))

# A session in these states may still own the position and hold the order lock.
_LIVE_ACTIVE_STATES = {LiveStatus.RUNNING.value, LiveStatus.STARTING.value, LiveStatus.STOPPING.value}


def _runner_base_url(live_id: int) -> str:
    return f"http://{RUNNER_CONTAINER_PREFIX}{live_id}:{RUNNER_PORT}"


def _position_broker_ids(position: LivePosition) -> set[str]:
    """Every id this projection row may be addressed by."""
    ids: set[str] = set()
    bucket = str(position.position_bucket or "")
    if bucket:
        ids.add(bucket)
        if bucket.startswith("position:"):
            ids.add(bucket[len("position:"):])
    extra = position.extra if isinstance(position.extra, dict) else {}
    snapshot = extra.get("snapshot_position")
    if isinstance(snapshot, dict):
        for key in ("broker_position_id", "position_id", "positionId"):
            value = snapshot.get(key)
            if value not in (None, ""):
                ids.add(str(value))
    if position.id is not None:
        ids.add(str(position.id))
    return ids


def find_open_position(session: Session, account_id: int, position_id: str) -> LivePosition | None:
    """Resolve an open projection row for this account by any of its ids."""
    target = str(position_id).strip()
    if not target:
        return None
    rows = session.exec(
        select(LivePosition).where(
            LivePosition.account_id == account_id,  # type: ignore[arg-type]
            LivePosition.status == "open",  # type: ignore[arg-type]
        )
    ).all()
    for row in rows:
        if target in _position_broker_ids(row):
            return row
    return None


def _find_active_live_session(
    session: Session,
    account_id: int,
    *,
    strategy_live_id: int | None,
    symbol: str | None,
) -> StrategyLive | None:
    """The live session that may still be commanding this position, if any."""
    if strategy_live_id is not None:
        live = session.get(StrategyLive, strategy_live_id)
        if live is not None and live.status in _LIVE_ACTIVE_STATES:
            return live
        # The owning session ended; another one may have taken over the symbol.
    if not symbol:
        return None
    stmt = select(StrategyLive).where(
        StrategyLive.account_id == account_id,  # type: ignore[arg-type]
        StrategyLive.symbol == symbol,  # type: ignore[arg-type]
        StrategyLive.status.in_(_LIVE_ACTIVE_STATES),  # type: ignore[union-attr]
    )
    return session.exec(stmt).first()


def _close_via_runner(live_id: int, position_id: str, quantity: float, reason: str) -> dict[str, Any]:
    payload = {"position_id": position_id, "quantity": quantity, "reason": reason}
    try:
        with httpx.Client(timeout=COMMAND_TIMEOUT) as client:
            resp = client.post(f"{_runner_base_url(live_id)}/position/close", json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Runner position close failed: {_error_detail(exc)}") from exc
    except Exception as exc:
        raise RuntimeError(f"Live runner {live_id} is not reachable: {exc}") from exc


async def _close_via_gateway(
    connection: Connection,
    broker_account_id: str | None,
    position_id: str,
    *,
    quantity: float,
    symbol: str | None,
    extra: dict[str, Any],
) -> dict[str, Any]:
    client = GatewayClient(connection.id, broker_type=connection.broker_type, timeout=COMMAND_TIMEOUT)
    try:
        return await client.close_position(
            position_id,
            quantity=quantity,
            account=broker_account_id,
            symbol=symbol,
            extra=extra,
        )
    except Exception as exc:
        raise RuntimeError(f"Gateway position close failed: {exc}") from exc


def _error_detail(exc: httpx.HTTPStatusError) -> Any:
    try:
        return exc.response.json()
    except Exception:
        return exc.response.text


async def close_account_position(
    session: Session,
    account: Any,
    position_id: str,
    *,
    quantity: float,
    symbol: str | None = None,
    reason: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Close one position of an account, routed to its authoritative commander."""
    if quantity <= 0:
        raise ValueError("quantity must be > 0")
    close_reason = reason or "account_position_close"
    command_extra = {
        "reason": close_reason,
        "source": "backend_account_api",
        **(extra or {}),
    }

    # Simulated account -> the backtest ledger owns execution.
    if account.account_type == SIMULATED_ACCOUNT_TYPE:
        backtest = session.exec(
            select(BacktestResult).where(BacktestResult.account_id == account.id)
        ).first()
        if backtest is None:
            raise ValueError("No backtest backs this simulated account")
        from app.services.backtest_runner_service import backtest_runner_service

        payload: dict[str, Any] = {"quantity": quantity, "extra": command_extra}
        if symbol:
            payload["symbol"] = symbol
        result = backtest_runner_service.close_backtest_position(backtest.id, position_id, payload)
        return {"venue": "backtest", "backtest_id": backtest.id, "result": result}

    row = find_open_position(session, account.id, position_id)
    resolved_symbol = symbol or (row.symbol if row is not None else None)
    live = _find_active_live_session(
        session,
        account.id,
        strategy_live_id=row.strategy_live_id if row is not None else None,
        symbol=resolved_symbol,
    )

    # A running runner stays the single commander so its order lock keeps this
    # close serialized against the rule engine's own TP/SL decisions.
    if live is not None and live.id is not None:
        try:
            result = _close_via_runner(live.id, position_id, quantity, close_reason)
            return {"venue": "runner", "live_id": live.id, "result": result}
        except RuntimeError as exc:
            # Marked active but unreachable (crashed / mid-restart): fall through
            # to the gateway rather than leaving the position uncloseable.
            logger.warning("Runner %s unreachable for position close, using gateway: %s", live.id, exc)

    connection = session.get(Connection, account.connection_id)
    if connection is None:
        raise ValueError("Account has no connection configured")
    result = await _close_via_gateway(
        connection,
        str(account.account_id) if account.account_id else None,
        position_id,
        quantity=quantity,
        symbol=resolved_symbol,
        extra=command_extra,
    )
    return {"venue": "gateway", "connection_id": connection.id, "result": result}
