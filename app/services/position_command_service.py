"""Single account-scoped write path for positions.

Callers (frontend, manager agent) get one endpoint and commands always reach the
broker the same way:

* simulated account -> the strategy-backtest ledger
* everything else   -> the broker gateway

The strategy-runner is deliberately not in this path. It owns strategy logic,
not position state: it learns about the resulting change from the canonical
projection (the order-aggregator's post-commit event), so there is one command
plane and one state plane instead of a runner-vs-backend split.

Sending an offsetting order is never a close: on hedging / ticket-based brokers
(e.g. cTrader) it opens a new contrary position. Every branch below closes by
broker position id through a real close path.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from sqlmodel import Session, select

from app.models.connection import Connection
from app.models.live_trading import LivePosition
from app.models.strategy import BacktestResult
from app.services.connection_service import SIMULATED_ACCOUNT_TYPE
from app.services.gateway_client import GatewayClient

logger = logging.getLogger(__name__)

BACKTEST_SERVICE_URL = os.getenv("BACKTEST_SERVICE_URL", "http://strategy-backtest:8080")
COMMAND_TIMEOUT = float(os.getenv("POSITION_COMMAND_TIMEOUT", "20"))


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def broker_position_id(position: LivePosition) -> str | None:
    """The id the broker itself knows this position by, or None if unknown.

    A projection row carries several ids; only this one is meaningful to the
    gateway. The row's own primary key never is, so it is deliberately not a
    fallback here — see `_position_broker_ids` for the addressing side.
    """
    extra = _as_dict(position.extra)
    snapshot = _as_dict(extra.get("snapshot_position"))
    adapter = _as_dict(snapshot.get("raw_position") or snapshot.get("rawPosition"))
    raw = _as_dict(snapshot.get("raw") or adapter.get("raw"))
    bucket = str(position.position_bucket or "")
    candidates = (
        snapshot.get("broker_position_id"),
        snapshot.get("position_id"),
        adapter.get("broker_position_id"),
        adapter.get("position_id"),
        raw.get("positionId"),
        raw.get("position_id"),
        bucket[len("position:"):] if bucket.startswith("position:") else None,
    )
    for candidate in candidates:
        text = str(candidate).strip() if candidate not in (None, "") else ""
        if text:
            return text
    return None


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
    resolved = broker_position_id(position)
    if resolved:
        ids.add(resolved)
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
    """Close one position of an account through the broker's own close path."""
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

    # Callers address a position by any of its ids (the row id included), but the
    # gateway only understands the broker's own id: translate once we have the row.
    row = find_open_position(session, account.id, position_id)
    resolved_symbol = symbol or (row.symbol if row is not None else None)
    resolved_position_id = str(position_id)
    if row is not None:
        resolved_position_id = broker_position_id(row) or resolved_position_id
    connection = session.get(Connection, account.connection_id)
    if connection is None:
        raise ValueError("Account has no connection configured")
    result = await _close_via_gateway(
        connection,
        str(account.account_id) if account.account_id else None,
        resolved_position_id,
        quantity=quantity,
        symbol=resolved_symbol,
        extra=command_extra,
    )
    return {"venue": "gateway", "connection_id": connection.id, "result": result}
