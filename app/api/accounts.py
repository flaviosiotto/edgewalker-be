"""Canonical consultative account APIs.

These endpoints expose account-scoped broker state for both the frontend
and delegated agent/n8n consumers without forcing them through runner
APIs or connection-management routes.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.db.database import get_session
from app.models.user import User
from app.schemas.connection import AccountListResponse, AccountRead
from app.schemas.live_trading import (
    LiveFillRead,
    LiveOrderRead,
    LivePositionRead,
    LiveTradeRead,
)
from app.services.connection_service import (
    SIMULATED_ACCOUNT_TYPE,
    get_account,
    list_accounts,
    list_all_accounts,
)
from app.services.connection_service import get_connection
from app.services.connection_manager import get_connection_manager, resolve_order_history_lookback_days
from app.services.live_trading_service import (
    list_account_fills,
    list_account_orders,
    list_account_positions,
    list_account_trades,
    purge_account_fills,
    purge_account_orders,
    purge_account_trades,
)
from app.utils.auth_utils import get_current_active_or_consultative_user

logger = logging.getLogger(__name__)

_BACKTEST_SERVICE_URL = os.getenv("BACKTEST_SERVICE_URL", "http://strategy-backtest:8080").rstrip("/")

router = APIRouter(prefix="/accounts", tags=["Accounts"])


def _resolve_simulated_account_read(session: Session, account, base: AccountRead) -> AccountRead:
    """Overlay live simulated balance/equity onto a backtest account.

    While the backtest runs, balance/equity come straight from the backtest
    service (proxy-on-read, always current). Once it ends, the persisted
    equity_final is authoritative. Any hiccup falls back to the stored row, so
    a slow/absent backtest service degrades to a stale read, never a 5xx.
    """
    from app.models.strategy import BacktestResult, BacktestStatus

    bt = session.exec(
        select(BacktestResult).where(BacktestResult.account_id == account.id)
    ).first()
    if bt is None:
        return base

    active = bt.status in {BacktestStatus.PENDING.value, BacktestStatus.RUNNING.value}
    if active:
        try:
            with httpx.Client(timeout=3.0) as client:
                resp = client.get(f"{_BACKTEST_SERVICE_URL}/backtests/{bt.id}/status")
                resp.raise_for_status()
                pos = (resp.json() or {}).get("position") or {}
            cash = pos.get("cash")
            equity = pos.get("equity")
            data = base.model_dump()
            if cash is not None:
                data["cash_balance"] = float(cash)
                data["available_funds"] = float(cash)
            if equity is not None:
                data["equity"] = float(equity)
                data["buying_power"] = float(equity)
            data["snapshot_at"] = datetime.now(timezone.utc)
            return AccountRead.model_validate(data)
        except Exception:
            logger.warning(
                "Simulated account %s: live status fetch failed, serving stored row",
                account.id,
            )
            return base

    # Finished backtest: the persisted equity_final is authoritative.
    if bt.equity_final is not None:
        data = base.model_dump()
        data["equity"] = float(bt.equity_final)
        return AccountRead.model_validate(data)
    return base


# ── Simulated backtest account: orders/positions proxy-on-read ───────────────
#
# A backtest runs against a simulated account (BT-{id}); its orders/positions
# are never written to the DB projections — the backtest service is the single
# source. So for simulated accounts we proxy /orders and /positions on-read to
# the coordinator, exactly like `_resolve_simulated_account_read` does for the
# balance snapshot. This lets the agent (and FE) query /accounts/{id}/orders in
# backtest with the same token and endpoint they use for live. Any coordinator
# hiccup (or a finished backtest whose in-memory run is gone) degrades to an
# empty list, never a 5xx — the same behaviour the runner's /orders had.

def _backtest_for_simulated_account(session: Session, account) -> Any | None:
    """Return the BacktestResult backing a simulated account, else None."""
    if account.account_type != SIMULATED_ACCOUNT_TYPE:
        return None
    from app.models.strategy import BacktestResult

    return session.exec(
        select(BacktestResult).where(BacktestResult.account_id == account.id)
    ).first()


def _fetch_backtest_json(path: str) -> dict | None:
    """GET a coordinator endpoint with a short timeout; None on any failure."""
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{_BACKTEST_SERVICE_URL}{path}")
            resp.raise_for_status()
            return resp.json() or {}
    except Exception:
        logger.warning("Backtest coordinator fetch failed: %s", path)
        return None


def _ms_to_dt(ms: Any) -> datetime | None:
    try:
        return datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc)
    except Exception:
        return None


def _synthetic_int_id(ref: str) -> int:
    """Stable int id from a coordinator ref (`bt-80-3` -> 3, else a hash)."""
    tail = ref.rsplit("-", 1)[-1] if ref else ""
    try:
        return int(tail)
    except (TypeError, ValueError):
        return abs(hash(ref)) % 1_000_000_000


def _coordinator_order_to_read(order: dict, account_id: int) -> LiveOrderRead:
    ref = str(order.get("id") or order.get("broker_order_id") or "")
    ts = _ms_to_dt((order.get("extra") or {}).get("bar_ts")) or datetime.now(timezone.utc)
    status = str(order.get("status") or "filled")
    return LiveOrderRead(
        id=_synthetic_int_id(ref),
        strategy_live_id=None,
        account_id=account_id,
        broker_order_id=ref or None,
        symbol=str(order.get("symbol") or ""),
        side=str(order.get("side") or ""),
        order_type=str(order.get("order_type") or "market"),
        quantity=float(order.get("quantity") or 0.0),
        limit_price=order.get("limit_price"),
        stop_price=order.get("stop_price"),
        filled_quantity=float(order.get("filled_quantity") or 0.0),
        avg_fill_price=order.get("avg_fill_price"),
        commission=order.get("commission"),
        status=status,
        status_message=order.get("status_message"),
        created_at=ts,
        submitted_at=ts,
        filled_at=ts if status == "filled" else None,
        cancelled_at=None,
        updated_at=ts,
        extra=order.get("extra"),
        fills=[],
    )


def _coordinator_position_to_read(pos: dict, account_id: int) -> LivePositionRead:
    ref = str(pos.get("position_id") or "")
    opened = _ms_to_dt(pos.get("opened_ts")) or datetime.now(timezone.utc)
    unrealized = pos.get("unrealized_pnl")
    return LivePositionRead(
        id=_synthetic_int_id(ref),
        strategy_live_id=None,
        account_id=account_id,
        position_key=ref or None,
        symbol=str(pos.get("symbol") or ""),
        position_bucket=pos.get("position_bucket"),
        side=str(pos.get("side") or "flat"),
        quantity=float(pos.get("quantity") or 0.0),
        avg_price=pos.get("avg_price"),
        unrealized_pnl=unrealized,
        status="open",
        opened_at=opened,
        updated_at=datetime.now(timezone.utc),
        extra=pos.get("extra"),
        last_price=pos.get("mark_price"),
        computed_unrealized_pnl=unrealized,
    )


class AccountOrdersResetRequest(BaseModel):
    lookback_days: int | None = Field(default=None, ge=1, le=90)
    lookback_hours: int | None = Field(default=None, ge=1, le=24 * 90, deprecated=True)


class AccountOrdersResetResponse(BaseModel):
    success: bool
    account_id: int
    connection_id: int
    deleted_count: int
    deleted_fill_count: int = 0
    orders_since: datetime
    published_count: int = 0
    latest_event_at: datetime | None = None
    fills_since: datetime | None = None
    fills_published_count: int = 0
    latest_fill_event_at: datetime | None = None
    positions_count: int = 0
    position_snapshots_published: int = 0
    message: str | None = None


@router.get("/", response_model=AccountListResponse)
def list_accounts_endpoint(
    connection_id: int | None = Query(default=None),
    active_only: bool = False,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    if connection_id is not None:
        accounts = list_accounts(
            session,
            connection_id,
            user_id=current_user.id,
            active_only=active_only,
        )
    else:
        accounts = list_all_accounts(
            session,
            current_user.id,
            active_only=active_only,
        )
    return AccountListResponse(
        accounts=[AccountRead.model_validate(account) for account in accounts],
        count=len(accounts),
    )


@router.get("/{account_id}", response_model=AccountRead)
def get_account_endpoint(
    account_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    account = get_account(session, account_id, current_user.id)
    if account is None:
        raise HTTPException(status_code=404, detail="Account not found")
    read = AccountRead.model_validate(account)
    if account.account_type == SIMULATED_ACCOUNT_TYPE:
        read = _resolve_simulated_account_read(session, account, read)
    return read


@router.get("/{account_id}/orders", response_model=list[LiveOrderRead])
def list_account_orders_endpoint(
    account_id: int,
    status: str | None = Query(default=None),
    active_only: bool = False,
    symbol: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    account = get_account(session, account_id, current_user.id)
    if account is None:
        raise HTTPException(status_code=404, detail="Account not found")

    bt = _backtest_for_simulated_account(session, account)
    if bt is not None:
        payload = _fetch_backtest_json(f"/backtests/{bt.id}/orders")
        raw = (payload or {}).get("orders") or []
        orders = [_coordinator_order_to_read(o, account.id) for o in raw]
        if symbol:
            wanted = symbol.upper()
            orders = [o for o in orders if o.symbol.upper() == wanted]
        if active_only:
            active_statuses = {"pending", "submitted", "partially_filled"}
            orders = [o for o in orders if o.status in active_statuses]
        elif status:
            orders = [o for o in orders if o.status == status]
        orders.sort(key=lambda o: o.created_at, reverse=True)
        return orders[:limit]

    orders = list_account_orders(
        session,
        account.id,
        status=status,
        active_only=active_only,
        symbol=symbol,
        limit=limit,
    )
    return orders


@router.post("/{account_id}/orders/reset", response_model=AccountOrdersResetResponse)
async def reset_account_orders_endpoint(
    account_id: int,
    payload: AccountOrdersResetRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    account = get_account(session, account_id, current_user.id)
    if account is None:
        raise HTTPException(status_code=404, detail="Account not found")

    connection = get_connection(session, account.connection_id, current_user.id)
    if connection is None:
        raise HTTPException(status_code=404, detail="Connection not found")

    manager = get_connection_manager()
    status_value = await manager.check_connection_status(connection.id)
    if status_value != "connected":
        raise HTTPException(
            status_code=409,
            detail="Connection must be connected to reset account orders",
        )

    lookback_days = payload.lookback_days
    if lookback_days is None and payload.lookback_hours is not None:
        lookback_days = max((payload.lookback_hours + 23) // 24, 1)
    if lookback_days is None:
        lookback_days = resolve_order_history_lookback_days(connection.config)

    orders_since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    fills_since = orders_since
    deleted_fill_count = purge_account_fills(session, account.id)
    deleted_count = purge_account_orders(session, account.id)
    purge_account_trades(session, account.id)

    try:
        client = manager.get_gateway_client(connection.id, connection.broker_type)
        order_result = await client.reread_orders(
            since=orders_since.isoformat(),
            account=account.account_id,
            persist_checkpoint=False,
        )
        fill_result = await client.reread_fills(
            since=fills_since.isoformat(),
            account=account.account_id,
            persist_checkpoint=False,
        )
        position_result = await client.reread_positions(
            account=account.account_id,
            force_publish=True,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to reload account data from broker: {exc}",
        ) from exc

    latest_event_at_raw = order_result.get("latest_event_at")
    latest_event_at = None
    if isinstance(latest_event_at_raw, str):
        latest_event_at = datetime.fromisoformat(latest_event_at_raw)

    latest_fill_event_at_raw = fill_result.get("latest_event_at")
    latest_fill_event_at = None
    if isinstance(latest_fill_event_at_raw, str):
        latest_fill_event_at = datetime.fromisoformat(latest_fill_event_at_raw)

    return AccountOrdersResetResponse(
        success=bool(order_result.get("success", True)) and bool(fill_result.get("success", True)),
        account_id=account.id,
        connection_id=connection.id,
        deleted_count=deleted_count,
        deleted_fill_count=deleted_fill_count,
        orders_since=orders_since,
        published_count=int(order_result.get("published_count") or 0),
        latest_event_at=latest_event_at,
        fills_since=fills_since,
        fills_published_count=int(fill_result.get("published_count") or 0),
        latest_fill_event_at=latest_fill_event_at,
        positions_count=int(position_result.get("positions_count") or 0),
        position_snapshots_published=int(position_result.get("snapshots_published") or 0),
        message=(
            f"Deleted {deleted_count} orders and {deleted_fill_count} fills, then triggered broker order/fill/position reread for account {account.account_id} from the last {lookback_days}d"
        ),
    )


@router.get("/{account_id}/fills", response_model=list[LiveFillRead])
def list_account_fills_endpoint(
    account_id: int,
    symbol: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    account = get_account(session, account_id, current_user.id)
    if account is None:
        raise HTTPException(status_code=404, detail="Account not found")

    fills = list_account_fills(
        session,
        account.id,
        symbol=symbol,
        limit=limit,
    )
    return [LiveFillRead.model_validate(fill) for fill in fills]


@router.get("/{account_id}/trades", response_model=list[LiveTradeRead])
def list_account_trades_endpoint(
    account_id: int,
    symbol: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    account = get_account(session, account_id, current_user.id)
    if account is None:
        raise HTTPException(status_code=404, detail="Account not found")

    trades = list_account_trades(
        session,
        account.id,
        symbol=symbol,
        limit=limit,
    )
    return [LiveTradeRead.model_validate(trade) for trade in trades]


@router.get("/{account_id}/positions", response_model=list[LivePositionRead])
def list_account_positions_endpoint(
    account_id: int,
    symbol: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    account = get_account(session, account_id, current_user.id)
    if account is None:
        raise HTTPException(status_code=404, detail="Account not found")

    bt = _backtest_for_simulated_account(session, account)
    if bt is not None:
        payload = _fetch_backtest_json(f"/backtests/{bt.id}/positions")
        raw = (payload or {}).get("positions") or []
        positions = [_coordinator_position_to_read(p, account.id) for p in raw]
        if symbol:
            wanted = symbol.upper()
            positions = [p for p in positions if p.symbol.upper() == wanted]
        return positions[:limit]

    positions = list_account_positions(
        session,
        account.id,
        symbol=symbol,
        limit=limit,
    )
    return [LivePositionRead.model_validate(position) for position in positions]


class AccountClosePositionRequest(BaseModel):
    """Close one position of an account by its broker position id."""
    quantity: float = Field(gt=0)
    symbol: str | None = None
    reason: str | None = None
    extra: dict[str, Any] | None = None


@router.post("/{account_id}/positions/{position_id}/close")
async def close_account_position_endpoint(
    account_id: int,
    position_id: str,
    payload: AccountClosePositionRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Close a position — the single write path for frontend and agent.

    Routed to whoever is authoritative: the backtest ledger for a simulated
    account, the live runner while one is running (so its order lock serializes
    the close against the rule engine), otherwise the broker gateway. Callers
    must never submit an offsetting order instead: on hedging / ticket-based
    brokers that opens a new contrary position rather than closing.
    """
    account = get_account(session, account_id, current_user.id)
    if account is None:
        raise HTTPException(status_code=404, detail="Account not found")

    from app.services.position_command_service import close_account_position

    try:
        return await close_account_position(
            session,
            account,
            position_id,
            quantity=payload.quantity,
            symbol=payload.symbol,
            reason=payload.reason,
            extra=payload.extra,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
