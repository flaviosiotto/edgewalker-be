"""Canonical consultative account APIs.

These endpoints expose account-scoped broker state for both the frontend
and delegated agent/n8n consumers without forcing them through runner
APIs or connection-management routes.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlmodel import Session

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

router = APIRouter(prefix="/accounts", tags=["Accounts"])


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
    return AccountRead.model_validate(account)


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

    orders = list_account_orders(
        session,
        account.id,
        status=status,
        active_only=active_only,
        symbol=symbol,
        limit=limit,
    )
    return orders


class AccountPlaceOrderRequest(BaseModel):
    symbol: str
    side: str
    order_type: str = "market"
    quantity: float = Field(gt=0)
    limit_price: float | None = Field(default=None, gt=0)
    stop_price: float | None = Field(default=None, gt=0)
    take_profit_price: float | None = Field(default=None, gt=0)
    stop_loss_price: float | None = Field(default=None, gt=0)
    # Attribution: with it the order carries the strategy's reference prefix,
    # exactly as one placed by the runner would.
    strategy_live_id: int | None = None
    extra: dict[str, Any] | None = None


@router.post("/{account_id}/orders")
async def place_account_order_endpoint(
    account_id: int,
    payload: AccountPlaceOrderRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Submit an order — the single write path for frontend and agent."""
    account = get_account(session, account_id, current_user.id)
    if account is None:
        raise HTTPException(status_code=404, detail="Account not found")

    from app.services.order_command_service import place_account_order

    try:
        return await place_account_order(
            session,
            account,
            symbol=payload.symbol,
            side=payload.side,
            order_type=payload.order_type,
            quantity=payload.quantity,
            limit_price=payload.limit_price,
            stop_price=payload.stop_price,
            take_profit_price=payload.take_profit_price,
            stop_loss_price=payload.stop_loss_price,
            strategy_live_id=payload.strategy_live_id,
            extra=payload.extra,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.delete("/{account_id}/orders/{order_id}")
async def cancel_account_order_endpoint(
    account_id: int,
    order_id: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Cancel a working order. `order_id` is the row id or the broker order id."""
    account = get_account(session, account_id, current_user.id)
    if account is None:
        raise HTTPException(status_code=404, detail="Account not found")

    from app.services.order_command_service import cancel_account_order

    try:
        return await cancel_account_order(session, account, order_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


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

    from app.services.position_command_service import broker_position_id

    positions = list_account_positions(
        session,
        account.id,
        symbol=symbol,
        limit=limit,
    )
    reads: list[LivePositionRead] = []
    for position in positions:
        read = LivePositionRead.model_validate(position)
        # Surface the close handle instead of making every caller dig it out of extra.
        read.broker_position_id = broker_position_id(position)
        reads.append(read)
    return reads


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
    account, the broker gateway otherwise. `position_id` is the position's
    `broker_position_id` (the row id is also accepted and translated). Callers
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


class AccountAmendPositionRequest(BaseModel):
    """Patch an open position's protection. Omitted legs keep their value."""
    take_profit_price: float | None = Field(default=None, gt=0)
    stop_loss_price: float | None = Field(default=None, gt=0)
    clear_take_profit: bool = False
    clear_stop_loss: bool = False


@router.patch("/{account_id}/positions/{position_id}")
async def amend_account_position_endpoint(
    account_id: int,
    position_id: str,
    payload: AccountAmendPositionRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Change an open position's take profit / stop loss.

    Patch semantics: a leg left out keeps whatever the broker currently holds,
    so moving the stop never silently drops the target. Use `clear_take_profit`
    / `clear_stop_loss` to remove one on purpose.
    """
    account = get_account(session, account_id, current_user.id)
    if account is None:
        raise HTTPException(status_code=404, detail="Account not found")

    from app.services.position_command_service import (
        ProtectionUnsupportedError,
        amend_account_position_protection,
    )

    try:
        return await amend_account_position_protection(
            session,
            account,
            position_id,
            take_profit_price=payload.take_profit_price,
            stop_loss_price=payload.stop_loss_price,
            clear_take_profit=payload.clear_take_profit,
            clear_stop_loss=payload.clear_stop_loss,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ProtectionUnsupportedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
