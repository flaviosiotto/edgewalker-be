"""Canonical consultative account APIs.

These endpoints expose account-scoped broker state for both the frontend
and delegated agent/n8n consumers without forcing them through runner
APIs or connection-management routes.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

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
)
from app.services.connection_service import get_account, list_accounts, list_all_accounts
from app.services.connection_service import get_connection
from app.services.connection_manager import get_connection_manager, resolve_order_history_lookback_hours
from app.services.live_trading_service import (
    list_account_fills,
    list_account_orders,
    list_account_positions,
    purge_account_fills,
    purge_account_orders,
)
from app.utils.auth_utils import get_current_active_or_consultative_user

router = APIRouter(prefix="/accounts", tags=["Accounts"])


class AccountOrdersResetRequest(BaseModel):
    lookback_hours: int | None = Field(default=None, ge=1, le=24 * 90)


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

    lookback_hours = payload.lookback_hours or resolve_order_history_lookback_hours(connection.config)
    orders_since = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    fills_since = orders_since
    deleted_fill_count = purge_account_fills(session, account.id)
    deleted_count = purge_account_orders(session, account.id)

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
        message=(
            f"Deleted {deleted_count} orders and {deleted_fill_count} fills, then triggered broker reread for account {account.account_id}"
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

    positions = list_account_positions(
        session,
        account.id,
        symbol=symbol,
        limit=limit,
    )
    return [LivePositionRead.model_validate(position) for position in positions]
