"""Canonical consultative account APIs.

These endpoints expose account-scoped broker state for both the frontend
and delegated agent/n8n consumers without forcing them through runner
APIs or connection-management routes.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session

from app.db.database import get_session
from app.models.user import User
from app.schemas.connection import AccountListResponse, AccountRead
from app.schemas.live_trading import LiveOrderRead
from app.services.connection_service import get_account, list_accounts, list_all_accounts
from app.services.live_trading_service import list_account_orders
from app.utils.auth_utils import get_current_active_or_consultative_user

router = APIRouter(prefix="/accounts", tags=["Accounts"])


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
        limit=limit,
    )
    return [LiveOrderRead.model_validate(order) for order in orders]
