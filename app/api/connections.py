"""
Connections & Accounts API Endpoints.

CRUD for broker connections.  Accounts are auto-discovered when a
connection is established — there is no manual account creation.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlmodel import Session

from app.db.database import get_session
from app.schemas.connection import (
    AccountListResponse,
    AccountRead,
    ConnectionCreate,
    ConnectionListResponse,
    ConnectionRead,
    ConnectionUpdate,
)
from app.services.connection_service import (
    create_connection,
    delete_connection,
    get_account,
    get_connection,
    list_accounts,
    list_all_accounts,
    list_connections,
    update_connection,
)
from app.services.connection_manager import get_connection_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/connections", tags=["Connections & Accounts"])


# ─── CONNECTIONS ───


@router.get("/", response_model=ConnectionListResponse)
def list_connections_endpoint(
    active_only: bool = False,
    session: Session = Depends(get_session),
):
    """List all broker connections."""
    conns = list_connections(session, active_only=active_only)
    return ConnectionListResponse(
        connections=[ConnectionRead.model_validate(c) for c in conns],
        count=len(conns),
    )


@router.get("/{connection_id}", response_model=ConnectionRead)
def get_connection_endpoint(
    connection_id: int,
    session: Session = Depends(get_session),
):
    """Get a single connection with its accounts."""
    conn = get_connection(session, connection_id)
    if conn is None:
        raise HTTPException(status_code=404, detail="Connection not found")
    return ConnectionRead.model_validate(conn)


@router.post("/", response_model=ConnectionRead, status_code=status.HTTP_201_CREATED)
def create_connection_endpoint(
    payload: ConnectionCreate,
    session: Session = Depends(get_session),
):
    """Create a new broker connection."""
    conn = create_connection(
        session,
        name=payload.name,
        broker_type=payload.broker_type,
        config=payload.config,
        is_active=payload.is_active,
    )
    return ConnectionRead.model_validate(conn)


@router.patch("/{connection_id}", response_model=ConnectionRead)
def update_connection_endpoint(
    connection_id: int,
    payload: ConnectionUpdate,
    session: Session = Depends(get_session),
):
    """Update a broker connection."""
    conn = update_connection(
        session,
        connection_id,
        **payload.model_dump(exclude_unset=True),
    )
    if conn is None:
        raise HTTPException(status_code=404, detail="Connection not found")
    return ConnectionRead.model_validate(conn)


@router.delete("/{connection_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_connection_endpoint(
    connection_id: int,
    session: Session = Depends(get_session),
):
    """Delete a broker connection (cascades to accounts)."""
    if not delete_connection(session, connection_id):
        raise HTTPException(status_code=404, detail="Connection not found")


# ─── CONNECT / DISCONNECT ───


class ConnectDisconnectResponse(BaseModel):
    """Response for connect / disconnect operations."""
    success: bool
    message: str | None = None
    accounts_discovered: int = 0


@router.post("/{connection_id}/connect", response_model=ConnectDisconnectResponse)
async def connect_endpoint(
    connection_id: int,
    session: Session = Depends(get_session),
):
    """Connect to the broker and auto-discover accounts."""
    conn = get_connection(session, connection_id)
    if conn is None:
        raise HTTPException(status_code=404, detail="Connection not found")

    manager = get_connection_manager()
    result = await manager.connect(connection_id)

    return ConnectDisconnectResponse(
        success=result.success,
        message=result.message,
        accounts_discovered=len(result.accounts) if result.accounts else 0,
    )


@router.post("/{connection_id}/disconnect", response_model=ConnectDisconnectResponse)
async def disconnect_endpoint(
    connection_id: int,
    session: Session = Depends(get_session),
):
    """Disconnect from the broker."""
    conn = get_connection(session, connection_id)
    if conn is None:
        raise HTTPException(status_code=404, detail="Connection not found")

    manager = get_connection_manager()
    result = await manager.disconnect(connection_id)

    return ConnectDisconnectResponse(
        success=result.success,
        message=result.message,
    )


# ─── ACCOUNTS (read-only) ───


@router.get("/{connection_id}/accounts", response_model=AccountListResponse)
def list_accounts_endpoint(
    connection_id: int,
    active_only: bool = False,
    session: Session = Depends(get_session),
):
    """List accounts for a connection."""
    conn = get_connection(session, connection_id)
    if conn is None:
        raise HTTPException(status_code=404, detail="Connection not found")
    accts = list_accounts(session, connection_id, active_only=active_only)
    return AccountListResponse(
        accounts=[AccountRead.model_validate(a) for a in accts],
        count=len(accts),
    )


@router.get("/accounts/all", response_model=AccountListResponse)
def list_all_accounts_endpoint(
    active_only: bool = False,
    session: Session = Depends(get_session),
):
    """List all accounts across all connections."""
    accts = list_all_accounts(session, active_only=active_only)
    return AccountListResponse(
        accounts=[AccountRead.model_validate(a) for a in accts],
        count=len(accts),
    )


@router.get("/accounts/{account_id}", response_model=AccountRead)
def get_account_endpoint(
    account_id: int,
    session: Session = Depends(get_session),
):
    """Get a single account."""
    account = get_account(session, account_id)
    if account is None:
        raise HTTPException(status_code=404, detail="Account not found")
    return AccountRead.model_validate(account)

