"""
Connections & Accounts API Endpoints.

CRUD for broker connections.  Accounts are auto-discovered when a
connection is established — there is no manual account creation.

Connection-scoped endpoints include symbol search via the connection's
gateway container.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlmodel import Session

from app.db.database import get_session
from app.models.user import User
from app.schemas.connection import (
    AccountListResponse,
    AccountRead,
    ConnectionCreate,
    ConnectionListResponse,
    ConnectionRead,
    ConnectionUpdate,
)
from app.schemas.marketdata import (
    AvailableSymbolsResponse,
    AssetType,
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
from app.services.client_portal_service import is_client_portal_transport
from app.services.connection_manager import get_connection_manager
from app.utils.auth_utils import get_current_active_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/connections", tags=["Connections & Accounts"])


# ─── CONNECTIONS ───


@router.get("/", response_model=ConnectionListResponse)
async def list_connections_endpoint(
    active_only: bool = False,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """List all broker connections.

    All active connections are probed in real-time against the broker
    so the returned status always reflects reality — including gateways
    that came back online after being marked disconnected.
    """
    manager = get_connection_manager()
    await manager.probe_active_connections()
    # Re-read after potential status updates
    session.expire_all()

    conns = list_connections(session, current_user.id, active_only=active_only)
    return ConnectionListResponse(
        connections=[ConnectionRead.model_validate(c) for c in conns],
        count=len(conns),
    )


@router.get("/{connection_id}", response_model=ConnectionRead)
async def get_connection_endpoint(
    connection_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Get a single connection with its accounts.

    If the connection is marked as *connected* it is probed against
    the broker to verify its real status before returning.
    """
    conn = get_connection(session, connection_id, current_user.id)
    if conn is None:
        raise HTTPException(status_code=404, detail="Connection not found")

    if conn.status == "connected":
        manager = get_connection_manager()
        await manager.check_connection_status(connection_id)
        session.refresh(conn)

    return ConnectionRead.model_validate(conn)


@router.post("/", response_model=ConnectionRead, status_code=status.HTTP_201_CREATED)
def create_connection_endpoint(
    payload: ConnectionCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Create a new broker connection."""
    conn = create_connection(
        session,
        user_id=current_user.id,
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
    current_user: User = Depends(get_current_active_user),
):
    """Update a broker connection."""
    conn = update_connection(
        session,
        connection_id,
        user_id=current_user.id,
        **payload.model_dump(exclude_unset=True),
    )
    if conn is None:
        raise HTTPException(status_code=404, detail="Connection not found")
    return ConnectionRead.model_validate(conn)


@router.delete("/{connection_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_connection_endpoint(
    connection_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Delete a broker connection (cascades to accounts)."""
    if not delete_connection(session, connection_id, current_user.id):
        raise HTTPException(status_code=404, detail="Connection not found")


# ─── CONNECT / DISCONNECT ───


class ConnectDisconnectResponse(BaseModel):
    """Response for connect / disconnect operations."""
    success: bool
    message: str | None = None
    accounts_discovered: int = 0
    auth_required: bool = False
    auth_url: str | None = None


class ClientPortalAuthStatusResponse(BaseModel):
    service_ready: bool
    gateway_session_ready: bool = False
    connected: bool = False
    session_authenticated: bool = False
    authenticated: bool
    established: bool = False
    competing: bool = False
    bridge_ready: bool = False
    ready_to_connect: bool = False
    gateway_started: bool
    connection_status: str
    auth_url: str | None = None
    message: str | None = None


@router.post("/{connection_id}/connect", response_model=ConnectDisconnectResponse)
async def connect_endpoint(
    connection_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Connect to the broker and auto-discover accounts."""
    conn = get_connection(session, connection_id, current_user.id)
    if conn is None:
        raise HTTPException(status_code=404, detail="Connection not found")

    manager = get_connection_manager()

    if conn.broker_type == "ibkr" and is_client_portal_transport(conn.config or {}):
        auth = await manager.begin_client_portal_auth(connection_id)
        if auth["ready_to_connect"]:
            result = await manager.complete_client_portal_connect(connection_id)
            return ConnectDisconnectResponse(
                success=result.success,
                message=result.message,
                accounts_discovered=len(result.accounts) if result.accounts else 0,
            )

        return ConnectDisconnectResponse(
            success=auth["service_ready"],
            message=auth["message"],
            auth_required=auth["service_ready"],
            auth_url=auth["auth_url"],
        )

    result = await manager.connect(connection_id)

    return ConnectDisconnectResponse(
        success=result.success,
        message=result.message,
        accounts_discovered=len(result.accounts) if result.accounts else 0,
    )


@router.get("/{connection_id}/client-portal/auth-status", response_model=ClientPortalAuthStatusResponse)
async def client_portal_auth_status_endpoint(
    connection_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    conn = get_connection(session, connection_id, current_user.id)
    if conn is None:
        raise HTTPException(status_code=404, detail="Connection not found")
    if conn.broker_type != "ibkr" or not is_client_portal_transport(conn.config or {}):
        raise HTTPException(status_code=400, detail="Connection is not configured for IBKR Client Portal")

    manager = get_connection_manager()
    payload = await manager.client_portal_auth_status(connection_id)
    return ClientPortalAuthStatusResponse(**payload)


@router.post("/{connection_id}/client-portal/connect", response_model=ConnectDisconnectResponse)
async def complete_client_portal_connect_endpoint(
    connection_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    conn = get_connection(session, connection_id, current_user.id)
    if conn is None:
        raise HTTPException(status_code=404, detail="Connection not found")
    if conn.broker_type != "ibkr" or not is_client_portal_transport(conn.config or {}):
        raise HTTPException(status_code=400, detail="Connection is not configured for IBKR Client Portal")

    manager = get_connection_manager()
    result = await manager.complete_client_portal_connect(connection_id)
    return ConnectDisconnectResponse(
        success=result.success,
        message=result.message,
        accounts_discovered=len(result.accounts) if result.accounts else 0,
    )


@router.post("/{connection_id}/disconnect", response_model=ConnectDisconnectResponse)
async def disconnect_endpoint(
    connection_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Disconnect from the broker."""
    conn = get_connection(session, connection_id, current_user.id)
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
    current_user: User = Depends(get_current_active_user),
):
    """List accounts for a connection."""
    conn = get_connection(session, connection_id, current_user.id)
    if conn is None:
        raise HTTPException(status_code=404, detail="Connection not found")
    accts = list_accounts(session, connection_id, user_id=current_user.id, active_only=active_only)
    return AccountListResponse(
        accounts=[AccountRead.model_validate(a) for a in accts],
        count=len(accts),
    )


@router.get("/accounts/all", response_model=AccountListResponse)
def list_all_accounts_endpoint(
    active_only: bool = False,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """List all accounts across all connections."""
    accts = list_all_accounts(session, current_user.id, active_only=active_only)
    return AccountListResponse(
        accounts=[AccountRead.model_validate(a) for a in accts],
        count=len(accts),
    )


@router.get("/accounts/{account_id}", response_model=AccountRead)
def get_account_endpoint(
    account_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Get a single account."""
    account = get_account(session, account_id, current_user.id)
    if account is None:
        raise HTTPException(status_code=404, detail="Account not found")
    return AccountRead.model_validate(account)


# ─── CONNECTION-SCOPED MARKET DATA ───


@router.get("/{connection_id}/symbols", response_model=AvailableSymbolsResponse)
def search_connection_symbols(
    connection_id: int,
    query: str = Query(
        ...,
        description="Search query — symbol pattern or company name (e.g., 'QQQ', 'Apple', 'NQ')",
    ),
    asset_type: Optional[AssetType] = Query(
        None,
        description="Filter by asset type: stock, futures, index, etf",
    ),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of results"),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Search symbols via a connection's gateway.

    Requires the connection to have a running gateway container
    (broker types with gateway support: ibkr, binance, etc.).
    """
    conn = get_connection(session, connection_id, current_user.id)
    if conn is None:
        raise HTTPException(status_code=404, detail="Connection not found")

    # Yahoo connections have no live gateway — use cached symbols only
    if conn.broker_type == "yahoo":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Connection {connection_id} is of type 'yahoo' which "
                   f"does not support live symbol search. Use the cached symbols endpoint.",
        )

    from app.services.symbol_sync_handler import search_gateway_symbols_by_id

    try:
        results = search_gateway_symbols_by_id(
            query=query,
            connection_id=connection_id,
            broker_type=conn.broker_type,
            asset_type=asset_type.value if asset_type else None,
            limit=limit,
        )
        return AvailableSymbolsResponse(
            symbols=results,
            source=conn.broker_type,
            asset_type=asset_type.value if asset_type else "all",
            count=len(results),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(e),
        )
    except Exception as e:
        logger.exception(f"Error searching symbols for connection {connection_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search symbols: {str(e)}",
        )

