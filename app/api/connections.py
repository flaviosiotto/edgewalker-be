"""
Connections & Accounts API Endpoints.

CRUD for broker connections.  Accounts are auto-discovered when a
connection is established — there is no manual account creation.

Connection-scoped endpoints include symbol search via the connection's
gateway container.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import logging
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlmodel import Session

from app.core.config import settings
from app.db.database import get_session
from app.models.user import User
from app.schemas.connection import (
    AccountListResponse,
    AccountRead,
    ConnectionCreate,
    ConnectionListResponse,
    ConnectionRead,
    ConnectionUpdate,
    CTraderAccountOption,
    CTraderAccountsRequest,
    CTraderAccountsResponse,
    CTraderOAuthConfigResponse,
    CTraderOAuthTokenRequest,
    CTraderOAuthTokenResponse,
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
from app.services.client_portal_service import is_client_portal_transport, is_tws_interactive_transport
from app.services.connection_manager import get_connection_manager
from app.services.connection_manager import resolve_order_history_lookback_days
from app.services.ctrader_accounts import CTraderAccountsError, fetch_ctrader_accounts
from app.utils.auth_utils import get_current_active_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/connections", tags=["Connections & Accounts"])

_CTRADER_TOKEN_URL = "https://openapi.ctrader.com/apps/token"


# ─── CONNECTIONS ───


@router.get("/ctrader/oauth/config", response_model=CTraderOAuthConfigResponse)
async def get_ctrader_oauth_config(
    current_user: User = Depends(get_current_active_user),
):
    """Return the public cTrader OAuth app configuration."""
    _ = current_user
    client_id = settings.CTRADER_OAUTH_CLIENT_ID.strip()
    client_secret = settings.CTRADER_OAUTH_CLIENT_SECRET.strip()
    return CTraderOAuthConfigResponse(
        configured=bool(client_id and client_secret),
        client_id=client_id or None,
    )


@router.post("/ctrader/oauth/token", response_model=CTraderOAuthTokenResponse)
async def exchange_ctrader_oauth_token(
    payload: CTraderOAuthTokenRequest,
    current_user: User = Depends(get_current_active_user),
):
    """Exchange a cTrader Open API authorisation code for access tokens."""
    _ = current_user
    client_id = settings.CTRADER_OAUTH_CLIENT_ID.strip()
    client_secret = settings.CTRADER_OAUTH_CLIENT_SECRET.strip()
    if not client_id or not client_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Applicazione OAuth cTrader non configurata sul backend",
        )

    params = {
        "grant_type": "authorization_code",
        "code": payload.code.strip(),
        "redirect_uri": payload.redirect_uri.strip(),
        "client_id": client_id,
        "client_secret": client_secret,
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(
                _CTRADER_TOKEN_URL,
                params=params,
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="cTrader ha rifiutato lo scambio del codice OAuth",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Impossibile contattare cTrader per lo scambio OAuth",
        ) from exc

    data = response.json()
    error_code = data.get("errorCode")
    if error_code:
        detail = data.get("description") or error_code
        if str(error_code).upper() == "ACCESS_DENIED":
            detail = (
                "cTrader ha rifiutato lo scambio OAuth. Genera un nuovo authorization code, "
                "verifica che il redirect URI corrisponda a quello registrato nell'app cTrader, "
                "e usa Scambia una sola volta prima che il codice scada."
            )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(detail))

    access_token = data.get("accessToken")
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Risposta cTrader senza access token",
        )

    return CTraderOAuthTokenResponse(
        access_token=str(access_token),
        refresh_token=data.get("refreshToken"),
        token_type=data.get("tokenType"),
        expires_in=data.get("expiresIn"),
    )


@router.post("/ctrader/accounts", response_model=CTraderAccountsResponse)
async def list_ctrader_accounts(
    payload: CTraderAccountsRequest,
    current_user: User = Depends(get_current_active_user),
):
    """List the trading accounts a cTrader access token can see.

    Powers the account pick-list in the connection form: the gateway's
    "Account ID" is a ``ctidTraderAccountId`` that appears nowhere in the broker
    UI, so we fetch it here (ApplicationAuth + account list) from the token the
    user just obtained. A single short-lived Open API round-trip — no gateway
    container is spawned.
    """
    _ = current_user
    client_id = settings.CTRADER_OAUTH_CLIENT_ID.strip()
    client_secret = settings.CTRADER_OAUTH_CLIENT_SECRET.strip()
    if not client_id or not client_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Applicazione OAuth cTrader non configurata sul backend",
        )

    try:
        accounts = await fetch_ctrader_accounts(
            access_token=payload.access_token.strip(),
            environment=payload.environment,
            client_id=client_id,
            client_secret=client_secret,
        )
    except CTraderAccountsError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"cTrader ha rifiutato la richiesta account: {exc}",
        ) from exc
    except (asyncio.TimeoutError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Impossibile contattare cTrader per la lista account",
        ) from exc

    options = [CTraderAccountOption(**account) for account in accounts]
    return CTraderAccountsResponse(accounts=options, count=len(options))


@router.get("/", response_model=ConnectionListResponse)
async def list_connections_endpoint(
    active_only: bool = False,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """List all broker connections.

    Reads connection status straight from the DB. Health is kept fresh by the
    connection manager's background loop (``_run_connection_health_loop``), so
    this endpoint stays O(1) in gateway round-trips. The returned
    ``last_checked_at``/``is_stale`` fields let callers detect a status that
    has gone stale (e.g. a gateway reporting "connected" from a cached flag).
    """
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

    If the connection is marked as *connected* or *degraded* it is probed against
    the broker to verify its real status before returning.
    """
    conn = get_connection(session, connection_id, current_user.id)
    if conn is None:
        raise HTTPException(status_code=404, detail="Connection not found")

    if conn.status in {"connected", "degraded"}:
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
    launch_url: str | None = None


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
    launch_url: str | None = None
    message: str | None = None


class ClientPortalFlowSignalResponse(BaseModel):
    success: bool
    message: str | None = None


class OrdersRereadRequest(BaseModel):
    lookback_days: int | None = Field(default=None, ge=1, le=90)
    lookback_hours: int | None = Field(default=None, ge=1, le=24 * 90, deprecated=True)


class OrdersRereadResponse(BaseModel):
    success: bool
    connection_id: int
    orders_since: datetime
    published_count: int = 0
    latest_event_at: datetime | None = None
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
        auth = await manager.begin_client_portal_auth(
            connection_id,
            user_id=current_user.id,
        )
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
            launch_url=auth.get("launch_url"),
        )

    if conn.broker_type == "ibkr" and is_tws_interactive_transport(conn.config or {}):
        auth = await manager.begin_tws_auth(
            connection_id,
            user_id=current_user.id,
        )
        if auth["ready_to_connect"]:
            result = await manager.complete_tws_connect(connection_id)
            return ConnectDisconnectResponse(
                success=result.success,
                message=result.message,
                accounts_discovered=len(result.accounts) if result.accounts else 0,
            )

        return ConnectDisconnectResponse(
            success=auth["service_ready"],
            message=auth["message"],
            auth_required=auth["service_ready"],
            launch_url=auth.get("launch_url"),
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
    payload = await manager.client_portal_auth_status(connection_id, user_id=current_user.id)
    return ClientPortalAuthStatusResponse(**payload)


@router.post("/{connection_id}/client-portal/dispatcher-received", response_model=ClientPortalFlowSignalResponse)
async def client_portal_dispatcher_received_endpoint(
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
    payload = await manager.mark_client_portal_dispatcher_received(connection_id)
    return ClientPortalFlowSignalResponse(**payload)


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


@router.get("/{connection_id}/tws/auth-status", response_model=ClientPortalAuthStatusResponse)
async def tws_auth_status_endpoint(
    connection_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    conn = get_connection(session, connection_id, current_user.id)
    if conn is None:
        raise HTTPException(status_code=404, detail="Connection not found")
    if conn.broker_type != "ibkr" or not is_tws_interactive_transport(conn.config or {}):
        raise HTTPException(status_code=400, detail="Connection is not configured for IBKR TWS interactive mode")

    manager = get_connection_manager()
    payload = await manager.tws_auth_status(connection_id, user_id=current_user.id)
    return ClientPortalAuthStatusResponse(**payload)


@router.post("/{connection_id}/tws/connect", response_model=ConnectDisconnectResponse)
async def complete_tws_connect_endpoint(
    connection_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    conn = get_connection(session, connection_id, current_user.id)
    if conn is None:
        raise HTTPException(status_code=404, detail="Connection not found")
    if conn.broker_type != "ibkr" or not is_tws_interactive_transport(conn.config or {}):
        raise HTTPException(status_code=400, detail="Connection is not configured for IBKR TWS interactive mode")

    manager = get_connection_manager()
    result = await manager.complete_tws_connect(connection_id)
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


@router.post("/{connection_id}/orders/reread", response_model=OrdersRereadResponse)
async def reread_orders_endpoint(
    connection_id: int,
    payload: OrdersRereadRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    conn = get_connection(session, connection_id, current_user.id)
    if conn is None:
        raise HTTPException(status_code=404, detail="Connection not found")

    manager = get_connection_manager()
    status_value = await manager.check_connection_status(connection_id)
    if status_value != "connected":
        raise HTTPException(
            status_code=409,
            detail="Connection must be connected to reread orders",
        )

    lookback_days = payload.lookback_days
    if lookback_days is None and payload.lookback_hours is not None:
        lookback_days = max((payload.lookback_hours + 23) // 24, 1)
    if lookback_days is None:
        lookback_days = resolve_order_history_lookback_days(conn.config)

    orders_since = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    try:
        client = manager.get_gateway_client(connection_id, conn.broker_type)
        result = await client.reread_orders(since=orders_since.isoformat())
    except Exception as exc:
        logger.warning(
            "Failed to trigger order reread for connection %s: %s",
            connection_id,
            exc,
        )
        raise HTTPException(
            status_code=502,
            detail=f"Failed to trigger order reread: {exc}",
        ) from exc

    latest_event_at_raw = result.get("latest_event_at")
    latest_event_at = None
    if isinstance(latest_event_at_raw, str):
        latest_event_at = datetime.fromisoformat(latest_event_at_raw)

    return OrdersRereadResponse(
        success=bool(result.get("success", True)),
        connection_id=connection_id,
        orders_since=orders_since,
        published_count=int(result.get("published_count") or 0),
        latest_event_at=latest_event_at,
        message=(
            f"Triggered order reread from the last {lookback_days}d"
            if result.get("success", True)
            else "Gateway did not accept the order reread request"
        ),
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

