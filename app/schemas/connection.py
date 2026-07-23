"""
Schemas for Connections & Accounts API.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field, computed_field


# A "connected"/"degraded" status older than this is considered stale: the
# background health loop should refresh well within it (default loop ~20s).
CONNECTION_STALE_TTL_SECONDS = max(
    15.0,
    float(os.getenv("CONNECTION_STALE_TTL_SECONDS", "60")),
)


# =============================================================================
# Connection Schemas
# =============================================================================

class ConnectionConfig(BaseModel):
    """Connection configuration (broker-specific)."""
    # IBKR
    transport: str | None = None
    host: str | None = None
    port: int | None = None
    client_id: int | str | None = None
    client_portal_enabled: bool | None = None
    order_history_lookback_days: int | None = None
    order_history_lookback_hours: int | None = None
    # Generic
    api_key: str | None = None
    api_secret: str | None = None
    # cTrader Open API
    access_token: str | None = None
    refresh_token: str | None = None
    account_id: str | None = None
    environment: str | None = None
    volume_scale: float | None = None

    class Config:
        extra = "allow"


class ConnectionCreate(BaseModel):
    """Schema for creating a new connection."""
    name: str = Field(..., min_length=1, max_length=100)
    broker_type: str = Field(..., description="Broker type: 'ibkr', 'binance', etc.")
    config: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True


class ConnectionUpdate(BaseModel):
    """Schema for updating a connection."""
    name: str | None = None
    config: dict[str, Any] | None = None
    is_active: bool | None = None


class AccountRead(BaseModel):
    """Response schema for an account."""
    id: int
    connection_id: int
    account_id: str
    display_name: str | None = None
    account_type: str | None = None
    currency: str
    cash_balance: float | None = None
    equity: float | None = None
    buying_power: float | None = None
    available_funds: float | None = None
    snapshot_at: datetime | None = None
    is_active: bool
    extra: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ConnectionRead(BaseModel):
    """Response schema for a connection."""
    id: int
    name: str
    broker_type: str
    config: dict[str, Any]
    is_active: bool
    status: str
    status_message: str | None = None
    last_connected_at: datetime | None = None
    last_checked_at: datetime | None = None
    last_ok_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    accounts: list[AccountRead] = []

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_stale(self) -> bool:
        """True when a connected/degraded status hasn't been re-probed recently.

        Lets the UI distinguish a freshly-confirmed "green" from one the gateway
        may be reporting from a cached flag after a silent broker drop.
        """
        if self.status not in {"connected", "degraded"}:
            return False
        if self.last_checked_at is None:
            return True
        checked = self.last_checked_at
        if checked.tzinfo is None:
            checked = checked.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - checked).total_seconds()
        return age > CONNECTION_STALE_TTL_SECONDS

    class Config:
        from_attributes = True


class ConnectionListResponse(BaseModel):
    """Response for listing connections."""
    connections: list[ConnectionRead]
    count: int


class CTraderOAuthConfigResponse(BaseModel):
    """Public cTrader OAuth app configuration for logged-in users."""
    configured: bool
    client_id: str | None = None


class CTraderOAuthTokenRequest(BaseModel):
    """Exchange a cTrader OAuth authorisation code for tokens."""
    code: str = Field(..., min_length=1)
    redirect_uri: str = Field(..., min_length=1)


class CTraderOAuthTokenResponse(BaseModel):
    """cTrader OAuth token response normalised for the frontend."""
    access_token: str
    refresh_token: str | None = None
    token_type: str | None = None
    expires_in: int | None = None


class CTraderAccountsRequest(BaseModel):
    """List the trading accounts a cTrader access token can see."""
    access_token: str = Field(..., min_length=1)
    environment: str = Field(default="demo", description="demo | live — only picks which host to dial")


class CTraderAccountOption(BaseModel):
    """One trading account visible to a cTrader access token."""
    ctid: int = Field(..., description="ctidTraderAccountId — the internal id used as Account ID")
    login: int | None = Field(default=None, description="Broker-facing login/account number")
    is_live: bool = Field(default=False, description="True when the account lives on the live host")


class CTraderAccountsResponse(BaseModel):
    """Accounts a cTrader access token can see, for the setup pick-list."""
    accounts: list[CTraderAccountOption]
    count: int


# =============================================================================
# Account Schemas
# =============================================================================

class AccountCreate(BaseModel):
    """Schema for creating an account under a connection."""
    account_id: str = Field(..., min_length=1, max_length=50, description="Broker account code")
    display_name: str | None = None
    account_type: str | None = None
    currency: str = "USD"
    cash_balance: float | None = None
    equity: float | None = None
    buying_power: float | None = None
    available_funds: float | None = None
    snapshot_at: datetime | None = None
    is_active: bool = True
    extra: dict[str, Any] | None = None


class AccountUpdate(BaseModel):
    """Schema for updating an account."""
    display_name: str | None = None
    account_type: str | None = None
    currency: str | None = None
    cash_balance: float | None = None
    equity: float | None = None
    buying_power: float | None = None
    available_funds: float | None = None
    snapshot_at: datetime | None = None
    is_active: bool | None = None
    extra: dict[str, Any] | None = None


class AccountListResponse(BaseModel):
    """Response for listing accounts."""
    accounts: list[AccountRead]
    count: int
