"""
Schemas for Connections & Accounts API.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Connection Schemas
# =============================================================================

class ConnectionConfig(BaseModel):
    """Connection configuration (broker-specific)."""
    # IBKR
    host: str | None = None
    port: int | None = None
    client_id: int | None = None
    # Generic
    api_key: str | None = None
    api_secret: str | None = None

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
    created_at: datetime
    updated_at: datetime
    accounts: list[AccountRead] = []

    class Config:
        from_attributes = True


class ConnectionListResponse(BaseModel):
    """Response for listing connections."""
    connections: list[ConnectionRead]
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
    is_active: bool = True
    extra: dict[str, Any] | None = None


class AccountUpdate(BaseModel):
    """Schema for updating an account."""
    display_name: str | None = None
    account_type: str | None = None
    currency: str | None = None
    is_active: bool | None = None
    extra: dict[str, Any] | None = None


class AccountListResponse(BaseModel):
    """Response for listing accounts."""
    accounts: list[AccountRead]
    count: int
