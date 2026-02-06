"""
Connection & Account Models.

A Connection represents a broker/exchange connection (IBKR, Binance, etc.).
An Account represents a trading account exposed by that connection.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from sqlalchemy import Column, ForeignKey, Integer, String, DateTime, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlmodel import Field, Relationship, SQLModel


class BrokerType(str, Enum):
    """Supported broker/exchange types."""
    IBKR = "ibkr"
    BINANCE = "binance"  # future


class ConnectionStatus(str, Enum):
    """Connection status."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class Connection(SQLModel, table=True):
    """Broker/exchange connection configuration."""

    __tablename__ = "connections"
    __allow_unmapped__ = True

    id: Optional[int] = Field(default=None, primary_key=True)

    name: str = Field(
        sa_column=Column(String(100), unique=True, nullable=False, index=True),
        description="Human-readable connection name",
    )
    broker_type: str = Field(
        sa_column=Column(String(30), nullable=False, index=True),
        description="Broker type: 'ibkr', 'binance', etc.",
    )

    # Connection settings (host, port, client_id, api_key, …)
    config: Any = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, server_default="'{}'"),
    )

    # Status
    is_active: bool = Field(default=True)
    status: str = Field(
        default=ConnectionStatus.DISCONNECTED.value,
        sa_column=Column(String(20), nullable=False),
    )
    status_message: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
    )
    last_connected_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )

    # Relationships
    accounts: list["Account"] = Relationship(
        sa_relationship=relationship(
            "Account",
            back_populates="connection",
            cascade="all, delete-orphan",
        )
    )


class Account(SQLModel, table=True):
    """Trading account exposed by a Connection."""

    __tablename__ = "accounts"
    __allow_unmapped__ = True

    id: Optional[int] = Field(default=None, primary_key=True)

    connection_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("connections.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )

    account_id: str = Field(
        sa_column=Column(String(50), nullable=False),
        description="Broker-specific account code (e.g. 'DU1234567')",
    )
    display_name: Optional[str] = Field(
        default=None,
        sa_column=Column(String(100), nullable=True),
    )
    account_type: Optional[str] = Field(
        default=None,
        sa_column=Column(String(30), nullable=True),
        description="paper, live, margin, etc.",
    )
    currency: str = Field(
        default="USD",
        sa_column=Column(String(10), nullable=False),
    )

    is_active: bool = Field(default=True)

    # Broker-specific extra data
    extra: Optional[Any] = Field(
        default=None,
        sa_column=Column(JSONB, nullable=True),
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )

    # Relationships
    connection: Connection | None = Relationship(
        sa_relationship=relationship("Connection", back_populates="accounts")
    )
