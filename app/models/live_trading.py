"""
Live Trading Models: Orders, Trades, and Positions.

These tables persist the live trading state for strategies connected
to broker accounts, enabling startup reconciliation and audit.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from sqlalchemy import Column, ForeignKey, Integer, String, DateTime, Float, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlmodel import Field, Relationship, SQLModel


# ── Enums ────────────────────────────────────────────────────────────


class OrderStatus(str, Enum):
    """Status of a live order."""
    PENDING = "pending"           # Created locally, not yet sent
    SUBMITTED = "submitted"       # Sent to broker
    FILLED = "filled"             # Fully filled
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"       # Cancelled by user or system
    REJECTED = "rejected"         # Rejected by broker
    EXPIRED = "expired"           # Expired (GTC/GTD timeout)
    ERROR = "error"               # Error state


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class PositionStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


# ── Models ───────────────────────────────────────────────────────────


class LiveOrder(SQLModel, table=True):
    """
    A live order placed by a strategy on a broker account.

    Tracks the full lifecycle: pending → submitted → filled/cancelled/rejected.
    """
    __tablename__ = "live_orders"
    __allow_unmapped__ = True

    id: Optional[int] = Field(default=None, primary_key=True)

    strategy_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("strategies.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )
    account_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("accounts.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
    )

    # Broker-assigned order identifier
    broker_order_id: Optional[str] = Field(
        default=None,
        sa_column=Column(String(100), nullable=True, index=True),
    )

    # Order details
    symbol: str = Field(sa_column=Column(String(32), nullable=False))
    side: str = Field(sa_column=Column(String(10), nullable=False))       # buy / sell
    order_type: str = Field(sa_column=Column(String(20), nullable=False)) # market / limit / stop / stop_limit
    quantity: float = Field(sa_column=Column(Float, nullable=False))

    # Prices
    limit_price: Optional[float] = Field(default=None, sa_column=Column(Float, nullable=True))
    stop_price: Optional[float] = Field(default=None, sa_column=Column(Float, nullable=True))

    # Fill information
    filled_quantity: float = Field(default=0.0, sa_column=Column(Float, nullable=False))
    avg_fill_price: Optional[float] = Field(default=None, sa_column=Column(Float, nullable=True))
    commission: Optional[float] = Field(default=None, sa_column=Column(Float, nullable=True))

    # Status
    status: str = Field(
        default=OrderStatus.PENDING.value,
        sa_column=Column(String(30), nullable=False, index=True),
    )
    status_message: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    submitted_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    filled_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    cancelled_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )

    # Extra data (broker-specific fields, TIF, etc.)
    extra: Optional[Any] = Field(default=None, sa_column=Column(JSONB, nullable=True))

    # Relationships
    trades: list["LiveTrade"] = Relationship(
        sa_relationship=relationship(
            "LiveTrade",
            back_populates="order",
            cascade="all, delete-orphan",
        )
    )


class LiveTrade(SQLModel, table=True):
    """
    An executed fill / trade from a live order.

    Each fill from the broker creates a LiveTrade record.
    Multiple fills may belong to a single order (partial fills).
    """
    __tablename__ = "live_trades"
    __allow_unmapped__ = True

    id: Optional[int] = Field(default=None, primary_key=True)

    strategy_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("strategies.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )
    account_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("accounts.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
    )
    order_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("live_orders.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
    )

    # Broker-assigned trade / execution identifier
    broker_trade_id: Optional[str] = Field(
        default=None,
        sa_column=Column(String(100), nullable=True, index=True),
    )

    # Trade details
    symbol: str = Field(sa_column=Column(String(32), nullable=False))
    side: str = Field(sa_column=Column(String(10), nullable=False))  # buy / sell
    quantity: float = Field(sa_column=Column(Float, nullable=False))
    price: float = Field(sa_column=Column(Float, nullable=False))
    commission: Optional[float] = Field(default=0.0, sa_column=Column(Float, nullable=True))

    # P&L (realized, computed at trade time)
    realized_pnl: Optional[float] = Field(default=None, sa_column=Column(Float, nullable=True))

    # Timestamps
    trade_time: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, index=True),
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )

    # Extra data
    extra: Optional[Any] = Field(default=None, sa_column=Column(JSONB, nullable=True))

    # Relationships
    order: LiveOrder | None = Relationship(
        sa_relationship=relationship("LiveOrder", back_populates="trades")
    )


class LivePosition(SQLModel, table=True):
    """
    Current or historical position held by a strategy on a broker account.

    Tracks open and closed positions for reconciliation and audit.
    """
    __tablename__ = "live_positions"
    __allow_unmapped__ = True

    id: Optional[int] = Field(default=None, primary_key=True)

    strategy_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("strategies.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )
    account_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("accounts.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
    )

    # Position details
    symbol: str = Field(sa_column=Column(String(32), nullable=False, index=True))
    side: str = Field(
        default=PositionSide.FLAT.value,
        sa_column=Column(String(10), nullable=False),
    )
    quantity: float = Field(default=0.0, sa_column=Column(Float, nullable=False))
    avg_price: Optional[float] = Field(default=None, sa_column=Column(Float, nullable=True))
    cost_basis: Optional[float] = Field(default=None, sa_column=Column(Float, nullable=True))

    # P&L
    unrealized_pnl: Optional[float] = Field(default=None, sa_column=Column(Float, nullable=True))
    realized_pnl: Optional[float] = Field(default=0.0, sa_column=Column(Float, nullable=True))
    market_value: Optional[float] = Field(default=None, sa_column=Column(Float, nullable=True))

    # Status
    status: str = Field(
        default=PositionStatus.OPEN.value,
        sa_column=Column(String(10), nullable=False, index=True),
    )

    # Timestamps
    opened_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    closed_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )

    # Extra data (broker-specific position details)
    extra: Optional[Any] = Field(default=None, sa_column=Column(JSONB, nullable=True))
