"""Live trading projections for broker accounts.

The canonical scope of orders, fills, and positions is the broker account.
`strategy_live_id` remains optional correlation metadata when a broker event
can be linked back to a specific live session.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import Column, ForeignKey, Integer, String, DateTime, Float, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlmodel import Field, Relationship, SQLModel

if TYPE_CHECKING:
    from app.models.strategy import StrategyLive


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
    __tablename__ = "orders"
    __allow_unmapped__ = True

    id: Optional[int] = Field(default=None, primary_key=True)

    strategy_live_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("strategy_live.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        )
    )
    account_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("accounts.id", ondelete="CASCADE"),
            nullable=False,
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
    strategy_live: Optional["StrategyLive"] = Relationship(
        sa_relationship=relationship("StrategyLive", back_populates="orders")
    )
    fills: list["LiveFill"] = Relationship(
        sa_relationship=relationship(
            "LiveFill",
            back_populates="order",
            cascade="all, delete-orphan",
        )
    )


class LiveFill(SQLModel, table=True):
    """
    An executed fill from a live order (immutable event).

    Pattern: Order (command) → Fill (event) → Position (state).
    Each fill from the broker creates a LiveFill record.
    Multiple fills may belong to a single order (partial fills).
    """
    __tablename__ = "fills"
    __allow_unmapped__ = True

    id: Optional[int] = Field(default=None, primary_key=True)

    strategy_live_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("strategy_live.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        )
    )
    account_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("accounts.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
    )
    order_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("orders.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
    )

    # Broker-assigned fill / execution identifier
    broker_fill_id: Optional[str] = Field(
        default=None,
        sa_column=Column(String(100), nullable=True, index=True),
    )

    # Fill details
    symbol: str = Field(sa_column=Column(String(32), nullable=False))
    side: str = Field(sa_column=Column(String(10), nullable=False))  # buy / sell
    quantity: float = Field(sa_column=Column(Float, nullable=False))
    price: float = Field(sa_column=Column(Float, nullable=False))
    commission: Optional[float] = Field(default=0.0, sa_column=Column(Float, nullable=True))

    # P&L (realized, computed at fill time)
    realized_pnl: Optional[float] = Field(default=None, sa_column=Column(Float, nullable=True))

    # Timestamps
    fill_time: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, index=True),
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )

    # Extra data
    extra: Optional[Any] = Field(default=None, sa_column=Column(JSONB, nullable=True))

    # Relationships
    strategy_live: Optional["StrategyLive"] = Relationship(
        sa_relationship=relationship("StrategyLive", back_populates="fills")
    )
    order: LiveOrder | None = Relationship(
        sa_relationship=relationship("LiveOrder", back_populates="fills")
    )


class LivePosition(SQLModel, table=True):
    """Broker-authoritative current position held on a broker account."""
    __tablename__ = "account_positions"
    __allow_unmapped__ = True

    id: Optional[int] = Field(default=None, primary_key=True)

    strategy_live_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("strategy_live.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        )
    )
    account_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("accounts.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
    )
    connection_id: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True, index=True))
    broker_account_id: str = Field(sa_column=Column(String(64), nullable=False, index=True))
    broker_type: str = Field(sa_column=Column(String(32), nullable=False, index=True))
    position_key: str = Field(sa_column=Column(String(255), nullable=False, index=True))
    instrument_key: str = Field(sa_column=Column(String(255), nullable=False))

    # Position details
    symbol: str = Field(sa_column=Column(String(64), nullable=False, index=True))
    asset_type: Optional[str] = Field(default=None, sa_column=Column(String(32), nullable=True, index=True))
    position_bucket: str = Field(default="net", sa_column=Column(String(32), nullable=False))
    side: str = Field(
        default=PositionSide.FLAT.value,
        sa_column=Column(String(10), nullable=False),
    )
    quantity: float = Field(default=0.0, sa_column=Column(Float, nullable=False))
    avg_price: Optional[float] = Field(default=None, sa_column=Column(Float, nullable=True))
    market_value: Optional[float] = Field(default=None, sa_column=Column(Float, nullable=True))
    currency: Optional[str] = Field(default=None, sa_column=Column(String(16), nullable=True))

    # Status
    status: str = Field(
        default=PositionStatus.OPEN.value,
        sa_column=Column(String(10), nullable=False, index=True),
    )
    snapshot_id: str = Field(sa_column=Column(String(255), nullable=False, index=True))
    observed_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, index=True),
    )

    # Timestamps
    opened_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )

    # Extra data (broker-specific position details)
    extra: Optional[Any] = Field(default=None, sa_column=Column(JSONB, nullable=True))

    # Relationships
    strategy_live: Optional["StrategyLive"] = Relationship(
        sa_relationship=relationship("StrategyLive", back_populates="positions")
    )

    @property
    def cost_basis(self) -> float | None:
        if self.avg_price is None:
            return None
        return float(self.avg_price) * float(self.quantity or 0.0)

    @property
    def realized_pnl(self) -> float:
        return 0.0

    @property
    def unrealized_pnl(self) -> float | None:
        return None

    @property
    def closed_at(self) -> datetime | None:
        return None
