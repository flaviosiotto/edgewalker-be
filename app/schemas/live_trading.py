"""
Schemas for Live Trading API: Orders, Trades, and Positions.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ── Status literals ──────────────────────────────────────────────────

OrderStatusType = Literal[
    "pending", "submitted", "filled", "partially_filled",
    "cancelled", "rejected", "expired", "error",
]
PositionStatusType = Literal["open", "closed"]
OrderSideType = Literal["buy", "sell"]
OrderTypeType = Literal["market", "limit", "stop", "stop_limit"]
PositionSideType = Literal["long", "short", "flat"]


# ── Order Schemas ────────────────────────────────────────────────────


class LiveOrderCreate(BaseModel):
    """Create a live order (called by the strategy runner)."""
    symbol: str
    side: OrderSideType
    order_type: OrderTypeType = "market"
    quantity: float
    limit_price: float | None = None
    stop_price: float | None = None
    broker_order_id: str | None = None
    extra: dict[str, Any] | None = None


class LiveOrderUpdate(BaseModel):
    """Update a live order (status change, fill info)."""
    status: OrderStatusType | None = None
    status_message: str | None = None
    broker_order_id: str | None = None
    filled_quantity: float | None = None
    avg_fill_price: float | None = None
    commission: float | None = None
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    cancelled_at: datetime | None = None
    extra: dict[str, Any] | None = None


class LiveOrderRead(BaseModel):
    """Response schema for a live order."""
    id: int
    strategy_live_id: int
    account_id: int | None = None
    broker_order_id: str | None = None
    symbol: str
    side: str
    order_type: str
    quantity: float
    limit_price: float | None = None
    stop_price: float | None = None
    filled_quantity: float
    avg_fill_price: float | None = None
    commission: float | None = None
    status: str
    status_message: str | None = None
    created_at: datetime
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    cancelled_at: datetime | None = None
    updated_at: datetime
    extra: dict[str, Any] | None = None
    trades: list["LiveTradeRead"] = []

    class Config:
        from_attributes = True


# ── Trade Schemas ────────────────────────────────────────────────────


class LiveTradeCreate(BaseModel):
    """Create a live trade / fill record."""
    symbol: str
    side: OrderSideType
    quantity: float
    price: float
    commission: float = 0.0
    realized_pnl: float | None = None
    trade_time: datetime
    order_id: int | None = None
    broker_trade_id: str | None = None
    extra: dict[str, Any] | None = None


class LiveTradeRead(BaseModel):
    """Response schema for a live trade."""
    id: int
    strategy_live_id: int
    account_id: int | None = None
    order_id: int | None = None
    broker_trade_id: str | None = None
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float | None = None
    realized_pnl: float | None = None
    trade_time: datetime
    created_at: datetime
    extra: dict[str, Any] | None = None

    class Config:
        from_attributes = True


# ── Position Schemas ─────────────────────────────────────────────────


class LivePositionCreate(BaseModel):
    """Create or update an open position."""
    symbol: str
    side: PositionSideType = "flat"
    quantity: float = 0.0
    avg_price: float | None = None
    cost_basis: float | None = None
    unrealized_pnl: float | None = None
    realized_pnl: float | None = None
    market_value: float | None = None
    extra: dict[str, Any] | None = None


class LivePositionUpdate(BaseModel):
    """Update a live position."""
    side: PositionSideType | None = None
    quantity: float | None = None
    avg_price: float | None = None
    cost_basis: float | None = None
    unrealized_pnl: float | None = None
    realized_pnl: float | None = None
    market_value: float | None = None
    status: PositionStatusType | None = None
    closed_at: datetime | None = None
    extra: dict[str, Any] | None = None


class LivePositionRead(BaseModel):
    """Response schema for a live position."""
    id: int
    strategy_live_id: int
    account_id: int | None = None
    symbol: str
    side: str
    quantity: float
    avg_price: float | None = None
    cost_basis: float | None = None
    unrealized_pnl: float | None = None
    realized_pnl: float | None = None
    market_value: float | None = None
    status: str
    opened_at: datetime
    closed_at: datetime | None = None
    updated_at: datetime
    extra: dict[str, Any] | None = None

    # Computed fields (enriched at response time, not persisted)
    last_price: float | None = None
    computed_unrealized_pnl: float | None = None
    computed_market_value: float | None = None
    total_commission: float | None = None
    net_pnl: float | None = None
    price_age_seconds: float | None = None

    class Config:
        from_attributes = True


# ── Reconciliation Schemas ───────────────────────────────────────────


class ReconciliationItem(BaseModel):
    """A single item from the reconciliation report."""
    entity: Literal["order", "position"]
    symbol: str
    issue: str  # e.g. "stale_order", "missing_position", "quantity_mismatch"
    db_state: dict[str, Any] | None = None
    broker_state: dict[str, Any] | None = None
    action_taken: str | None = None  # e.g. "cancelled", "synced", "flagged"


class ReconciliationReport(BaseModel):
    """Report returned after startup reconciliation."""
    strategy_live_id: int
    account_id: int | None = None
    checked_at: datetime
    items: list[ReconciliationItem] = []
    summary: str
