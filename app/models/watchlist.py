"""Watchlist (migr. 079): the symbols a user keeps an eye on.

A symbol is the broker's own name for an instrument, so an item belongs to a
connection: ``BTC-USDT-SWAP`` on an OKX connection and ``US500.cash`` on a
cTrader one are two items. Deleting the connection deletes its items.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class WatchlistItem(SQLModel, table=True):
    __tablename__ = "watchlist_items"
    __table_args__ = (
        UniqueConstraint("user_id", "connection_id", "symbol", name="uq_watchlist_items_user_conn_symbol"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(
        sa_column=Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    connection_id: int = Field(
        sa_column=Column(Integer, ForeignKey("connections.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    #: written the way the broker spells it (``US500.cash``)
    symbol: str = Field(sa_column=Column(String(64), nullable=False))
    name: Optional[str] = Field(default=None, sa_column=Column(String(255), nullable=True))
    asset_type: Optional[str] = Field(default=None, sa_column=Column(String(32), nullable=True))
    exchange: Optional[str] = Field(default=None, sa_column=Column(String(64), nullable=True))
    currency: Optional[str] = Field(default=None, sa_column=Column(String(16), nullable=True))
    #: the broker's description when the symbol was added (tick size, expiry…):
    #: the row and its chart open without asking the gateway again
    info: Any = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False))
    created_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))
