"""Schemas of the watchlist and of the symbol catalogue reads."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class WatchlistItemCreate(BaseModel):
    connection_id: int
    symbol: str = Field(min_length=1, max_length=64)
    name: Optional[str] = Field(default=None, max_length=255)
    asset_type: Optional[str] = Field(default=None, max_length=32)
    exchange: Optional[str] = Field(default=None, max_length=64)
    currency: Optional[str] = Field(default=None, max_length=16)
    info: dict[str, Any] = Field(default_factory=dict)

    @field_validator("symbol")
    @classmethod
    def _strip_symbol(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("symbol is required")
        return value


class WatchlistItemRead(BaseModel):
    id: int
    connection_id: int
    symbol: str
    name: Optional[str] = None
    asset_type: Optional[str] = None
    exchange: Optional[str] = None
    currency: Optional[str] = None
    info: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class SymbolCatalogResponse(BaseModel):
    """Every instrument of a connection (gateway ``UniverseRow`` rows, untouched)."""
    connection_id: int
    broker_type: str
    market_type: Optional[str] = None
    #: false = the broker has no catalogue: its symbols are found by search only
    supported: bool
    #: rows carry last / change / volume (the venue has a bulk ticker)
    has_prices: bool
    #: false = the rows are a part of what the broker trades (public listings):
    #: what is not there is still found by ``/marketdata/symbols``
    complete: bool = True
    #: ms, when the gateway read the venue
    as_of: Optional[int] = None
    #: the gateway did not answer: this is the last copy read
    stale: bool = False
    count: int
    rows: list[dict[str, Any]]


class SymbolQuotesResponse(BaseModel):
    connection_id: int
    supported: bool
    as_of: Optional[int] = None
    stale: bool = False
    #: symbol → {last, change_24h_pct, …}; ``last_from_bar`` when read from the last closed bar
    quotes: dict[str, dict[str, Any]]
