"""API shapes of the symbol scans (docs/valutazione-scanner-simboli.md)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class ScanCreate(BaseModel):
    #: strategy whose primary-chart indicators and rules are read on each symbol
    strategy_id: int
    #: bars to read; default: the timeframe of the strategy's primary chart
    timeframe: Optional[str] = Field(default=None, max_length=16)
    #: explicit symbols; default: the market of the strategy's connection
    symbols: Optional[list[str]] = Field(default=None, max_length=1000)
    #: lower bound on the 24h turnover, in quote currency
    min_turnover_24h: Optional[float] = Field(default=None, ge=0)
    #: how many symbols at most, largest turnover first
    max_symbols: int = Field(default=300, ge=1, le=1000)


class ScanSummary(BaseModel):
    """A scan without its definition snapshot (lists, polling)."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    strategy_id: Optional[int] = None
    strategy_name: Optional[str] = None
    connection_id: Optional[int] = None
    broker_type: Optional[str] = None
    market_type: Optional[str] = None
    timeframe: str
    mode: str
    status: str
    symbols_total: Optional[int] = None
    symbols_done: int = 0
    symbols_failed: int = 0
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class ScanRead(ScanSummary):
    request: Any = None
    #: rules as evaluated, indicators, window, notes (None until the scan starts)
    plan: Any = None
    #: field paths present in the rows' ``values``, for column pickers
    value_keys: list[str] = []


class ScanListPage(BaseModel):
    items: list[ScanSummary]
    total: int
    limit: int
    offset: int


class ScanRowRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol: str
    name: Optional[str] = None
    bar_ts: Optional[int] = None
    market: Any = None
    values: Any = None
    rules: Any = None
    matched_rules: int = 0
    conditions_true: int = 0
    conditions_total: int = 0
    error: Optional[str] = None


class ScanRowsPage(BaseModel):
    items: list[ScanRowRead]
    #: rows after the filters, before the page cut
    total: int
    limit: int
    offset: int
    #: the scan's progress, so one poll serves both
    status: str
    symbols_total: Optional[int] = None
    symbols_done: int = 0
