"""API shapes of the symbol scans (docs/valutazione-scanner-simboli.md)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator, computed_field


class ScanCreate(BaseModel):
    """Two ways to say what to read on each symbol.

    From a strategy (``strategy_id`` alone): its primary-chart indicators and
    its enabled rules. Or freely (``indicators``, with optional
    ``conditions``): the classic screener — a few indicators and a filter over
    them — on the connection given by ``connection_id`` or, when a
    ``strategy_id`` comes with it, on that strategy's connection.
    """
    strategy_id: Optional[int] = None
    connection_id: Optional[int] = None
    #: title of a free scan, shown where a strategy's name would be
    name: Optional[str] = Field(default=None, max_length=120)
    #: free scan: indicator instances, ``{"name", "type", "params"}``
    indicators: Optional[list[dict[str, Any]]] = Field(default=None, max_length=12)
    #: free scan: conditions over price.* / indicators.*, same shape as a rule's
    conditions: Optional[list[dict[str, Any]]] = Field(default=None, max_length=12)
    #: free scan: how the conditions combine
    logic: Literal["and", "or"] = "and"
    #: bars to read; default: the timeframe of the strategy's primary chart
    timeframe: Optional[str] = Field(default=None, max_length=16)
    #: explicit symbols; default: the market of the connection
    symbols: Optional[list[str]] = Field(default=None, max_length=1000)
    #: lower bound on the 24h turnover, in quote currency
    min_turnover_24h: Optional[float] = Field(default=None, ge=0)
    #: how many symbols at most, largest turnover first
    max_symbols: int = Field(default=300, ge=1, le=1000)
    #: closed bars the conditions are evaluated on, the last one included:
    #: a row says when each rule was true in that window, not only whether it is now
    lookback: int = Field(default=20, ge=1, le=200)

    @model_validator(mode="after")
    def _one_source(self) -> "ScanCreate":
        free = self.indicators is not None or self.conditions is not None
        if not free and self.strategy_id is None:
            raise ValueError("Give a strategy_id, or indicators for a free scan")
        if free:
            if not self.indicators and not self.conditions:
                raise ValueError("A free scan needs at least one indicator or one condition")
            if self.strategy_id is None and self.connection_id is None:
                raise ValueError("A free scan needs the connection_id of the market to read")
            if not str(self.timeframe or "").strip():
                raise ValueError("A free scan needs a timeframe")
        return self

    @property
    def is_free(self) -> bool:
        return self.indicators is not None or self.conditions is not None


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
    #: True when the scan reads free indicators and conditions, not a strategy's
    free: bool = False


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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def lookback(self) -> Optional[int]:
        """Closed bars evaluated for this symbol (None: a scan of the last bar only)."""
        for outcome in self.rules if isinstance(self.rules, list) else []:
            if isinstance(outcome, dict) and isinstance(outcome.get("lookback"), int):
                return outcome["lookback"]
        return None


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
