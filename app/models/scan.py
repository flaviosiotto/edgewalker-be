"""Symbol scans (migr. 077, docs/valutazione-scanner-simboli.md).

A scan reads one strategy's indicators and rule conditions on the last closed
bar of many symbols of a connection: a table with one row per symbol, measures
only. The backend writes the request (``Scan``); the coordinator
(strategy-backtest) writes the progress and every ``ScanRow``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from sqlalchemy import BigInteger, Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ScanStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ScanMode(str, Enum):
    #: computed once, when asked. A scan refreshed at every bar close would
    #: write the same tables under another mode.
    ON_DEMAND = "on_demand"


class Scan(SQLModel, table=True):
    __tablename__ = "scans"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(
        sa_column=Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    strategy_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("strategies.id", ondelete="SET NULL"), nullable=True, index=True),
    )
    #: name of the strategy when the scan was asked (the strategy may be deleted)
    strategy_name: Optional[str] = Field(default=None, sa_column=Column(String(255), nullable=True))
    connection_id: Optional[int] = Field(
        default=None, sa_column=Column(Integer, ForeignKey("connections.id", ondelete="SET NULL"), nullable=True)
    )
    broker_type: Optional[str] = Field(default=None, sa_column=Column(String(32), nullable=True))
    market_type: Optional[str] = Field(default=None, sa_column=Column(String(32), nullable=True))
    timeframe: str = Field(sa_column=Column(String(16), nullable=False))
    mode: str = Field(default=ScanMode.ON_DEMAND.value, sa_column=Column(String(16), nullable=False))
    #: what was asked: {"symbols", "min_turnover_24h", "max_symbols"}
    request: Any = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False))
    #: snapshot of the strategy definition the scan evaluates
    config: Any = Field(default=None, sa_column=Column(JSONB, nullable=False))
    #: written by the coordinator: {"rules", "indicators", "window", "warmup", "notes"}
    plan: Optional[Any] = Field(default=None, sa_column=Column(JSONB, nullable=True))
    status: str = Field(default=ScanStatus.PENDING.value, sa_column=Column(String(16), nullable=False))
    symbols_total: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))
    symbols_done: int = Field(default=0, sa_column=Column(Integer, nullable=False, default=0))
    symbols_failed: int = Field(default=0, sa_column=Column(Integer, nullable=False, default=0))
    error_message: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    created_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))
    started_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True), nullable=True))
    completed_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True), nullable=True))


class ScanRow(SQLModel, table=True):
    """One symbol of a scan, written by the coordinator only."""
    __tablename__ = "scan_rows"
    __table_args__ = (UniqueConstraint("scan_id", "symbol", name="uq_scan_rows_scan_symbol"),)

    id: Optional[int] = Field(default=None, sa_column=Column(BigInteger, primary_key=True, autoincrement=True))
    scan_id: int = Field(
        sa_column=Column(Integer, ForeignKey("scans.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    symbol: str = Field(sa_column=Column(String(64), nullable=False))
    name: Optional[str] = Field(default=None, sa_column=Column(String(128), nullable=True))
    #: open time (ms) of the closed bar the values refer to
    bar_ts: Optional[int] = Field(default=None, sa_column=Column(BigInteger, nullable=True))
    #: venue ticker: last, change_24h_pct, volume_24h, turnover_24h, turnover_estimated
    market: Optional[Any] = Field(default=None, sa_column=Column(JSONB, nullable=True))
    #: measures by field path: price.close, indicators.<key>
    values: Optional[Any] = Field(default=None, sa_column=Column("values", JSONB, nullable=True))
    #: per rule: {"name", "matched", "conditions": [{"passed", "actual", ...}]}; null = not evaluable
    rules: Optional[Any] = Field(default=None, sa_column=Column(JSONB, nullable=True))
    matched_rules: int = Field(default=0, sa_column=Column(Integer, nullable=False, default=0))
    conditions_true: int = Field(default=0, sa_column=Column(Integer, nullable=False, default=0))
    conditions_total: int = Field(default=0, sa_column=Column(Integer, nullable=False, default=0))
    error: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    updated_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))
