from __future__ import annotations

from datetime import datetime, date, timezone
from typing import Any, Optional

from sqlalchemy import Column, ForeignKey, String, Date, DateTime, Integer, Numeric
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlmodel import Field, Relationship, SQLModel


class Strategy(SQLModel, table=True):
    __tablename__ = "strategies"
    __allow_unmapped__ = True

    id: Optional[int] = Field(default=None, primary_key=True)

    name: str = Field(sa_column=Column(String(255), unique=True, nullable=False, index=True))
    description: Optional[str] = Field(default=None, sa_column=Column(String(2000), nullable=True))

    # Declarative strategy syntax/config
    definition: Any = Field(sa_column=Column(JSONB, nullable=False))

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )

    backtests: list["BacktestResult"] = Relationship(
        sa_relationship=relationship(
            "BacktestResult",
            back_populates="strategy",
            cascade="all, delete-orphan",
        )
    )


class BacktestResult(SQLModel, table=True):
    __tablename__ = "strategy_backtests"
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

    symbol: str = Field(sa_column=Column(String(32), nullable=False, index=True))
    start_date: date = Field(sa_column=Column(Date, nullable=False))
    end_date: date = Field(sa_column=Column(Date, nullable=False))

    # Optional: store any run parameters, metrics, and report payloads
    parameters: Optional[Any] = Field(default=None, sa_column=Column(JSONB, nullable=True))
    metrics: Optional[Any] = Field(default=None, sa_column=Column(JSONB, nullable=True))

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )

    strategy: Strategy | None = Relationship(
        sa_relationship=relationship("Strategy", back_populates="backtests")
    )

    trades: list["BacktestTrade"] = Relationship(
        sa_relationship=relationship(
            "BacktestTrade",
            back_populates="backtest",
            cascade="all, delete-orphan",
        )
    )


class BacktestTrade(SQLModel, table=True):
    __tablename__ = "strategy_backtest_trades"
    __allow_unmapped__ = True

    id: Optional[int] = Field(default=None, primary_key=True)

    backtest_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("strategy_backtests.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )

    # Redundant but useful for filtering without join
    strategy_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("strategies.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )

    ts_open: datetime = Field(sa_column=Column(DateTime(timezone=True), nullable=False, index=True))
    ts_close: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True), nullable=True))

    side: str = Field(sa_column=Column(String(8), nullable=False))  # long|short
    quantity: float = Field(sa_column=Column(Numeric(20, 8), nullable=False))

    entry_price: float = Field(sa_column=Column(Numeric(20, 8), nullable=False))
    exit_price: Optional[float] = Field(default=None, sa_column=Column(Numeric(20, 8), nullable=True))

    pnl: Optional[float] = Field(default=None, sa_column=Column(Numeric(20, 8), nullable=True))
    fees: Optional[float] = Field(default=None, sa_column=Column(Numeric(20, 8), nullable=True))

    meta: Optional[Any] = Field(default=None, sa_column=Column(JSONB, nullable=True))

    backtest: BacktestResult | None = Relationship(
        sa_relationship=relationship("BacktestResult", back_populates="trades")
    )
