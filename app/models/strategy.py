from __future__ import annotations

from datetime import datetime, date, timezone
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import Column, ForeignKey, String, Date, DateTime, Integer, Numeric, Text, Float
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlmodel import Field, Relationship, SQLModel

if TYPE_CHECKING:
    from app.models.agent import Chat


class BacktestStatus(str, Enum):
    """Status of a backtest execution.
    
    State transitions:
    - pending: Initial state when backtest is created
    - running: Set by n8n workflow when execution starts
    - completed: Set by n8n workflow on successful completion
    - failed: Set by n8n workflow on execution failure
    - error: Set by backend if webhook call fails (cannot start workflow)
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ERROR = "error"  # Backend error (webhook unreachable, etc.)


class LiveStatus(str, Enum):
    """Status of a live strategy execution.
    
    State transitions:
    - stopped: Initial state / after manual stop
    - starting: Container is being created
    - running: Container is running and processing events
    - stopping: Container stop requested
    - error: Container failed or crashed
    """
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


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

    # ── LIVE TRADING STATE ──
    live_status: str = Field(
        default=LiveStatus.STOPPED.value,
        sa_column=Column(String(20), nullable=False, index=True),
    )
    live_container_id: Optional[str] = Field(
        default=None,
        sa_column=Column(String(64), nullable=True),
    )
    live_symbol: Optional[str] = Field(
        default=None,
        sa_column=Column(String(32), nullable=True),
    )
    live_timeframe: Optional[str] = Field(
        default=None,
        sa_column=Column(String(10), nullable=True),
    )
    live_started_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    live_stopped_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    live_error_message: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
    )
    # Trading account for live execution (nullable)
    live_account_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("accounts.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
    )
    # Runtime metrics snapshot (updated periodically)
    live_metrics: Optional[Any] = Field(
        default=None,
        sa_column=Column(JSONB, nullable=True),
    )

    backtests: list["BacktestResult"] = Relationship(
        sa_relationship=relationship(
            "BacktestResult",
            back_populates="strategy",
            cascade="all, delete-orphan",
        )
    )

    chats: list["Chat"] = Relationship(
        sa_relationship=relationship(
            "Chat",
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

    # Agent that executes this backtest via n8n webhook
    agent_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("agent.id_agent", ondelete="SET NULL"),
            nullable=True,
            index=True,
        )
    )

    # ── INPUT (immutable after creation) ──
    symbol: str = Field(sa_column=Column(String(32), nullable=False, index=True))
    start_date: date = Field(sa_column=Column(Date, nullable=False))
    end_date: date = Field(sa_column=Column(Date, nullable=False))
    
    # Data source parameters (for fetch)
    source: Optional[str] = Field(default="ibkr", sa_column=Column(String(20), nullable=True))
    timeframe: Optional[str] = Field(default="5m", sa_column=Column(String(10), nullable=True))
    asset: Optional[str] = Field(default="stock", sa_column=Column(String(20), nullable=True))
    rth: Optional[bool] = Field(default=True, sa_column=Column(String(10), nullable=True))  # stored as string for simplicity
    
    # IBKR-specific parameters
    ibkr_config: Optional[str] = Field(default=None, sa_column=Column(String(255), nullable=True))
    exchange: Optional[str] = Field(default="SMART", sa_column=Column(String(20), nullable=True))
    currency: Optional[str] = Field(default="USD", sa_column=Column(String(10), nullable=True))
    expiry: Optional[str] = Field(default=None, sa_column=Column(String(20), nullable=True))
    
    # Backtest execution parameters
    initial_capital: Optional[float] = Field(default=100000.0, sa_column=Column(Float, nullable=True))
    commission: Optional[float] = Field(default=0.0, sa_column=Column(Float, nullable=True))
    
    # Additional config overrides (JSONB)
    parameters: Optional[Any] = Field(default=None, sa_column=Column(JSONB, nullable=True))

    # Full strategy configuration snapshot at creation time (immutable)
    config: Optional[Any] = Field(default=None, sa_column=Column(JSONB, nullable=True))

    # ── STATUS ──
    status: str = Field(
        default=BacktestStatus.PENDING.value,
        sa_column=Column(String(20), nullable=False, index=True),
    )
    started_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    error_message: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
    )

    # ── OUTPUT (populated on completion by n8n workflow) ──
    # Raw stats dict from edgewalker/backtesting.py
    metrics: Optional[Any] = Field(default=None, sa_column=Column(JSONB, nullable=True))
    
    # Key metrics extracted for easy querying (from edgewalker BacktestResult.stats)
    return_pct: Optional[float] = Field(default=None, sa_column=Column(Float, nullable=True))
    sharpe_ratio: Optional[float] = Field(default=None, sa_column=Column(Float, nullable=True))
    max_drawdown_pct: Optional[float] = Field(default=None, sa_column=Column(Float, nullable=True))
    win_rate_pct: Optional[float] = Field(default=None, sa_column=Column(Float, nullable=True))
    profit_factor: Optional[float] = Field(default=None, sa_column=Column(Float, nullable=True))
    total_trades: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))
    equity_final: Optional[float] = Field(default=None, sa_column=Column(Float, nullable=True))
    equity_peak: Optional[float] = Field(default=None, sa_column=Column(Float, nullable=True))
    
    # HTML report path/URL
    html_report_url: Optional[str] = Field(
        default=None,
        sa_column=Column(String(500), nullable=True),
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )

    strategy: Strategy | None = Relationship(
        sa_relationship=relationship("Strategy", back_populates="backtests")
    )

    agent: "Agent | None" = Relationship(
        sa_relationship=relationship("Agent")
    )

    trades: list["BacktestTrade"] = Relationship(
        sa_relationship=relationship(
            "BacktestTrade",
            back_populates="backtest",
            cascade="all, delete-orphan",
        )
    )


class BacktestTrade(SQLModel, table=True):
    """Trade record from backtest execution.
    
    Aligned with edgewalker's TradeRecord dataclass.
    """
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

    # Trade timing (from edgewalker TradeRecord)
    entry_time: datetime = Field(sa_column=Column(DateTime(timezone=True), nullable=False, index=True))
    exit_time: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True), nullable=True))
    
    # Trade direction: 'long' or 'short'
    direction: str = Field(sa_column=Column(String(8), nullable=False))
    
    # Position size
    size: float = Field(sa_column=Column(Numeric(20, 8), nullable=False))

    # Prices
    entry_price: float = Field(sa_column=Column(Numeric(20, 8), nullable=False))
    exit_price: Optional[float] = Field(default=None, sa_column=Column(Numeric(20, 8), nullable=True))

    # P&L
    pnl: Optional[float] = Field(default=None, sa_column=Column(Numeric(20, 8), nullable=True))
    pnl_pct: Optional[float] = Field(default=None, sa_column=Column(Float, nullable=True))
    
    # Session context (for session-based strategies)
    session_date: Optional[date] = Field(default=None, sa_column=Column(Date, nullable=True))
    
    # Exit reason (stop_loss, take_profit, end_of_session, etc.)
    exit_reason: Optional[str] = Field(default=None, sa_column=Column(String(50), nullable=True))
    
    # Additional data as JSON (called 'extra' to avoid SQLAlchemy reserved 'metadata')
    extra: Optional[Any] = Field(default=None, sa_column=Column(JSONB, nullable=True))

    backtest: BacktestResult | None = Relationship(
        sa_relationship=relationship("BacktestResult", back_populates="trades")
    )


# Import at end to avoid circular imports
from app.models.agent import Agent  # noqa: E402, F401
