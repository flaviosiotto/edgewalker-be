from datetime import datetime, date
from typing import Any, Literal, Optional

from pydantic import BaseModel

from app.schemas.chat import ChatRead


# Live status enum values
LiveStatus = Literal["stopped", "starting", "running", "stopping", "error"]


class StrategyCreate(BaseModel):
    name: str
    description: Optional[str] = None
    definition: Any


class StrategyRead(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    definition: Any
    created_at: datetime
    updated_at: datetime
    chats: list[ChatRead] = []
    
    # Live trading state
    live_status: Optional[LiveStatus] = "stopped"
    live_container_id: Optional[str] = None
    live_symbol: Optional[str] = None
    live_timeframe: Optional[str] = None
    live_started_at: Optional[datetime] = None
    live_stopped_at: Optional[datetime] = None
    live_error_message: Optional[str] = None
    live_metrics: Optional[dict[str, Any]] = None
    live_account_id: Optional[int] = None


class StrategyUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    definition: Optional[Any] = None


# ─── BACKTEST SCHEMAS ───


class BacktestCreate(BaseModel):
    """Input parameters for creating a new backtest (status=pending).
    
    Includes all parameters for both fetch and backtest operations.
    """
    # Required parameters
    symbol: str
    start_date: date
    end_date: date
    
    # Agent to execute this backtest via n8n
    agent_id: Optional[int] = None
    
    # Data source parameters (for fetch)
    source: Literal["ibkr", "yahoo"] = "ibkr"
    timeframe: str = "5m"  # e.g., "1m", "5m", "15m", "1h", "1d"
    asset: Literal["stock", "future"] = "stock"
    rth: bool = True  # True = Regular Trading Hours only
    
    # IBKR-specific parameters
    ibkr_config: Optional[str] = "configs/ibkr.yaml"
    exchange: str = "SMART"
    currency: str = "USD"
    
    # Futures-specific parameters
    expiry: Optional[str] = None  # YYYYMM, YYYYMMDD, or "auto"
    
    # Backtest execution parameters
    initial_capital: float = 100000.0
    commission: float = 0.0
    
    # Additional config overrides (JSONB)
    parameters: Optional[dict[str, Any]] = None

    # Full strategy configuration snapshot (auto-captured from strategy.definition)
    config: Optional[dict[str, Any]] = None


class BacktestRead(BaseModel):
    """Full backtest representation including status and results."""
    id: int
    strategy_id: int
    agent_id: Optional[int] = None
    
    # Input parameters
    symbol: str
    start_date: date
    end_date: date
    
    # Data source parameters
    source: Optional[str] = None
    timeframe: Optional[str] = None
    asset: Optional[str] = None
    rth: Optional[bool] = None
    
    # IBKR-specific parameters
    ibkr_config: Optional[str] = None
    exchange: Optional[str] = None
    currency: Optional[str] = None
    expiry: Optional[str] = None
    
    # Backtest execution parameters
    initial_capital: Optional[float] = None
    commission: Optional[float] = None
    
    # Additional config overrides
    parameters: Optional[dict[str, Any]] = None

    # Full strategy configuration snapshot
    config: Optional[dict[str, Any]] = None
    
    # Status (pending | running | completed | failed | error)
    status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # Typed metrics from edgewalker (populated on completion)
    return_pct: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    win_rate_pct: Optional[float] = None
    profit_factor: Optional[float] = None
    total_trades: Optional[int] = None
    equity_final: Optional[float] = None
    equity_peak: Optional[float] = None
    
    # Extra metrics (JSONB for additional data)
    metrics: Optional[dict[str, Any]] = None
    html_report_url: Optional[str] = None
    
    created_at: datetime


class BacktestUpdate(BaseModel):
    """Schema for updating backtest status and results (used by n8n callback).
    
    The n8n workflow should populate typed metrics directly.
    """
    status: Optional[str] = None
    error_message: Optional[str] = None
    
    # Typed metrics (preferred - populate these from edgewalker results)
    return_pct: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    win_rate_pct: Optional[float] = None
    profit_factor: Optional[float] = None
    total_trades: Optional[int] = None
    equity_final: Optional[float] = None
    equity_peak: Optional[float] = None
    
    # Extra metrics JSONB (for any additional data)
    metrics: Optional[dict[str, Any]] = None
    html_report_url: Optional[str] = None


# ─── TRADE SCHEMAS ───


class TradeCreate(BaseModel):
    """Input for creating a trade record (aligned with edgewalker TradeRecord)."""
    entry_time: datetime
    exit_time: Optional[datetime] = None
    direction: str  # "long" or "short"
    size: float  # Position size
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None  # Return percentage
    session_date: Optional[date] = None  # Trading session date
    exit_reason: Optional[str] = None  # e.g., "stop_loss", "take_profit", "eod"
    extra: Optional[Any] = None  # Extra data


class TradeRead(BaseModel):
    id: int
    backtest_id: int
    strategy_id: int
    entry_time: datetime
    exit_time: Optional[datetime] = None
    direction: str
    size: float
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    session_date: Optional[date] = None
    exit_reason: Optional[str] = None
    extra: Optional[Any] = None


# ─── RULE TRIGGER SCHEMAS ───


class RuleTriggerRequest(BaseModel):
    """Request to trigger an agent from a rule during backtest."""
    agent_id: int
    chat_id: int
    rule_context: dict[str, Any]
    webhook_url: Optional[str] = None  # Optional override for backtest scenarios


class RuleTriggerResponse(BaseModel):
    """Response from agent webhook after rule trigger."""
    status: str
    agent_response: Optional[dict[str, Any]] = None
