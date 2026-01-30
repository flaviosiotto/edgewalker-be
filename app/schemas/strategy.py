from datetime import datetime, date
from typing import Any, Optional

from pydantic import BaseModel


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


class StrategyUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    definition: Optional[Any] = None


# ─── BACKTEST SCHEMAS ───


class BacktestCreate(BaseModel):
    """Input parameters for creating a new backtest (status=pending)."""
    symbol: str
    start_date: date
    end_date: date
    agent_id: Optional[int] = None  # Agent to execute this backtest via n8n
    parameters: Optional[dict[str, Any]] = None
    # parameters can include: initial_capital, commission, slippage, config overrides...


class BacktestRead(BaseModel):
    """Full backtest representation including status and results."""
    id: int
    strategy_id: int
    agent_id: Optional[int] = None
    
    # Input
    symbol: str
    start_date: date
    end_date: date
    parameters: Optional[dict[str, Any]] = None
    
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
