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


class BacktestCreate(BaseModel):
    symbol: str
    start_date: date
    end_date: date
    parameters: Optional[Any] = None
    metrics: Optional[Any] = None


class BacktestRead(BaseModel):
    id: int
    strategy_id: int
    symbol: str
    start_date: date
    end_date: date
    parameters: Optional[Any] = None
    metrics: Optional[Any] = None
    created_at: datetime


class TradeCreate(BaseModel):
    ts_open: datetime
    ts_close: Optional[datetime] = None
    side: str
    quantity: float
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    fees: Optional[float] = None
    meta: Optional[Any] = None


class TradeRead(BaseModel):
    id: int
    backtest_id: int
    strategy_id: int
    ts_open: datetime
    ts_close: Optional[datetime] = None
    side: str
    quantity: float
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    fees: Optional[float] = None
    meta: Optional[Any] = None
