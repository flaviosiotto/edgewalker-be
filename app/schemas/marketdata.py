"""
Schemas for Market Data API.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Optional

from pydantic import BaseModel, Field


class OHLCHistoryRequest(BaseModel):
    """Request parameters for OHLC history endpoint."""
    symbol: str = Field(..., description="Trading symbol (e.g., 'QQQ', 'SPY')")
    start_date: Optional[date] = Field(None, description="Start date (inclusive)")
    end_date: Optional[date] = Field(None, description="End date (inclusive)")
    timeframe: str = Field("5min", description="Bar timeframe: 1min, 5min, 15min, 30min, 1h, 1d")
    indicators: Optional[list[str]] = Field(
        None, 
        description="List of indicators to compute: sma_20, ema_50, rsi_14, bbands_20_2, macd_12_26_9, vwap"
    )


class CandleData(BaseModel):
    """Single candlestick data point for lightweight-charts."""
    time: int = Field(..., description="Unix timestamp in seconds")
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None


class LineDataPoint(BaseModel):
    """Single line indicator data point."""
    time: int = Field(..., description="Unix timestamp in seconds")
    value: float


class BandsDataPoint(BaseModel):
    """Bollinger Bands or similar multi-line indicator data point."""
    time: int
    upper: Optional[float] = None
    middle: float
    lower: Optional[float] = None


class MACDDataPoint(BaseModel):
    """MACD indicator data point."""
    time: int
    macd: float
    signal: Optional[float] = None
    histogram: Optional[float] = None


class OHLCHistoryResponse(BaseModel):
    """Response from OHLC history endpoint."""
    symbol: str
    timeframe: str
    count: int = Field(..., description="Number of candles returned")
    candles: list[CandleData]
    indicators: dict[str, list[Any]] = Field(
        default_factory=dict,
        description="Computed indicators keyed by name (e.g., 'sma_20')"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "QQQ",
                "timeframe": "5min",
                "count": 100,
                "candles": [
                    {"time": 1704067200, "open": 400.0, "high": 401.5, "low": 399.5, "close": 401.0, "volume": 10000}
                ],
                "indicators": {
                    "sma_20": [{"time": 1704067200, "value": 400.5}],
                    "rsi_14": [{"time": 1704067200, "value": 55.2}]
                }
            }
        }


class AvailableSymbolsResponse(BaseModel):
    """Response listing available symbols."""
    symbols: list[str]
    data_directory: str
