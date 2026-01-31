"""
Schemas for Market Data API.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class DataSourceType(str, Enum):
    """Data source for market data."""
    IBKR = "ibkr"
    YAHOO = "yahoo"


class AssetType(str, Enum):
    """Type of asset/instrument."""
    STOCK = "stock"
    FUTURES = "futures"
    INDEX = "index"
    ETF = "etf"


class IndicatorConfig(BaseModel):
    """Configuration for a single indicator."""
    type: str = Field(..., description="Indicator type (e.g., 'SMA', 'MACD', 'vwap')")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Indicator parameters (e.g., {'timeperiod': 20})"
    )
    name: Optional[str] = Field(
        None,
        description="Custom name for the indicator output (defaults to type_params)"
    )


class OHLCHistoryRequest(BaseModel):
    """Request parameters for OHLC history endpoint."""
    symbol: str = Field(..., description="Trading symbol (e.g., 'QQQ', 'SPY')")
    source: DataSourceType = Field(
        DataSourceType.YAHOO,
        description="Data source to fetch from"
    )
    asset_type: AssetType = Field(
        AssetType.STOCK,
        description="Type of asset"
    )
    start_date: Optional[date] = Field(None, description="Start date (inclusive)")
    end_date: Optional[date] = Field(None, description="End date (inclusive)")
    timeframe: str = Field("5min", description="Bar timeframe: 1min, 5min, 15min, 30min, 1h, 1d")
    indicators: Optional[list[IndicatorConfig]] = Field(
        None, 
        description="List of indicators to compute with full configuration"
    )
    # Legacy support for simple indicator strings
    indicator_strings: Optional[list[str]] = Field(
        None,
        alias="indicator_list",
        description="Legacy: comma-separated indicator specs like 'sma_20,rsi_14'"
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
    symbols: list[dict[str, Any]] = Field(
        ...,
        description="List of symbol info dicts with name, description, type, etc."
    )
    source: str = Field(..., description="Data source queried")
    asset_type: str = Field(..., description="Asset type filter used")
    count: int = Field(..., description="Number of symbols returned")


class SymbolSearchRequest(BaseModel):
    """Request parameters for symbol search."""
    query: str = Field(..., description="Search query (symbol pattern or name)")
    source: DataSourceType = Field(
        DataSourceType.YAHOO,
        description="Data source to search"
    )
    asset_type: Optional[AssetType] = Field(
        None,
        description="Filter by asset type"
    )
    limit: int = Field(50, description="Maximum number of results", ge=1, le=500)


class IndicatorParameterInfo(BaseModel):
    """Information about an indicator parameter."""
    type: str = Field(..., description="Parameter type: integer, number, string")
    default: Any = Field(..., description="Default value")
    min: Optional[float] = Field(None, description="Minimum value (for numbers)")
    max: Optional[float] = Field(None, description="Maximum value (for numbers)")
    options: Optional[list[str]] = Field(None, description="Valid options (for strings)")
    description: Optional[str] = Field(None, description="Parameter description")


class IndicatorInfo(BaseModel):
    """Detailed information about a technical indicator."""
    name: str = Field(..., description="Indicator name (e.g., 'SMA', 'MACD')")
    display_name: str = Field(..., description="Human-readable name")
    group: str = Field(..., description="Category group")
    description: str = Field(..., description="Brief description")
    overlay: bool = Field(..., description="Whether it overlays on price chart")
    inputs: dict[str, list[str]] = Field(
        ...,
        description="Required input data types"
    )
    parameters: dict[str, dict[str, Any]] = Field(
        ...,
        description="Configurable parameters with type/default/constraints"
    )
    outputs: list[str] = Field(..., description="Output names")


class IndicatorsListResponse(BaseModel):
    """Response listing all available indicators."""
    indicators: list[IndicatorInfo]
    groups: dict[str, list[str]] = Field(
        ...,
        description="Indicators grouped by category"
    )
    count: int = Field(..., description="Total number of indicators")


class FetchResultResponse(BaseModel):
    """Response from a fetch operation."""
    success: bool
    symbol: str
    source: str
    output_path: Optional[str] = None
    rows_fetched: int
    rows_total: int
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    interval: str
    updated: bool
    message: Optional[str] = None
    error: Optional[str] = None
    duration_seconds: Optional[float] = None
