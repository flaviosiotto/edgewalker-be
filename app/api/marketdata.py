"""
Market Data API - OHLCV history and indicators.

Provides flexible access to historical market data from parquet files
with optional technical indicator computation.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status

from app.schemas.marketdata import (
    OHLCHistoryResponse,
    AvailableSymbolsResponse,
)
from app.services.marketdata_service import (
    get_ohlc_history,
    MarketDataError,
    EDGEWALKER_DATA_DIR,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/marketdata", tags=["Market Data"])


@router.get("/ohlc-history", response_model=OHLCHistoryResponse)
def get_ohlc_history_endpoint(
    symbol: str = Query(..., description="Trading symbol (e.g., 'QQQ', 'SPY')"),
    start_date: Optional[date] = Query(None, alias="start", description="Start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, alias="end", description="End date (YYYY-MM-DD)"),
    timeframe: str = Query("5min", description="Bar timeframe: 1min, 5min, 15min, 30min, 1h, 1d"),
    indicators: Optional[str] = Query(
        None, 
        description="Comma-separated indicators: sma_20,ema_50,rsi_14,bbands_20_2,macd_12_26_9,vwap"
    ),
):
    """
    Get OHLC history with optional technical indicators.
    
    **Timeframes:**
    - `1min`, `5min`, `15min`, `30min`: Intraday bars
    - `1h`: Hourly bars
    - `1d`: Daily bars
    
    **Indicators:**
    - `sma_N`: Simple Moving Average with period N (e.g., sma_20)
    - `ema_N`: Exponential Moving Average with period N (e.g., ema_50)
    - `rsi_N`: Relative Strength Index with period N (e.g., rsi_14)
    - `bbands_N_S`: Bollinger Bands with period N and S std devs (e.g., bbands_20_2)
    - `macd_F_S_G`: MACD with fast F, slow S, signal G (e.g., macd_12_26_9)
    - `vwap`: Volume Weighted Average Price (resets daily)
    
    **Response format:**
    All timestamps are Unix timestamps in seconds, compatible with TradingView Lightweight Charts.
    """
    # Parse indicators from comma-separated string
    indicator_list = None
    if indicators:
        indicator_list = [i.strip() for i in indicators.split(",") if i.strip()]
    
    try:
        result = get_ohlc_history(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            indicators=indicator_list,
        )
        return result
    except MarketDataError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.exception(f"Error fetching OHLC history for {symbol}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load market data: {str(e)}",
        )


@router.get("/symbols", response_model=AvailableSymbolsResponse)
def list_available_symbols():
    """
    List all available symbols (parquet files in data directory).
    """
    try:
        data_dir = Path(EDGEWALKER_DATA_DIR)
        if not data_dir.exists():
            return AvailableSymbolsResponse(
                symbols=[],
                data_directory=str(data_dir),
            )
        
        # Find all parquet files and extract symbol names
        symbols = set()
        for f in data_dir.glob("*.parquet"):
            # Extract symbol from filename (e.g., QQQ.parquet or QQQ_train.parquet)
            name = f.stem
            # Remove common suffixes
            for suffix in ["_train", "_test", "_backtest", "_live"]:
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
                    break
            symbols.add(name.upper())
        
        return AvailableSymbolsResponse(
            symbols=sorted(symbols),
            data_directory=str(data_dir),
        )
    except Exception as e:
        logger.exception("Error listing available symbols")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list symbols: {str(e)}",
        )


@router.get("/indicators")
def list_available_indicators():
    """
    List all supported technical indicators with their parameters schema.
    
    This endpoint is used by the frontend to dynamically build indicator configuration forms.
    Each indicator has:
    - type: The indicator type code (used in strategy definition)
    - name: Human-readable name
    - description: Brief explanation
    - overlay: Whether it overlays on price chart (True) or separate panel (False)
    - parameters: List of configurable parameters with type, default, min, max, etc.
    """
    return {
        "indicators": [
            {
                "type": "SMA",
                "name": "Simple Moving Average",
                "description": "Average price over N periods",
                "overlay": True,
                "parameters": [
                    {
                        "name": "period",
                        "label": "Period",
                        "type": "integer",
                        "default": 20,
                        "min": 1,
                        "max": 500,
                        "required": True,
                    }
                ],
            },
            {
                "type": "EMA",
                "name": "Exponential Moving Average",
                "description": "Exponentially weighted average price",
                "overlay": True,
                "parameters": [
                    {
                        "name": "period",
                        "label": "Period",
                        "type": "integer",
                        "default": 20,
                        "min": 1,
                        "max": 500,
                        "required": True,
                    }
                ],
            },
            {
                "type": "RSI",
                "name": "Relative Strength Index",
                "description": "Momentum oscillator (0-100)",
                "overlay": False,
                "parameters": [
                    {
                        "name": "period",
                        "label": "Period",
                        "type": "integer",
                        "default": 14,
                        "min": 2,
                        "max": 100,
                        "required": True,
                    }
                ],
            },
            {
                "type": "BBANDS",
                "name": "Bollinger Bands",
                "description": "Volatility bands around moving average",
                "overlay": True,
                "parameters": [
                    {
                        "name": "period",
                        "label": "Period",
                        "type": "integer",
                        "default": 20,
                        "min": 2,
                        "max": 500,
                        "required": True,
                    },
                    {
                        "name": "std_dev",
                        "label": "Std Deviations",
                        "type": "number",
                        "default": 2.0,
                        "min": 0.1,
                        "max": 5.0,
                        "step": 0.1,
                        "required": False,
                    },
                ],
            },
            {
                "type": "MACD",
                "name": "MACD",
                "description": "Moving Average Convergence Divergence",
                "overlay": False,
                "parameters": [
                    {
                        "name": "fast_period",
                        "label": "Fast Period",
                        "type": "integer",
                        "default": 12,
                        "min": 1,
                        "max": 100,
                        "required": True,
                    },
                    {
                        "name": "slow_period",
                        "label": "Slow Period",
                        "type": "integer",
                        "default": 26,
                        "min": 1,
                        "max": 200,
                        "required": True,
                    },
                    {
                        "name": "signal_period",
                        "label": "Signal Period",
                        "type": "integer",
                        "default": 9,
                        "min": 1,
                        "max": 100,
                        "required": True,
                    },
                ],
            },
            {
                "type": "VWAP",
                "name": "VWAP",
                "description": "Volume Weighted Average Price (resets daily)",
                "overlay": True,
                "parameters": [],
            },
            {
                "type": "ATR",
                "name": "Average True Range",
                "description": "Volatility indicator",
                "overlay": False,
                "parameters": [
                    {
                        "name": "period",
                        "label": "Period",
                        "type": "integer",
                        "default": 14,
                        "min": 1,
                        "max": 100,
                        "required": True,
                    }
                ],
            },
            {
                "type": "STOCH",
                "name": "Stochastic Oscillator",
                "description": "Momentum oscillator comparing close to high-low range",
                "overlay": False,
                "parameters": [
                    {
                        "name": "k_period",
                        "label": "%K Period",
                        "type": "integer",
                        "default": 14,
                        "min": 1,
                        "max": 100,
                        "required": True,
                    },
                    {
                        "name": "d_period",
                        "label": "%D Period",
                        "type": "integer",
                        "default": 3,
                        "min": 1,
                        "max": 50,
                        "required": True,
                    },
                ],
            },
            {
                "type": "ADX",
                "name": "Average Directional Index",
                "description": "Trend strength indicator",
                "overlay": False,
                "parameters": [
                    {
                        "name": "period",
                        "label": "Period",
                        "type": "integer",
                        "default": 14,
                        "min": 1,
                        "max": 100,
                        "required": True,
                    }
                ],
            },
            {
                "type": "CCI",
                "name": "Commodity Channel Index",
                "description": "Momentum oscillator measuring deviation from average",
                "overlay": False,
                "parameters": [
                    {
                        "name": "period",
                        "label": "Period",
                        "type": "integer",
                        "default": 20,
                        "min": 1,
                        "max": 100,
                        "required": True,
                    }
                ],
            },
            {
                "type": "WMA",
                "name": "Weighted Moving Average",
                "description": "Linearly weighted moving average",
                "overlay": True,
                "parameters": [
                    {
                        "name": "period",
                        "label": "Period",
                        "type": "integer",
                        "default": 20,
                        "min": 1,
                        "max": 500,
                        "required": True,
                    }
                ],
            },
            {
                "type": "DEMA",
                "name": "Double Exponential Moving Average",
                "description": "Faster-reacting EMA variant",
                "overlay": True,
                "parameters": [
                    {
                        "name": "period",
                        "label": "Period",
                        "type": "integer",
                        "default": 20,
                        "min": 1,
                        "max": 500,
                        "required": True,
                    }
                ],
            },
            {
                "type": "TEMA",
                "name": "Triple Exponential Moving Average",
                "description": "Even faster-reacting EMA variant",
                "overlay": True,
                "parameters": [
                    {
                        "name": "period",
                        "label": "Period",
                        "type": "integer",
                        "default": 20,
                        "min": 1,
                        "max": 500,
                        "required": True,
                    }
                ],
            },
        ]
    }
