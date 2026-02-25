"""
Market Data API - OHLCV history, symbols, and indicators.

Provides flexible access to historical market data from multiple sources
(IBKR, Yahoo) with dynamic technical indicator computation using the
same engine as the edgewalker library.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Request, status

from app.schemas.marketdata import (
    OHLCHistoryResponse,
    AvailableSymbolsResponse,
    IndicatorsListResponse,
    IndicatorInfo,
    AssetType,
)
from app.services.marketdata_service import (
    get_ohlc_history,
    MarketDataError,
)
from app.services.indicator_registry import (
    get_all_indicators,
    get_indicator_by_name,
    get_indicator_groups,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/marketdata", tags=["Market Data"])


@router.get("/ohlc-history", response_model=OHLCHistoryResponse)
def get_ohlc_history_endpoint(
    request: Request,
    symbol: str = Query(..., description="Trading symbol (e.g., 'QQQ', 'SPY', 'NQ')"),
    asset_type: AssetType = Query(
        AssetType.STOCK,
        description="Asset type: stock, futures, index, etf"
    ),
    start_date: Optional[date] = Query(None, alias="start", description="Start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, alias="end", description="End date (YYYY-MM-DD)"),
    timeframe: str = Query("5min", description="Bar timeframe: 1min, 5min, 15min, 30min, 1h, 1d"),
    indicators: Optional[str] = Query(
        None, 
        description=(
            "Indicators in JSON format or comma-separated simple format. "
            "Simple: 'SMA:timeperiod=20,MACD:fastperiod=12;slowperiod=26;signalperiod=9,RSI:timeperiod=14' "
            "JSON: '[{\"type\":\"SMA\",\"params\":{\"timeperiod\":20}}]'"
        )
    ),
    fetch: bool = Query(
        False,
        description="If true, fetch fresh data from source before returning. Otherwise use cached files."
    ),
    extended_hours: bool = Query(
        False,
        description="Include pre/after-market data. False = RTH only (09:30-16:00 ET)."
    ),
    connection_id: Optional[int] = Query(
        None,
        description="Connection ID scoping data storage. If None, auto-detects from available data."
    ),
):
    """
    Get OHLC history with optional technical indicators.
    
    Data source (IBKR vs Yahoo) is auto-detected from the connection_id
    by the datasource-historical service.
    
    **Asset Types:**
    - `stock`: Equities (e.g., AAPL, MSFT)
    - `futures`: Futures contracts (e.g., NQ, ES)
    - `index`: Market indices (e.g., SPX)
    - `etf`: ETFs (e.g., QQQ, SPY)
    
    **Timeframes:**
    - `1min`, `5min`, `15min`, `30min`: Intraday bars
    - `1h`: Hourly bars
    - `1d`: Daily bars
    
    **Indicators:**
    All TA-Lib indicators are supported plus custom indicators (vwap, pivot, session_high, session_low).
    Use /marketdata/indicators to get the full list with parameters.
    
    Format: `TYPE:param1=value1;param2=value2` (comma-separated for multiple)
    
    Examples:
    - `SMA:timeperiod=20,EMA:timeperiod=50`
    - `MACD:fastperiod=12;slowperiod=26;signalperiod=9`
    - `BBANDS:timeperiod=20;nbdevup=2;nbdevdn=2`
    - `RSI:timeperiod=14`
    - `vwap` (no params needed)
    
    **Response format:**
    All timestamps are Unix timestamps in seconds, compatible with TradingView Lightweight Charts.
    """
    # ── Track ID propagation (end-to-end tracing) ──
    track_id = request.headers.get("X-Track-Id")
    if track_id:
        logger.info(json.dumps({
            "track_id": track_id,
            "action": "ohlc_history_request",
            "service": "backend",
            "symbol": symbol,
            "asset_type": asset_type.value,
            "timeframe": timeframe,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "fetch": fetch,
        }))

    # Parse indicators from string
    indicator_list = _parse_indicator_string(indicators)
    
    try:
        # NOTE: fetch=True is handled by datasource-historical (Traefik
        # routes /api/marketdata/ohlc-history there with priority 200).
        # The backend only serves from pre-existing partitioned data.
        result = get_ohlc_history(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            indicators=indicator_list,
            connection_id=str(connection_id) if connection_id is not None else None,
            extended_hours=extended_hours,
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
def search_symbols_endpoint(
    query: str = Query(
        ...,
        description="Search query - symbol pattern or company name (e.g., 'QQQ', 'Apple', 'NQ')"
    ),
    connection_id: Optional[int] = Query(
        None,
        description=(
            "Connection ID to scope the search. "
            "For IBKR connections, performs a live search via the gateway. "
            "For Yahoo connections, searches that connection's cached symbols. "
            "If omitted, searches all cached symbols."
        ),
    ),
    asset_type: Optional[AssetType] = Query(
        None,
        description="Filter by asset type: stock, futures, index, etf"
    ),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of results"),
):
    """
    Search for available symbols.
    
    When ``connection_id`` points to an IBKR connection with a running
    gateway container, the search is done **live** via ``reqMatchingSymbols``.
    Otherwise, results come from the local symbol cache.
    
    **Examples:**
    - ``/marketdata/symbols?query=QQQ`` – Search all cached symbols
    - ``/marketdata/symbols?query=Apple&asset_type=stock`` – Search Apple stocks
    - ``/marketdata/symbols?query=NQ&connection_id=1`` – Live search NQ on IBKR connection 1
    """
    try:
        # If connection_id is provided, check if it's an IBKR connection for live search
        if connection_id is not None:
            from app.models.connection import Connection
            from app.db.database import get_session_context
            from sqlmodel import select as sql_select

            with get_session_context() as session:
                conn = session.get(Connection, connection_id)

            if conn and conn.broker_type == "ibkr":
                from app.services.symbol_sync_handler import search_ibkr_symbols_by_id

                results = search_ibkr_symbols_by_id(
                    query=query,
                    connection_id=connection_id,
                    asset_type=asset_type.value if asset_type else None,
                    limit=limit,
                )
                return AvailableSymbolsResponse(
                    symbols=results,
                    source="ibkr",
                    asset_type=asset_type.value if asset_type else "all",
                    count=len(results),
                )

        # Local cache search (all connections or specific one)
        from app.services.symbol_sync_handler import search_cached_symbols
        
        results = search_cached_symbols(
            query=query,
            connection_id=connection_id,
            asset_type=asset_type.value if asset_type else None,
            limit=limit,
        )
        
        return AvailableSymbolsResponse(
            symbols=results,
            source="cache",
            asset_type=asset_type.value if asset_type else "all",
            count=len(results),
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(e),
        )
    except Exception as e:
        logger.exception(f"Error searching symbols for '{query}'")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search symbols: {str(e)}",
        )


@router.get("/indicators", response_model=IndicatorsListResponse)
def list_available_indicators(
    group: Optional[str] = Query(
        None,
        description="Filter by group (e.g., 'Momentum Indicators', 'Overlap Studies', 'Custom')"
    ),
):
    """
    List all available technical indicators with their parameters.
    
    Returns the dynamic list of all indicators available in the system,
    including all TA-Lib indicators and custom edgewalker indicators.
    
    Each indicator includes:
    - **name**: Technical identifier (e.g., 'SMA', 'MACD')
    - **display_name**: Human-readable name
    - **group**: Category (e.g., 'Momentum Indicators', 'Overlap Studies')
    - **description**: Brief explanation
    - **overlay**: Whether it should overlay on price chart (vs separate panel)
    - **inputs**: Required input data types (e.g., close, high/low/close)
    - **parameters**: Configurable parameters with types, defaults, and constraints
    - **outputs**: List of output names (e.g., ['macd', 'signal', 'hist'] for MACD)
    
    **Groups:**
    - Cycle Indicators
    - Math Operators  
    - Math Transform
    - Momentum Indicators
    - Overlap Studies
    - Pattern Recognition
    - Price Transform
    - Statistic Functions
    - Volatility Indicators
    - Volume Indicators
    - Custom (edgewalker-specific)
    """
    try:
        all_indicators = get_all_indicators()
        groups = get_indicator_groups()
        
        # Filter by group if specified
        if group:
            all_indicators = [
                ind for ind in all_indicators 
                if ind.get("group", "").lower() == group.lower()
            ]
        
        return IndicatorsListResponse(
            indicators=[IndicatorInfo(**ind) for ind in all_indicators],
            groups=groups,
            count=len(all_indicators),
        )
    except Exception as e:
        logger.exception("Error listing indicators")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list indicators: {str(e)}",
        )


@router.get("/indicators/{indicator_name}")
def get_indicator_info(indicator_name: str):
    """
    Get detailed information about a specific indicator.
    
    **Parameters:**
    - `indicator_name`: Indicator type (e.g., 'SMA', 'MACD', 'vwap')
    
    Returns full parameter schema and usage information.
    """
    try:
        info = get_indicator_by_name(indicator_name)
        if not info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Indicator '{indicator_name}' not found",
            )
        return info
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting indicator info for '{indicator_name}'")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get indicator info: {str(e)}",
        )


def _parse_indicator_string(indicators_str: Optional[str]) -> Optional[list[dict[str, Any]]]:
    """Parse indicator string into list of indicator configs.
    
    Supports two formats:
    1. Simple: "SMA:timeperiod=20,MACD:fastperiod=12;slowperiod=26"
    2. JSON: '[{"type":"SMA","params":{"timeperiod":20}}]'
    3. Legacy: "sma_20,rsi_14" (underscore format)
    
    Returns list of {"type": str, "params": dict} dicts.
    """
    if not indicators_str:
        return None
    
    indicators_str = indicators_str.strip()
    
    # Try JSON format first
    if indicators_str.startswith("["):
        try:
            parsed = json.loads(indicators_str)
            return parsed
        except json.JSONDecodeError:
            pass
    
    result = []
    for part in indicators_str.split(","):
        part = part.strip()
        if not part:
            continue
        
        if ":" in part:
            # New format: TYPE:param1=val1;param2=val2
            type_part, params_part = part.split(":", 1)
            params = {}
            for param in params_part.split(";"):
                if "=" in param:
                    key, val = param.split("=", 1)
                    # Try to convert to number
                    try:
                        params[key.strip()] = int(val)
                    except ValueError:
                        try:
                            params[key.strip()] = float(val)
                        except ValueError:
                            params[key.strip()] = val.strip()
            result.append({"type": type_part.strip().upper(), "params": params})
        elif "_" in part:
            # Legacy format: sma_20, macd_12_26_9, bbands_20_2
            parts = part.lower().split("_")
            indicator_type = parts[0].upper()
            
            # Map legacy params to proper names based on indicator type
            if indicator_type in ("SMA", "EMA", "RSI", "ATR", "WMA", "DEMA", "TEMA", "ADX", "CCI"):
                params = {"timeperiod": int(parts[1])} if len(parts) > 1 else {}
            elif indicator_type == "BBANDS":
                params = {}
                if len(parts) > 1:
                    params["timeperiod"] = int(parts[1])
                if len(parts) > 2:
                    params["nbdevup"] = float(parts[2])
                    params["nbdevdn"] = float(parts[2])
            elif indicator_type == "MACD":
                params = {}
                if len(parts) > 1:
                    params["fastperiod"] = int(parts[1])
                if len(parts) > 2:
                    params["slowperiod"] = int(parts[2])
                if len(parts) > 3:
                    params["signalperiod"] = int(parts[3])
            elif indicator_type == "STOCH":
                params = {}
                if len(parts) > 1:
                    params["fastk_period"] = int(parts[1])
                if len(parts) > 2:
                    params["slowk_period"] = int(parts[2])
                if len(parts) > 3:
                    params["slowd_period"] = int(parts[3])
            else:
                params = {}
            
            result.append({"type": indicator_type, "params": params})
        else:
            # No params (e.g., just "vwap")
            result.append({"type": part.upper(), "params": {}})
    
    return result if result else None
