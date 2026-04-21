"""Market Data API - symbols and indicator catalog."""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status

from app.schemas.marketdata import (
    AvailableSymbolsResponse,
    IndicatorsListResponse,
    IndicatorInfo,
    AssetType,
)
from app.services.indicator_registry import (
    get_all_indicators,
    get_indicator_by_name,
    get_indicator_groups,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/marketdata", tags=["Market Data"])


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

            if conn and conn.broker_type in ("ibkr", "binance"):
                from app.services.symbol_sync_handler import search_gateway_symbols_by_id

                results = search_gateway_symbols_by_id(
                    query=query,
                    connection_id=connection_id,
                    broker_type=conn.broker_type,
                    asset_type=asset_type.value if asset_type else None,
                    limit=limit,
                )
                return AvailableSymbolsResponse(
                    symbols=results,
                    source=conn.broker_type,
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
