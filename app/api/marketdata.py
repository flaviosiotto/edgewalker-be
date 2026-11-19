"""Market Data API - symbols, option chains (gateway proxy) and indicator catalog."""
from __future__ import annotations
from edgewalker_platform.brokers import has_gateway

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.models.user import User
from app.utils.auth_utils import get_current_active_user
from app.schemas.marketdata import (
    AvailableSymbolsResponse,
    IndicatorsListResponse,
    IndicatorInfo,
    AssetType,
)
from app.schemas.options import OptionPayoffRequest, OptionPayoffResponse
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
            "For gateway-backed connections, performs a live search via the gateway. "
            "For Yahoo connections, searches that connection's cached symbols. "
            "If omitted, searches all cached symbols."
        ),
    ),
    asset_type: Optional[AssetType] = Query(
        None,
        description="Filter by asset type: stock, forex, futures, index, etf"
    ),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of results"),
):
    """
    Search for available symbols.
    
    When ``connection_id`` points to a gateway-backed connection, the search is
    done **live** via that broker's symbol search API.
    Otherwise, results come from the local symbol cache.
    
    **Examples:**
    - ``/marketdata/symbols?query=QQQ`` – Search all cached symbols
    - ``/marketdata/symbols?query=Apple&asset_type=stock`` – Search Apple stocks
    - ``/marketdata/symbols?query=NQ&connection_id=1`` – Live search NQ on IBKR connection 1
    """
    try:
        # If connection_id is provided, use live gateway search when available.
        if connection_id is not None:
            from app.models.connection import Connection
            from app.db.database import get_session_context
            from sqlmodel import select as sql_select

            with get_session_context() as session:
                conn = session.get(Connection, connection_id)

            if conn and has_gateway(conn.broker_type):
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


# ── Options (proxied to the connection's gateway) ──────────────────────────
# Contract: devops docs/valutazione-opzioni.md §11. The body is the gateway's,
# untouched. Errors: 404 connection/account not the caller's, 409 with an
# ``OPTIONS_*`` detail (broker without options, or the gateway's own 409),
# 400 invalid parameters, 424 gateway unreachable / broker not connected.


async def _options_read(current_user: User, connection_id: int, account_id: int | None, read):
    from app.services import options_service as svc
    from app.services.gateway_client import GatewayError

    try:
        target = svc.resolve_options_target(
            user_id=current_user.id, connection_id=connection_id, account_id=account_id
        )
    except svc.OptionsNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except svc.OptionsRequestError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except svc.OptionsUnsupportedError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=exc.detail) from exc

    try:
        return await read(svc, target)
    except GatewayError as exc:
        status_code, detail = svc.http_error_for(exc)
        if status_code >= 424:
            logger.warning("Options read on connection %s failed: %s", connection_id, exc)
        raise HTTPException(status_code=status_code, detail=detail) from exc


@router.get("/options/chain")
async def get_option_chain_endpoint(
    connection_id: int = Query(..., description="Connection whose gateway serves the chain"),
    underlying: str = Query(..., min_length=1, max_length=32, description="Underlying symbol, e.g. MNQZ6"),
    current_user: User = Depends(get_current_active_user),
):
    """Option chain definition of an underlying: expiries, strikes, multiplier."""
    return await _options_read(
        current_user, connection_id, None,
        lambda svc, target: svc.get_option_chain(target, underlying=underlying),
    )


@router.get("/options/quotes")
async def get_option_quotes_endpoint(
    connection_id: int = Query(..., description="Connection whose gateway serves the quotes"),
    underlying: str = Query(..., min_length=1, max_length=32, description="Underlying symbol, e.g. MNQZ6"),
    expiry: Optional[str] = Query(None, description="Expiry (YYYY-MM-DD) of the strike window"),
    strikes_around_atm: Optional[int] = Query(
        None, ge=1, le=25, description="Strikes per side around the underlying price (gateway default 10)"
    ),
    symbols: Optional[str] = Query(
        None, description="Comma-separated canonical option symbols to include anyway"
    ),
    current_user: User = Depends(get_current_active_user),
):
    """Quotes snapshot of a strike window (prices, greeks, market-data state)."""
    return await _options_read(
        current_user, connection_id, None,
        lambda svc, target: svc.get_option_quotes(
            target,
            underlying=underlying,
            expiry=expiry,
            strikes_around_atm=strikes_around_atm,
            symbols=symbols,
        ),
    )


@router.get("/options/readiness")
async def get_option_readiness_endpoint(
    connection_id: int = Query(..., description="Connection whose gateway runs the checks"),
    underlying: str = Query(..., min_length=1, max_length=32, description="Underlying symbol, e.g. MNQZ6"),
    account_id: Optional[int] = Query(
        None, description="Internal account id (must belong to the connection) for the trading-permission check"
    ),
    current_user: User = Depends(get_current_active_user),
):
    """Whether chain, option market data and option trading are usable."""
    return await _options_read(
        current_user, connection_id, account_id,
        lambda svc, target: svc.get_option_readiness(target, underlying=underlying),
    )


@router.post("/options/payoff", response_model=OptionPayoffResponse)
def option_payoff_endpoint(
    request: OptionPayoffRequest,
    current_user: User = Depends(get_current_active_user),
):
    """Model value (Black-76) of a set of option legs before expiry.

    Pure computation on the request body — no gateway, no broker: P&L of the
    set over a grid of underlying prices at ``valuation_at``, and its net
    greeks at the current underlying price. Position greeks are in currency
    (× quantity × side × multiplier): theta per calendar day, vega per
    volatility point. Legs that cannot be valued are listed in ``excluded``.
    400: legs on different underlyings, ``valuation_at`` in the past.
    """
    from app.services.option_payoff_service import OptionPayoffError, compute_payoff

    try:
        return compute_payoff(request)
    except OptionPayoffError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


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
    - **outputs**: List of output names (e.g., ['macd', 'signal', 'histogram'] for MACD)
    
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
