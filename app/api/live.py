"""
Live Trading API Endpoints.

Manages live strategy execution via Docker containers.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlmodel import Session

from app.db.database import get_session
from app.models.strategy import LiveStatus
from app.services.live_runner_service import live_runner_service
from app.services.strategy_service import get_strategy

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/live", tags=["Live Trading"])


# ─── REQUEST/RESPONSE MODELS ───


class LiveStartRequest(BaseModel):
    """Request to start a live strategy."""
    symbol: str
    timeframe: str = "5s"  # Bar aggregation interval
    tick_eval: bool = True  # Evaluate rules on every tick
    debug_rules: bool = False  # Log per-condition evaluation detail
    account_id: int | None = None  # Trading account to send orders to


class LiveStartResponse(BaseModel):
    """Response after starting a live strategy."""
    status: str
    container_id: str | None = None
    container_name: str | None = None
    symbol: str | None = None
    timeframe: str | None = None
    message: str | None = None


class LiveStatusResponse(BaseModel):
    """Status of a live strategy runner."""
    # Database state
    live_status: str
    symbol: str | None = None
    timeframe: str | None = None
    started_at: str | None = None
    stopped_at: str | None = None
    error_message: str | None = None
    metrics: dict | None = None
    # Container state
    container_status: str | None = None
    container_id: str | None = None
    container_name: str | None = None
    container_health: str | None = None
    strategy_id: int | None = None


class LiveStopResponse(BaseModel):
    """Response after stopping a live strategy."""
    status: str
    message: str | None = None


class LiveLogsResponse(BaseModel):
    """Container logs response."""
    logs: str
    strategy_id: int


class RunningStrategyInfo(BaseModel):
    """Info about a running strategy."""
    container_id: str
    container_name: str
    strategy_id: str | None = None
    symbol: str | None = None
    status: str


# ─── ENDPOINTS ───


@router.post("/strategies/{strategy_id}/start", response_model=LiveStartResponse)
def start_live_strategy(
    strategy_id: int,
    request: LiveStartRequest,
    session: Session = Depends(get_session),
):
    """
    Start a live strategy runner.
    
    Creates a Docker container for the strategy that:
    - Subscribes to Redis for real-time market data
    - Executes the strategy logic (same as backtest)
    - Exposes an HTTP endpoint for status/control
    
    The container is routed via Traefik at /api/runners/{strategy_id}/
    """
    # Verify strategy exists
    strategy = get_strategy(session, strategy_id)
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found",
        )
    
    # Check if already running
    if strategy.live_status == LiveStatus.RUNNING.value:
        return LiveStartResponse(
            status="already_running",
            container_id=strategy.live_container_id,
            symbol=strategy.live_symbol,
            timeframe=strategy.live_timeframe,
            message="Strategy is already running",
        )
    
    try:
        # Update status to starting
        strategy.live_status = LiveStatus.STARTING.value
        strategy.live_symbol = request.symbol
        strategy.live_timeframe = request.timeframe
        strategy.live_account_id = request.account_id
        strategy.live_error_message = None
        session.add(strategy)
        session.commit()
        
        result = live_runner_service.start_strategy(
            strategy_id=strategy_id,
            strategy_config=strategy.definition,
            symbol=request.symbol,
            timeframe=request.timeframe,
            tick_eval=request.tick_eval,
            debug_rules=request.debug_rules,
        )
        
        if result["status"] == "already_running":
            # Sync DB with actual container state
            strategy.live_status = LiveStatus.RUNNING.value
            strategy.live_container_id = result.get("container_id")
            session.add(strategy)
            session.commit()
            
            return LiveStartResponse(
                status="already_running",
                container_id=result.get("container_id"),
                container_name=result.get("container_name"),
                message="Strategy is already running",
            )
        
        # Update DB with running state
        strategy.live_status = LiveStatus.RUNNING.value
        strategy.live_container_id = result.get("container_id")
        strategy.live_started_at = datetime.now(timezone.utc)
        strategy.live_stopped_at = None
        session.add(strategy)
        session.commit()
        
        return LiveStartResponse(
            status="started",
            container_id=result.get("container_id"),
            container_name=result.get("container_name"),
            symbol=request.symbol,
            timeframe=request.timeframe,
        )
        
    except Exception as e:
        # Update DB with error state
        strategy.live_status = LiveStatus.ERROR.value
        strategy.live_error_message = str(e)
        session.add(strategy)
        session.commit()
        
        logger.exception(f"Failed to start strategy {strategy_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/strategies/{strategy_id}/stop", response_model=LiveStopResponse)
def stop_live_strategy(
    strategy_id: int,
    remove: bool = True,
    session: Session = Depends(get_session),
):
    """
    Stop a live strategy runner.
    
    Stops and optionally removes the Docker container.
    """
    # Verify strategy exists
    strategy = get_strategy(session, strategy_id)
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found",
        )
    
    try:
        # Update status to stopping
        strategy.live_status = LiveStatus.STOPPING.value
        session.add(strategy)
        session.commit()
        
        result = live_runner_service.stop_strategy(
            strategy_id=strategy_id,
            remove=remove,
        )
        
        # Update DB with stopped state
        strategy.live_status = LiveStatus.STOPPED.value
        strategy.live_container_id = None
        strategy.live_stopped_at = datetime.now(timezone.utc)
        session.add(strategy)
        session.commit()
        
        if result["status"] == "not_found":
            return LiveStopResponse(
                status="stopped",
                message=f"Strategy {strategy_id} was not running (container not found)",
            )
        
        return LiveStopResponse(
            status=result["status"],
            message=f"Strategy {strategy_id} stopped successfully",
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Update DB with error state
        strategy.live_status = LiveStatus.ERROR.value
        strategy.live_error_message = str(e)
        session.add(strategy)
        session.commit()
        
        logger.exception(f"Failed to stop strategy {strategy_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("/strategies/{strategy_id}/status", response_model=LiveStatusResponse)
def get_live_strategy_status(
    strategy_id: int,
    session: Session = Depends(get_session),
):
    """
    Get status of a live strategy runner.
    
    Returns both database state and container state.
    """
    # Get strategy from DB
    strategy = get_strategy(session, strategy_id)
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found",
        )
    
    # Get container status
    container_info = live_runner_service.get_strategy_status(strategy_id)
    
    # Sync DB if container state differs
    if container_info["status"] == "not_found" and strategy.live_status == LiveStatus.RUNNING.value:
        # Container died unexpectedly
        strategy.live_status = LiveStatus.ERROR.value
        strategy.live_error_message = "Container not found (crashed or removed)"
        strategy.live_container_id = None
        session.add(strategy)
        session.commit()
    
    return LiveStatusResponse(
        live_status=strategy.live_status,
        symbol=strategy.live_symbol,
        timeframe=strategy.live_timeframe,
        started_at=strategy.live_started_at.isoformat() if strategy.live_started_at else None,
        stopped_at=strategy.live_stopped_at.isoformat() if strategy.live_stopped_at else None,
        error_message=strategy.live_error_message,
        metrics=strategy.live_metrics,
        container_status=container_info.get("status"),
        container_id=container_info.get("container_id"),
        container_name=container_info.get("container_name"),
        container_health=container_info.get("health_status"),
        strategy_id=strategy_id,
    )


@router.get("/strategies/{strategy_id}/logs", response_model=LiveLogsResponse)
def get_live_strategy_logs(
    strategy_id: int,
    tail: int = 100,
    session: Session = Depends(get_session),
):
    """
    Get logs from a live strategy runner container.
    """
    try:
        logs = live_runner_service.get_container_logs(
            strategy_id=strategy_id,
            tail=tail,
        )
        
        return LiveLogsResponse(
            logs=logs,
            strategy_id=strategy_id,
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.exception(f"Failed to get logs for strategy {strategy_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("/strategies", response_model=list[RunningStrategyInfo])
def list_running_strategies():
    """
    List all running live strategy containers.
    """
    try:
        result = live_runner_service.list_running_strategies()
        return [RunningStrategyInfo(**info) for info in result]
        
    except Exception as e:
        logger.exception("Failed to list running strategies")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
