"""
Live Trading API Endpoints.

Manages live strategy execution via Docker containers.
Includes endpoints for orders, trades, positions, and reconciliation.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlmodel import Session

from app.db.database import get_session
from app.models.strategy import LiveStatus
from app.schemas.live_trading import (
    LiveOrderCreate,
    LiveOrderRead,
    LiveOrderUpdate,
    LivePositionCreate,
    LivePositionRead,
    LivePositionUpdate,
    LiveTradeCreate,
    LiveTradeRead,
    ReconciliationReport,
)
from app.services.live_runner_service import live_runner_service
from app.services.live_trading_service import (
    create_live_order,
    create_live_trade,
    get_live_order,
    list_active_orders,
    list_live_orders,
    list_live_trades,
    list_open_positions,
    list_positions,
    reconcile_on_startup,
    update_live_order,
    upsert_position,
    validate_account_for_live,
)
from app.services.strategy_service import get_strategy
from app.services.strategy_service import (
    notify_manager_live_start,
    post_manager_message,
    get_or_create_live_chat,
)

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
    
    # Validate account if provided
    account_config: dict[str, Any] | None = None
    broker_type: str | None = None
    if request.account_id is not None:
        try:
            account, connection = validate_account_for_live(session, request.account_id)
            account_config = {
                "account_id": account.account_id,  # broker-specific code (e.g. DU1234567)
                "db_account_id": account.id,
                "connection_id": connection.id,
                "connection_name": connection.name,
                **(connection.config or {}),
            }
            broker_type = connection.broker_type
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
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
        
        # Run reconciliation before starting
        if request.account_id is not None:
            recon_report = reconcile_on_startup(
                session,
                strategy_id=strategy_id,
                account_id=request.account_id,
            )
            if recon_report.items:
                logger.info(
                    "Pre-start reconciliation for strategy %s: %s",
                    strategy_id, recon_report.summary,
                )
        
        # Resolve manager agent webhook URL for the runner
        _mgr_webhook: str | None = None
        _mgr_session_id: str | None = None
        if strategy.manager_agent_id:
            from app.models.agent import Agent
            _mgr_agent = session.get(Agent, strategy.manager_agent_id)
            if _mgr_agent:
                _mgr_webhook = _mgr_agent.n8n_webhook
            # Pre-create the live chat so the runner has its session_id
            _live_chat = get_or_create_live_chat(session, strategy_id)
            _mgr_session_id = _live_chat.n8n_session_id

        result = live_runner_service.start_strategy(
            strategy_id=strategy_id,
            strategy_config=strategy.definition,
            symbol=request.symbol,
            timeframe=request.timeframe,
            tick_eval=request.tick_eval,
            debug_rules=request.debug_rules,
            account_config=account_config,
            broker_type=broker_type,
            manager_webhook_url=_mgr_webhook,
            manager_chat_session_id=_mgr_session_id,
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
        
        # Notify the manager agent about the live start
        try:
            notify_manager_live_start(
                session=session,
                strategy_id=strategy_id,
                symbol=request.symbol,
                timeframe=request.timeframe,
                account_config=account_config,
            )
        except Exception as notify_err:
            logger.warning(
                "Failed to notify manager agent for strategy %s: %s",
                strategy_id, notify_err,
            )
        
        return LiveStartResponse(
            status="started",
            container_id=result.get("container_id"),
            container_name=result.get("container_name"),
            symbol=request.symbol,
            timeframe=request.timeframe,
        )
        
    except HTTPException:
        raise
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


# ═════════════════════════════════════════════════════════════════════
# LIVE ORDERS
# ═════════════════════════════════════════════════════════════════════


@router.get("/strategies/{strategy_id}/orders", response_model=list[LiveOrderRead])
def list_strategy_orders(
    strategy_id: int,
    status: str | None = Query(None, description="Filter by order status"),
    limit: int = Query(100, ge=1, le=1000),
    session: Session = Depends(get_session),
):
    """List live orders for a strategy."""
    strategy = get_strategy(session, strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
    orders = list_live_orders(session, strategy_id, status=status, limit=limit)
    return [LiveOrderRead.model_validate(o) for o in orders]


@router.get("/strategies/{strategy_id}/orders/active", response_model=list[LiveOrderRead])
def list_strategy_active_orders(
    strategy_id: int,
    session: Session = Depends(get_session),
):
    """List active (pending/submitted/partially_filled) orders for a strategy."""
    strategy = get_strategy(session, strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
    orders = list_active_orders(session, strategy_id)
    return [LiveOrderRead.model_validate(o) for o in orders]


@router.post("/strategies/{strategy_id}/orders", response_model=LiveOrderRead, status_code=201)
def create_strategy_order(
    strategy_id: int,
    payload: LiveOrderCreate,
    session: Session = Depends(get_session),
):
    """Create a live order for a strategy (typically called by the runner)."""
    strategy = get_strategy(session, strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
    order = create_live_order(session, strategy_id, strategy.live_account_id, payload)
    return LiveOrderRead.model_validate(order)


@router.patch("/orders/{order_id}", response_model=LiveOrderRead)
def update_order(
    order_id: int,
    payload: LiveOrderUpdate,
    session: Session = Depends(get_session),
):
    """Update a live order (status, fill info, etc.)."""
    order = update_live_order(session, order_id, payload)
    if order is None:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
    return LiveOrderRead.model_validate(order)


@router.get("/orders/{order_id}", response_model=LiveOrderRead)
def get_order(
    order_id: int,
    session: Session = Depends(get_session),
):
    """Get a single live order by ID."""
    order = get_live_order(session, order_id)
    if order is None:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
    return LiveOrderRead.model_validate(order)


# ═════════════════════════════════════════════════════════════════════
# LIVE TRADES
# ═════════════════════════════════════════════════════════════════════


@router.get("/strategies/{strategy_id}/trades", response_model=list[LiveTradeRead])
def list_strategy_trades(
    strategy_id: int,
    limit: int = Query(200, ge=1, le=1000),
    session: Session = Depends(get_session),
):
    """List live trades / fills for a strategy."""
    strategy = get_strategy(session, strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
    trades = list_live_trades(session, strategy_id, limit=limit)
    return [LiveTradeRead.model_validate(t) for t in trades]


@router.post("/strategies/{strategy_id}/trades", response_model=LiveTradeRead, status_code=201)
def create_strategy_trade(
    strategy_id: int,
    payload: LiveTradeCreate,
    session: Session = Depends(get_session),
):
    """Record a live trade / fill (typically called by the runner)."""
    strategy = get_strategy(session, strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
    trade = create_live_trade(session, strategy_id, strategy.live_account_id, payload)
    return LiveTradeRead.model_validate(trade)


# ═════════════════════════════════════════════════════════════════════
# LIVE POSITIONS
# ═════════════════════════════════════════════════════════════════════


@router.get("/strategies/{strategy_id}/positions", response_model=list[LivePositionRead])
def list_strategy_positions(
    strategy_id: int,
    status: str | None = Query(None, description="Filter by status: open/closed"),
    limit: int = Query(100, ge=1, le=1000),
    session: Session = Depends(get_session),
):
    """List live positions for a strategy."""
    strategy = get_strategy(session, strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
    positions = list_positions(session, strategy_id, status=status, limit=limit)
    return [LivePositionRead.model_validate(p) for p in positions]


@router.get("/strategies/{strategy_id}/positions/open", response_model=list[LivePositionRead])
def list_strategy_open_positions(
    strategy_id: int,
    session: Session = Depends(get_session),
):
    """List open positions for a strategy."""
    strategy = get_strategy(session, strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
    positions = list_open_positions(session, strategy_id)
    return [LivePositionRead.model_validate(p) for p in positions]


@router.post("/strategies/{strategy_id}/positions", response_model=LivePositionRead, status_code=201)
def upsert_strategy_position(
    strategy_id: int,
    payload: LivePositionCreate,
    session: Session = Depends(get_session),
):
    """Create or update a position for a strategy (typically called by the runner)."""
    strategy = get_strategy(session, strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
    position = upsert_position(session, strategy_id, strategy.live_account_id, payload)
    return LivePositionRead.model_validate(position)


# ═════════════════════════════════════════════════════════════════════
# RECONCILIATION
# ═════════════════════════════════════════════════════════════════════


@router.post("/strategies/{strategy_id}/reconcile", response_model=ReconciliationReport)
def reconcile_strategy(
    strategy_id: int,
    broker_orders: list[dict[str, Any]] | None = None,
    broker_positions: list[dict[str, Any]] | None = None,
    session: Session = Depends(get_session),
):
    """
    Run reconciliation between DB state and broker state for a strategy.

    Can be called manually or automatically on startup.
    Optionally accepts broker state; if not provided, reconciles
    only local DB state (cancels stale orders, etc.).
    """
    strategy = get_strategy(session, strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")

    report = reconcile_on_startup(
        session,
        strategy_id=strategy_id,
        account_id=strategy.live_account_id,
        broker_orders=broker_orders,
        broker_positions=broker_positions,
    )
    return report


# ═════════════════════════════════════════════════════════════════════
# MANAGER AGENT  (AI Agent Manager)
# ═════════════════════════════════════════════════════════════════════


class ManagerMessageRequest(BaseModel):
    """Message to post in the strategy's live chat."""
    message: str
    sender: str = "agent"  # 'agent' | 'system' | 'user'


class ManagerMessageResponse(BaseModel):
    """Response after posting a message."""
    status: str
    chat_id: int


@router.post(
    "/strategies/{strategy_id}/manager/message",
    response_model=ManagerMessageResponse,
)
def post_manager_message_endpoint(
    strategy_id: int,
    payload: ManagerMessageRequest,
    session: Session = Depends(get_session),
):
    """Post a message to the strategy's Live chat.

    Typically called by the strategy-runner container to report
    trade actions, status updates, or errors to the manager agent's
    live chat.
    """
    _ = get_strategy(session, strategy_id)
    chat = post_manager_message(
        session=session,
        strategy_id=strategy_id,
        message=payload.message,
        sender=payload.sender,
    )
    return ManagerMessageResponse(status="ok", chat_id=chat.id)


@router.get("/strategies/{strategy_id}/manager/chat")
def get_manager_live_chat(
    strategy_id: int,
    session: Session = Depends(get_session),
):
    """Get (or create) the dedicated Live chat for a strategy."""
    from app.schemas.chat import ChatRead

    chat = get_or_create_live_chat(session, strategy_id)
    return ChatRead.model_validate(chat)
