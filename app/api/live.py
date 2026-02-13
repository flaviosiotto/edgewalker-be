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
from app.models.strategy import LiveStatus, StrategyLive
from app.schemas.live_trading import (
    LiveOrderRead,
    LiveOrderUpdate,
    LivePositionRead,
    LiveTradeRead,
    ReconciliationReport,
)
from app.services.live_runner_service import live_runner_service
from app.services.live_trading_service import (
    get_live_order,
    list_active_orders,
    list_live_orders,
    list_live_trades,
    list_open_positions,
    list_positions,
    reconcile_on_startup,
    update_live_order,
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
    live_id: int | None = None
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


# ─── HELPERS ───


def _get_strategy_or_404(session: Session, strategy_id: int):
    """Fetch strategy or raise 404."""
    strategy = get_strategy(session, strategy_id)
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found",
        )
    return strategy


def _get_current_live(strategy) -> StrategyLive | None:
    """Get the current (active or latest) StrategyLive session."""
    return strategy.live


def _require_current_live(strategy) -> StrategyLive:
    """Get the current StrategyLive or raise 404."""
    sl = _get_current_live(strategy)
    if sl is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No live session found for strategy {strategy.id}",
        )
    return sl


def _get_live_or_404(session: Session, live_id: int) -> StrategyLive:
    """Fetch a StrategyLive by ID or raise 404."""
    sl = session.get(StrategyLive, live_id)
    if sl is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Live session {live_id} not found",
        )
    return sl


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
    strategy = _get_strategy_or_404(session, strategy_id)
    
    # Check if already running via current_live
    current = _get_current_live(strategy)
    if current and current.status == LiveStatus.RUNNING.value:
        return LiveStartResponse(
            status="already_running",
            container_id=current.container_id,
            symbol=current.symbol,
            timeframe=current.timeframe,
            message="Strategy is already running",
        )
    
    # Account is mandatory for live trading
    if request.account_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="account_id è obbligatorio per il live trading",
        )
    
    # Validate account
    account_config: dict[str, Any] | None = None
    broker_type: str | None = None
    try:
        account, connection = validate_account_for_live(session, request.account_id)
        account_config = {
            "account_id": account.account_id,
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
        # Create a new StrategyLive session
        sl = StrategyLive(
            strategy_id=strategy_id,
            status=LiveStatus.STARTING.value,
            symbol=request.symbol,
            timeframe=request.timeframe,
            account_id=request.account_id,
        )
        session.add(sl)
        session.commit()
        session.refresh(sl)
        
        # Run reconciliation before starting
        recon_report = reconcile_on_startup(
            session,
            strategy_live_id=sl.id,
            account_id=request.account_id,
        )
        if recon_report.items:
            logger.info(
                "Pre-start reconciliation for strategy_live %s: %s",
                sl.id, recon_report.summary,
            )
        
        # Resolve manager agent webhook URL for the runner
        _mgr_webhook: str | None = None
        _mgr_session_id: str | None = None
        if strategy.manager_agent_id:
            from app.models.agent import Agent
            _mgr_agent = session.get(Agent, strategy.manager_agent_id)
            if _mgr_agent:
                _mgr_webhook = _mgr_agent.n8n_webhook
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
            strategy_live_id=sl.id,
        )
        
        if result["status"] == "already_running":
            sl.status = LiveStatus.RUNNING.value
            sl.container_id = result.get("container_id")
            session.add(sl)
            session.commit()
            
            return LiveStartResponse(
                status="already_running",
                container_id=result.get("container_id"),
                container_name=result.get("container_name"),
                message="Strategy is already running",
            )
        
        # Update StrategyLive with running state
        sl.status = LiveStatus.RUNNING.value
        sl.container_id = result.get("container_id")
        sl.started_at = datetime.now(timezone.utc)
        session.add(sl)
        session.commit()
        
        # Notify manager agent
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
        # Update StrategyLive with error state
        if sl and sl.id:
            sl.status = LiveStatus.ERROR.value
            sl.error_message = str(e)
            session.add(sl)
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
    strategy = _get_strategy_or_404(session, strategy_id)
    sl = _require_current_live(strategy)
    
    try:
        sl.status = LiveStatus.STOPPING.value
        session.add(sl)
        session.commit()
        
        result = live_runner_service.stop_strategy(
            strategy_id=strategy_id,
            remove=remove,
        )
        
        sl.status = LiveStatus.STOPPED.value
        sl.container_id = None
        sl.stopped_at = datetime.now(timezone.utc)
        session.add(sl)
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
        sl.status = LiveStatus.ERROR.value
        sl.error_message = str(e)
        session.add(sl)
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
    strategy = _get_strategy_or_404(session, strategy_id)
    sl = _get_current_live(strategy)
    
    # Get container status
    container_info = live_runner_service.get_strategy_status(strategy_id)
    
    # Sync DB if container state differs
    if sl and container_info["status"] == "not_found" and sl.status == LiveStatus.RUNNING.value:
        sl.status = LiveStatus.ERROR.value
        sl.error_message = "Container not found (crashed or removed)"
        sl.container_id = None
        session.add(sl)
        session.commit()
    
    return LiveStatusResponse(
        live_id=sl.id if sl else None,
        live_status=sl.status if sl else LiveStatus.STOPPED.value,
        symbol=sl.symbol if sl else None,
        timeframe=sl.timeframe if sl else None,
        started_at=sl.started_at.isoformat() if sl and sl.started_at else None,
        stopped_at=sl.stopped_at.isoformat() if sl and sl.stopped_at else None,
        error_message=sl.error_message if sl else None,
        metrics=sl.metrics if sl else None,
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
# LIVE ORDERS  (read-only — submission via runner: POST /api/runners/{id}/orders)
# ═════════════════════════════════════════════════════════════════════


@router.get("/sessions/{live_id}/orders", response_model=list[LiveOrderRead])
def list_session_orders(
    live_id: int,
    status: str | None = Query(None, description="Filter by order status"),
    limit: int = Query(100, ge=1, le=1000),
    session: Session = Depends(get_session),
):
    """List live orders for a session."""
    _get_live_or_404(session, live_id)
    orders = list_live_orders(session, live_id, status=status, limit=limit)
    return [LiveOrderRead.model_validate(o) for o in orders]


@router.get("/sessions/{live_id}/orders/active", response_model=list[LiveOrderRead])
def list_session_active_orders(
    live_id: int,
    session: Session = Depends(get_session),
):
    """List active (pending/submitted/partially_filled) orders for a session."""
    _get_live_or_404(session, live_id)
    orders = list_active_orders(session, live_id)
    return [LiveOrderRead.model_validate(o) for o in orders]


@router.patch("/orders/{order_id}", response_model=LiveOrderRead)
def update_order(
    order_id: int,
    payload: LiveOrderUpdate,
    session: Session = Depends(get_session),
):
    """Update a live order (status, fill info — manual corrections)."""
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
# LIVE TRADES  (read-only — runner writes directly to DB)
# ═════════════════════════════════════════════════════════════════════


@router.get("/sessions/{live_id}/trades", response_model=list[LiveTradeRead])
def list_session_trades(
    live_id: int,
    limit: int = Query(200, ge=1, le=1000),
    session: Session = Depends(get_session),
):
    """List live trades / fills for a session."""
    _get_live_or_404(session, live_id)
    trades = list_live_trades(session, live_id, limit=limit)
    return [LiveTradeRead.model_validate(t) for t in trades]


# ═════════════════════════════════════════════════════════════════════
# LIVE POSITIONS
# ═════════════════════════════════════════════════════════════════════


def _enrich_position(pos) -> LivePositionRead:
    """Enrich a LivePosition DB model with computed real-time PnL fields.

    Reads the last market price from the in-memory Redis cache and
    computes:
      - ``computed_unrealized_pnl`` = (last_price - avg_price) * qty * direction
      - ``computed_market_value``   = last_price * qty
      - ``total_commission``        = sum of commissions from fills
      - ``net_pnl``                 = realized + unrealized - commissions
    """
    from app.services.market_price_cache import price_cache

    data = LivePositionRead.model_validate(pos)

    # Extract total commission from position extra
    total_comm = (pos.extra or {}).get("total_commission", 0) or 0
    data.total_commission = total_comm

    if pos.status != "open" or pos.quantity == 0 or pos.avg_price is None:
        data.net_pnl = (pos.realized_pnl or 0) - total_comm
        return data

    # Look up last market price
    price_info = price_cache.get_last_price(pos.symbol)
    if price_info and price_info.get("price") is not None:
        last = price_info["price"]
        data.last_price = last
        data.price_age_seconds = price_info.get("age_seconds")
        data.computed_market_value = last * pos.quantity

        # Unrealized PnL
        if pos.side == "long":
            data.computed_unrealized_pnl = (last - pos.avg_price) * pos.quantity
        elif pos.side == "short":
            data.computed_unrealized_pnl = (pos.avg_price - last) * pos.quantity
        else:
            data.computed_unrealized_pnl = 0

        # Net PnL = realized + unrealized - commissions
        data.net_pnl = (
            (pos.realized_pnl or 0)
            + (data.computed_unrealized_pnl or 0)
            - total_comm
        )
    else:
        # No live price available — return DB values only
        data.net_pnl = (pos.realized_pnl or 0) - total_comm

    return data


@router.get("/sessions/{live_id}/positions", response_model=list[LivePositionRead])
def list_session_positions(
    live_id: int,
    status: str | None = Query(None, description="Filter by status: open/closed"),
    limit: int = Query(100, ge=1, le=1000),
    session: Session = Depends(get_session),
):
    """List live positions for a session (enriched with real-time PnL)."""
    _get_live_or_404(session, live_id)
    positions = list_positions(session, live_id, status=status, limit=limit)
    return [_enrich_position(p) for p in positions]


@router.get("/sessions/{live_id}/positions/open", response_model=list[LivePositionRead])
def list_session_open_positions(
    live_id: int,
    session: Session = Depends(get_session),
):
    """List open positions for a session (enriched with real-time PnL)."""
    _get_live_or_404(session, live_id)
    positions = list_open_positions(session, live_id)
    return [_enrich_position(p) for p in positions]


# ═════════════════════════════════════════════════════════════════════
# RECONCILIATION
# ═════════════════════════════════════════════════════════════════════


@router.post("/sessions/{live_id}/reconcile", response_model=ReconciliationReport)
def reconcile_session(
    live_id: int,
    broker_orders: list[dict[str, Any]] | None = None,
    broker_positions: list[dict[str, Any]] | None = None,
    session: Session = Depends(get_session),
):
    """
    Run reconciliation between DB state and broker state for a live session.

    Can be called manually or automatically on startup.
    Optionally accepts broker state; if not provided, reconciles
    only local DB state (cancels stale orders, etc.).
    """
    sl = _get_live_or_404(session, live_id)

    report = reconcile_on_startup(
        session,
        strategy_live_id=sl.id,
        account_id=sl.account_id,
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
