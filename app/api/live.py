"""
Live Trading API Endpoints.

Manages live strategy execution via Docker containers.
Includes endpoints for orders, trades, positions, and reconciliation.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlmodel import Session

from app.db.database import get_session
from app.models.agent import Agent, Chat
from app.models.strategy import LiveStatus, StrategyLive
from app.schemas.chat import ChatRead
from app.schemas.live_trading import (
    LiveAlertCreate,
    LiveAlertRead,
    LiveAlertUpdate,
    LiveOrderRead,
    LiveOrderUpdate,
    LivePositionRead,
    LiveFillRead,
    ReconciliationReport,
)
import redis as _redis

from app.services.live_runner_service import live_runner_service
from app.services.live_trading_service import (
    get_live_order,
    list_active_orders,
    list_live_orders,
    list_live_fills,
    list_open_positions,
    list_positions,
    list_strategy_orders,
    list_strategy_positions,
    list_strategy_fills,
    reconcile_on_startup,
    update_live_order,
    validate_account_for_live,
)
from app.services.live_alert_service import (
    create_live_alert,
    delete_live_alert,
    get_live_alert,
    list_live_alerts,
    update_live_alert,
)
from app.services.gateway_client import GatewayClient
from app.services.strategy_service import get_strategy
from app.services.strategy_service import (
    post_manager_message,
    get_or_create_live_chat,
    list_live_session_chats,
    resolve_strategy_manager_agent_id,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/live", tags=["Live Trading"])

# Redis connection for config-live cleanup on stop
_REDIS_HOST = os.getenv("REDIS_HOST", "redis")
_REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
_CHECKPOINT_LIMIT = 100


def _clear_live_config(
    symbol: str,
    connection_id: int | str,
) -> None:
    """Delete the ``config-live:subscribe`` Hash (safety net on stop).

    The strategy-runner writes and manages this Hash during its
    lifecycle.  On stop the backend deletes it as a safety net in case
    the runner was killed before it could clean up.
    """
    import json as _json

    key = f"config-live:subscribe:{symbol.upper()}:{connection_id}"
    r = _redis.Redis(host=_REDIS_HOST, port=_REDIS_PORT, decode_responses=True)
    try:
        r.delete(key)
        r.publish(
            "subscribe:changed",
            _json.dumps({"symbol": symbol.upper(), "connection_id": str(connection_id)}),
        )
        logger.info(
            "Deleted config-live Hash for %s conn=%s (safety net)",
            symbol, connection_id,
        )
    finally:
        r.close()


# ─── REQUEST/RESPONSE MODELS ───


class LiveStartRequest(BaseModel):
    """Request to start a live strategy."""
    symbol: str
    timeframe: str = "5s"  # Bar aggregation interval
    eval_in_progress: bool = True  # Evaluate rules on in-progress bars
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


class LayoutConfigUpdate(BaseModel):
    """Schema for updating the layout_config field."""
    layout_config: dict[str, Any]


class LiveLayoutResponse(BaseModel):
    """Response after updating live session layout."""
    live_id: int
    layout_config: dict[str, Any] | None = None


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


def _append_live_checkpoint(
    session: Session,
    live_id: int,
    *,
    source: str,
    kind: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> None:
    """Persist a lightweight debug checkpoint on the live session."""
    sl = session.get(StrategyLive, live_id)
    if sl is None:
        return

    metrics = sl.metrics if isinstance(sl.metrics, dict) else {}
    checkpoints = metrics.get("debug_checkpoints")
    if not isinstance(checkpoints, list):
        checkpoints = []

    checkpoints.append({
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "kind": kind,
        "message": message,
        "details": details or {},
    })
    metrics["debug_checkpoints"] = checkpoints[-_CHECKPOINT_LIMIT:]

    sl.metrics = metrics
    sl.updated_at = datetime.now(timezone.utc)
    session.add(sl)
    session.commit()


def _get_live_checkpoints(sl: StrategyLive, limit: int = 100) -> list[dict[str, Any]]:
    metrics = sl.metrics if isinstance(sl.metrics, dict) else {}
    checkpoints = metrics.get("debug_checkpoints")
    if not isinstance(checkpoints, list):
        return []
    return checkpoints[-limit:]


def _mark_live_error(session: Session, live_id: int, message: str) -> None:
    sl = session.get(StrategyLive, live_id)
    if sl is None:
        return
    sl.status = LiveStatus.ERROR.value
    sl.error_message = message
    sl.updated_at = datetime.now(timezone.utc)
    session.add(sl)
    session.commit()


async def _fetch_broker_snapshot(
    *,
    connection_id: int,
    broker_type: str,
    broker_account_id: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch a normalized broker snapshot through the common gateway API."""
    client = GatewayClient(connection_id=connection_id, broker_type=broker_type)

    orders = await client.list_open_orders()
    positions = await client.list_positions(account=broker_account_id)

    if broker_account_id:
        orders = [
            order for order in orders
            if not order.get("account") or str(order.get("account")) == str(broker_account_id)
        ]

    return orders, positions


def _serialize_chat_read(
    session: Session,
    chat: Chat,
    fallback_agent_id: int | None = None,
) -> ChatRead:
    resolved_agent_id = chat.id_agent or fallback_agent_id
    agent = session.get(Agent, resolved_agent_id) if resolved_agent_id else None
    return ChatRead(
        id=chat.id,
        id_agent=resolved_agent_id,
        agent_name=agent.agent_name if agent else chat.agent_name,
        agent_webhook_url=agent.n8n_webhook if agent else chat.agent_webhook_url,
        user_id=chat.user_id,
        strategy_id=chat.strategy_id,
        nome=chat.nome,
        descrizione=chat.descrizione,
        chat_type=chat.chat_type,
        created_at=chat.created_at,
        n8n_session_id=chat.n8n_session_id,
    )


# ─── ENDPOINTS ───


@router.post("/strategies/{strategy_id}/start", response_model=LiveStartResponse)
async def start_live_strategy(
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
        live_chat = get_or_create_live_chat(session, strategy_id)

        # Create a new StrategyLive session
        sl = StrategyLive(
            strategy_id=strategy_id,
            chat_id=live_chat.id,
            manager_agent_id=strategy.manager_agent_id,
            status=LiveStatus.STARTING.value,
            symbol=request.symbol,
            timeframe=request.timeframe,
            account_id=request.account_id,
            connection_id=connection.id if connection else None,
            definition=strategy.definition,  # snapshot, decoupled from design
        )
        session.add(sl)
        session.commit()
        session.refresh(sl)

        _append_live_checkpoint(
            session,
            sl.id,
            source="backend",
            kind="live_session_created",
            message="Live session created before broker-aware reconciliation.",
            details={
                "strategy_id": strategy_id,
                "strategy_live_id": sl.id,
                "symbol": request.symbol,
                "timeframe": request.timeframe,
                "connection_id": connection.id if connection else None,
                "account_id": request.account_id,
                "broker_type": broker_type,
            },
        )

        try:
            broker_orders, broker_positions = await _fetch_broker_snapshot(
                connection_id=connection.id,
                broker_type=broker_type,
                broker_account_id=account.account_id,
            )
        except Exception as exc:
            error_message = (
                "Broker snapshot unavailable before runner start. "
                f"Start aborted to avoid running without alignment: {exc}"
            )
            _mark_live_error(session, sl.id, error_message)
            _append_live_checkpoint(
                session,
                sl.id,
                source="backend",
                kind="broker_snapshot_failed",
                message="Failed to fetch broker snapshot before start.",
                details={
                    "connection_id": connection.id,
                    "broker_type": broker_type,
                    "broker_account_id": account.account_id,
                    "error": str(exc),
                },
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=error_message,
            ) from exc

        _append_live_checkpoint(
            session,
            sl.id,
            source="backend",
            kind="broker_snapshot_fetched",
            message="Fetched broker snapshot through the common gateway before start.",
            details={
                "connection_id": connection.id,
                "broker_type": broker_type,
                "broker_account_id": account.account_id,
                "orders_count": len(broker_orders),
                "positions_count": len(broker_positions),
            },
        )
        
        # Run reconciliation before starting
        recon_report = reconcile_on_startup(
            session,
            strategy_live_id=sl.id,
            account_id=request.account_id,
            broker_orders=broker_orders,
            broker_positions=broker_positions,
        )
        _append_live_checkpoint(
            session,
            sl.id,
            source="backend",
            kind="pre_start_reconciliation",
            message="Completed broker-aware reconciliation before runner start.",
            details={
                "items_count": len(recon_report.items),
                "summary": recon_report.summary,
                "issues": [item.issue for item in recon_report.items[:20]],
            },
        )
        if recon_report.items:
            logger.info(
                "Pre-start reconciliation for strategy_live %s: %s",
                sl.id, recon_report.summary,
            )
        
        # Resolve manager agent webhook URL for the runner
        _mgr_webhook: str | None = None
        _mgr_session_id: str | None = live_chat.n8n_session_id
        if strategy.manager_agent_id:
            from app.models.agent import Agent
            _mgr_agent = session.get(Agent, strategy.manager_agent_id)
            if _mgr_agent:
                _mgr_webhook = _mgr_agent.n8n_webhook

        result = live_runner_service.start_strategy(
            strategy_id=strategy_id,
            strategy_config=sl.definition,
            symbol=request.symbol,
            timeframe=request.timeframe,
            eval_in_progress=request.eval_in_progress,
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

        _append_live_checkpoint(
            session,
            sl.id,
            source="backend",
            kind="runner_start_requested",
            message="Runner container started after broker-aware reconciliation.",
            details={
                "container_id": result.get("container_id"),
                "container_name": result.get("container_name"),
            },
        )

        # NOTE: The runner writes its own config-live:subscribe Hash with
        # live_timeframes and indicators.  The backend no longer sets
        # live_timeframes — the runner owns the live pipeline config.

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

        # Clear config-live Hash so ticks-aggregator stops writing live:bars
        try:
            if sl.connection_id and sl.symbol:
                _clear_live_config(sl.symbol, sl.connection_id)
        except Exception as lt_err:
            logger.warning(
                "Failed to clear config-live Hash for strategy %s: %s",
                strategy_id, lt_err,
            )

        # Close any open DB positions for this session — safety net for
        # cases where the runner didn't close them (crash, gateway DNS
        # failure, forced container removal, etc.).
        try:
            from app.models.live_trading import LivePosition, PositionStatus
            from sqlmodel import select as _select
            open_positions = session.exec(
                _select(LivePosition)
                .where(LivePosition.strategy_live_id == sl.id)
                .where(LivePosition.status == PositionStatus.OPEN.value)
            ).all()
            for pos in open_positions:
                pos.status = PositionStatus.CLOSED.value
                pos.side = "flat"
                pos.quantity = 0
                pos.closed_at = datetime.now(timezone.utc)
                pos.updated_at = datetime.now(timezone.utc)
                pos.extra = {**(pos.extra or {}), "closed_by": "backend_stop"}
                session.add(pos)
            if open_positions:
                logger.info(
                    "Closed %d orphan open position(s) on strategy %s stop",
                    len(open_positions), strategy_id,
                )
        except Exception as pos_err:
            logger.warning(
                "Failed to close orphan positions on stop for strategy %s: %s",
                strategy_id, pos_err,
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
# LIVE FILLS  (read-only — runner writes directly to DB)
# ═════════════════════════════════════════════════════════════════════


@router.get("/sessions/{live_id}/fills", response_model=list[LiveFillRead])
def list_session_fills(
    live_id: int,
    limit: int = Query(200, ge=1, le=1000),
    session: Session = Depends(get_session),
):
    """List live fills for a session."""
    _get_live_or_404(session, live_id)
    fills = list_live_fills(session, live_id, limit=limit)
    return [LiveFillRead.model_validate(f) for f in fills]


# ═════════════════════════════════════════════════════════════════════
# LIVE ALERTS  (persistent source of truth for runner-managed alerts)
# ═════════════════════════════════════════════════════════════════════


@router.get("/sessions/{live_id}/alerts", response_model=list[LiveAlertRead])
def list_session_alerts(
    live_id: int,
    enabled: bool | None = Query(None, description="Filter by enabled flag"),
    status: str | None = Query(None, description="Filter by alert status"),
    session: Session = Depends(get_session),
):
    """List persistent alerts for a live session."""
    _get_live_or_404(session, live_id)
    alerts = list_live_alerts(session, live_id, enabled=enabled, status=status)
    return [LiveAlertRead.model_validate(a) for a in alerts]


@router.post("/sessions/{live_id}/alerts", response_model=LiveAlertRead, status_code=status.HTTP_201_CREATED)
def create_session_alert(
    live_id: int,
    payload: LiveAlertCreate,
    session: Session = Depends(get_session),
):
    """Create a persistent alert for a live session."""
    _get_live_or_404(session, live_id)
    alert = create_live_alert(session, live_id, payload)
    return LiveAlertRead.model_validate(alert)


@router.get("/alerts/{alert_id}", response_model=LiveAlertRead)
def get_alert(
    alert_id: int,
    session: Session = Depends(get_session),
):
    """Get a single persistent live alert by ID."""
    alert = get_live_alert(session, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    return LiveAlertRead.model_validate(alert)


@router.patch("/alerts/{alert_id}", response_model=LiveAlertRead)
def patch_alert(
    alert_id: int,
    payload: LiveAlertUpdate,
    session: Session = Depends(get_session),
):
    """Update a persistent live alert."""
    alert = update_live_alert(session, alert_id, payload)
    if alert is None:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    return LiveAlertRead.model_validate(alert)


@router.delete("/alerts/{alert_id}")
def remove_alert(
    alert_id: int,
    session: Session = Depends(get_session),
):
    """Delete a persistent live alert."""
    deleted = delete_live_alert(session, alert_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    return {"status": "deleted", "id": alert_id}


# ═════════════════════════════════════════════════════════════════════
# LIVE POSITIONS
# ═════════════════════════════════════════════════════════════════════


def _enrich_position(pos) -> LivePositionRead:
    """Enrich a LivePosition DB model with computed PnL fields.

    Computes:
      - ``total_commission`` = sum of commissions from fills
      - ``net_pnl``          = realized - commissions
    """
    data = LivePositionRead.model_validate(pos)

    # Extract total commission from position extra
    total_comm = (pos.extra or {}).get("total_commission", 0) or 0
    data.total_commission = total_comm

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
# STRATEGY-LEVEL HISTORY (aggregate across all sessions)
# ═════════════════════════════════════════════════════════════════════


@router.get("/strategies/{strategy_id}/orders", response_model=list[LiveOrderRead])
def list_all_strategy_orders(
    strategy_id: int,
    status: str | None = Query(None, description="Filter by order status"),
    limit: int = Query(200, ge=1, le=2000),
    session: Session = Depends(get_session),
):
    """List orders across ALL live sessions for a strategy."""
    _get_strategy_or_404(session, strategy_id)
    orders = list_strategy_orders(session, strategy_id, status=status, limit=limit)
    return [LiveOrderRead.model_validate(o) for o in orders]


@router.get("/strategies/{strategy_id}/fills", response_model=list[LiveFillRead])
def list_all_strategy_fills(
    strategy_id: int,
    limit: int = Query(500, ge=1, le=5000),
    session: Session = Depends(get_session),
):
    """List fills across ALL live sessions for a strategy."""
    _get_strategy_or_404(session, strategy_id)
    fills = list_strategy_fills(session, strategy_id, limit=limit)
    return [LiveFillRead.model_validate(f) for f in fills]


@router.get("/strategies/{strategy_id}/positions", response_model=list[LivePositionRead])
def list_all_strategy_positions(
    strategy_id: int,
    status: str | None = Query(None, description="Filter by status: open/closed"),
    limit: int = Query(200, ge=1, le=2000),
    session: Session = Depends(get_session),
):
    """List positions across ALL live sessions for a strategy (enriched with PnL)."""
    _get_strategy_or_404(session, strategy_id)
    positions = list_strategy_positions(session, strategy_id, status=status, limit=limit)
    return [_enrich_position(p) for p in positions]


@router.get("/sessions/{live_id}/checkpoints")
def list_live_checkpoints(
    live_id: int,
    limit: int = Query(100, ge=1, le=500),
    session: Session = Depends(get_session),
):
    """Read persisted checkpoints for debugging startup, reconcile, and order flow."""
    sl = _get_live_or_404(session, live_id)
    checkpoints = _get_live_checkpoints(sl, limit=limit)
    return {
        "live_id": sl.id,
        "count": len(checkpoints),
        "checkpoints": checkpoints,
    }


# ═════════════════════════════════════════════════════════════════════
# RECONCILIATION
# ═════════════════════════════════════════════════════════════════════


@router.post("/sessions/{live_id}/reconcile", response_model=ReconciliationReport)
async def reconcile_session(
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

    if broker_orders is None or broker_positions is None:
        if sl.account_id is None or sl.connection_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Live session has no account/connection configured for broker-aware reconciliation.",
            )

        try:
            account, connection = validate_account_for_live(session, sl.account_id)
            broker_orders, broker_positions = await _fetch_broker_snapshot(
                connection_id=connection.id,
                broker_type=connection.broker_type,
                broker_account_id=account.account_id,
            )
            _append_live_checkpoint(
                session,
                sl.id,
                source="backend",
                kind="manual_broker_snapshot_fetched",
                message="Fetched broker snapshot for explicit reconciliation.",
                details={
                    "orders_count": len(broker_orders),
                    "positions_count": len(broker_positions),
                    "connection_id": connection.id,
                    "broker_type": connection.broker_type,
                },
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except Exception as exc:
            _append_live_checkpoint(
                session,
                sl.id,
                source="backend",
                kind="manual_broker_snapshot_failed",
                message="Failed to fetch broker snapshot for reconciliation.",
                details={"error": str(exc)},
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Broker snapshot unavailable for reconciliation: {exc}",
            ) from exc

    report = reconcile_on_startup(
        session,
        strategy_live_id=sl.id,
        account_id=sl.account_id,
        broker_orders=broker_orders,
        broker_positions=broker_positions,
    )
    _append_live_checkpoint(
        session,
        sl.id,
        source="backend",
        kind="manual_reconciliation",
        message="Manual broker-aware reconciliation completed.",
        details={
            "items_count": len(report.items),
            "summary": report.summary,
            "issues": [item.issue for item in report.items[:20]],
        },
    )
    return report


# ═════════════════════════════════════════════════════════════════════
# LAYOUT PERSISTENCE
# ═════════════════════════════════════════════════════════════════════


@router.patch("/sessions/{live_id}/layout", response_model=LiveLayoutResponse)
def update_live_layout_endpoint(
    live_id: int,
    payload: LayoutConfigUpdate,
    session: Session = Depends(get_session),
):
    """Update only the UI layout configuration for a live session."""
    from app.services.strategy_service import update_live_layout
    sl = update_live_layout(session, live_id, payload)
    return LiveLayoutResponse(
        live_id=sl.id,
        layout_config=sl.layout_config,
    )


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
    chat = get_or_create_live_chat(session, strategy_id)
    fallback_agent_id = resolve_strategy_manager_agent_id(session, strategy_id)
    return _serialize_chat_read(session, chat, fallback_agent_id)


@router.get("/sessions/{live_id}/chats", response_model=list[ChatRead])
def list_live_session_chats_endpoint(
    live_id: int,
    session: Session = Depends(get_session),
):
    """List chats relevant to a live session for the LiveStrategy page."""
    live_session = _get_live_or_404(session, live_id)
    fallback_agent_id = resolve_strategy_manager_agent_id(
        session,
        live_session.strategy_id,
        live_session=live_session,
    )
    chats = list_live_session_chats(session, live_id)
    return [_serialize_chat_read(session, chat, fallback_agent_id) for chat in chats]
