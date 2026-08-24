import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session

from app.db.database import get_session
from app.models.user import User
from app.models.strategy import BacktestStatus
from app.schemas.strategy import (
    BacktestCreateRequest,
    BacktestListPage,
    BacktestPlaybackControl,
    BacktestRead,
    BacktestRuntimeOrderRequest,
    BacktestRuntimePositionCloseRequest,
    BacktestSummary,
    LayoutConfigUpdate,
    TradeRead,
)
from app.schemas.chat import ChatRead
from app.services.strategy_service import (
    create_backtest,
    delete_backtest,
    get_backtest,
    get_or_create_backtest_chat,
    get_strategy,
    list_all_backtests,
    list_trades,
    run_backtest,
    update_backtest_layout,
)
from app.utils.auth_utils import get_current_active_or_consultative_user

logger = logging.getLogger(__name__)

# The single backtest API: every backtest is addressed by its own id, the
# owning strategy is resolved (and ownership-checked) server side. The old
# /strategies/{sid}/backtests/* routes were removed in favour of this router.
# Auth mirrors /accounts: user JWT, PAT, delegated n8n token, or the agent's
# consultative token — the in-backtest manager agent reads and trades through
# the /runtime/* endpoints with its agent_backend_consult token.
router = APIRouter(prefix="/backtests", tags=["Backtests"])

_ACTIVE_STATUSES = {BacktestStatus.PENDING.value, BacktestStatus.RUNNING.value}


@router.get("/", response_model=BacktestListPage)
def list_all_backtests_endpoint(
    status_filter: str | None = Query(default=None, alias="status"),
    strategy_id: int | None = Query(default=None),
    symbol: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    with_progress: bool = Query(default=True),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """List backtests across all the user's strategies, newest first.

    ``status`` accepts a comma-separated list (e.g. ``running,pending``).
    Rows in an active status carry live ``phase``/``progress`` from the
    coordinator (unless ``with_progress=false``); a row stuck in ``running``
    with no coordinator state and no runner container is flagged ``stale``.
    """
    from app.services.backtest_runner_service import backtest_runner_service

    statuses = None
    if status_filter:
        statuses = [s.strip() for s in status_filter.split(",") if s.strip()]

    rows, total = list_all_backtests(
        session,
        current_user.id,
        statuses=statuses,
        strategy_id=strategy_id,
        symbol=symbol,
        limit=limit,
        offset=offset,
    )
    items = []
    for backtest, strategy_name in rows:
        item = BacktestSummary.model_validate(backtest, from_attributes=True)
        item.strategy_name = strategy_name
        items.append(item)
    # Release the pooled DB connection before the per-row HTTP probes.
    session.close()

    if with_progress:
        for item in items:
            if item.status not in _ACTIVE_STATUSES:
                continue
            probe = backtest_runner_service.get_backtest_progress(item.id)
            if probe is not None:
                item.phase = probe.get("phase")
                progress = probe.get("progress")
                item.progress = float(progress) if progress is not None else None
            elif item.status == BacktestStatus.RUNNING.value:
                item.stale = not backtest_runner_service.is_backtest_container_running(item.id)

    return BacktestListPage(items=items, total=total, limit=limit, offset=offset)


@router.post("/", response_model=BacktestRead)
def create_backtest_endpoint(
    payload: BacktestCreateRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Create a new backtest with status=pending. Call /run to execute."""
    return create_backtest(session, payload.strategy_id, payload, current_user.id)


@router.get("/{backtest_id}", response_model=BacktestRead)
def get_backtest_endpoint(
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Get backtest details including status and results if completed."""
    return get_backtest(session, backtest_id, current_user.id)


@router.delete("/{backtest_id}")
def delete_backtest_endpoint(
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Delete a backtest and all its trades."""
    delete_backtest(session, backtest_id, current_user.id)
    return {"status": "ok"}


@router.get("/{backtest_id}/chat", response_model=ChatRead)
def get_backtest_chat_endpoint(
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Get or create the dedicated chat for a backtest instance."""
    _ = get_backtest(session, backtest_id, current_user.id)
    return get_or_create_backtest_chat(session, backtest_id, current_user.id)


@router.post("/{backtest_id}/run", response_model=BacktestRead)
def run_backtest_endpoint(
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Start backtest execution.

    Spawns strategy-runner in backtest mode. The always-on strategy-backtest
    service prepares/replays data, records simulated orders, and writes results.
    """
    return run_backtest(session, backtest_id, current_user.id)


@router.post("/{backtest_id}/control")
def control_backtest_playback_endpoint(
    backtest_id: int,
    payload: BacktestPlaybackControl,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Control runtime backtest playback: pause, resume, speed, or step."""
    from app.services.backtest_runner_service import backtest_runner_service

    backtest = get_backtest(session, backtest_id, current_user.id)
    try:
        return backtest_runner_service.control_backtest_playback(
            backtest.id,
            payload.model_dump(exclude_none=True),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@router.post("/{backtest_id}/stop")
def stop_backtest_endpoint(
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Stop a running backtest runner and request replay cancellation."""
    from app.services.backtest_runner_service import backtest_runner_service

    backtest = get_backtest(session, backtest_id, current_user.id)
    result = backtest_runner_service.stop_backtest(backtest.id)

    # Update DB status
    if backtest.status in (BacktestStatus.PENDING.value, BacktestStatus.RUNNING.value):
        backtest.status = BacktestStatus.FAILED.value
        backtest.completed_at = datetime.now(timezone.utc)
        backtest.error_message = "Stopped by user"
        session.add(backtest)
        session.commit()

    return result


@router.get("/{backtest_id}/status")
def get_backtest_runtime_status_endpoint(
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Get runner container and replay-service progress for a backtest."""
    from app.services.backtest_runner_service import backtest_runner_service

    backtest = get_backtest(session, backtest_id, current_user.id)
    bt_id = backtest.id
    strategy = get_strategy(session, backtest.strategy_id, current_user.id)
    connection_id = strategy.connection_id
    # Release the pooled DB connection before the blocking HTTP call to the
    # backtest container: this endpoint is polled once per second per client
    # during playback, and holding the connection across a slow (up to 5s)
    # httpx call exhausts the pool.
    session.close()

    status_payload = backtest_runner_service.get_backtest_status(bt_id)
    service_status = status_payload.setdefault("service", {})
    if isinstance(service_status, dict):
        service_status.setdefault("backtest_id", bt_id)
        service_status.setdefault("stream_id", f"backtest-{bt_id}")
        service_status.setdefault("bars_stream", f"bars:backtest-{bt_id}")
        if connection_id is not None:
            service_status.setdefault("connection_id", str(connection_id))
    return status_payload


@router.get("/{backtest_id}/runtime/orders")
def get_backtest_runtime_orders_endpoint(
    backtest_id: int,
    active_only: bool = Query(default=False),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """List source-of-truth simulated orders recorded by strategy-backtest."""
    from app.services.backtest_runner_service import backtest_runner_service

    backtest = get_backtest(session, backtest_id, current_user.id)
    bt_id = backtest.id
    session.close()  # release the pooled DB connection before the blocking HTTP call
    try:
        return backtest_runner_service.list_backtest_orders(bt_id, active_only=active_only)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@router.post("/{backtest_id}/runtime/orders")
def submit_backtest_runtime_order_endpoint(
    backtest_id: int,
    payload: BacktestRuntimeOrderRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Submit a manual order to the source-of-truth strategy-backtest ledger."""
    from app.services.backtest_runner_service import backtest_runner_service

    backtest = get_backtest(session, backtest_id, current_user.id)
    strategy_id = backtest.strategy_id
    order_payload = payload.model_dump(exclude_none=True)
    extra = dict(order_payload.get("extra") or {})
    extra.setdefault("reason", "manual_backtest_order")
    extra.setdefault("source", "backtest_detail_ui")
    extra.setdefault("strategy_id", strategy_id)
    extra.setdefault("backtest_id", backtest.id)
    order_payload["extra"] = extra
    order_payload.setdefault(
        "order_ref",
        f"strategy-{strategy_id}:backtest-{backtest.id}:manual:{int(datetime.now(timezone.utc).timestamp() * 1000)}",
    )
    try:
        return backtest_runner_service.place_backtest_order(backtest.id, order_payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@router.delete("/{backtest_id}/runtime/orders/{order_id}")
def cancel_backtest_runtime_order_endpoint(
    backtest_id: int,
    order_id: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Cancel an order recorded by strategy-backtest when it is still cancellable."""
    from app.services.backtest_runner_service import backtest_runner_service

    backtest = get_backtest(session, backtest_id, current_user.id)
    try:
        return backtest_runner_service.cancel_backtest_order(
            backtest.id,
            order_id,
            status_message="Cancelled from backtest detail UI",
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@router.get("/{backtest_id}/runtime/positions")
def get_backtest_runtime_positions_endpoint(
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Get source-of-truth simulated position recorded by strategy-backtest."""
    from app.services.backtest_runner_service import backtest_runner_service

    backtest = get_backtest(session, backtest_id, current_user.id)
    bt_id = backtest.id
    session.close()  # release the pooled DB connection before the blocking HTTP call
    try:
        return backtest_runner_service.get_backtest_position(bt_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@router.post("/{backtest_id}/runtime/positions/{position_id}/close")
def close_backtest_runtime_position_endpoint(
    backtest_id: int,
    position_id: str,
    payload: BacktestRuntimePositionCloseRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Close one simulated position through the strategy-backtest ledger."""
    from app.services.backtest_runner_service import backtest_runner_service

    backtest = get_backtest(session, backtest_id, current_user.id)
    close_payload: dict = {
        "quantity": payload.quantity,
        "extra": {
            "reason": payload.reason or "backtest_position_close",
            "source": "backend_backtest_api",
            **(payload.extra or {}),
        },
    }
    if payload.symbol:
        close_payload["symbol"] = payload.symbol
    try:
        result = backtest_runner_service.close_backtest_position(
            backtest.id, position_id, close_payload
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    return {"venue": "backtest", "backtest_id": backtest.id, "result": result}


@router.get("/{backtest_id}/runtime/trades")
def get_backtest_runtime_trades_endpoint(
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """List source-of-truth closed trades computed from the runtime backtest ledger."""
    from app.services.backtest_runner_service import backtest_runner_service

    backtest = get_backtest(session, backtest_id, current_user.id)
    bt_id = backtest.id
    session.close()  # release the pooled DB connection before the blocking HTTP call
    try:
        return backtest_runner_service.list_backtest_trades(bt_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@router.get("/{backtest_id}/runtime/equity")
def get_backtest_runtime_equity_endpoint(
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """List runtime equity snapshots recorded by strategy-backtest."""
    from app.services.backtest_runner_service import backtest_runner_service

    backtest = get_backtest(session, backtest_id, current_user.id)
    bt_id = backtest.id
    session.close()  # release the pooled DB connection before the blocking HTTP call
    try:
        return backtest_runner_service.list_backtest_equity(bt_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@router.get("/{backtest_id}/runtime/alerts")
def get_backtest_runtime_alerts_endpoint(
    backtest_id: int,
    active_only: bool = Query(default=False),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """List structured alerts tracked by the active backtest runner."""
    from app.services.backtest_runner_service import backtest_runner_service

    backtest = get_backtest(session, backtest_id, current_user.id)
    bt_id = backtest.id
    session.close()  # release the pooled DB connection before the blocking HTTP call
    try:
        return backtest_runner_service.list_backtest_alerts(bt_id, active_only=active_only)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@router.get("/{backtest_id}/runtime/agent-calls")
def get_backtest_agent_calls_endpoint(
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """List manager-agent invocations recorded by the runner for this backtest.

    Read from the ``agent_call`` table (written by the runner at dispatch
    time), so the history survives the end of the run. ``bar_ts`` is the
    simulated replay timestamp used to place each call on the chart.
    """
    from sqlmodel import select

    from app.models.agent_call import AgentCall

    backtest = get_backtest(session, backtest_id, current_user.id)
    calls = session.exec(
        select(AgentCall)
        .where(AgentCall.backtest_id == backtest.id)
        .order_by(AgentCall.id)
    ).all()
    return {"agent_calls": [call.model_dump(mode="json") for call in calls]}


@router.patch("/{backtest_id}/layout", response_model=BacktestRead)
def update_backtest_layout_endpoint(
    backtest_id: int,
    payload: LayoutConfigUpdate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Update only the UI layout configuration for a backtest."""
    return update_backtest_layout(session, backtest_id, payload, current_user.id)


@router.get("/{backtest_id}/trades", response_model=list[TradeRead])
def list_trades_endpoint(
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """List all trades for a backtest."""
    return list_trades(session, backtest_id, current_user.id)
