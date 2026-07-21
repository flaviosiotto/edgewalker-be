from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session

from app.db.database import get_session
from app.models.user import User
from app.models.strategy import BacktestStatus
from app.schemas.strategy import (
    StrategyCreate,
    StrategyRead,
    StrategyUpdate,
    BacktestCreate,
    BacktestRead,
    BacktestUpdate,
    BacktestPlaybackControl,
    BacktestRuntimeOrderRequest,
    TradeRead,
    RuleTriggerRequest,
    RuleTriggerResponse,
    LayoutConfigUpdate,
)
from app.schemas.chat import ChatCreate, ChatRead
from app.services.live_summary_service import build_live_summary
from app.services.strategy_service import (
    create_strategy,
    delete_strategy,
    get_strategy,
    list_strategies,
    update_strategy,
    create_backtest,
    list_backtests,
    get_backtest,
    get_or_create_backtest_chat,
    run_backtest,
    update_backtest,
    delete_backtest,
    list_trades,
    list_strategy_chats,
    create_strategy_chat,
    get_strategy_chat,
    update_strategy_chat,
    delete_strategy_chat,
    trigger_rule_agent,
    update_strategy_layout,
    update_backtest_layout,
)
from app.utils.auth_utils import (
    get_current_active_user,
)

router = APIRouter(prefix="/strategies", tags=["Strategies"])


def _serialize_strategy_with_live(session: Session, strategy) -> StrategyRead:
    """Build a StrategyRead and attach `live_summary` if a live session exists."""
    payload = StrategyRead.model_validate(strategy)
    sl = strategy.live
    if sl is not None:
        payload.live_summary = build_live_summary(session, sl)
    return payload


@router.post("/", response_model=StrategyRead)
def create_strategy_endpoint(
    payload: StrategyCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    strategy = create_strategy(session, payload, current_user.id)
    return _serialize_strategy_with_live(session, strategy)


@router.get("/", response_model=list[StrategyRead])
def list_strategies_endpoint(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    strategies = list_strategies(session, current_user.id)
    return [_serialize_strategy_with_live(session, s) for s in strategies]


@router.get("/{strategy_id}", response_model=StrategyRead)
def get_strategy_endpoint(
    strategy_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    strategy = get_strategy(session, strategy_id, current_user.id)
    return _serialize_strategy_with_live(session, strategy)


@router.patch("/{strategy_id}", response_model=StrategyRead)
def update_strategy_endpoint(
    strategy_id: int,
    payload: StrategyUpdate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    strategy = update_strategy(session, strategy_id, payload, current_user.id)
    return _serialize_strategy_with_live(session, strategy)


@router.delete("/{strategy_id}")
def delete_strategy_endpoint(
    strategy_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    delete_strategy(session, strategy_id, current_user.id)
    return {"status": "ok"}


@router.patch("/{strategy_id}/layout", response_model=StrategyRead)
def update_strategy_layout_endpoint(
    strategy_id: int,
    payload: LayoutConfigUpdate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Update only the UI layout configuration for a strategy."""
    strategy = update_strategy_layout(session, strategy_id, payload, current_user.id)
    return _serialize_strategy_with_live(session, strategy)


@router.post("/{strategy_id}/backtests", response_model=BacktestRead)
def create_backtest_endpoint(
    strategy_id: int,
    payload: BacktestCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Create a new backtest with status=pending. Call /run to execute."""
    return create_backtest(session, strategy_id, payload, current_user.id)


@router.get("/{strategy_id}/backtests", response_model=list[BacktestRead])
def list_backtests_endpoint(
    strategy_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """List all backtests for a strategy."""
    return list_backtests(session, strategy_id, current_user.id)


@router.get("/{strategy_id}/backtests/{backtest_id}", response_model=BacktestRead)
def get_backtest_endpoint(
    strategy_id: int,
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Get backtest details including status and results if completed."""
    return get_backtest(session, backtest_id, current_user.id)


@router.get("/{strategy_id}/backtests/{backtest_id}/chat", response_model=ChatRead)
def get_backtest_chat_endpoint(
    strategy_id: int,
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Get or create the dedicated chat for a backtest instance."""
    backtest = get_backtest(session, backtest_id, current_user.id)
    if backtest.strategy_id != strategy_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Backtest not found")
    return get_or_create_backtest_chat(session, backtest_id, current_user.id)


@router.post("/{strategy_id}/backtests/{backtest_id}/run", response_model=BacktestRead)
def run_backtest_endpoint(
    strategy_id: int,
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Start backtest execution.

    Spawns strategy-runner in backtest mode. The always-on strategy-backtest
    service prepares/replays data, records simulated orders, and writes results.
    """
    return run_backtest(session, backtest_id, current_user.id)


@router.post("/{strategy_id}/backtests/{backtest_id}/control")
def control_backtest_playback_endpoint(
    strategy_id: int,
    backtest_id: int,
    payload: BacktestPlaybackControl,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
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


@router.post("/{strategy_id}/backtests/{backtest_id}/stop")
def stop_backtest_endpoint(
    strategy_id: int,
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
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


@router.get("/{strategy_id}/backtests/{backtest_id}/status")
def get_backtest_runtime_status_endpoint(
    strategy_id: int,
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
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


@router.get("/{strategy_id}/backtests/{backtest_id}/runtime/orders")
def get_backtest_runtime_orders_endpoint(
    strategy_id: int,
    backtest_id: int,
    active_only: bool = Query(default=False),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
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


@router.post("/{strategy_id}/backtests/{backtest_id}/runtime/orders")
def submit_backtest_runtime_order_endpoint(
    strategy_id: int,
    backtest_id: int,
    payload: BacktestRuntimeOrderRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Submit a manual order to the source-of-truth strategy-backtest ledger."""
    from app.services.backtest_runner_service import backtest_runner_service

    backtest = get_backtest(session, backtest_id, current_user.id)
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



@router.delete("/{strategy_id}/backtests/{backtest_id}/runtime/orders/{order_id}")
def cancel_backtest_runtime_order_endpoint(
    strategy_id: int,
    backtest_id: int,
    order_id: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
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


@router.get("/{strategy_id}/backtests/{backtest_id}/runtime/positions")
def get_backtest_runtime_positions_endpoint(
    strategy_id: int,
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
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


@router.get("/{strategy_id}/backtests/{backtest_id}/runtime/trades")
def get_backtest_runtime_trades_endpoint(
    strategy_id: int,
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
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


@router.get("/{strategy_id}/backtests/{backtest_id}/runtime/equity")
def get_backtest_runtime_equity_endpoint(
    strategy_id: int,
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
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


@router.get("/{strategy_id}/backtests/{backtest_id}/runtime/alerts")
def get_backtest_runtime_alerts_endpoint(
    strategy_id: int,
    backtest_id: int,
    active_only: bool = Query(default=False),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
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


@router.get("/{strategy_id}/backtests/{backtest_id}/logs")
def get_backtest_logs_endpoint(
    strategy_id: int,
    backtest_id: int,
    tail: int = 200,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Get runner container logs for a running or recently finished backtest."""
    from app.services.backtest_runner_service import backtest_runner_service

    _ = get_backtest(session, backtest_id, current_user.id)  # validate exists
    session.close()  # release the pooled DB connection before the blocking docker call
    logs = backtest_runner_service.get_container_logs(backtest_id, tail=tail)
    return {"backtest_id": backtest_id, "logs": logs}


@router.patch("/{strategy_id}/backtests/{backtest_id}", response_model=BacktestRead)
def update_backtest_endpoint(
    strategy_id: int,
    backtest_id: int,
    payload: BacktestUpdate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Update backtest status and results."""
    return update_backtest(session, backtest_id, payload, current_user.id)


@router.delete("/{strategy_id}/backtests/{backtest_id}")
def delete_backtest_endpoint(
    strategy_id: int,
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Delete a backtest and all its trades."""
    delete_backtest(session, backtest_id, current_user.id)
    return {"status": "ok"}


@router.patch("/{strategy_id}/backtests/{backtest_id}/layout", response_model=BacktestRead)
def update_backtest_layout_endpoint(
    strategy_id: int,
    backtest_id: int,
    payload: LayoutConfigUpdate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Update only the UI layout configuration for a backtest."""
    return update_backtest_layout(session, backtest_id, payload, current_user.id)


@router.get("/{strategy_id}/backtests/{backtest_id}/trades", response_model=list[TradeRead])
def list_trades_endpoint(
    strategy_id: int,
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """List all trades for a backtest."""
    return list_trades(session, backtest_id, current_user.id)


# ─── CHAT ENDPOINTS ───


@router.get("/{strategy_id}/chats", response_model=list[ChatRead])
def list_strategy_chats_endpoint(
    strategy_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """List all chats for a strategy."""
    return list_strategy_chats(session, strategy_id, current_user.id)


@router.post("/{strategy_id}/chats", response_model=ChatRead)
def create_strategy_chat_endpoint(
    strategy_id: int,
    payload: ChatCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Create a new chat for a strategy."""
    return create_strategy_chat(session, strategy_id, payload, current_user.id)


@router.get("/{strategy_id}/chats/{chat_id}", response_model=ChatRead)
def get_strategy_chat_endpoint(
    strategy_id: int,
    chat_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Get a specific chat for a strategy."""
    return get_strategy_chat(session, strategy_id, chat_id, current_user.id)


@router.patch("/{strategy_id}/chats/{chat_id}", response_model=ChatRead)
def update_strategy_chat_endpoint(
    strategy_id: int,
    chat_id: int,
    payload: ChatCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Update a chat for a strategy."""
    return update_strategy_chat(session, strategy_id, chat_id, payload, current_user.id)


@router.delete("/{strategy_id}/chats/{chat_id}")
def delete_strategy_chat_endpoint(
    strategy_id: int,
    chat_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Delete a chat from a strategy."""
    delete_strategy_chat(session, strategy_id, chat_id, current_user.id)
    return {"status": "ok"}


# ─── RULE TRIGGER ENDPOINTS ───


@router.post("/trigger-agent", response_model=RuleTriggerResponse)
def trigger_agent_endpoint(
    payload: RuleTriggerRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Trigger an agent webhook when an ask_agent rule is activated.
    
    This endpoint is called during backtest execution when a rule with
    action='ask_agent' has its conditions satisfied.
    
    The rule_context should contain:
    - rule_name: Name of the triggered rule
    - timestamp: When the rule was triggered
    - bar_data: Current bar OHLCV data
    - indicators: Current indicator values
    - position: Current position info
    - conditions_matched: List of matched conditions
    """
    result = trigger_rule_agent(
        session=session,
        agent_id=payload.agent_id,
        chat_id=payload.chat_id,
        rule_context=payload.rule_context,
        webhook_url=payload.webhook_url,
        user_id=current_user.id,
    )
    return RuleTriggerResponse(status="ok", agent_response=result)
