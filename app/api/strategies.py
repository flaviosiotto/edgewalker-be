from datetime import datetime, timezone

from fastapi import APIRouter, Depends
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
    TradeRead,
    RuleTriggerRequest,
    RuleTriggerResponse,
    LayoutConfigUpdate,
)
from app.schemas.chat import ChatCreate, ChatRead
from app.services.strategy_service import (
    create_strategy,
    delete_strategy,
    get_strategy,
    list_strategies,
    update_strategy,
    create_backtest,
    list_backtests,
    get_backtest,
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
from app.utils.auth_utils import get_current_active_user

router = APIRouter(prefix="/strategies", tags=["Strategies"])


@router.post("/", response_model=StrategyRead)
def create_strategy_endpoint(
    payload: StrategyCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    return create_strategy(session, payload, current_user.id)


@router.get("/", response_model=list[StrategyRead])
def list_strategies_endpoint(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    return list_strategies(session, current_user.id)


@router.get("/{strategy_id}", response_model=StrategyRead)
def get_strategy_endpoint(
    strategy_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    return get_strategy(session, strategy_id, current_user.id)


@router.patch("/{strategy_id}", response_model=StrategyRead)
def update_strategy_endpoint(
    strategy_id: int,
    payload: StrategyUpdate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    return update_strategy(session, strategy_id, payload, current_user.id)


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
    return update_strategy_layout(session, strategy_id, payload, current_user.id)


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


@router.post("/{strategy_id}/backtests/{backtest_id}/run", response_model=BacktestRead)
def run_backtest_endpoint(
    strategy_id: int,
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Start backtest execution.

    Spawns a strategy-backtest Docker container that reads params from
    the DB, verifies data coverage, runs the backtest with edgewalker,
    and writes results directly to the database.
    """
    return run_backtest(session, backtest_id, current_user.id)


@router.post("/{strategy_id}/backtests/{backtest_id}/stop")
def stop_backtest_endpoint(
    strategy_id: int,
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Stop a running backtest container."""
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


@router.get("/{strategy_id}/backtests/{backtest_id}/logs")
def get_backtest_logs_endpoint(
    strategy_id: int,
    backtest_id: int,
    tail: int = 200,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Get container logs for a running or recently finished backtest."""
    from app.services.backtest_runner_service import backtest_runner_service

    _ = get_backtest(session, backtest_id, current_user.id)  # validate exists
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
