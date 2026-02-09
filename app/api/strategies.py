from fastapi import APIRouter, Depends, BackgroundTasks
from sqlmodel import Session

from app.db.database import get_session
from app.schemas.strategy import (
    StrategyCreate,
    StrategyRead,
    StrategyUpdate,
    BacktestCreate,
    BacktestRead,
    BacktestUpdate,
    TradeCreate,
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
    create_trade,
    create_trades_bulk,
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

router = APIRouter(prefix="/strategies", tags=["Strategies"])


@router.post("/", response_model=StrategyRead)
def create_strategy_endpoint(payload: StrategyCreate, session: Session = Depends(get_session)):
    return create_strategy(session, payload)


@router.get("/", response_model=list[StrategyRead])
def list_strategies_endpoint(session: Session = Depends(get_session)):
    return list_strategies(session)


@router.get("/{strategy_id}", response_model=StrategyRead)
def get_strategy_endpoint(strategy_id: int, session: Session = Depends(get_session)):
    return get_strategy(session, strategy_id)


@router.patch("/{strategy_id}", response_model=StrategyRead)
def update_strategy_endpoint(
    strategy_id: int,
    payload: StrategyUpdate,
    session: Session = Depends(get_session),
):
    return update_strategy(session, strategy_id, payload)


@router.delete("/{strategy_id}")
def delete_strategy_endpoint(strategy_id: int, session: Session = Depends(get_session)):
    delete_strategy(session, strategy_id)
    return {"status": "ok"}


@router.patch("/{strategy_id}/layout", response_model=StrategyRead)
def update_strategy_layout_endpoint(
    strategy_id: int,
    payload: LayoutConfigUpdate,
    session: Session = Depends(get_session),
):
    """Update only the UI layout configuration for a strategy."""
    return update_strategy_layout(session, strategy_id, payload)


@router.post("/{strategy_id}/backtests", response_model=BacktestRead)
def create_backtest_endpoint(
    strategy_id: int,
    payload: BacktestCreate,
    session: Session = Depends(get_session),
):
    """Create a new backtest with status=pending. Call /run to execute."""
    return create_backtest(session, strategy_id, payload)


@router.get("/{strategy_id}/backtests", response_model=list[BacktestRead])
def list_backtests_endpoint(strategy_id: int, session: Session = Depends(get_session)):
    """List all backtests for a strategy."""
    return list_backtests(session, strategy_id)


@router.get("/backtests/{backtest_id}", response_model=BacktestRead)
def get_backtest_endpoint(backtest_id: int, session: Session = Depends(get_session)):
    """Get backtest details including status and results if completed."""
    return get_backtest(session, backtest_id)


@router.post("/backtests/{backtest_id}/run", response_model=BacktestRead)
def run_backtest_endpoint(
    backtest_id: int,
    session: Session = Depends(get_session),
):
    """Start backtest execution via n8n webhook.
    
    Sets status to 'running' and calls the agent's webhook.
    The agent_id must be set on the backtest.
    """
    return run_backtest(session, backtest_id)


@router.patch("/backtests/{backtest_id}", response_model=BacktestRead)
def update_backtest_endpoint(
    backtest_id: int,
    payload: BacktestUpdate,
    session: Session = Depends(get_session),
):
    """Update backtest status and results (callback endpoint for n8n)."""
    return update_backtest(session, backtest_id, payload)


@router.delete("/backtests/{backtest_id}")
def delete_backtest_endpoint(backtest_id: int, session: Session = Depends(get_session)):
    """Delete a backtest and all its trades."""
    delete_backtest(session, backtest_id)
    return {"status": "ok"}


@router.patch("/backtests/{backtest_id}/layout", response_model=BacktestRead)
def update_backtest_layout_endpoint(
    backtest_id: int,
    payload: LayoutConfigUpdate,
    session: Session = Depends(get_session),
):
    """Update only the UI layout configuration for a backtest."""
    return update_backtest_layout(session, backtest_id, payload)


@router.post("/backtests/{backtest_id}/trades", response_model=TradeRead)
def create_trade_endpoint(
    backtest_id: int,
    payload: TradeCreate,
    session: Session = Depends(get_session),
):
    """Create a single trade record for a backtest."""
    return create_trade(session, backtest_id, payload)


@router.post("/backtests/{backtest_id}/trades/bulk", response_model=list[TradeRead])
def create_trades_bulk_endpoint(
    backtest_id: int,
    payload: list[TradeCreate],
    session: Session = Depends(get_session),
):
    """Create multiple trade records for a backtest in bulk."""
    return create_trades_bulk(session, backtest_id, payload)


@router.get("/backtests/{backtest_id}/trades", response_model=list[TradeRead])
def list_trades_endpoint(backtest_id: int, session: Session = Depends(get_session)):
    """List all trades for a backtest."""
    return list_trades(session, backtest_id)


# ─── CHAT ENDPOINTS ───


@router.get("/{strategy_id}/chats", response_model=list[ChatRead])
def list_strategy_chats_endpoint(strategy_id: int, session: Session = Depends(get_session)):
    """List all chats for a strategy."""
    return list_strategy_chats(session, strategy_id)


@router.post("/{strategy_id}/chats", response_model=ChatRead)
def create_strategy_chat_endpoint(
    strategy_id: int,
    payload: ChatCreate,
    session: Session = Depends(get_session),
):
    """Create a new chat for a strategy."""
    return create_strategy_chat(session, strategy_id, payload)


@router.get("/{strategy_id}/chats/{chat_id}", response_model=ChatRead)
def get_strategy_chat_endpoint(
    strategy_id: int,
    chat_id: int,
    session: Session = Depends(get_session),
):
    """Get a specific chat for a strategy."""
    return get_strategy_chat(session, strategy_id, chat_id)


@router.patch("/{strategy_id}/chats/{chat_id}", response_model=ChatRead)
def update_strategy_chat_endpoint(
    strategy_id: int,
    chat_id: int,
    payload: ChatCreate,
    session: Session = Depends(get_session),
):
    """Update a chat for a strategy."""
    return update_strategy_chat(session, strategy_id, chat_id, payload)


@router.delete("/{strategy_id}/chats/{chat_id}")
def delete_strategy_chat_endpoint(
    strategy_id: int,
    chat_id: int,
    session: Session = Depends(get_session),
):
    """Delete a chat from a strategy."""
    delete_strategy_chat(session, strategy_id, chat_id)
    return {"status": "ok"}


# ─── RULE TRIGGER ENDPOINTS ───


@router.post("/trigger-agent", response_model=RuleTriggerResponse)
def trigger_agent_endpoint(
    payload: RuleTriggerRequest,
    session: Session = Depends(get_session),
):
    """Trigger an agent webhook when an ask_agent rule is activated.
    
    This endpoint is called during backtest execution when a rule with
    action='ask_agent' has its conditions satisfied (event-based or time-based).
    
    The rule_context should contain:
    - rule_name: Name of the triggered rule
    - trigger_type: 'event' | 'time' | 'both'
    - timestamp: When the rule was triggered
    - bar_data: Current bar OHLCV data
    - indicators: Current indicator values
    - position: Current position info
    - conditions_matched: List of matched conditions (for event triggers)
    - time_trigger: Time trigger info (for time-based triggers)
    """
    result = trigger_rule_agent(
        session=session,
        agent_id=payload.agent_id,
        chat_id=payload.chat_id,
        rule_context=payload.rule_context,
        webhook_url=payload.webhook_url,
    )
    return RuleTriggerResponse(status="ok", agent_response=result)
