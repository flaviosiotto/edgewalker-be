from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import httpx
from fastapi import HTTPException, status
from sqlalchemy.orm import selectinload
from sqlmodel import Session, select

from app.models.agent import Agent, Chat
from app.models.strategy import Strategy, BacktestResult, BacktestTrade, BacktestStatus
from app.schemas.strategy import (
    StrategyCreate,
    StrategyUpdate,
    BacktestCreate,
    BacktestUpdate,
    TradeCreate,
)
from app.schemas.chat import ChatCreate

if TYPE_CHECKING:
    from sqlmodel import Session


def create_strategy(session: Session, payload: StrategyCreate) -> Strategy:
    name = (payload.name or "").strip()
    if not name:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Strategy name is required",
        )

    existing = session.exec(select(Strategy).where(Strategy.name == name)).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Strategy name already exists",
        )

    now = datetime.now(timezone.utc)
    strategy = Strategy(
        name=name,
        description=payload.description,
        definition=payload.definition,
        created_at=now,
        updated_at=now,
    )
    session.add(strategy)
    session.commit()
    session.refresh(strategy)
    return strategy


def list_strategies(session: Session) -> list[Strategy]:
    stmt = select(Strategy).options(selectinload(Strategy.chats)).order_by(Strategy.id.desc())
    return list(session.exec(stmt).all())


def get_strategy(session: Session, strategy_id: int) -> Strategy:
    stmt = select(Strategy).options(selectinload(Strategy.chats)).where(Strategy.id == strategy_id)
    strategy = session.exec(stmt).first()
    if not strategy:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Strategy not found")
    return strategy


def update_strategy(session: Session, strategy_id: int, payload: StrategyUpdate) -> Strategy:
    strategy = get_strategy(session, strategy_id)

    if payload.name is not None and payload.name != strategy.name:
        name = (payload.name or "").strip()
        if not name:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Strategy name is required",
            )

        existing = session.exec(select(Strategy).where(Strategy.name == name)).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Strategy name already exists",
            )
        strategy.name = name

    if payload.description is not None:
        strategy.description = payload.description

    if payload.definition is not None:
        strategy.definition = payload.definition

    strategy.updated_at = datetime.now(timezone.utc)

    session.add(strategy)
    session.commit()
    session.refresh(strategy)
    return strategy


def delete_strategy(session: Session, strategy_id: int) -> None:
    strategy = get_strategy(session, strategy_id)
    session.delete(strategy)
    session.commit()


def create_backtest(session: Session, strategy_id: int, payload: BacktestCreate) -> BacktestResult:
    """Create a new backtest with status=pending."""
    _ = get_strategy(session, strategy_id)
    
    # Validate agent_id if provided
    if payload.agent_id is not None:
        agent = session.get(Agent, payload.agent_id)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent with id {payload.agent_id} not found",
            )
    
    backtest = BacktestResult(
        strategy_id=strategy_id,
        agent_id=payload.agent_id,
        # Required parameters
        symbol=payload.symbol,
        start_date=payload.start_date,
        end_date=payload.end_date,
        # Data source parameters
        source=payload.source,
        timeframe=payload.timeframe,
        asset=payload.asset,
        rth=payload.rth,
        # IBKR-specific parameters
        ibkr_config=payload.ibkr_config,
        exchange=payload.exchange,
        currency=payload.currency,
        expiry=payload.expiry,
        # Backtest execution parameters
        initial_capital=payload.initial_capital,
        commission=payload.commission,
        # Additional config overrides
        parameters=payload.parameters,
        status=BacktestStatus.PENDING.value,
    )
    session.add(backtest)
    session.commit()
    session.refresh(backtest)
    return backtest


def list_backtests(session: Session, strategy_id: int) -> list[BacktestResult]:
    _ = get_strategy(session, strategy_id)
    return list(
        session.exec(
            select(BacktestResult)
            .where(BacktestResult.strategy_id == strategy_id)
            .order_by(BacktestResult.id.desc())
        ).all()
    )


def get_backtest(session: Session, backtest_id: int) -> BacktestResult:
    backtest = session.get(BacktestResult, backtest_id)
    if not backtest:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Backtest not found")
    return backtest


def run_backtest(session: Session, backtest_id: int) -> BacktestResult:
    """Start backtest execution via n8n webhook.
    
    Sends a minimal request to the agent's webhook with backtest_id and dates.
    The n8n workflow fetches full details from DB and should:
    1. Execute the backtest using edgewalker
    2. Call PATCH /strategies/backtests/{id} with results
    3. Call POST /strategies/backtests/{id}/trades/bulk with trades
    """
    backtest = get_backtest(session, backtest_id)
    
    if backtest.status != BacktestStatus.PENDING.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot run backtest with status '{backtest.status}'. Only 'pending' backtests can be run.",
        )
    
    if backtest.agent_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Backtest has no agent assigned. Set agent_id when creating the backtest.",
        )
    
    agent = session.get(Agent, backtest.agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent with id {backtest.agent_id} not found",
        )
    
    # Update status to running
    backtest.status = BacktestStatus.RUNNING.value
    backtest.started_at = datetime.now(timezone.utc)
    session.add(backtest)
    session.commit()
    session.refresh(backtest)
    
    # Prepare payload for n8n webhook with all parameters for fetch and backtest
    webhook_payload = {
        "action": "backtest",
        "backtest_id": backtest.id,
        "strategy_id": backtest.strategy_id,
        # Required parameters
        "symbol": backtest.symbol,
        "start_date": str(backtest.start_date),
        "end_date": str(backtest.end_date),
        # Data source parameters (for fetch)
        "source": backtest.source,
        "timeframe": backtest.timeframe,
        "asset": backtest.asset,
        "rth": backtest.rth,
        # IBKR-specific parameters
        "ibkr_config": backtest.ibkr_config,
        "exchange": backtest.exchange,
        "currency": backtest.currency,
        "expiry": backtest.expiry,
        # Backtest execution parameters
        "initial_capital": backtest.initial_capital,
        "commission": backtest.commission,
        # Additional config overrides
        "parameters": backtest.parameters,
    }
    
    # Call n8n webhook (fire and forget - don't wait for backtest to complete)
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(agent.n8n_webhook, json=webhook_payload)
            response.raise_for_status()
    except httpx.ConnectError as e:
        # Connection failed - n8n might be down or webhook URL is wrong
        backtest.status = BacktestStatus.FAILED.value
        backtest.completed_at = datetime.now(timezone.utc)
        backtest.error_message = f"Cannot connect to n8n webhook: {agent.n8n_webhook}"
        session.add(backtest)
        session.commit()
        session.refresh(backtest)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Cannot connect to n8n webhook at {agent.n8n_webhook}. Is n8n running?",
        )
    except httpx.HTTPStatusError as e:
        # Webhook returned an error status
        backtest.status = BacktestStatus.FAILED.value
        backtest.completed_at = datetime.now(timezone.utc)
        error_detail = f"Webhook returned {e.response.status_code}"
        try:
            error_body = e.response.text[:500]  # Limit error body size
            error_detail += f": {error_body}"
        except Exception:
            pass
        backtest.error_message = error_detail
        session.add(backtest)
        session.commit()
        session.refresh(backtest)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"n8n webhook error: {error_detail}",
        )
    except httpx.HTTPError as e:
        # Revert status on webhook failure
        backtest.status = BacktestStatus.FAILED.value
        backtest.completed_at = datetime.now(timezone.utc)
        backtest.error_message = f"Failed to call n8n webhook: {str(e)}"
        session.add(backtest)
        session.commit()
        session.refresh(backtest)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to trigger n8n workflow: {str(e)}",
        )
    
    return backtest


def update_backtest(
    session: Session, backtest_id: int, payload: BacktestUpdate
) -> BacktestResult:
    """Update backtest status and results (callback from n8n)."""
    backtest = get_backtest(session, backtest_id)
    
    if payload.status is not None:
        # Validate status transition
        valid_statuses = [s.value for s in BacktestStatus]
        if payload.status not in valid_statuses:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status '{payload.status}'. Must be one of: {valid_statuses}",
            )
        backtest.status = payload.status
        
        # Set completed_at when transitioning to terminal state
        if payload.status in [BacktestStatus.COMPLETED.value, BacktestStatus.FAILED.value, BacktestStatus.ERROR.value]:
            backtest.completed_at = datetime.now(timezone.utc)
    
    if payload.error_message is not None:
        backtest.error_message = payload.error_message
    
    # Update typed metrics
    if payload.return_pct is not None:
        backtest.return_pct = payload.return_pct
    if payload.sharpe_ratio is not None:
        backtest.sharpe_ratio = payload.sharpe_ratio
    if payload.max_drawdown_pct is not None:
        backtest.max_drawdown_pct = payload.max_drawdown_pct
    if payload.win_rate_pct is not None:
        backtest.win_rate_pct = payload.win_rate_pct
    if payload.profit_factor is not None:
        backtest.profit_factor = payload.profit_factor
    if payload.total_trades is not None:
        backtest.total_trades = payload.total_trades
    if payload.equity_final is not None:
        backtest.equity_final = payload.equity_final
    if payload.equity_peak is not None:
        backtest.equity_peak = payload.equity_peak
    
    # Extra metrics JSONB
    if payload.metrics is not None:
        backtest.metrics = payload.metrics
    
    if payload.html_report_url is not None:
        backtest.html_report_url = payload.html_report_url
    
    session.add(backtest)
    session.commit()
    session.refresh(backtest)
    return backtest


def delete_backtest(session: Session, backtest_id: int) -> None:
    """Delete a backtest and all its trades."""
    backtest = get_backtest(session, backtest_id)
    session.delete(backtest)
    session.commit()


def create_trade(session: Session, backtest_id: int, payload: TradeCreate) -> BacktestTrade:
    backtest = get_backtest(session, backtest_id)

    trade = BacktestTrade(
        backtest_id=backtest_id,
        strategy_id=backtest.strategy_id,
        entry_time=payload.entry_time,
        exit_time=payload.exit_time,
        direction=payload.direction,
        size=payload.size,
        entry_price=payload.entry_price,
        exit_price=payload.exit_price,
        pnl=payload.pnl,
        pnl_pct=payload.pnl_pct,
        session_date=payload.session_date,
        exit_reason=payload.exit_reason,
        extra=payload.extra,
    )
    session.add(trade)
    session.commit()
    session.refresh(trade)
    return trade


def list_trades(session: Session, backtest_id: int) -> list[BacktestTrade]:
    _ = get_backtest(session, backtest_id)
    return list(
        session.exec(
            select(BacktestTrade)
            .where(BacktestTrade.backtest_id == backtest_id)
            .order_by(BacktestTrade.id.asc())
        ).all()
    )


def create_trades_bulk(
    session: Session, backtest_id: int, trades: list[TradeCreate]
) -> list[BacktestTrade]:
    """Create multiple trades for a backtest in bulk."""
    backtest = get_backtest(session, backtest_id)
    
    created_trades = []
    for payload in trades:
        trade = BacktestTrade(
            backtest_id=backtest_id,
            strategy_id=backtest.strategy_id,
            entry_time=payload.entry_time,
            exit_time=payload.exit_time,
            direction=payload.direction,
            size=payload.size,
            entry_price=payload.entry_price,
            exit_price=payload.exit_price,
            pnl=payload.pnl,
            pnl_pct=payload.pnl_pct,
            session_date=payload.session_date,
            exit_reason=payload.exit_reason,
            extra=payload.extra,
        )
        session.add(trade)
        created_trades.append(trade)
    
    session.commit()
    for trade in created_trades:
        session.refresh(trade)
    
    return created_trades


# ─── CHAT FUNCTIONS ───


def list_strategy_chats(session: Session, strategy_id: int) -> list[Chat]:
    """List all chats for a strategy."""
    _ = get_strategy(session, strategy_id)
    return list(
        session.exec(
            select(Chat)
            .where(Chat.strategy_id == strategy_id)
            .order_by(Chat.created_at.desc())
        ).all()
    )


def create_strategy_chat(
    session: Session, strategy_id: int, payload: ChatCreate
) -> Chat:
    """Create a new chat for a strategy."""
    _ = get_strategy(session, strategy_id)
    
    now = datetime.now(timezone.utc)
    chat = Chat(
        strategy_id=strategy_id,
        id_agent=payload.id_agent,
        user_id=payload.user_id,
        nome=payload.nome,
        descrizione=payload.descrizione,
        chat_type=payload.chat_type or Chat.ChatType.GENERIC,
        created_at=now,
    )
    session.add(chat)
    session.commit()
    session.refresh(chat)
    return chat


def get_strategy_chat(session: Session, strategy_id: int, chat_id: int) -> Chat:
    """Get a specific chat for a strategy."""
    _ = get_strategy(session, strategy_id)
    chat = session.exec(
        select(Chat)
        .where(Chat.strategy_id == strategy_id)
        .where(Chat.id == chat_id)
    ).first()
    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
    return chat


def update_strategy_chat(
    session: Session, strategy_id: int, chat_id: int, payload: ChatCreate
) -> Chat:
    """Update a chat for a strategy."""
    chat = get_strategy_chat(session, strategy_id, chat_id)
    
    if payload.nome is not None:
        chat.nome = payload.nome
    if payload.descrizione is not None:
        chat.descrizione = payload.descrizione
    if payload.id_agent is not None:
        chat.id_agent = payload.id_agent
    if payload.chat_type is not None:
        chat.chat_type = payload.chat_type
    
    session.add(chat)
    session.commit()
    session.refresh(chat)
    return chat


def delete_strategy_chat(session: Session, strategy_id: int, chat_id: int) -> None:
    """Delete a chat from a strategy."""
    chat = get_strategy_chat(session, strategy_id, chat_id)
    session.delete(chat)
    session.commit()


# ─── RULE AGENT TRIGGER ───


def trigger_rule_agent(
    session: Session,
    agent_id: int,
    chat_id: int,
    rule_context: dict,
    webhook_url: str | None = None,
) -> dict:
    """Trigger an agent webhook when an ask_agent rule is activated.
    
    This function is called during backtest execution when a rule with
    action='ask_agent' has its conditions satisfied.
    
    Args:
        session: Database session
        agent_id: ID of the agent to trigger
        chat_id: ID of the chat to use for the conversation
        rule_context: Context data from the rule evaluation including:
            - rule_name: Name of the triggered rule
            - trigger_type: 'event' | 'time' | 'both'
            - timestamp: When the rule was triggered
            - bar_data: Current bar OHLCV data
            - indicators: Current indicator values
            - position: Current position info
            - conditions_matched: List of matched conditions
        webhook_url: Optional override for webhook URL (for backtest scenarios)
    
    Returns:
        Response from the agent webhook
    """
    # Get agent
    agent = session.get(Agent, agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent with id {agent_id} not found",
        )
    
    # Get chat and its session_id
    chat = session.get(Chat, chat_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat with id {chat_id} not found",
        )
    
    # Use provided webhook_url or fall back to agent's webhook
    target_webhook = webhook_url or agent.n8n_webhook
    
    # Build payload for the agent
    webhook_payload = {
        "action": "rule_trigger",
        "agent_id": agent_id,
        "chat_id": chat_id,
        "sessionId": chat.n8n_session_id,
        "rule_context": rule_context,
    }
    
    # Call agent webhook
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(target_webhook, json=webhook_payload)
            response.raise_for_status()
            return response.json() if response.text else {"status": "ok"}
    except httpx.ConnectError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Cannot connect to agent webhook at {target_webhook}: {str(e)}",
        )
    except httpx.HTTPStatusError as e:
        error_detail = f"Agent webhook returned {e.response.status_code}"
        try:
            error_body = e.response.text[:500]
            error_detail += f": {error_body}"
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=error_detail,
        )
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to call agent webhook: {str(e)}",
        )
