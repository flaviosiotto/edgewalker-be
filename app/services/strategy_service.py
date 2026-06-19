from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, cast

import httpx
from fastapi import HTTPException, status
from sqlalchemy.orm import selectinload
from sqlmodel import Session, select

from app.core.config import settings
from app.models.agent import Agent, Chat
from app.models.connection import Connection
from app.models.n8n_chat_history import N8nChatHistory
from app.models.strategy import Strategy, StrategyLive, BacktestResult, BacktestTrade, BacktestStatus
from app.schemas.strategy import (
    StrategyCreate,
    StrategyUpdate,
    BacktestCreate,
    BacktestUpdate,
    TradeCreate,
    LayoutConfigUpdate,
)
from app.schemas.chat import ChatCreate
from app.services.n8n_auth import (
    build_n8n_api_auth_metadata,
    build_n8n_backend_api_metadata,
    build_n8n_webhook_auth_headers,
    issue_n8n_api_access_token,
    issue_n8n_webhook_auth_token,
)
from app.utils.auth_utils import create_user_delegated_token

if TYPE_CHECKING:
    from sqlmodel import Session

logger = logging.getLogger(__name__)


def _coerce_position_accounting_mode(value: Any) -> str | None:
    text = str(value or "").strip().lower().replace("-", "_")
    if not text:
        return None
    aliases = {
        "net": "netting",
        "netting": "netting",
        "hedge": "hedging",
        "hedging": "hedging",
        "dual_side": "hedging",
        "ticket": "ticket_based",
        "ticket_based": "ticket_based",
        "position_id": "ticket_based",
    }
    return aliases.get(text)


def _resolve_backtest_position_accounting_mode(connection: Connection | None) -> str:
    config = connection.config if connection and isinstance(connection.config, dict) else {}
    for key in ("position_accounting_mode", "accounting_mode", "position_mode"):
        mode = _coerce_position_accounting_mode(config.get(key))
        if mode:
            return mode
    broker_type = str(connection.broker_type if connection else "").strip().lower()
    if broker_type in {"ctrader", "spotware"}:
        return "ticket_based"
    return "netting"


def _with_backtest_accounting_snapshot(config: Any, *, broker_type: str, position_accounting_mode: str) -> Any:
    if not isinstance(config, dict):
        return config
    snapshot = dict(config)
    backtest_cfg = dict(snapshot.get("backtest") or {}) if isinstance(snapshot.get("backtest"), dict) else {}
    backtest_cfg["broker_type"] = broker_type
    backtest_cfg["position_accounting_mode"] = position_accounting_mode
    snapshot["backtest"] = backtest_cfg
    return snapshot

_INDICATOR_OUTPUT_NAME_MAP = {
    "upperband": "upper",
    "middleband": "middle",
    "lowerband": "lower",
    "macdsignal": "signal",
    "macdhist": "histogram",
}

_INDICATOR_FIELD_PATTERN = re.compile(
    r"\b(?P<namespace>prev_indicators|indicators)\."
    r"(?P<name>[A-Za-z0-9_-]+)\."
    r"(?P<output>upperband|middleband|lowerband|macdsignal|macdhist)\b",
    re.IGNORECASE,
)


def _normalize_indicator_field_reference(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        output = match.group("output").lower()
        mapped_output = _INDICATOR_OUTPUT_NAME_MAP.get(output)
        if not mapped_output:
            return match.group(0)
        return f"{match.group('namespace')}.{match.group('name')}.{mapped_output}"

    return _INDICATOR_FIELD_PATTERN.sub(replace, value)


def _normalize_strategy_indicator_field_references(value: Any) -> Any:
    if isinstance(value, str):
        return _normalize_indicator_field_reference(value)
    if isinstance(value, list):
        items = cast(list[Any], value)
        return [_normalize_strategy_indicator_field_references(item) for item in items]
    if isinstance(value, dict):
        mapping = cast(dict[str, Any], value)
        return {
            key: _normalize_strategy_indicator_field_references(item)
            for key, item in mapping.items()
        }
    return value


def _get_owned_agent(session: Session, agent_id: int, user_id: int | None = None) -> Agent:
    agent = session.get(Agent, agent_id)
    if not agent or (user_id is not None and agent.user_id != user_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent with id {agent_id} not found",
        )
    return agent


def _get_owned_connection(session: Session, connection_id: int, user_id: int | None = None) -> Connection:
    connection = session.get(Connection, connection_id)
    if not connection or (user_id is not None and connection.user_id != user_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Connection with id {connection_id} not found",
        )
    return connection


def _chat_session_id(chat: Chat) -> str:
    if chat.n8n_session_id:
        return chat.n8n_session_id
    if chat.id is not None:
        return str(chat.id)
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Chat session id is not available",
    )


def _build_rule_trigger_chat_input(rule_context: dict) -> str:
    rule_name = str(rule_context.get("rule_name") or "ask_agent")
    timestamp = rule_context.get("timestamp")
    conditions = rule_context.get("conditions_matched")

    parts = [f"Rule '{rule_name}' triggered."]
    if timestamp:
        parts.append(f"Timestamp: {timestamp}.")
    if isinstance(conditions, list) and conditions:
        matched = ", ".join(str(condition) for condition in conditions[:5])
        parts.append(f"Matched conditions: {matched}.")
    parts.append("Review metadata.rule_context and decide the next action.")
    return " ".join(parts)


def _load_chat_with_agent(session: Session, chat_id: int) -> Chat:
    chat = session.exec(
        select(Chat)
        .options(selectinload(Chat.agent))
        .where(Chat.id == chat_id)
    ).first()
    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
    return chat


def resolve_strategy_manager_agent_id(
    session: Session,
    strategy_id: int,
    *,
    live_session: StrategyLive | None = None,
    user_id: int | None = None,
) -> int | None:
    """Resolve the effective manager agent for a strategy/live session.

    Historical live sessions should prefer their own manager snapshot. When that
    snapshot is missing, fall back to the strategy manager. As a final fallback,
    infer the manager from the most recent strategy-bound chat that is already
    linked to an agent.
    """
    if live_session and live_session.manager_agent_id is not None:
        return live_session.manager_agent_id

    strategy = get_strategy(session, strategy_id, user_id)
    if strategy.manager_agent_id is not None:
        return strategy.manager_agent_id

    inferred_chat = session.exec(
        select(Chat)
        .options(selectinload(Chat.agent))
        .where(Chat.strategy_id == strategy_id)
        .where(Chat.id_agent.is_not(None))
        .order_by(Chat.created_at.desc(), Chat.id.desc())
    ).first()
    return inferred_chat.id_agent if inferred_chat else None


def create_strategy(session: Session, payload: StrategyCreate, user_id: int) -> Strategy:
    name = (payload.name or "").strip()
    if not name:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Strategy name is required",
        )

    existing = session.exec(
        select(Strategy)
        .where(Strategy.user_id == user_id)
        .where(Strategy.name == name)
    ).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Strategy name already exists",
        )

    if payload.manager_agent_id is not None:
        _get_owned_agent(session, payload.manager_agent_id, user_id)

    if payload.connection_id is not None:
        _get_owned_connection(session, payload.connection_id, user_id)

    now = datetime.now(timezone.utc)
    strategy = Strategy(
        user_id=user_id,
        name=name,
        description=payload.description,
        definition=_normalize_strategy_indicator_field_references(payload.definition),
        manager_agent_id=payload.manager_agent_id,
        connection_id=payload.connection_id,
        created_at=now,
        updated_at=now,
    )
    session.add(strategy)
    session.commit()
    session.refresh(strategy)
    return strategy


def list_strategies(session: Session, user_id: int) -> list[Strategy]:
    stmt = select(Strategy).options(
        selectinload(Strategy.chats),
        selectinload(Strategy.live_sessions),
    ).where(Strategy.user_id == user_id).order_by(Strategy.id.desc())
    return list(session.exec(stmt).all())


def get_strategy(session: Session, strategy_id: int, user_id: int | None = None) -> Strategy:
    stmt = select(Strategy).options(
        selectinload(Strategy.chats),
        selectinload(Strategy.live_sessions),
    ).where(Strategy.id == strategy_id)
    if user_id is not None:
        stmt = stmt.where(Strategy.user_id == user_id)
    strategy = session.exec(stmt).first()
    if not strategy:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Strategy not found")
    return strategy


def update_strategy(session: Session, strategy_id: int, payload: StrategyUpdate, user_id: int | None = None) -> Strategy:
    strategy = get_strategy(session, strategy_id, user_id)

    if payload.name is not None and payload.name != strategy.name:
        name = (payload.name or "").strip()
        if not name:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Strategy name is required",
            )

        existing = session.exec(
            select(Strategy)
            .where(Strategy.user_id == strategy.user_id)
            .where(Strategy.name == name)
            .where(Strategy.id != strategy_id)
        ).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Strategy name already exists",
            )
        strategy.name = name

    if payload.description is not None:
        strategy.description = payload.description

    if payload.definition is not None:
        strategy.definition = _normalize_strategy_indicator_field_references(payload.definition)

    if payload.layout_config is not None:
        strategy.layout_config = payload.layout_config

    if payload.manager_agent_id is not None:
        # Validate agent exists
        if payload.manager_agent_id != 0:  # 0 means unset
            _get_owned_agent(session, payload.manager_agent_id, strategy.user_id)
            strategy.manager_agent_id = payload.manager_agent_id
        else:
            strategy.manager_agent_id = None

    if payload.connection_id is not None:
        if payload.connection_id == 0:
            strategy.connection_id = None
        else:
            _get_owned_connection(session, payload.connection_id, strategy.user_id)
            strategy.connection_id = payload.connection_id

    strategy.updated_at = datetime.now(timezone.utc)

    session.add(strategy)
    session.commit()
    session.refresh(strategy)
    return strategy


def delete_strategy(session: Session, strategy_id: int, user_id: int | None = None) -> None:
    strategy = get_strategy(session, strategy_id, user_id)
    session.delete(strategy)
    session.commit()


def create_backtest(
    session: Session,
    strategy_id: int,
    payload: BacktestCreate,
    user_id: int | None = None,
) -> BacktestResult:
    """Create a new backtest with status=pending.
    
    Automatically snapshots the strategy definition into the config field
    so the backtest retains the exact configuration used at creation time.
    """
    strategy = get_strategy(session, strategy_id, user_id)
    allowed_sources = {"ibkr", "yahoo", "binance", "ctrader"}
    source = str(payload.source or "").strip().lower()
    connection_source = ""
    if strategy.connection_id is not None:
        connection = _get_owned_connection(session, strategy.connection_id, strategy.user_id)
        connection_source = str(connection.broker_type or "").strip().lower()
    if not source or (source == "ibkr" and connection_source not in {"", "ibkr"}):
        source = connection_source
    if source not in allowed_sources:
        source = "ibkr"
    
    # Snapshot the manager agent used by this backtest instance.
    resolved_agent_id = payload.agent_id if payload.agent_id is not None else strategy.manager_agent_id
    if resolved_agent_id is not None:
        _get_owned_agent(session, resolved_agent_id, strategy.user_id)
    
    config_snapshot = _normalize_strategy_indicator_field_references(
        payload.config if payload.config else strategy.definition
    )

    backtest = BacktestResult(
        strategy_id=strategy_id,
        agent_id=resolved_agent_id,
        # Required parameters
        symbol=payload.symbol,
        start_date=payload.start_date,
        end_date=payload.end_date,
        # Data source parameters
        source=source,
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
        # Strategy configuration snapshot (explicit override or auto-capture)
        config=config_snapshot,
        # UI layout config (timezone, extended hours, etc.)
        layout_config=payload.layout_config,
        status=BacktestStatus.PENDING.value,
    )
    session.add(backtest)
    session.flush()
    chat = _create_backtest_chat(session, strategy, backtest, resolved_agent_id)
    backtest.chat_id = chat.id
    session.add(backtest)
    session.commit()
    session.refresh(backtest)
    return backtest


def list_backtests(session: Session, strategy_id: int, user_id: int | None = None) -> list[BacktestResult]:
    _ = get_strategy(session, strategy_id, user_id)
    return list(
        session.exec(
            select(BacktestResult)
            .where(BacktestResult.strategy_id == strategy_id)
            .order_by(BacktestResult.id.desc())
        ).all()
    )


def get_backtest(session: Session, backtest_id: int, user_id: int | None = None) -> BacktestResult:
    backtest = session.get(BacktestResult, backtest_id)
    if not backtest:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Backtest not found")
    if user_id is not None:
        _ = get_strategy(session, backtest.strategy_id, user_id)
    return backtest


def _resolve_backtest_agent_id(strategy: Strategy, backtest: BacktestResult) -> int | None:
    return backtest.agent_id if backtest.agent_id is not None else strategy.manager_agent_id


def _create_backtest_chat(
    session: Session,
    strategy: Strategy,
    backtest: BacktestResult,
    agent_id: int | None,
) -> Chat:
    if backtest.id is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Backtest id is not available",
        )

    chat = Chat(
        strategy_id=backtest.strategy_id,
        user_id=strategy.user_id,
        id_agent=agent_id,
        nome=f"Backtest #{backtest.id}",
        descrizione=f"Chat backtest {backtest.symbol} {backtest.start_date} - {backtest.end_date}",
        chat_type=Chat.ChatType.BACKTEST,
        created_at=datetime.now(timezone.utc),
    )
    session.add(chat)
    session.flush()
    logger.info("Created Backtest chat (id=%s) for backtest %s", chat.id, backtest.id)
    return chat


def get_or_create_backtest_chat(session: Session, backtest_id: int, user_id: int | None = None) -> Chat:
    """Get or create the dedicated chat for a single backtest instance."""
    backtest = get_backtest(session, backtest_id, user_id)
    strategy = get_strategy(session, backtest.strategy_id, user_id)
    resolved_agent_id = _resolve_backtest_agent_id(strategy, backtest)
    if resolved_agent_id is not None:
        _get_owned_agent(session, resolved_agent_id, strategy.user_id)

    if backtest.chat_id is not None:
        chat = _load_chat_with_agent(session, backtest.chat_id)
        changed = False
        if chat.chat_type != Chat.ChatType.BACKTEST:
            chat.chat_type = Chat.ChatType.BACKTEST
            changed = True
        if chat.strategy_id != backtest.strategy_id:
            chat.strategy_id = backtest.strategy_id
            changed = True
        if chat.id_agent != resolved_agent_id:
            chat.id_agent = resolved_agent_id
            changed = True
        if backtest.agent_id is None and resolved_agent_id is not None:
            backtest.agent_id = resolved_agent_id
            session.add(backtest)
            changed = True
        if changed:
            session.add(chat)
            session.commit()
            session.refresh(chat)
        return _load_chat_with_agent(session, chat.id)

    chat = _create_backtest_chat(session, strategy, backtest, resolved_agent_id)
    backtest.chat_id = chat.id
    if backtest.agent_id is None and resolved_agent_id is not None:
        backtest.agent_id = resolved_agent_id
    session.add(backtest)
    session.commit()
    session.refresh(backtest)
    return _load_chat_with_agent(session, chat.id)


def run_backtest(session: Session, backtest_id: int, user_id: int | None = None) -> BacktestResult:
    """Start backtest execution by spawning strategy-runner in backtest mode.

    ``strategy-backtest`` is an always-on service. It prepares historical data,
    publishes ``bars:backtest-{id}``, receives simulated orders from the runner,
    and writes final results. The backend only starts the runner container.
    """
    from app.services.backtest_runner_service import backtest_runner_service

    backtest = get_backtest(session, backtest_id, user_id)

    if backtest.status != BacktestStatus.PENDING.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot run backtest with status '{backtest.status}'. Only 'pending' backtests can be run.",
        )

    # Resolve connection_id from the strategy
    strategy = get_strategy(session, backtest.strategy_id, user_id)
    connection_id = strategy.connection_id if strategy else None

    if not connection_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Strategy has no connection_id assigned. "
                "Assign a data connection to the strategy before running a backtest."
            ),
        )
    connection = _get_owned_connection(session, connection_id, strategy.user_id)
    broker_type = str(connection.broker_type or "").strip().lower()
    position_accounting_mode = _resolve_backtest_position_accounting_mode(connection)

    try:
        backtest_chat = get_or_create_backtest_chat(session, backtest.id, user_id)
        runner_auth_token = create_user_delegated_token(
            session,
            user_id=strategy.user_id,
            audience=settings.RUNNER_TOKEN_AUDIENCE,
            purpose="runner_backend",
            extra_claims={
                "strategy_id": backtest.strategy_id,
                "backtest_id": backtest.id,
                "scopes": ["runner:read"],
            },
            no_expiry=True,
        )
        manager_webhook_url: str | None = None
        manager_webhook_auth_token: str | None = None
        manager_agent_id = backtest_chat.id_agent
        if manager_agent_id:
            manager_agent = session.get(Agent, manager_agent_id)
            if manager_agent:
                manager_webhook_url = manager_agent.n8n_webhook
                manager_webhook_auth_token = create_user_delegated_token(
                    session,
                    user_id=strategy.user_id,
                    audience=settings.N8N_TOKEN_AUDIENCE,
                    purpose="n8n_runner_webhook",
                    extra_claims={
                        "agent_id": manager_agent.id_agent,
                        "chat_id": backtest_chat.id,
                        "strategy_id": backtest.strategy_id,
                        "backtest_id": backtest.id,
                    },
                )

        raw_strategy_config = backtest.config or strategy.definition
        strategy_config = _normalize_strategy_indicator_field_references(raw_strategy_config)
        strategy_config = _with_backtest_accounting_snapshot(
            strategy_config,
            broker_type=broker_type,
            position_accounting_mode=position_accounting_mode,
        )
        if strategy_config != raw_strategy_config:
            backtest.config = strategy_config
            logger.info(
                "Normalized backtest %d config before runner start",
                backtest.id,
            )

        result = backtest_runner_service.start_backtest(
            backtest_id=backtest.id,
            connection_id=connection_id,
            strategy_id=backtest.strategy_id,
            strategy_config=strategy_config,
            symbol=backtest.symbol,
            timeframe=backtest.timeframe or "5m",
            broker_type=broker_type,
            position_accounting_mode=position_accounting_mode,
            manager_webhook_url=manager_webhook_url,
            backend_auth_token=runner_auth_token,
            manager_webhook_auth_token=manager_webhook_auth_token,
            manager_chat_session_id=_chat_session_id(backtest_chat),
            owner_user_id=strategy.user_id,
        )
        logger.info(
            "Started backtest runner for backtest %d: %s",
            backtest.id, result,
        )
        backtest.status = BacktestStatus.RUNNING.value
        backtest.started_at = datetime.now(timezone.utc)
        backtest.completed_at = None
        backtest.error_message = None
        session.add(backtest)
        session.commit()
        session.refresh(backtest)
    except RuntimeError as e:
        backtest.status = BacktestStatus.ERROR.value
        backtest.completed_at = datetime.now(timezone.utc)
        backtest.error_message = f"Failed to start backtest runner: {e}"
        session.add(backtest)
        session.commit()
        session.refresh(backtest)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start backtest runner: {e}",
        )

    return backtest


def update_backtest(
    session: Session,
    backtest_id: int,
    payload: BacktestUpdate,
    user_id: int | None = None,
) -> BacktestResult:
    """Update backtest status and results."""
    backtest = get_backtest(session, backtest_id, user_id)
    
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
    
    if payload.layout_config is not None:
        backtest.layout_config = payload.layout_config
    
    session.add(backtest)
    session.commit()
    session.refresh(backtest)
    return backtest


def delete_backtest(session: Session, backtest_id: int, user_id: int | None = None) -> None:
    """Delete a backtest and all its trades."""
    backtest = get_backtest(session, backtest_id, user_id)
    session.delete(backtest)
    session.commit()


def update_strategy_layout(
    session: Session,
    strategy_id: int,
    payload: LayoutConfigUpdate,
    user_id: int | None = None,
) -> Strategy:
    """Update only the layout_config of a strategy."""
    strategy = get_strategy(session, strategy_id, user_id)
    strategy.layout_config = payload.layout_config
    session.add(strategy)
    session.commit()
    session.refresh(strategy)
    return strategy


def update_backtest_layout(
    session: Session,
    backtest_id: int,
    payload: LayoutConfigUpdate,
    user_id: int | None = None,
) -> BacktestResult:
    """Update only the layout_config of a backtest."""
    backtest = get_backtest(session, backtest_id, user_id)
    backtest.layout_config = payload.layout_config
    session.add(backtest)
    session.commit()
    session.refresh(backtest)
    return backtest


def update_live_layout(
    session: Session,
    live_id: int,
    payload: LayoutConfigUpdate,
    user_id: int | None = None,
) -> StrategyLive:
    """Update only the layout_config of a live session."""
    sl = session.get(StrategyLive, live_id)
    if not sl:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Live session {live_id} not found",
        )
    if user_id is not None:
        _ = get_strategy(session, sl.strategy_id, user_id)
    sl.layout_config = payload.layout_config
    session.add(sl)
    session.commit()
    session.refresh(sl)
    return sl


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


def list_trades(session: Session, backtest_id: int, user_id: int | None = None) -> list[BacktestTrade]:
    _ = get_backtest(session, backtest_id, user_id)
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


def list_strategy_chats(session: Session, strategy_id: int, user_id: int | None = None) -> list[Chat]:
    """List all chats for a strategy."""
    strategy = get_strategy(session, strategy_id, user_id)
    return list(
        session.exec(
            select(Chat)
            .options(selectinload(Chat.agent))
            .where(Chat.strategy_id == strategy_id)
            .where(Chat.user_id == strategy.user_id)
            .order_by(Chat.created_at.desc())
        ).all()
    )


def create_strategy_chat(
    session: Session,
    strategy_id: int,
    payload: ChatCreate,
    user_id: int,
) -> Chat:
    """Create a new chat for a strategy."""
    strategy = get_strategy(session, strategy_id, user_id)

    if payload.id_agent is not None:
        _get_owned_agent(session, payload.id_agent, strategy.user_id)
    
    now = datetime.now(timezone.utc)
    chat = Chat(
        strategy_id=strategy_id,
        id_agent=payload.id_agent,
        user_id=strategy.user_id,
        nome=payload.nome,
        descrizione=payload.descrizione,
        chat_type=payload.chat_type or Chat.ChatType.GENERIC,
        created_at=now,
    )
    session.add(chat)
    session.commit()
    session.refresh(chat)
    return _load_chat_with_agent(session, chat.id)


def get_strategy_chat(
    session: Session,
    strategy_id: int,
    chat_id: int,
    user_id: int | None = None,
) -> Chat:
    """Get a specific chat for a strategy."""
    strategy = get_strategy(session, strategy_id, user_id)
    chat = session.exec(
        select(Chat)
        .options(selectinload(Chat.agent))
        .where(Chat.strategy_id == strategy_id)
        .where(Chat.id == chat_id)
        .where(Chat.user_id == strategy.user_id)
    ).first()
    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
    return chat


def update_strategy_chat(
    session: Session,
    strategy_id: int,
    chat_id: int,
    payload: ChatCreate,
    user_id: int | None = None,
) -> Chat:
    """Update a chat for a strategy."""
    chat = get_strategy_chat(session, strategy_id, chat_id, user_id)
    
    if payload.nome is not None:
        chat.nome = payload.nome
    if payload.descrizione is not None:
        chat.descrizione = payload.descrizione
    if payload.id_agent is not None:
        _get_owned_agent(session, payload.id_agent, chat.user_id)
        chat.id_agent = payload.id_agent
    if payload.chat_type is not None:
        chat.chat_type = payload.chat_type
    
    session.add(chat)
    session.commit()
    session.refresh(chat)
    return _load_chat_with_agent(session, chat.id)


def delete_strategy_chat(
    session: Session,
    strategy_id: int,
    chat_id: int,
    user_id: int | None = None,
) -> None:
    """Delete a chat from a strategy."""
    chat = get_strategy_chat(session, strategy_id, chat_id, user_id)
    session.delete(chat)
    session.commit()


# ─── RULE AGENT TRIGGER ───


def trigger_rule_agent(
    session: Session,
    agent_id: int,
    chat_id: int,
    rule_context: dict,
    webhook_url: str | None = None,
    user_id: int | None = None,
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
    agent = _get_owned_agent(session, agent_id, user_id)
    
    # Get chat and its session_id
    chat = session.get(Chat, chat_id)
    if not chat or (user_id is not None and chat.user_id != user_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat with id {chat_id} not found",
        )
    session_id = _chat_session_id(chat)
    request_id = str(uuid.uuid4())
    
    # Use provided webhook_url or fall back to agent's webhook
    target_webhook = webhook_url or agent.n8n_webhook

    webhook_auth_token = issue_n8n_webhook_auth_token(
        session,
        user_id=chat.user_id,
        purpose="n8n_rule_trigger",
        extra_claims={
            "agent_id": agent_id,
            "chat_id": chat_id,
            "request_id": request_id,
            "session_id": session_id,
        },
    )
    api_auth_token, api_auth_expires_at = issue_n8n_api_access_token(
        session,
        user_id=chat.user_id,
        purpose="n8n_rule_trigger_api_access",
        extra_claims={
            "agent_id": agent_id,
            "chat_id": chat_id,
            "session_id": session_id,
        },
    )
    api_auth_metadata = build_n8n_api_auth_metadata(
        user_id=chat.user_id,
        purpose="n8n_rule_trigger_api_access",
        token=api_auth_token,
        expires_at=api_auth_expires_at,
        backend_api=build_n8n_backend_api_metadata(
            session, user_id=chat.user_id, chat_id=chat_id,
        ),
    )
    
    # Build payload using the same sendMessage envelope as chat and runner flows.
    webhook_payload = {
        "action": "sendMessage",
        "sessionId": session_id,
        "chatInput": _build_rule_trigger_chat_input(rule_context),
        "metadata": {
            "chat_id": session_id,
            "edgewalker_chat_id": chat.id,
            "request_id": request_id,
            "message_type": "rule_trigger",
            "requested_action": "rule_trigger",
            "agent_id": agent_id,
            "rule_context": rule_context,
            "api_auth": api_auth_metadata,
        },
    }
    
    # Call agent webhook
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                target_webhook,
                json=webhook_payload,
                headers=build_n8n_webhook_auth_headers(webhook_auth_token),
            )
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


# ─── AI AGENT MANAGER ───


def get_or_create_live_chat(session: Session, strategy_id: int, user_id: int | None = None) -> Chat:
    """Get or create the dedicated 'Live' chat for a strategy.

    Every strategy has exactly one chat with ``chat_type='live'``.
    If none exists it is created automatically, linked to the strategy's
    manager agent.
    """
    strategy = get_strategy(session, strategy_id, user_id)
    resolved_manager_agent_id = resolve_strategy_manager_agent_id(session, strategy_id)

    # Look for existing live chat
    live_chat = session.exec(
        select(Chat)
        .where(Chat.strategy_id == strategy_id)
        .where(Chat.chat_type == Chat.ChatType.LIVE)
    ).first()

    if live_chat:
        # Ensure it points to the current manager agent
        if live_chat.id_agent != resolved_manager_agent_id:
            live_chat.id_agent = resolved_manager_agent_id
            session.add(live_chat)
            session.commit()
            session.refresh(live_chat)
        return _load_chat_with_agent(session, live_chat.id)

    # Create a new live chat
    live_chat = Chat(
        strategy_id=strategy_id,
        user_id=strategy.user_id,
        id_agent=resolved_manager_agent_id,
        nome=f"Live",
        descrizione="Chat live della strategia — l'agent manager riporta le attività in tempo reale",
        chat_type=Chat.ChatType.LIVE,
        created_at=datetime.now(timezone.utc),
    )
    session.add(live_chat)
    session.commit()
    session.refresh(live_chat)
    logger.info("Created Live chat (id=%s) for strategy %s", live_chat.id, strategy_id)
    return _load_chat_with_agent(session, live_chat.id)


def list_live_session_chats(session: Session, live_id: int, user_id: int | None = None) -> list[Chat]:
    """List chats relevant to a live session.

    The live page only shows the dedicated LIVE chat plus chats bound to the
    manager agent snapshot captured on the live session. This keeps the view
    historically coherent even if the strategy manager changes later.
    """
    live_session = session.get(StrategyLive, live_id)
    if not live_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Live session {live_id} not found",
        )

    live_chat = get_or_create_live_chat(session, live_session.strategy_id, user_id)
    manager_agent_id = resolve_strategy_manager_agent_id(
        session,
        live_session.strategy_id,
        live_session=live_session,
        user_id=user_id,
    )

    if manager_agent_id is not None and live_chat.id_agent != manager_agent_id:
        live_chat.id_agent = manager_agent_id
        session.add(live_chat)
        session.commit()
        session.refresh(live_chat)

    chats = list(
        session.exec(
            select(Chat)
            .options(selectinload(Chat.agent))
            .where(Chat.strategy_id == live_session.strategy_id)
            .order_by(Chat.created_at.desc())
        ).all()
    )

    relevant: dict[int, Chat] = {live_chat.id: live_chat} if live_chat.id is not None else {}
    for chat in chats:
        if chat.id is None:
            continue
        if chat.chat_type == Chat.ChatType.LIVE:
            relevant[chat.id] = chat
            continue
        if manager_agent_id is not None and chat.id_agent == manager_agent_id:
            relevant[chat.id] = chat

    return sorted(
        relevant.values(),
        key=lambda chat: (
            0 if chat.chat_type == Chat.ChatType.LIVE else 1,
            -(chat.created_at.timestamp() if chat.created_at else 0),
            -(chat.id or 0),
        ),
    )


def notify_manager_live_start(
    session: Session,
    strategy_id: int,
    symbol: str,
    timeframe: str,
    account_config: dict | None = None,
    user_id: int | None = None,
) -> dict | None:
    """Send a context message to the strategy's manager agent when going live.

    Creates a "Live" chat if it doesn't exist, writes a system message
    with the strategy context, and fires the manager's webhook so the
    agent is aware of the live session.

    Returns the webhook response dict, or ``None`` if no manager is set.
    """
    strategy = get_strategy(session, strategy_id, user_id)

    if not strategy.manager_agent_id:
        logger.info("Strategy %s has no manager agent — skipping notification", strategy_id)
        return None

    agent = _get_owned_agent(session, strategy.manager_agent_id, strategy.user_id)
    if not agent:
        logger.warning("Manager agent %s not found for strategy %s", strategy.manager_agent_id, strategy_id)
        return None

    # Ensure live chat exists
    live_chat = get_or_create_live_chat(session, strategy_id, strategy.user_id)

    # Build context payload
    context = {
        "strategy_id": strategy_id,
        "strategy_name": strategy.name,
        "symbol": symbol,
        "timeframe": timeframe,
        "definition": strategy.definition,
        "live_started_at": datetime.now(timezone.utc).isoformat(),
    }
    if account_config:
        context["account"] = {
            k: v for k, v in account_config.items()
            if k not in ("password", "secret", "token")
        }

    # Fire webhook (fire-and-forget style, but log errors)
    chat_input = (
        f"🚀 **Strategia avviata in Live**\n\n"
        f"- **Strategia**: {strategy.name}\n"
        f"- **Simbolo**: {symbol}\n"
        f"- **Timeframe**: {timeframe}\n"
        f"- **Avvio**: {context['live_started_at']}\n\n"
        f"Sono il manager di questa strategia. "
        f"Riporterò le attività in tempo reale e risponderò alle domande."
    )
    webhook_payload = {
        "action": "sendMessage",
        "sessionId": live_chat.n8n_session_id or "",
        "chatInput": chat_input,
        "metadata": {
            "chat_id": live_chat.n8n_session_id or "",
        },
    }
    try:
        from app.services.live_runner_service import _rewrite_webhook_for_docker
        webhook_url = _rewrite_webhook_for_docker(agent.n8n_webhook)
        webhook_auth_token = issue_n8n_webhook_auth_token(
            session,
            user_id=strategy.user_id,
            purpose="n8n_live_start",
            extra_claims={
                "agent_id": agent.id_agent,
                "chat_id": live_chat.id,
                "strategy_id": strategy_id,
                "session_id": live_chat.n8n_session_id or "",
            },
        )
        api_auth_token, api_auth_expires_at = issue_n8n_api_access_token(
            session,
            user_id=strategy.user_id,
            purpose="n8n_live_start_api_access",
            extra_claims={
                "agent_id": agent.id_agent,
                "chat_id": live_chat.id,
                "strategy_id": strategy_id,
                "session_id": live_chat.n8n_session_id or "",
            },
        )
        webhook_payload["metadata"]["api_auth"] = build_n8n_api_auth_metadata(
            user_id=strategy.user_id,
            purpose="n8n_live_start_api_access",
            token=api_auth_token,
            expires_at=api_auth_expires_at,
            backend_api=build_n8n_backend_api_metadata(
                session,
                user_id=strategy.user_id,
                chat_id=live_chat.id,
                strategy_id=strategy_id,
            ),
        )
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                webhook_url,
                json=webhook_payload,
                headers=build_n8n_webhook_auth_headers(webhook_auth_token),
            )
            response.raise_for_status()
            logger.info(
                "Notified manager agent '%s' about live start for strategy %s",
                agent.agent_name, strategy_id,
            )
            return response.json() if response.text else {"status": "ok"}
    except Exception as e:
        logger.warning(
            "Failed to notify manager agent '%s' for strategy %s: %s",
            agent.agent_name, strategy_id, e,
        )
        return None


def post_manager_message(
    session: Session,
    strategy_id: int,
    message: str,
    sender: str = "system",
    forward_to_webhook: bool = False,
    user_id: int | None = None,
) -> Chat:
    """Post a message to the strategy's live chat.

    Used by the strategy runner to send status updates to the manager.
    The message is written directly to the n8n chat history so it
    appears in the ChatWidget.

    When ``forward_to_webhook`` is True the message is also POSTed to
    the manager agent's n8n webhook (fire-and-forget).

    Args:
        session: DB session
        strategy_id: Strategy ID
        message: Text content of the message
        sender: 'system' | 'user' | 'ai' — maps to n8n message type
        forward_to_webhook: Also call the manager agent's webhook

    Returns:
        The live Chat object
    """
    live_chat = get_or_create_live_chat(session, strategy_id, user_id)

    msg_type = "ai" if sender in ("system", "ai", "agent") else "human"
    history_entry = N8nChatHistory(
        session_id=live_chat.n8n_session_id,
        message={
            "type": msg_type,
            "text": message,
        },
    )
    session.add(history_entry)
    session.commit()

    # Optionally forward to the manager agent's webhook
    if forward_to_webhook:
        strategy = get_strategy(session, strategy_id)
        if strategy.manager_agent_id:
            agent = session.get(Agent, strategy.manager_agent_id)
            if agent and agent.n8n_webhook:
                from app.services.live_runner_service import _rewrite_webhook_for_docker
                webhook_url = _rewrite_webhook_for_docker(agent.n8n_webhook)
                webhook_payload = {
                    "action": "sendMessage",
                    "sessionId": live_chat.n8n_session_id or "",
                    "chatInput": message,
                    "metadata": {
                        "chat_id": live_chat.n8n_session_id or "",
                    },
                }
                try:
                    webhook_auth_token = issue_n8n_webhook_auth_token(
                        session,
                        user_id=live_chat.user_id,
                        purpose="n8n_manager_message",
                        extra_claims={
                            "agent_id": agent.id_agent,
                            "chat_id": live_chat.id,
                            "strategy_id": strategy_id,
                            "session_id": live_chat.n8n_session_id or "",
                        },
                    )
                    api_auth_token, api_auth_expires_at = issue_n8n_api_access_token(
                        session,
                        user_id=live_chat.user_id,
                        purpose="n8n_manager_message_api_access",
                        extra_claims={
                            "agent_id": agent.id_agent,
                            "chat_id": live_chat.id,
                            "strategy_id": strategy_id,
                            "session_id": live_chat.n8n_session_id or "",
                        },
                    )
                    webhook_payload["metadata"]["api_auth"] = build_n8n_api_auth_metadata(
                        user_id=live_chat.user_id,
                        purpose="n8n_manager_message_api_access",
                        token=api_auth_token,
                        expires_at=api_auth_expires_at,
                        backend_api=build_n8n_backend_api_metadata(
                            session,
                            user_id=live_chat.user_id,
                            chat_id=live_chat.id,
                            strategy_id=strategy_id,
                        ),
                    )
                    headers = build_n8n_webhook_auth_headers(webhook_auth_token)
                    with httpx.Client(timeout=10.0) as client:
                        resp = client.post(webhook_url, json=webhook_payload, headers=headers)
                        resp.raise_for_status()
                except Exception as e:
                    logger.warning(
                        "Failed to forward message to manager webhook for strategy %s: %s",
                        strategy_id, e,
                    )

    return live_chat
