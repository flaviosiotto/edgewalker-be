"""Event sources for outbound webhooks (migr. 067).

The rows that mean "something happened" are written by four services
(backtest coordinator, strategy runner, order-aggregator, agent-svc) besides
the backend. Postgres triggers raise ``NOTIFY ew_webhook_events`` with
``{table, op, id}``; this module keeps a LISTEN connection in a thread
(same pattern as ``chat_realtime``), loads the row and turns it into an
event through :func:`handle_notification`, which is also what the tests
call directly. Every handler is idempotent thanks to ``dedupe_key``: two
backend replicas listening at once enqueue each delivery only once.

Credits exhaustion has no table write of its own: ``entitlement_service``
and ``wallet_service`` call ``emit_event`` directly where they send the
e-mail.
"""
from __future__ import annotations

import contextlib
import json
import logging
import select
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import psycopg2
import psycopg2.extensions
from sqlmodel import Session

from app.core.config import settings
from app.db.database import get_session_context
from app.models.agent import Agent
from app.models.agent_turn import AgentTurn
from app.models.connection import Account, Connection
from app.models.live_trading import LiveTrade
from app.models.strategy import BacktestResult, LiveAlert, Strategy, StrategyLive
from app.services.webhook_service import emit_event

logger = logging.getLogger(__name__)

NOTIFY_CHANNEL = "ew_webhook_events"

#: A trade re-materialised long after it closed is not news: the
#: order-aggregator rewrites whole (account, symbol) groups, so only trades
#: closed recently count as "trade closed".
TRADE_FRESHNESS = timedelta(minutes=30)

#: Connection statuses that deserve an event.
_CONNECTION_ALERT_STATUSES = {"disconnected", "error", "degraded"}
_LIVE_STATUSES_OF_INTEREST = {"running", "paused", "stopped", "error"}

_listener_thread: Optional[threading.Thread] = None
_listener_stop = threading.Event()


def _aware(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _strategy_of_live(session: Session, live: StrategyLive) -> Optional[Strategy]:
    return session.get(Strategy, live.strategy_id)


def _links(**paths: str) -> dict[str, str]:
    base = (settings.FRONTEND_BASE_URL or "").rstrip("/") if hasattr(settings, "FRONTEND_BASE_URL") else ""
    return {k: f"{base}{v}" if base else v for k, v in paths.items()}


# ── handlers ───────────────────────────────────────────────────────────────


def on_strategy_live(session: Session, live_id: int) -> int:
    live = session.get(StrategyLive, live_id)
    if live is None or live.status not in _LIVE_STATUSES_OF_INTEREST:
        return 0
    strategy = _strategy_of_live(session, live)
    if strategy is None:
        return 0
    stamp = _aware(live.updated_at) if getattr(live, "updated_at", None) else datetime.now(timezone.utc)
    rows = emit_event(
        session,
        user_id=strategy.user_id,
        event_type="live.status.changed",
        data={
            "live_id": live.id,
            "strategy_id": strategy.id,
            "strategy_name": strategy.name,
            "status": live.status,
            "previous_status": None,
            "symbol": live.symbol,
            "timeframe": live.timeframe,
            "account_id": live.account_id,
            "error_message": live.error_message,
            "started_at": live.started_at,
            "stopped_at": live.stopped_at,
        },
        links=_links(live=f"/strategies/{strategy.id}?live={live.id}"),
        # one event per (live, status, minute): a flapping status does not spam
        dedupe_key=f"live:{live.id}:{live.status}:{(stamp or datetime.now(timezone.utc)).strftime('%Y%m%d%H%M')}",
        commit=True,
    )
    return len(rows)


def on_strategy_backtest(session: Session, backtest_id: int) -> int:
    bt = session.get(BacktestResult, backtest_id)
    if bt is None or bt.status not in {"completed", "failed"}:
        return 0
    strategy = session.get(Strategy, bt.strategy_id)
    if strategy is None:
        return 0
    if bt.status == "completed":
        metrics = bt.metrics if isinstance(bt.metrics, dict) else {}
        # scalar metrics only: the ledger never travels in a webhook
        scalars = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool)) or v is None}
        data: dict[str, Any] = {
            "backtest_id": bt.id,
            "strategy_id": strategy.id,
            "strategy_name": strategy.name,
            "symbol": bt.symbol,
            "timeframe": bt.timeframe,
            "start_date": bt.start_date,
            "end_date": bt.end_date,
            "completed_at": bt.completed_at,
            "metrics": scalars,
            "return_pct": bt.return_pct,
            "sharpe_ratio": bt.sharpe_ratio,
            "max_drawdown_pct": bt.max_drawdown_pct,
            "win_rate_pct": bt.win_rate_pct,
            "profit_factor": bt.profit_factor,
        }
        event_type = "backtest.completed"
    else:
        data = {
            "backtest_id": bt.id,
            "strategy_id": strategy.id,
            "strategy_name": strategy.name,
            "error_message": bt.error_message,
            "completed_at": bt.completed_at,
        }
        event_type = "backtest.failed"
    rows = emit_event(
        session,
        user_id=strategy.user_id,
        event_type=event_type,
        data=data,
        links=_links(backtest=f"/strategies/{strategy.id}?backtest={bt.id}"),
        dedupe_key=f"backtest:{bt.id}:{bt.status}",
        commit=True,
    )
    return len(rows)


def on_live_alert(session: Session, alert_id: int) -> int:
    alert = session.get(LiveAlert, alert_id)
    if alert is None or alert.last_triggered_at is None:
        return 0
    live = session.get(StrategyLive, alert.strategy_live_id)
    if live is None:
        return 0
    strategy = _strategy_of_live(session, live)
    if strategy is None:
        return 0
    triggered_at = _aware(alert.last_triggered_at)
    rows = emit_event(
        session,
        user_id=strategy.user_id,
        event_type="live.alert.triggered",
        data={
            "live_id": live.id,
            "strategy_id": strategy.id,
            "strategy_name": strategy.name,
            "alert_id": alert.id,
            "alert_name": alert.name,
            "alert_type": alert.trigger_type,
            "trigger": alert.trigger,
            "message": alert.message,
            "symbol": live.symbol,
            "price": alert.last_triggered_price,
            "recipient": alert.recipient,
            "trigger_count": alert.trigger_count,
            "triggered_at": triggered_at,
        },
        links=_links(live=f"/strategies/{strategy.id}?live={live.id}"),
        dedupe_key=f"alert:{alert.id}:{triggered_at.isoformat() if triggered_at else alert.trigger_count}",
        commit=True,
    )
    return len(rows)


def on_trade(session: Session, trade_id: int) -> int:
    trade = session.get(LiveTrade, trade_id)
    if trade is None:
        return 0
    exit_time = _aware(trade.exit_time)
    if exit_time is None or datetime.now(timezone.utc) - exit_time > TRADE_FRESHNESS:
        return 0
    account = session.get(Account, trade.account_id)
    if account is None:
        return 0
    connection = session.get(Connection, account.connection_id)
    if connection is None:
        return 0
    live = session.get(StrategyLive, trade.strategy_live_id) if trade.strategy_live_id else None
    strategy = _strategy_of_live(session, live) if live is not None else None
    rows = emit_event(
        session,
        user_id=connection.user_id,
        event_type="live.trade.closed",
        data={
            "live_id": live.id if live else None,
            "strategy_id": strategy.id if strategy else None,
            "strategy_name": strategy.name if strategy else None,
            "account_id": account.id,
            "trade_id": trade.id,
            "symbol": trade.symbol,
            "side": trade.direction,
            "quantity": trade.quantity,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "realized_pnl": trade.realized_pnl,
            "net_pnl": trade.net_pnl,
            "commission": trade.commission,
            "pnl_currency": trade.currency or account.currency,
            "entry_time": trade.entry_time,
            "exit_time": exit_time,
            "exit_reason": (trade.extra or {}).get("exit_reason") if isinstance(trade.extra, dict) else None,
        },
        links=_links(account=f"/accounts/{account.id}"),
        # rows are rewritten by the projection: the fill (or the exit time) is the identity
        dedupe_key=f"trade:{account.id}:{trade.symbol}:{trade.exit_fill_id or exit_time.isoformat()}",
        commit=True,
    )
    return len(rows)


def on_connection(session: Session, connection_id: int) -> int:
    conn = session.get(Connection, connection_id)
    if conn is None or str(conn.status or "").lower() not in _CONNECTION_ALERT_STATUSES:
        return 0
    stamp = _aware(conn.updated_at) or datetime.now(timezone.utc)
    rows = emit_event(
        session,
        user_id=conn.user_id,
        event_type="connection.stale",
        data={
            "connection_id": conn.id,
            "connection_name": conn.name,
            "broker_type": conn.broker_type,
            "status": conn.status,
            "status_message": conn.status_message,
            "last_checked_at": conn.last_checked_at,
            "last_ok_at": conn.last_ok_at,
        },
        links=_links(connections="/settings?tab=connections"),
        dedupe_key=f"connection:{conn.id}:{conn.status}:{stamp.strftime('%Y%m%d%H%M')}",
        commit=True,
    )
    return len(rows)


def on_agent_turn(session: Session, turn_id: str) -> int:
    turn = session.get(AgentTurn, turn_id)
    if turn is None or turn.finished_at is None or turn.user_id is None:
        return 0
    agent = session.get(Agent, turn.agent_id) if turn.agent_id else None
    response = (turn.response or "").strip()
    summary = response[:500] + ("…" if len(response) > 500 else "")
    rows = emit_event(
        session,
        user_id=turn.user_id,
        event_type="agent.turn.completed",
        data={
            "turn_id": turn.turn_id,
            "chat_id": int(turn.session_id) if str(turn.session_id).isdigit() else None,
            "session_id": turn.session_id,
            "agent_id": turn.agent_id,
            "agent_name": agent.agent_name if agent else None,
            "strategy_id": turn.strategy_id,
            "live_id": turn.strategy_live_id,
            "backtest_id": turn.backtest_id,
            "message_type": turn.trigger_type or turn.kind,
            "status": turn.status,
            "error": turn.error,
            "summary": summary,
            "tool_calls": turn.tool_calls,
            "tokens_input": turn.tokens_input,
            "tokens_output": turn.tokens_output,
            "duration_ms": turn.duration_ms,
            "finished_at": turn.finished_at,
        },
        links=_links(chat=f"/strategies/{turn.strategy_id}" if turn.strategy_id else "/agents"),
        dedupe_key=f"turn:{turn.turn_id}",
        commit=True,
    )
    return len(rows)


_HANDLERS = {
    "strategy_live": on_strategy_live,
    "strategy_backtests": on_strategy_backtest,
    "live_alert": on_live_alert,
    "trades": on_trade,
    "connections": on_connection,
    "agent_turn": on_agent_turn,
}


def handle_notification(payload: dict[str, Any]) -> int:
    """Route one NOTIFY payload to its handler with a fresh session.
    Returns the number of deliveries enqueued. Never raises."""
    table = payload.get("table")
    row_id = payload.get("id")
    handler = _HANDLERS.get(str(table))
    if handler is None or row_id is None:
        return 0
    try:
        with get_session_context() as session:
            return int(handler(session, row_id))
    except Exception:  # noqa: BLE001
        logger.exception("webhook source handler failed: table=%s id=%s", table, row_id)
        return 0


# ── LISTEN thread ──────────────────────────────────────────────────────────


def _listener_main() -> None:
    backoff = 1.0
    while not _listener_stop.is_set():
        conn: Optional[psycopg2.extensions.connection] = None
        try:
            conn = psycopg2.connect(settings.DATABASE_URL)
            conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            with conn.cursor() as cur:
                cur.execute(f"LISTEN {NOTIFY_CHANNEL};")
            logger.info("Webhook sources listener connected (LISTEN %s)", NOTIFY_CHANNEL)
            backoff = 1.0
            while not _listener_stop.is_set():
                readable, _, _ = select.select([conn], [], [], 5.0)
                if not readable:
                    continue
                conn.poll()
                while conn.notifies:
                    notify = conn.notifies.pop(0)
                    try:
                        payload = json.loads(notify.payload)
                    except (ValueError, TypeError):
                        continue
                    if isinstance(payload, dict):
                        handle_notification(payload)
        except Exception as exc:  # noqa: BLE001
            if _listener_stop.is_set():
                break
            logger.warning("Webhook sources listener error: %s; reconnecting in %.1fs", exc, backoff)
            _listener_stop.wait(backoff)
            backoff = min(backoff * 2, 30.0)
        finally:
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.close()


def start_webhook_sources() -> None:
    global _listener_thread
    if _listener_thread is not None and _listener_thread.is_alive():
        return
    _listener_stop.clear()
    _listener_thread = threading.Thread(target=_listener_main, name="webhook-sources-listener", daemon=True)
    _listener_thread.start()


def stop_webhook_sources() -> None:
    global _listener_thread
    _listener_stop.set()
    if _listener_thread is not None:
        _listener_thread.join(timeout=10)
        _listener_thread = None
