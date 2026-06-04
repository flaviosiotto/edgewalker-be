"""Helpers to build live-instance summaries.

Extracted from app.api.live to allow reuse from the strategies API
(so GET /strategies/ can include a per-strategy live_summary).
"""
from __future__ import annotations

from typing import Any

from sqlmodel import Session, select

from app.models.connection import Account, Connection
from app.models.live_trading import LiveFill, LivePosition, LiveTrade, PositionStatus
from app.models.strategy import LiveStatus, Strategy, StrategyLive
from app.schemas.live_strategy import LivePerformanceSummary, LiveStrategySummaryRead
from app.services.live_runner_service import live_runner_service


def compute_live_performance_summary(session: Session, sl: StrategyLive) -> LivePerformanceSummary:
    # Broker-synced orders/fills/trades are account-scoped (strategy_live_id is
    # NULL), so the card must aggregate from the canonical account+symbol
    # projection (trades + account_positions), consistent with the dashboard and
    # the Trade History view. Falls back to strategy_live scope when the session
    # has no account binding.
    if sl.account_id is not None:
        return _compute_account_scoped_summary(session, sl)
    return _compute_strategy_scoped_summary(session, sl)


def _compute_account_scoped_summary(session: Session, sl: StrategyLive) -> LivePerformanceSummary:
    trades = session.exec(
        select(LiveTrade)
        .where(LiveTrade.account_id == sl.account_id)
        .where(LiveTrade.symbol == sl.symbol)
        .order_by(LiveTrade.exit_time.desc())
    ).all()
    positions = session.exec(
        select(LivePosition)
        .where(LivePosition.account_id == sl.account_id)
        .where(LivePosition.symbol == sl.symbol)
        .order_by(LivePosition.updated_at.desc())
    ).all()

    open_positions = [
        pos for pos in positions
        if pos.status == PositionStatus.OPEN.value and abs(float(pos.quantity or 0.0)) > 1e-9
    ]
    realized_trades = [trade for trade in trades if trade.realized_pnl is not None]

    realized_pnl = sum((trade.realized_pnl or 0.0) for trade in realized_trades)
    commission = sum((trade.commission or 0.0) for trade in realized_trades)
    net_pnl = realized_pnl - commission
    unrealized_pnl = 0.0
    total_trades = len(realized_trades)
    wins = sum(1 for trade in realized_trades if (trade.realized_pnl or 0.0) > 0)
    win_rate = (wins / total_trades * 100.0) if total_trades else None
    position_side = open_positions[0].side if open_positions else "flat"

    last_activity_candidates = [sl.started_at, sl.updated_at]
    if trades:
        last_activity_candidates.append(trades[0].exit_time)
    if positions:
        last_activity_candidates.append(positions[0].updated_at)

    last_activity_at = max(ts for ts in last_activity_candidates if ts is not None)

    return LivePerformanceSummary(
        total_pnl=net_pnl + unrealized_pnl,
        total_trades=total_trades,
        win_rate=win_rate,
        position_side=position_side if position_side in {"long", "short", "flat"} else "flat",
        has_open_position=bool(open_positions),
        open_positions=len(open_positions),
        last_activity_at=last_activity_at,
    )


def _compute_strategy_scoped_summary(session: Session, sl: StrategyLive) -> LivePerformanceSummary:
    positions = session.exec(
        select(LivePosition)
        .where(LivePosition.strategy_live_id == sl.id)
        .order_by(LivePosition.updated_at.desc())
    ).all()
    fills = session.exec(
        select(LiveFill)
        .where(LiveFill.strategy_live_id == sl.id)
        .order_by(LiveFill.fill_time.desc())
    ).all()

    open_positions = [pos for pos in positions if pos.status == PositionStatus.OPEN.value]
    realized_fills = [fill for fill in fills if fill.realized_pnl is not None]

    realized_pnl = sum((fill.realized_pnl or 0.0) for fill in realized_fills)
    commission = sum((fill.commission or 0.0) for fill in realized_fills)
    net_pnl = realized_pnl - commission
    unrealized_pnl = 0.0
    total_trades = len(realized_fills)
    wins = sum(1 for fill in realized_fills if (fill.realized_pnl or 0.0) > 0)
    win_rate = (wins / total_trades * 100.0) if total_trades else None
    position_side = open_positions[0].side if open_positions else "flat"

    last_activity_candidates = [sl.started_at, sl.updated_at]
    if fills:
        last_activity_candidates.append(fills[0].fill_time)
    if positions:
        last_activity_candidates.append(positions[0].updated_at)

    last_activity_at = max(ts for ts in last_activity_candidates if ts is not None)

    return LivePerformanceSummary(
        total_pnl=net_pnl + unrealized_pnl,
        total_trades=total_trades,
        win_rate=win_rate,
        position_side=position_side if position_side in {"long", "short", "flat"} else "flat",
        has_open_position=bool(open_positions),
        open_positions=len(open_positions),
        last_activity_at=last_activity_at,
    )


def load_container_info_for_live(sl: StrategyLive) -> dict[str, Any]:
    if sl.id is None:
        return {"status": "not_found", "running": False}
    return live_runner_service.get_live_instance_status(sl.id)


def derive_sync_state(sl: StrategyLive, container_info: dict[str, Any]) -> str:
    container_status = container_info.get("status")
    if container_status == "not_found":
        if sl.status in {
            LiveStatus.RUNNING.value,
            LiveStatus.STARTING.value,
            LiveStatus.STOPPING.value,
        }:
            return "missing_container"
        return "aligned"

    if sl.status == LiveStatus.RUNNING.value and container_status != "running":
        return "stale"
    if sl.status == LiveStatus.STARTING.value and container_status not in {"created", "running", "restarting"}:
        return "stale"
    if sl.status == LiveStatus.STOPPING.value and container_status not in {"exited", "dead", "not_found"}:
        return "stale"
    return "aligned"


def build_live_summary_payload(session: Session, sl: StrategyLive) -> dict[str, Any]:
    strategy = session.get(Strategy, sl.strategy_id)
    account = session.get(Account, sl.account_id) if sl.account_id else None
    connection = session.get(Connection, sl.connection_id) if sl.connection_id else None

    account_display: str | None = None
    if account is not None:
        account_display = account.display_name or account.account_id
        if connection is not None:
            account_display = f"{account_display} ({connection.name})"

    return {
        "id": sl.id,
        "strategy_id": sl.strategy_id,
        "strategy_name": strategy.name if strategy else f"Strategy {sl.strategy_id}",
        "status": sl.status,
        "symbol": sl.symbol,
        "timeframe": sl.timeframe,
        "account_id": sl.account_id,
        "account_display": account_display,
        "connection_id": sl.connection_id,
        "connection_name": connection.name if connection else None,
        "started_at": sl.started_at,
        "stopped_at": sl.stopped_at,
        "error_message": sl.error_message,
        "performance_summary": compute_live_performance_summary(session, sl),
        "created_at": sl.created_at,
        "updated_at": sl.updated_at,
    }


def serialize_live_summary_from_payload(
    sl: StrategyLive,
    payload: dict[str, Any],
    *,
    container_info: dict[str, Any] | None = None,
) -> LiveStrategySummaryRead:
    resolved_container_info = container_info or load_container_info_for_live(sl)

    return LiveStrategySummaryRead(
        **payload,
        sync_state=derive_sync_state(sl, resolved_container_info),
        container_id=resolved_container_info.get("container_id") or sl.container_id,
        container_name=resolved_container_info.get("container_name"),
        container_status=resolved_container_info.get("status"),
        container_health=resolved_container_info.get("health_status"),
    )


def build_live_summary(session: Session, sl: StrategyLive) -> LiveStrategySummaryRead:
    payload = build_live_summary_payload(session, sl)
    return serialize_live_summary_from_payload(sl, payload)
