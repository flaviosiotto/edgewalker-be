"""Helpers to build live-instance summaries.

Extracted from app.api.live to allow reuse from the strategies API
(so GET /strategies/ can include a per-strategy live_summary).
"""
from __future__ import annotations

from typing import Any

from sqlmodel import Session, select

from app.models.connection import Account, Connection
from app.models.live_trading import LivePosition, PositionStatus
from app.models.strategy import LiveStatus, Strategy, StrategyLive
from app.schemas.live_strategy import LivePerformanceSummary, LiveStrategySummaryRead
from app.services.live_runner_service import live_runner_service
from app.services.performance_service import compute_live_performance


def compute_live_performance_summary(session: Session, sl: StrategyLive) -> LivePerformanceSummary:
    """Card/status summary of a live session from the single performance ledger.

    Scope = trades attributed to this session (``trades.strategy_live_id``),
    window = the session lifetime. Unrealized P&L is not derived from the DB:
    it stays ``None`` and the FE overlays the broker mark-to-market from the
    realtime portfolio plane. ``total_pnl`` is the ledger net (gross realized +
    swap − costs).
    """
    stats = compute_live_performance(session, live=sl, daily=False)
    totals = stats.totals

    positions: list[LivePosition] = []
    if sl.account_id is not None and sl.symbol:
        positions = session.exec(
            select(LivePosition)
            .where(LivePosition.account_id == sl.account_id)
            .where(LivePosition.symbol == sl.symbol)
            .order_by(LivePosition.updated_at.desc())  # type: ignore[union-attr]
        ).all()
    open_positions = [
        pos for pos in positions
        if pos.status == PositionStatus.OPEN.value and abs(float(pos.quantity or 0.0)) > 1e-9
    ]
    position_side = open_positions[0].side if open_positions else "flat"

    last_activity_candidates = [sl.started_at, sl.updated_at, totals.last_exit_at]
    if positions:
        last_activity_candidates.append(positions[0].updated_at)
    last_activity_at = max(ts for ts in last_activity_candidates if ts is not None)

    return LivePerformanceSummary(
        total_pnl=totals.net if totals.trades else (0.0 if totals.unreconciled_trades == 0 else None),
        realized_gross=totals.realized_gross,
        commission=totals.commission,
        swap=totals.swap,
        net_pnl=totals.net,
        net_pnl_account_ccy=totals.net_account_ccy,
        currency=totals.currency,
        unrealized_pnl=None,
        total_trades=totals.trades,
        unreconciled_trades=totals.unreconciled_trades,
        win_rate=totals.win_rate,
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
        if sl.status in LiveStatus.active_values():
            return "missing_container"
        return "aligned"

    if sl.status in {LiveStatus.RUNNING.value, LiveStatus.PAUSED.value} and container_status != "running":
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
