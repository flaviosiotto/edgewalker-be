"""Live trading read-side services and reporting helpers.

Canonical order, fill, and position writes are owned by the runtime
order-aggregator service. Backend services should treat these tables as
read-only projections.
"""
from __future__ import annotations

from calendar import monthrange
from collections import defaultdict
import logging
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from sqlalchemy import delete
from sqlmodel import Session, select

from app.models.connection import Account, Connection, ConnectionStatus
from app.models.live_trading import (
    LiveFill,
    LiveOrder,
    LivePosition,
    LiveTrade,
    OrderStatus,
    PositionStatus,
)
from app.services.performance_service import compute_account_performance
from app.models.strategy import LiveStatus, Strategy, StrategyLive
from app.schemas.live_trading import (
    LiveOrderCreate,
    LiveOrderRead,
    LiveOrderUpdate,
    LivePositionCreate,
    LivePositionUpdate,
    LiveFillCreate,
    ReconciliationItem,
    ReconciliationReport,
)
from app.schemas.live_strategy import (
    LiveDashboardAccountBreakdownRead,
    LiveDashboardDailyResultRead,
    LiveDashboardDateRange,
    LiveDashboardEquityPointRead,
    LiveDashboardOverviewRead,
    LiveDashboardSummaryRead,
)

logger = logging.getLogger(__name__)


def _coerce_float(value: Any) -> float | None:
    try:
        if value in (None, "", "N/A"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _projection_write_disabled() -> None:
    raise RuntimeError(
        "live trading projection is read-only in the backend; order-aggregator owns writes"
    )


# ═════════════════════════════════════════════════════════════════════
# ORDERS
# ═════════════════════════════════════════════════════════════════════


def create_live_order(
    session: Session,
    strategy_live_id: int,
    account_id: int | None,
    payload: LiveOrderCreate,
) -> LiveOrder:
    _ = (session, strategy_live_id, account_id, payload)
    _projection_write_disabled()


def update_live_order(
    session: Session,
    order_id: int,
    payload: LiveOrderUpdate,
) -> LiveOrder | None:
    _ = (session, order_id, payload)
    _projection_write_disabled()


def get_live_order(session: Session, order_id: int) -> LiveOrder | None:
    return session.get(LiveOrder, order_id)


def list_live_orders(
    session: Session,
    strategy_live_id: int,
    *,
    status: str | None = None,
    limit: int = 100,
) -> list[LiveOrder]:
    """List live orders for a strategy live session, optionally filtered by status."""
    stmt = (
        select(LiveOrder)
        .where(LiveOrder.strategy_live_id == strategy_live_id)
        .order_by(LiveOrder.created_at.desc())  # type: ignore[union-attr]
        .limit(limit)
    )
    if status:
        stmt = stmt.where(LiveOrder.status == status)
    return list(session.exec(stmt).all())


def list_active_orders(session: Session, strategy_live_id: int) -> list[LiveOrder]:
    """List orders that are still active (pending, submitted, partially_filled)."""
    active_statuses = [
        OrderStatus.PENDING.value,
        OrderStatus.SUBMITTED.value,
        OrderStatus.PARTIALLY_FILLED.value,
    ]
    stmt = (
        select(LiveOrder)
        .where(LiveOrder.strategy_live_id == strategy_live_id)
        .where(LiveOrder.status.in_(active_statuses))  # type: ignore[union-attr]
        .order_by(LiveOrder.created_at.desc())  # type: ignore[union-attr]
    )
    return list(session.exec(stmt).all())


def list_account_orders(
    session: Session,
    account_id: int,
    *,
    status: str | None = None,
    active_only: bool = False,
    symbol: str | None = None,
    limit: int = 100,
) -> list[LiveOrderRead]:
    """List persisted orders for a broker account across all live sessions."""
    stmt = (
        select(LiveOrder)
        .where(LiveOrder.account_id == account_id)
        .order_by(LiveOrder.created_at.desc())  # type: ignore[union-attr]
        .limit(limit)
    )
    if symbol:
        stmt = stmt.where(LiveOrder.symbol == symbol.upper())
    if active_only:
        stmt = stmt.where(
            LiveOrder.status.in_([
                OrderStatus.PENDING.value,
                OrderStatus.SUBMITTED.value,
                OrderStatus.PARTIALLY_FILLED.value,
            ])
        )  # type: ignore[union-attr]
    elif status:
        stmt = stmt.where(LiveOrder.status == status)

    orders = list(session.exec(stmt).all())
    if not orders:
        return []

    live_ids = sorted({order.strategy_live_id for order in orders if order.strategy_live_id})
    if not live_ids:
        return [LiveOrderRead.model_validate(order) for order in orders]

    strategy_name_rows = session.exec(
        select(StrategyLive.id, Strategy.name)
        .join(Strategy, StrategyLive.strategy_id == Strategy.id)
        .where(StrategyLive.id.in_(live_ids))  # type: ignore[union-attr]
    ).all()
    strategy_names_by_live_id = {
        int(live_id): strategy_name
        for live_id, strategy_name in strategy_name_rows
    }

    return [
        LiveOrderRead.model_validate(order).model_copy(
            update={"strategy_name": strategy_names_by_live_id.get(order.strategy_live_id)},
        )
        for order in orders
    ]


def purge_account_orders(session: Session, account_id: int) -> int:
    """Delete persisted orders for one broker account without dropping fills."""
    order_ids = list(session.exec(
        select(LiveOrder.id).where(LiveOrder.account_id == account_id)
    ).all())
    if not order_ids:
        return 0

    session.exec(
        delete(LiveOrder).where(LiveOrder.account_id == account_id)
    )
    session.commit()
    return len(order_ids)


def purge_account_fills(session: Session, account_id: int) -> int:
    """Delete persisted fills for one broker account."""
    fill_ids = list(session.exec(
        select(LiveFill.id).where(LiveFill.account_id == account_id)
    ).all())
    if not fill_ids:
        return 0

    session.exec(
        delete(LiveFill).where(LiveFill.account_id == account_id)
    )
    session.commit()
    return len(fill_ids)


def purge_account_trades(session: Session, account_id: int) -> int:
    """Delete materialized round-trip trades for one broker account."""
    trade_ids = list(session.exec(
        select(LiveTrade.id).where(LiveTrade.account_id == account_id)
    ).all())
    if not trade_ids:
        return 0

    session.exec(
        delete(LiveTrade).where(LiveTrade.account_id == account_id)
    )
    session.commit()
    return len(trade_ids)


# ═════════════════════════════════════════════════════════════════════
# FILLS
# ═════════════════════════════════════════════════════════════════════


def create_live_fill(
    session: Session,
    strategy_live_id: int,
    account_id: int | None,
    payload: LiveFillCreate,
) -> LiveFill:
    _ = (session, strategy_live_id, account_id, payload)
    _projection_write_disabled()


def list_live_fills(
    session: Session,
    strategy_live_id: int,
    *,
    limit: int = 200,
) -> list[LiveFill]:
    """List live fills for a strategy live session."""
    stmt = (
        select(LiveFill)
        .where(LiveFill.strategy_live_id == strategy_live_id)
        .order_by(LiveFill.fill_time.desc())  # type: ignore[union-attr]
        .limit(limit)
    )
    return list(session.exec(stmt).all())


def list_account_fills(
    session: Session,
    account_id: int,
    *,
    symbol: str | None = None,
    limit: int = 200,
) -> list[LiveFill]:
    """List persisted fills for a broker account across all live sessions."""
    stmt = (
        select(LiveFill)
        .where(LiveFill.account_id == account_id)
        .order_by(LiveFill.fill_time.desc())  # type: ignore[union-attr]
        .limit(limit)
    )
    if symbol:
        stmt = stmt.where(LiveFill.symbol == symbol.upper())
    return list(session.exec(stmt).all())

def list_account_trades(
    session: Session,
    account_id: int,
    *,
    symbol: str | None = None,
    limit: int = 200,
) -> list[LiveTrade]:
    """List materialized FIFO trades (closed lots) for a broker account."""
    stmt = (
        select(LiveTrade)
        .where(LiveTrade.account_id == account_id)
        .order_by(LiveTrade.exit_time.desc(), LiveTrade.id.desc())  # type: ignore[union-attr]
        .limit(limit)
    )
    if symbol:
        stmt = stmt.where(LiveTrade.symbol == symbol.upper())
    return list(session.exec(stmt).all())

# ═════════════════════════════════════════════════════════════════════
# POSITIONS
# ═════════════════════════════════════════════════════════════════════


def get_open_position(
    session: Session,
    strategy_live_id: int,
    symbol: str,
    account_id: int | None = None,
) -> LivePosition | None:
    """Get the open position for a strategy_live+symbol+account (max one by DB constraint)."""
    stmt = (
        select(LivePosition)
        .where(LivePosition.strategy_live_id == strategy_live_id)
        .where(LivePosition.symbol == symbol)
        .where(LivePosition.status == PositionStatus.OPEN.value)
    )
    if account_id is not None:
        stmt = stmt.where(LivePosition.account_id == account_id)
    return session.exec(stmt).first()


def upsert_position(
    session: Session,
    strategy_live_id: int,
    account_id: int | None,
    payload: LivePositionCreate,
) -> LivePosition:
    _ = (session, strategy_live_id, account_id, payload)
    _projection_write_disabled()


def list_positions(
    session: Session,
    strategy_live_id: int,
    *,
    status: str | None = None,
    limit: int = 100,
) -> list[LivePosition]:
    """List positions for a strategy live session."""
    stmt = (
        select(LivePosition)
        .where(LivePosition.strategy_live_id == strategy_live_id)
        .order_by(LivePosition.opened_at.desc())  # type: ignore[union-attr]
        .limit(limit)
    )
    if status:
        stmt = stmt.where(LivePosition.status == status)
    return list(session.exec(stmt).all())


def list_open_positions(session: Session, strategy_live_id: int) -> list[LivePosition]:
    """List all open positions for a strategy live session."""
    return list_positions(session, strategy_live_id, status=PositionStatus.OPEN.value)


def list_account_positions(
    session: Session,
    account_id: int,
    *,
    symbol: str | None = None,
    limit: int = 200,
) -> list[LivePosition]:
    """List broker-authoritative current positions for an account."""
    stmt = (
        select(LivePosition)
        .where(LivePosition.account_id == account_id)
        .order_by(LivePosition.symbol.asc(), LivePosition.position_key.asc())  # type: ignore[union-attr]
        .limit(limit)
    )
    if symbol:
        stmt = stmt.where(LivePosition.symbol == symbol.upper())
    return list(session.exec(stmt).all())


def _live_ids_for_strategy(session: Session, strategy_id: int) -> list[int]:
    """Return all strategy_live IDs for a given strategy."""
    stmt = select(StrategyLive.id).where(StrategyLive.strategy_id == strategy_id)
    return list(session.exec(stmt).all())


def list_strategy_orders(
    session: Session,
    strategy_id: int,
    *,
    status: str | None = None,
    limit: int = 200,
) -> list[LiveOrder]:
    """List orders across ALL sessions for a strategy."""
    live_ids = _live_ids_for_strategy(session, strategy_id)
    if not live_ids:
        return []
    stmt = (
        select(LiveOrder)
        .where(LiveOrder.strategy_live_id.in_(live_ids))  # type: ignore[union-attr]
        .order_by(LiveOrder.created_at.desc())  # type: ignore[union-attr]
        .limit(limit)
    )
    if status:
        stmt = stmt.where(LiveOrder.status == status)
    return list(session.exec(stmt).all())


def list_strategy_fills(
    session: Session,
    strategy_id: int,
    *,
    limit: int = 500,
) -> list[LiveFill]:
    """List fills across ALL sessions for a strategy."""
    live_ids = _live_ids_for_strategy(session, strategy_id)
    if not live_ids:
        return []
    stmt = (
        select(LiveFill)
        .where(LiveFill.strategy_live_id.in_(live_ids))  # type: ignore[union-attr]
        .order_by(LiveFill.fill_time.desc())  # type: ignore[union-attr]
        .limit(limit)
    )
    return list(session.exec(stmt).all())


def list_strategy_positions(
    session: Session,
    strategy_id: int,
    *,
    status: str | None = None,
    limit: int = 200,
) -> list[LivePosition]:
    """List positions across ALL sessions for a strategy."""
    live_ids = _live_ids_for_strategy(session, strategy_id)
    if not live_ids:
        return []
    stmt = (
        select(LivePosition)
        .where(LivePosition.strategy_live_id.in_(live_ids))  # type: ignore[union-attr]
        .order_by(LivePosition.opened_at.desc())  # type: ignore[union-attr]
        .limit(limit)
    )
    if status:
        stmt = stmt.where(LivePosition.status == status)
    return list(session.exec(stmt).all())


def _resolve_dashboard_range(
    start_date: date | None,
    end_date: date | None,
) -> tuple[date, date]:
    today = datetime.now(timezone.utc).date()

    if start_date is None and end_date is None:
        start_date = today.replace(day=1)
        end_date = date(today.year, today.month, monthrange(today.year, today.month)[1])
    elif start_date is None and end_date is not None:
        start_date = end_date.replace(day=1)
    elif start_date is not None and end_date is None:
        end_date = date(start_date.year, start_date.month, monthrange(start_date.year, start_date.month)[1])

    assert start_date is not None
    assert end_date is not None

    if end_date < start_date:
        raise ValueError("end_date must be greater than or equal to start_date")

    return start_date, end_date


def _account_scope_for_user(
    session: Session,
    user_id: int,
    account_ids: list[int] | None,
) -> list[tuple[Account, Connection]]:
    stmt = (
        select(Account, Connection)
        .join(Connection, Account.connection_id == Connection.id)
        .where(Connection.user_id == user_id)
        .order_by(Connection.name, Account.account_id)
    )
    if account_ids:
        stmt = stmt.where(Account.id.in_(account_ids))  # type: ignore[union-attr]

    account_rows = list(session.exec(stmt).all())

    if account_ids:
        found_ids = {account.id for account, _connection in account_rows if account.id is not None}
        missing_ids = sorted(set(account_ids) - found_ids)
        if missing_ids:
            raise ValueError(f"Accounts not found or not accessible: {', '.join(str(v) for v in missing_ids)}")

    return account_rows


def _aggregate_account_state(values: list[float | None], currencies: list[str]) -> float | None:
    if not values:
        return None
    if len(set(currencies)) != 1:
        return None
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def get_live_dashboard_overview(
    session: Session,
    user_id: int,
    *,
    account_ids: list[int] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> LiveDashboardOverviewRead:
    resolved_start, resolved_end = _resolve_dashboard_range(start_date, end_date)
    range_start = datetime.combine(resolved_start, time.min, tzinfo=timezone.utc)
    range_end = datetime.combine(resolved_end + timedelta(days=1), time.min, tzinfo=timezone.utc)

    account_rows = _account_scope_for_user(session, user_id, account_ids)
    scoped_account_ids = [account.id for account, _connection in account_rows if account.id is not None]

    if not scoped_account_ids:
        date_cursor = resolved_start
        daily_results: list[LiveDashboardDailyResultRead] = []
        while date_cursor <= resolved_end:
            daily_results.append(LiveDashboardDailyResultRead(date=date_cursor))
            date_cursor += timedelta(days=1)

        return LiveDashboardOverviewRead(
            date_range=LiveDashboardDateRange(start_date=resolved_start, end_date=resolved_end),
            selected_account_ids=[],
            summary=LiveDashboardSummaryRead(),
            equity_curve=[
                LiveDashboardEquityPointRead(
                    date=result.date,
                    realized_pnl=result.realized_pnl,
                    commission=result.commission,
                    net_pnl=result.net_pnl,
                    cumulative_pnl=0.0,
                    trade_count=result.trade_count,
                )
                for result in daily_results
            ],
            daily_results=daily_results,
            accounts=[],
        )

    base_session_stmt = (
        select(StrategyLive)
        .join(Strategy, StrategyLive.strategy_id == Strategy.id)
        .where(Strategy.user_id == user_id)
        .where(StrategyLive.account_id.in_(scoped_account_ids))  # type: ignore[union-attr]
    )
    live_sessions = list(session.exec(base_session_stmt.order_by(StrategyLive.started_at.desc(), StrategyLive.id.desc())).all())

    open_positions_stmt = (
        select(LivePosition)
        .where(LivePosition.account_id.in_(scoped_account_ids))  # type: ignore[union-attr]
        .where(LivePosition.status == PositionStatus.OPEN.value)
        .where(LivePosition.quantity != 0)
        .order_by(LivePosition.updated_at.desc())  # type: ignore[union-attr]
    )
    open_positions = list(session.exec(open_positions_stmt).all())

    # Every PnL number comes from the single ledger aggregation: one call for
    # the selected accounts (daily series, per-live breakdown, reconciliation)
    # and one per account for the breakdown cards. Unrealized is never derived
    # from the DB (the FE overlays the realtime portfolio plane).
    overall = compute_account_performance(
        session,
        account_ids=scoped_account_ids,
        start=range_start,
        end=range_end,
        include_breakdown=True,
        include_reconciliation=True,
    )
    per_account = {
        account_id: compute_account_performance(
            session,
            account_ids=[account_id],
            start=range_start,
            end=range_end,
            include_breakdown=False,
            include_reconciliation=False,
        )
        for account_id in scoped_account_ids
    }
    reconciliation_by_account = {item.account_id: item for item in overall.reconciliation}

    session_counts: dict[int, dict[str, Any]] = {}
    for live_session in live_sessions:
        if live_session.account_id is None:
            continue
        item = session_counts.setdefault(live_session.account_id, {"session_count": 0, "running_session_count": 0, "last_activity_at": None})
        item["session_count"] += 1
        if live_session.status in {
            LiveStatus.RUNNING.value,
            LiveStatus.STARTING.value,
            LiveStatus.STOPPING.value,
        }:
            item["running_session_count"] += 1
        candidates = [ts for ts in [live_session.started_at, live_session.updated_at, live_session.stopped_at] if ts is not None]
        if candidates:
            session_activity = max(candidates)
            if item["last_activity_at"] is None or session_activity > item["last_activity_at"]:
                item["last_activity_at"] = session_activity

    open_positions_by_account: dict[int, list[LivePosition]] = {}
    for position in open_positions:
        if position.account_id is not None:
            open_positions_by_account.setdefault(position.account_id, []).append(position)

    account_items: list[LiveDashboardAccountBreakdownRead] = []
    for account, connection in account_rows:
        if account.id is None:
            continue
        totals = per_account[account.id].totals
        counts = session_counts.get(account.id, {"session_count": 0, "running_session_count": 0, "last_activity_at": None})
        positions_for_account = open_positions_by_account.get(account.id, [])
        activity_candidates = [ts for ts in [counts["last_activity_at"], totals.last_exit_at] if ts is not None]
        activity_candidates.extend(position.updated_at for position in positions_for_account)
        reconciliation = reconciliation_by_account.get(account.id)
        account_items.append(LiveDashboardAccountBreakdownRead(
            account_id=account.id,
            account_code=account.account_id,
            account_display=account.display_name or account.account_id,
            connection_id=connection.id,
            connection_name=connection.name,
            currency=account.currency,
            cash_balance=float(account.cash_balance) if account.cash_balance is not None else None,
            equity=float(account.equity) if account.equity is not None else None,
            buying_power=float(account.buying_power) if account.buying_power is not None else None,
            available_funds=float(account.available_funds) if account.available_funds is not None else None,
            snapshot_at=account.snapshot_at,
            session_count=int(counts["session_count"]),
            running_session_count=int(counts["running_session_count"]),
            open_positions=len(positions_for_account),
            realized_pnl=totals.realized_gross,
            unrealized_pnl=None,
            net_pnl=totals.net,
            commission=totals.commission,
            swap=totals.swap,
            net_pnl_account_ccy=totals.net_account_ccy,
            pnl_currency=totals.currency,
            pnl_mixed_currency=totals.mixed_currency,
            total_trades=totals.trades,
            unreconciled_trades=totals.unreconciled_trades,
            winning_trades=totals.wins,
            losing_trades=totals.losses,
            win_rate=totals.win_rate,
            last_activity_at=max(activity_candidates) if activity_candidates else None,
            reconciliation_status=reconciliation.status if reconciliation else "unknown",
            reconciliation_gap=reconciliation.gap if reconciliation else None,
        ))

    account_items.sort(key=lambda item: (item.net_pnl, item.account_display), reverse=True)

    daily_results: list[LiveDashboardDailyResultRead] = []
    equity_curve: list[LiveDashboardEquityPointRead] = []
    for point in overall.daily:
        daily_results.append(LiveDashboardDailyResultRead(
            date=point.date,
            realized_pnl=point.realized_gross,
            commission=point.commission,
            net_pnl=point.net,
            trade_count=point.trades,
            winning_trades=point.wins,
            losing_trades=point.losses,
            win_rate=point.win_rate,
        ))
        equity_curve.append(LiveDashboardEquityPointRead(
            date=point.date,
            realized_pnl=point.realized_gross,
            commission=point.commission,
            net_pnl=point.net,
            cumulative_pnl=point.cumulative_net,
            trade_count=point.trades,
        ))

    totals = overall.totals
    summary_last_activity_candidates = [
        item.last_activity_at for item in account_items if item.last_activity_at is not None
    ]
    summary_currencies = [item.currency for item in account_items]

    return LiveDashboardOverviewRead(
        date_range=LiveDashboardDateRange(start_date=resolved_start, end_date=resolved_end),
        selected_account_ids=scoped_account_ids,
        summary=LiveDashboardSummaryRead(
            account_count=len(account_items),
            session_count=sum(item.session_count for item in account_items),
            running_session_count=sum(item.running_session_count for item in account_items),
            open_positions=sum(item.open_positions for item in account_items),
            active_days=sum(1 for result in daily_results if result.trade_count > 0 or result.net_pnl != 0),
            cash_balance=_aggregate_account_state([item.cash_balance for item in account_items], summary_currencies),
            equity=_aggregate_account_state([item.equity for item in account_items], summary_currencies),
            buying_power=_aggregate_account_state([item.buying_power for item in account_items], summary_currencies),
            available_funds=_aggregate_account_state([item.available_funds for item in account_items], summary_currencies),
            realized_pnl=totals.realized_gross,
            unrealized_pnl=None,
            net_pnl=totals.net,
            commission=totals.commission,
            swap=totals.swap,
            net_pnl_account_ccy=totals.net_account_ccy,
            pnl_currency=totals.currency,
            pnl_mixed_currency=totals.mixed_currency,
            account_currency=summary_currencies[0] if len(set(summary_currencies)) == 1 else None,
            total_trades=totals.trades,
            unreconciled_trades=totals.unreconciled_trades,
            winning_trades=totals.wins,
            losing_trades=totals.losses,
            win_rate=totals.win_rate,
            last_activity_at=max(summary_last_activity_candidates) if summary_last_activity_candidates else None,
        ),
        equity_curve=equity_curve,
        daily_results=daily_results,
        accounts=account_items,
        reconciliation=overall.reconciliation,
        breakdown=overall.breakdown,
    )


# ═════════════════════════════════════════════════════════════════════
# ACCOUNT VALIDATION
# ═════════════════════════════════════════════════════════════════════


def validate_account_for_live(
    session: Session,
    account_id: int,
    user_id: int | None = None,
) -> tuple[Account, Connection]:
    """
    Validate that an account exists, is active, and its connection is
    in a connected state.  Raises ValueError if validation fails.

    Returns:
        (Account, Connection) tuple
    """
    account = session.get(Account, account_id)
    if account is None:
        raise ValueError(f"Account {account_id} not found")

    if not account.is_active:
        raise ValueError(f"Account {account.account_id} is not active")

    connection = session.get(Connection, account.connection_id)
    if connection is None:
        raise ValueError(f"Connection for account {account.account_id} not found")

    if user_id is not None and connection.user_id != user_id:
        raise ValueError(f"Account {account_id} not found")

    if connection.status != ConnectionStatus.CONNECTED.value:
        raise ValueError(
            f"Connection '{connection.name}' is not connected "
            f"(status: {connection.status}). Connect it first."
        )

    if not connection.is_active:
        raise ValueError(f"Connection '{connection.name}' is not active")

    return account, connection


# ═════════════════════════════════════════════════════════════════════
# RECONCILIATION
# ═════════════════════════════════════════════════════════════════════


def reconcile_on_startup(
    session: Session,
    strategy_live_id: int,
    account_id: int | None,
    broker_orders: list[dict[str, Any]] | None = None,
    broker_positions: list[dict[str, Any]] | None = None,
) -> ReconciliationReport:
    """
    Reconcile local DB state with broker account state on strategy startup.

    This function:
    1. Cancels stale local orders that are no longer active on the broker
    2. Flags positions that differ between the canonical projection and broker

    Args:
        strategy_live_id: The strategy_live session being started
        account_id: The broker account
        broker_orders: Active orders from broker API
                       (list of dicts with keys: order_id, symbol, side, qty, status)
        broker_positions: Open positions from broker API
                          (list of dicts with keys: symbol, side, quantity, avg_price)

    Returns:
        ReconciliationReport with all discrepancies found and actions taken
    """
    now = datetime.now(timezone.utc)
    items: list[ReconciliationItem] = []

    broker_orders = broker_orders or []
    broker_positions = broker_positions or []

    # ── 1. Reconcile Orders ──────────────────────────────────────────

    # Get all active orders from DB
    db_active_orders = list_active_orders(session, strategy_live_id)
    broker_order_ids = {o.get("order_id") or o.get("broker_order_id") for o in broker_orders}

    for db_order in db_active_orders:
        # If the order is not in the broker's active list, it's stale
        if db_order.broker_order_id and db_order.broker_order_id not in broker_order_ids:
            items.append(ReconciliationItem(
                entity="order",
                symbol=db_order.symbol,
                issue="stale_order",
                db_state={
                    "order_id": db_order.id,
                    "broker_order_id": db_order.broker_order_id,
                    "status": db_order.status,
                    "side": db_order.side,
                    "quantity": db_order.quantity,
                },
                broker_state=None,
                action_taken="cancelled",
            ))
            # Mark as cancelled
            db_order.status = OrderStatus.CANCELLED.value
            db_order.status_message = "Cancelled during startup reconciliation (not found on broker)"
            db_order.cancelled_at = now
            db_order.updated_at = now
            session.add(db_order)

        elif not db_order.broker_order_id:
            # Order was never submitted — cancel it
            items.append(ReconciliationItem(
                entity="order",
                symbol=db_order.symbol,
                issue="unsubmitted_order",
                db_state={
                    "order_id": db_order.id,
                    "status": db_order.status,
                    "side": db_order.side,
                    "quantity": db_order.quantity,
                },
                action_taken="cancelled",
            ))
            db_order.status = OrderStatus.CANCELLED.value
            db_order.status_message = "Cancelled during startup reconciliation (never submitted)"
            db_order.cancelled_at = now
            db_order.updated_at = now
            session.add(db_order)

    # ── 2. Reconcile Positions ───────────────────────────────────────

    # Build map of broker positions by symbol
    broker_pos_map: dict[str, dict[str, Any]] = {}
    for bp in broker_positions:
        sym = bp.get("symbol", "")
        if sym:
            broker_pos_map[sym] = bp

    # Check DB open positions against broker
    db_open_positions = list_open_positions(session, strategy_live_id)
    checked_symbols: set[str] = set()

    for db_pos in db_open_positions:
        checked_symbols.add(db_pos.symbol)
        broker_pos = broker_pos_map.get(db_pos.symbol)

        if broker_pos is None:
            # DB says we have a position, broker says we don't
            items.append(ReconciliationItem(
                entity="position",
                symbol=db_pos.symbol,
                issue="stale_position",
                db_state={
                    "side": db_pos.side,
                    "quantity": db_pos.quantity,
                    "avg_price": db_pos.avg_price,
                },
                broker_state=None,
                action_taken="closed",
            ))
            continue
        else:
            # Both exist — check for quantity / side mismatch
            broker_qty = broker_pos.get("quantity", 0)
            broker_side = broker_pos.get("side", "flat")
            broker_avg = broker_pos.get("avg_price")

            if db_pos.quantity != broker_qty or db_pos.side != broker_side:
                items.append(ReconciliationItem(
                    entity="position",
                    symbol=db_pos.symbol,
                    issue="quantity_mismatch",
                    db_state={
                        "side": db_pos.side,
                        "quantity": db_pos.quantity,
                        "avg_price": db_pos.avg_price,
                    },
                    broker_state={
                        "side": broker_side,
                        "quantity": broker_qty,
                        "avg_price": broker_avg,
                    },
                    action_taken="reported",
                ))
                continue

    # Check for broker positions not in DB
    for sym, bp in broker_pos_map.items():
        if sym not in checked_symbols:
            broker_qty = bp.get("quantity", 0)
            broker_side = bp.get("side", "flat")
            if broker_qty > 0 and broker_side != "flat":
                items.append(ReconciliationItem(
                    entity="position",
                    symbol=sym,
                    issue="missing_position",
                    db_state=None,
                    broker_state={
                        "side": broker_side,
                        "quantity": broker_qty,
                        "avg_price": bp.get("avg_price"),
                    },
                    action_taken="reported",
                ))

    session.commit()

    # Build summary
    if not items:
        summary = "No discrepancies found — DB and broker state are aligned."
    else:
        stale = sum(1 for i in items if "stale" in i.issue)
        mismatches = sum(1 for i in items if "mismatch" in i.issue)
        missing = sum(1 for i in items if "missing" in i.issue)
        parts = []
        if stale:
            parts.append(f"{stale} stale")
        if mismatches:
            parts.append(f"{mismatches} mismatched")
        if missing:
            parts.append(f"{missing} missing")
        summary = f"Found {len(items)} discrepancies ({', '.join(parts)}). All resolved."

    logger.info(
        "Reconciliation for strategy_live %s (account %s): %s",
        strategy_live_id, account_id, summary,
    )

    return ReconciliationReport(
        strategy_live_id=strategy_live_id,
        account_id=account_id,
        checked_at=now,
        items=items,
        summary=summary,
    )

# NOTE: Order submission to the broker gateway and position auto-update
# on fill have been moved to the strategy runner.  The runner writes
# directly to the database and talks to the gateway without routing
# through the backend.  See:
#   strategy-runner/app/broker_client.py  — BrokerClient.place_order()
#   strategy-runner/app/broker_client.py  — BrokerClient._update_position_from_fill()

