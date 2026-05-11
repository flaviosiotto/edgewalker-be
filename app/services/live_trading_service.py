"""
Live Trading Service.

CRUD operations for live orders, trades, and positions.
Includes startup reconciliation logic to detect stale states
between the broker account and the local database.

All operations are keyed to a StrategyLive session (strategy_live_id),
NOT directly to a Strategy.
"""
from __future__ import annotations

from calendar import monthrange
from collections import defaultdict
import logging
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from sqlmodel import Session, select

from app.models.connection import Account, Connection, ConnectionStatus
from app.models.live_trading import (
    LiveOrder,
    LivePosition,
    LiveFill,
    OrderStatus,
    PositionStatus,
)
from app.schemas.live_trading import (
    LiveOrderCreate,
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


# ═════════════════════════════════════════════════════════════════════
# ORDERS
# ═════════════════════════════════════════════════════════════════════


def create_live_order(
    session: Session,
    strategy_live_id: int,
    account_id: int | None,
    payload: LiveOrderCreate,
) -> LiveOrder:
    """Create a new live order."""
    order = LiveOrder(
        strategy_live_id=strategy_live_id,
        account_id=account_id,
        symbol=payload.symbol,
        side=payload.side,
        order_type=payload.order_type,
        quantity=payload.quantity,
        limit_price=payload.limit_price,
        stop_price=payload.stop_price,
        broker_order_id=payload.broker_order_id,
        extra=payload.extra,
    )
    session.add(order)
    session.commit()
    session.refresh(order)
    logger.info(
        "Created live order %s: %s %s %s qty=%s (strategy_live=%s)",
        order.id, order.side, order.order_type, order.symbol, order.quantity, strategy_live_id,
    )
    return order


def update_live_order(
    session: Session,
    order_id: int,
    payload: LiveOrderUpdate,
) -> LiveOrder | None:
    """Update a live order (status, fill info, etc.)."""
    order = session.get(LiveOrder, order_id)
    if order is None:
        return None

    now = datetime.now(timezone.utc)
    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        setattr(order, key, value)
    order.updated_at = now

    session.add(order)
    session.commit()
    session.refresh(order)
    return order


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


# ═════════════════════════════════════════════════════════════════════
# FILLS
# ═════════════════════════════════════════════════════════════════════


def create_live_fill(
    session: Session,
    strategy_live_id: int,
    account_id: int | None,
    payload: LiveFillCreate,
) -> LiveFill:
    """Record a live fill (immutable event)."""
    fill = LiveFill(
        strategy_live_id=strategy_live_id,
        account_id=account_id,
        order_id=payload.order_id,
        symbol=payload.symbol,
        side=payload.side,
        quantity=payload.quantity,
        price=payload.price,
        commission=payload.commission,
        realized_pnl=payload.realized_pnl,
        fill_time=payload.fill_time,
        broker_fill_id=payload.broker_fill_id,
        extra=payload.extra,
    )
    session.add(fill)
    session.commit()
    session.refresh(fill)
    logger.info(
        "Recorded live fill %s: %s %s qty=%s @%s (strategy_live=%s)",
        fill.id, fill.side, fill.symbol, fill.quantity, fill.price, strategy_live_id,
    )
    return fill


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
    """Create or update an open position for the strategy_live+symbol+account."""
    pos = get_open_position(session, strategy_live_id, payload.symbol, account_id)
    now = datetime.now(timezone.utc)

    if pos is None:
        pos = LivePosition(
            strategy_live_id=strategy_live_id,
            account_id=account_id,
            symbol=payload.symbol,
            side=payload.side,
            quantity=payload.quantity,
            avg_price=payload.avg_price,
            cost_basis=payload.cost_basis,
            unrealized_pnl=payload.unrealized_pnl,
            realized_pnl=payload.realized_pnl,
            market_value=payload.market_value,
            extra=payload.extra,
        )
        session.add(pos)
    else:
        pos.side = payload.side
        pos.quantity = payload.quantity
        if payload.avg_price is not None:
            pos.avg_price = payload.avg_price
        if payload.cost_basis is not None:
            pos.cost_basis = payload.cost_basis
        if payload.unrealized_pnl is not None:
            pos.unrealized_pnl = payload.unrealized_pnl
        if payload.realized_pnl is not None:
            pos.realized_pnl = payload.realized_pnl
        if payload.market_value is not None:
            pos.market_value = payload.market_value
        if payload.extra is not None:
            pos.extra = payload.extra
        pos.updated_at = now

        # Close position if flat
        if payload.side == "flat" or payload.quantity == 0:
            pos.status = PositionStatus.CLOSED.value
            pos.closed_at = now

    session.commit()
    session.refresh(pos)
    return pos


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


# ═════════════════════════════════════════════════════════════════════
# STRATEGY-LEVEL QUERIES (aggregate across all sessions)
# ═════════════════════════════════════════════════════════════════════

from app.models.strategy import LiveStatus, Strategy, StrategyLive  # noqa: E402


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

    fills_stmt = (
        select(LiveFill)
        .join(StrategyLive, LiveFill.strategy_live_id == StrategyLive.id)
        .join(Strategy, StrategyLive.strategy_id == Strategy.id)
        .where(Strategy.user_id == user_id)
        .where(LiveFill.account_id.in_(scoped_account_ids))  # type: ignore[union-attr]
        .where(LiveFill.fill_time >= range_start)
        .where(LiveFill.fill_time < range_end)
        .order_by(LiveFill.fill_time.asc())  # type: ignore[union-attr]
    )
    fills = list(session.exec(fills_stmt).all())

    closed_positions_stmt = (
        select(LivePosition)
        .join(StrategyLive, LivePosition.strategy_live_id == StrategyLive.id)
        .join(Strategy, StrategyLive.strategy_id == Strategy.id)
        .where(Strategy.user_id == user_id)
        .where(LivePosition.account_id.in_(scoped_account_ids))  # type: ignore[union-attr]
        .where(LivePosition.status == PositionStatus.CLOSED.value)
        .where(LivePosition.closed_at.is_not(None))
        .where(LivePosition.closed_at >= range_start)
        .where(LivePosition.closed_at < range_end)
        .order_by(LivePosition.closed_at.asc())  # type: ignore[union-attr]
    )
    closed_positions = list(session.exec(closed_positions_stmt).all())

    open_positions_stmt = (
        select(LivePosition)
        .join(StrategyLive, LivePosition.strategy_live_id == StrategyLive.id)
        .join(Strategy, StrategyLive.strategy_id == Strategy.id)
        .where(Strategy.user_id == user_id)
        .where(LivePosition.account_id.in_(scoped_account_ids))  # type: ignore[union-attr]
        .where(LivePosition.status == PositionStatus.OPEN.value)
        .order_by(LivePosition.updated_at.desc())  # type: ignore[union-attr]
    )
    open_positions = list(session.exec(open_positions_stmt).all())

    daily_buckets: dict[date, dict[str, float | int]] = {}
    account_breakdown: dict[int, dict[str, Any]] = {}

    for account, connection in account_rows:
        if account.id is None:
            continue
        account_breakdown[account.id] = {
            "account_id": account.id,
            "account_code": account.account_id,
            "account_display": account.display_name or account.account_id,
            "connection_id": connection.id,
            "connection_name": connection.name,
            "currency": account.currency,
            "session_count": 0,
            "running_session_count": 0,
            "open_positions": 0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "net_pnl": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "last_activity_at": None,
        }

    for live_session in live_sessions:
        if live_session.account_id is None or live_session.account_id not in account_breakdown:
            continue
        item = account_breakdown[live_session.account_id]
        item["session_count"] += 1
        if live_session.status in {
            LiveStatus.RUNNING.value,
            LiveStatus.STARTING.value,
            LiveStatus.STOPPING.value,
        }:
            item["running_session_count"] += 1
        session_activity = max(
            ts for ts in [live_session.started_at, live_session.updated_at, live_session.stopped_at] if ts is not None
        ) if any(ts is not None for ts in [live_session.started_at, live_session.updated_at, live_session.stopped_at]) else None
        if session_activity and (item["last_activity_at"] is None or session_activity > item["last_activity_at"]):
            item["last_activity_at"] = session_activity

    for fill in fills:
        if fill.account_id is None or fill.account_id not in account_breakdown:
            continue
        bucket_date = fill.fill_time.astimezone(timezone.utc).date()
        bucket = daily_buckets.setdefault(bucket_date, {
            "realized_pnl": 0.0,
            "commission": 0.0,
            "net_pnl": 0.0,
            "trade_count": 0,
            "winning_trades": 0,
            "losing_trades": 0,
        })

        realized_pnl = float(fill.realized_pnl or 0.0)
        commission = float(fill.commission or 0.0)
        bucket["realized_pnl"] += realized_pnl
        bucket["commission"] += commission
        bucket["net_pnl"] += realized_pnl - commission

        item = account_breakdown[fill.account_id]
        item["realized_pnl"] += realized_pnl
        item["net_pnl"] += realized_pnl - commission
        if item["last_activity_at"] is None or fill.fill_time > item["last_activity_at"]:
            item["last_activity_at"] = fill.fill_time

    for position in closed_positions:
        if position.account_id is None or position.account_id not in account_breakdown:
            continue
        if position.closed_at is None:
            continue

        bucket_date = position.closed_at.astimezone(timezone.utc).date()
        bucket = daily_buckets.setdefault(bucket_date, {
            "realized_pnl": 0.0,
            "commission": 0.0,
            "net_pnl": 0.0,
            "trade_count": 0,
            "winning_trades": 0,
            "losing_trades": 0,
        })
        bucket["trade_count"] += 1

        realized_pnl = float(position.realized_pnl or 0.0)
        if realized_pnl > 0:
            bucket["winning_trades"] += 1
        elif realized_pnl < 0:
            bucket["losing_trades"] += 1

        item = account_breakdown[position.account_id]
        item["total_trades"] += 1
        if realized_pnl > 0:
            item["winning_trades"] += 1
        elif realized_pnl < 0:
            item["losing_trades"] += 1
        if item["last_activity_at"] is None or position.closed_at > item["last_activity_at"]:
            item["last_activity_at"] = position.closed_at

    for position in open_positions:
        if position.account_id is None or position.account_id not in account_breakdown:
            continue
        unrealized_pnl = float(position.unrealized_pnl or 0.0)
        item = account_breakdown[position.account_id]
        item["open_positions"] += 1
        item["unrealized_pnl"] += unrealized_pnl
        item["net_pnl"] += unrealized_pnl
        if item["last_activity_at"] is None or position.updated_at > item["last_activity_at"]:
            item["last_activity_at"] = position.updated_at

    daily_results: list[LiveDashboardDailyResultRead] = []
    equity_curve: list[LiveDashboardEquityPointRead] = []
    cumulative_pnl = 0.0
    cursor = resolved_start
    while cursor <= resolved_end:
        bucket = daily_buckets.get(cursor, {
            "realized_pnl": 0.0,
            "commission": 0.0,
            "net_pnl": 0.0,
            "trade_count": 0,
            "winning_trades": 0,
            "losing_trades": 0,
        })
        trade_count = int(bucket["trade_count"])
        winning_trades = int(bucket["winning_trades"])
        losing_trades = int(bucket["losing_trades"])
        net_pnl = float(bucket["net_pnl"])
        cumulative_pnl += net_pnl

        daily_results.append(LiveDashboardDailyResultRead(
            date=cursor,
            realized_pnl=float(bucket["realized_pnl"]),
            commission=float(bucket["commission"]),
            net_pnl=net_pnl,
            trade_count=trade_count,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=(winning_trades / trade_count * 100.0) if trade_count else None,
        ))
        equity_curve.append(LiveDashboardEquityPointRead(
            date=cursor,
            realized_pnl=float(bucket["realized_pnl"]),
            commission=float(bucket["commission"]),
            net_pnl=net_pnl,
            cumulative_pnl=cumulative_pnl,
            trade_count=trade_count,
        ))
        cursor += timedelta(days=1)

    account_items: list[LiveDashboardAccountBreakdownRead] = []
    for item in account_breakdown.values():
        total_trades = int(item["total_trades"])
        winning_trades = int(item["winning_trades"])
        losing_trades = int(item["losing_trades"])
        account_items.append(LiveDashboardAccountBreakdownRead(
            account_id=item["account_id"],
            account_code=item["account_code"],
            account_display=item["account_display"],
            connection_id=item["connection_id"],
            connection_name=item["connection_name"],
            currency=item["currency"],
            session_count=int(item["session_count"]),
            running_session_count=int(item["running_session_count"]),
            open_positions=int(item["open_positions"]),
            realized_pnl=float(item["realized_pnl"]),
            unrealized_pnl=float(item["unrealized_pnl"]),
            net_pnl=float(item["net_pnl"]),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=(winning_trades / total_trades * 100.0) if total_trades else None,
            last_activity_at=item["last_activity_at"],
        ))

    account_items.sort(key=lambda item: (item.net_pnl, item.account_display), reverse=True)

    summary_total_trades = sum(item.total_trades for item in account_items)
    summary_winning_trades = sum(item.winning_trades for item in account_items)
    summary_losing_trades = sum(item.losing_trades for item in account_items)
    summary_last_activity_candidates = [
        item.last_activity_at for item in account_items if item.last_activity_at is not None
    ]

    return LiveDashboardOverviewRead(
        date_range=LiveDashboardDateRange(start_date=resolved_start, end_date=resolved_end),
        selected_account_ids=scoped_account_ids,
        summary=LiveDashboardSummaryRead(
            account_count=len(account_items),
            session_count=sum(item.session_count for item in account_items),
            running_session_count=sum(item.running_session_count for item in account_items),
            open_positions=sum(item.open_positions for item in account_items),
            active_days=sum(1 for result in daily_results if result.trade_count > 0 or result.net_pnl != 0),
            realized_pnl=sum(item.realized_pnl for item in account_items),
            unrealized_pnl=sum(item.unrealized_pnl for item in account_items),
            net_pnl=sum(item.net_pnl for item in account_items),
            total_trades=summary_total_trades,
            winning_trades=summary_winning_trades,
            losing_trades=summary_losing_trades,
            win_rate=(summary_winning_trades / summary_total_trades * 100.0) if summary_total_trades else None,
            last_activity_at=max(summary_last_activity_candidates) if summary_last_activity_candidates else None,
        ),
        equity_curve=equity_curve,
        daily_results=daily_results,
        accounts=account_items,
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
    2. Flags positions that differ between DB and broker
    3. Syncs broker positions that are missing from DB

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
            db_pos.status = PositionStatus.CLOSED.value
            db_pos.closed_at = now
            db_pos.quantity = 0
            db_pos.side = "flat"
            db_pos.updated_at = now
            db_pos.extra = {**(db_pos.extra or {}), "reconciled": True, "reconciled_at": now.isoformat()}
            session.add(db_pos)
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
                    action_taken="synced_to_broker",
                ))
                # Sync to broker state (broker is source of truth)
                db_pos.side = broker_side
                db_pos.quantity = broker_qty
                if broker_avg is not None:
                    db_pos.avg_price = broker_avg
                db_pos.updated_at = now
                db_pos.extra = {**(db_pos.extra or {}), "reconciled": True, "reconciled_at": now.isoformat()}
                session.add(db_pos)

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
                    action_taken="created",
                ))
                # Create position in DB to match broker
                new_pos = LivePosition(
                    strategy_live_id=strategy_live_id,
                    account_id=account_id,
                    symbol=sym,
                    side=broker_side,
                    quantity=broker_qty,
                    avg_price=bp.get("avg_price"),
                    cost_basis=bp.get("cost_basis"),
                    unrealized_pnl=bp.get("unrealized_pnl"),
                    market_value=bp.get("market_value"),
                    extra={"reconciled": True, "reconciled_at": now.isoformat()},
                )
                session.add(new_pos)

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

