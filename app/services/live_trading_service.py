"""
Live Trading Service.

CRUD operations for live orders, trades, and positions.
Includes startup reconciliation logic to detect stale states
between the broker account and the local database.

All operations are keyed to a StrategyLive session (strategy_live_id),
NOT directly to a Strategy.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlmodel import Session, select

from app.models.connection import Account, Connection, ConnectionStatus
from app.models.live_trading import (
    LiveOrder,
    LivePosition,
    LiveTrade,
    OrderStatus,
    PositionStatus,
)
from app.schemas.live_trading import (
    LiveOrderCreate,
    LiveOrderUpdate,
    LivePositionCreate,
    LivePositionUpdate,
    LiveTradeCreate,
    ReconciliationItem,
    ReconciliationReport,
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
# TRADES
# ═════════════════════════════════════════════════════════════════════


def create_live_trade(
    session: Session,
    strategy_live_id: int,
    account_id: int | None,
    payload: LiveTradeCreate,
) -> LiveTrade:
    """Record a live trade / fill."""
    trade = LiveTrade(
        strategy_live_id=strategy_live_id,
        account_id=account_id,
        order_id=payload.order_id,
        symbol=payload.symbol,
        side=payload.side,
        quantity=payload.quantity,
        price=payload.price,
        commission=payload.commission,
        realized_pnl=payload.realized_pnl,
        trade_time=payload.trade_time,
        broker_trade_id=payload.broker_trade_id,
        extra=payload.extra,
    )
    session.add(trade)
    session.commit()
    session.refresh(trade)
    logger.info(
        "Recorded live trade %s: %s %s qty=%s @%s (strategy_live=%s)",
        trade.id, trade.side, trade.symbol, trade.quantity, trade.price, strategy_live_id,
    )
    return trade


def list_live_trades(
    session: Session,
    strategy_live_id: int,
    *,
    limit: int = 200,
) -> list[LiveTrade]:
    """List live trades for a strategy live session."""
    stmt = (
        select(LiveTrade)
        .where(LiveTrade.strategy_live_id == strategy_live_id)
        .order_by(LiveTrade.trade_time.desc())  # type: ignore[union-attr]
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
# ACCOUNT VALIDATION
# ═════════════════════════════════════════════════════════════════════


def validate_account_for_live(
    session: Session,
    account_id: int,
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

