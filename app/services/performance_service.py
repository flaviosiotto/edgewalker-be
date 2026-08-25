"""Single performance aggregation over the ``trades`` ledger.

Every performance number the platform shows (strategy card, live Performance
tab, dashboard, MCP tools) is produced here, from one scope and one window:

* ``account``       – all ledger rows of the given accounts, with a breakdown
                      per live session plus an *unattributed* bucket (manual /
                      external trades). Breakdown sums to the totals.
* ``strategy_live`` – rows attributed to one live session
                      (``trades.strategy_live_id``), window defaulting to the
                      session lifetime.
* ``strategy``      – rows of every live session of a strategy.

Ledger semantics (migration 044): ``realized_pnl`` is gross, ``commission``
the total cost, ``swap`` financing, ``net_pnl = realized + swap − commission``.
Rows with ``realized_pnl IS NULL`` are *unreconciled*: they are counted
(``unreconciled_trades``) and never silently dropped, but contribute nothing
to the sums.

Unrealized P&L is never derived from the DB (mark-to-market lives on the
realtime portfolio plane); ``totals.unrealized`` stays ``None`` unless the
caller supplies a broker value.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Iterable

from sqlmodel import Session, select

from app.models.connection import Account, AccountSnapshot
from app.models.live_trading import LiveTrade
from app.models.strategy import Strategy, StrategyLive
from app.schemas.performance import (
    AccountReconciliation,
    PerformanceBreakdownItem,
    PerformanceDailyPoint,
    PerformanceScope,
    PerformanceStats,
    PerformanceTotals,
    PerformanceWindow,
)

_RECONCILIATION_TOLERANCE = 0.01


# ── Public API ────────────────────────────────────────────────────────


def compute_account_performance(
    session: Session,
    *,
    account_ids: list[int],
    start: datetime | None = None,
    end: datetime | None = None,
    include_breakdown: bool = True,
    include_reconciliation: bool = True,
) -> PerformanceStats:
    scope = PerformanceScope(kind="account", account_ids=sorted(set(account_ids)))
    window = PerformanceWindow(start=start, end=end)
    if not account_ids:
        return _empty_stats(scope, window)

    stmt = select(LiveTrade).where(LiveTrade.account_id.in_(account_ids))  # type: ignore[union-attr]
    trades = _load_trades(session, stmt, start, end)
    stats = _build_stats(scope, window, trades, daily=True)

    if include_breakdown:
        stats.breakdown = _breakdown_by_live(session, trades)
    if include_reconciliation:
        stats.reconciliation = [
            _reconcile_account(session, account_id, trades, start, end) for account_id in scope.account_ids
        ]
    return stats


def compute_live_performance(
    session: Session,
    *,
    live: StrategyLive,
    start: datetime | None = None,
    end: datetime | None = None,
    daily: bool = True,
) -> PerformanceStats:
    """Performance of one live session = ledger rows attributed to it.

    Window defaults to the session lifetime ``[started_at, stopped_at)`` so a
    restarted container does not lose its history and a stopped session stops
    accruing.
    """
    window_start = start if start is not None else live.started_at
    window_end = end if end is not None else live.stopped_at
    scope = PerformanceScope(
        kind="strategy_live",
        account_ids=[live.account_id] if live.account_id is not None else [],
        strategy_live_id=live.id,
        strategy_id=live.strategy_id,
    )
    window = PerformanceWindow(start=window_start, end=window_end)
    if live.id is None:
        return _empty_stats(scope, window)
    stmt = select(LiveTrade).where(LiveTrade.strategy_live_id == live.id)
    trades = _load_trades(session, stmt, window_start, window_end)
    return _build_stats(scope, window, trades, daily=daily)


def compute_strategy_performance(
    session: Session,
    *,
    strategy_id: int,
    start: datetime | None = None,
    end: datetime | None = None,
    daily: bool = True,
    breakdown: bool = True,
) -> PerformanceStats:
    """Performance of a strategy = ledger rows of every live session it ever ran.

    This is the figure shown on the strategy card and in the Performance tab,
    live or not: it does not depend on a session being active.
    """
    scope = PerformanceScope(kind="strategy", strategy_id=strategy_id)
    window = PerformanceWindow(start=start, end=end)
    live_ids = [
        row for row in session.exec(select(StrategyLive.id).where(StrategyLive.strategy_id == strategy_id)).all()
        if row is not None
    ]
    if not live_ids:
        return _empty_stats(scope, window)
    stmt = select(LiveTrade).where(LiveTrade.strategy_live_id.in_(live_ids))  # type: ignore[union-attr]
    trades = _load_trades(session, stmt, start, end)
    stats = _build_stats(scope, window, trades, daily=daily)
    if breakdown:
        stats.breakdown = _breakdown_by_live(session, trades)
    return stats


def strategy_live_session_count(session: Session, strategy_id: int) -> int:
    return len(session.exec(select(StrategyLive.id).where(StrategyLive.strategy_id == strategy_id)).all())


# ── Aggregation ───────────────────────────────────────────────────────


def _load_trades(session: Session, stmt, start: datetime | None, end: datetime | None) -> list[LiveTrade]:
    if start is not None:
        stmt = stmt.where(LiveTrade.exit_time >= start)
    if end is not None:
        stmt = stmt.where(LiveTrade.exit_time < end)
    stmt = stmt.order_by(LiveTrade.exit_time.asc(), LiveTrade.id.asc())  # type: ignore[union-attr]
    return list(session.exec(stmt).all())


def _totals(trades: Iterable[LiveTrade]) -> PerformanceTotals:
    totals = PerformanceTotals()
    net_ccy_sum = 0.0
    net_ccy_complete = True
    currencies: set[str] = set()
    first: datetime | None = None
    last: datetime | None = None
    for trade in trades:
        if trade.currency:
            currencies.add(str(trade.currency).upper())
        if first is None or trade.exit_time < first:
            first = trade.exit_time
        if last is None or trade.exit_time > last:
            last = trade.exit_time
        if trade.realized_pnl is None:
            totals.unreconciled_trades += 1
            net_ccy_complete = False
            continue
        realized = float(trade.realized_pnl)
        commission = float(trade.commission or 0.0)
        swap = float(trade.swap or 0.0)
        net = float(trade.net_pnl) if trade.net_pnl is not None else realized + swap - commission
        totals.realized_gross += realized
        totals.commission += commission
        totals.swap += swap
        totals.net += net
        totals.trades += 1
        if net > 0:
            totals.wins += 1
        elif net < 0:
            totals.losses += 1
        if trade.net_pnl_account_ccy is not None:
            net_ccy_sum += float(trade.net_pnl_account_ccy)
        else:
            net_ccy_complete = False
    totals.win_rate = (totals.wins / totals.trades * 100.0) if totals.trades else None
    totals.net_account_ccy = net_ccy_sum if (totals.trades and net_ccy_complete) else None
    totals.currency = next(iter(currencies)) if len(currencies) == 1 else None
    totals.mixed_currency = len(currencies) > 1
    totals.first_exit_at = first
    totals.last_exit_at = last
    return totals


def _daily_series(
    trades: list[LiveTrade],
    start: datetime | None,
    end: datetime | None,
) -> list[PerformanceDailyPoint]:
    """Calendar-day (UTC) buckets over the window, gaps filled with zeros."""
    by_day: dict[date, list[LiveTrade]] = defaultdict(list)
    for trade in trades:
        by_day[trade.exit_time.astimezone(timezone.utc).date()].append(trade)

    if start is not None:
        first_day = start.astimezone(timezone.utc).date()
    elif by_day:
        first_day = min(by_day)
    else:
        return []
    if end is not None:
        last_day = (end - timedelta(microseconds=1)).astimezone(timezone.utc).date()
    else:
        last_day = max(by_day) if by_day else first_day
    if last_day < first_day:
        return []

    series: list[PerformanceDailyPoint] = []
    cumulative = 0.0
    cursor = first_day
    while cursor <= last_day:
        day_totals = _totals(by_day.get(cursor, []))
        cumulative += day_totals.net
        series.append(
            PerformanceDailyPoint(
                date=cursor,
                realized_gross=day_totals.realized_gross,
                commission=day_totals.commission,
                swap=day_totals.swap,
                net=day_totals.net,
                cumulative_net=cumulative,
                trades=day_totals.trades,
                wins=day_totals.wins,
                losses=day_totals.losses,
                win_rate=day_totals.win_rate,
            )
        )
        cursor += timedelta(days=1)
    return series


def _build_stats(
    scope: PerformanceScope,
    window: PerformanceWindow,
    trades: list[LiveTrade],
    *,
    daily: bool,
) -> PerformanceStats:
    return PerformanceStats(
        scope=scope,
        window=window,
        totals=_totals(trades),
        daily=_daily_series(trades, window.start, window.end) if daily else [],
        computed_at=datetime.now(timezone.utc),
    )


def _empty_stats(scope: PerformanceScope, window: PerformanceWindow) -> PerformanceStats:
    return PerformanceStats(scope=scope, window=window, totals=PerformanceTotals(), computed_at=datetime.now(timezone.utc))


def _breakdown_by_live(session: Session, trades: list[LiveTrade]) -> list[PerformanceBreakdownItem]:
    by_live: dict[int | None, list[LiveTrade]] = defaultdict(list)
    for trade in trades:
        by_live[trade.strategy_live_id].append(trade)
    live_ids = [live_id for live_id in by_live if live_id is not None]
    live_meta: dict[int, tuple[StrategyLive, str | None]] = {}
    if live_ids:
        rows = session.exec(
            select(StrategyLive, Strategy.name)
            .join(Strategy, StrategyLive.strategy_id == Strategy.id)
            .where(StrategyLive.id.in_(live_ids))  # type: ignore[union-attr]
        ).all()
        for live, name in rows:
            if live.id is not None:
                live_meta[live.id] = (live, name)

    items: list[PerformanceBreakdownItem] = []
    for live_id, rows in by_live.items():
        meta = live_meta.get(live_id) if live_id is not None else None
        items.append(
            PerformanceBreakdownItem(
                strategy_live_id=live_id,
                strategy_id=meta[0].strategy_id if meta else None,
                strategy_name=meta[1] if meta else None,
                symbol=meta[0].symbol if meta else None,
                totals=_totals(rows),
            )
        )
    items.sort(key=lambda item: (item.strategy_live_id is None, -(item.totals.last_exit_at.timestamp() if item.totals.last_exit_at else 0)))
    return items


# ── Reconciliation against the broker account ─────────────────────────


def _snapshot_at_or_before(session: Session, account_id: int, at: datetime | None) -> AccountSnapshot | None:
    stmt = select(AccountSnapshot).where(AccountSnapshot.account_id == account_id)
    if at is not None:
        stmt = stmt.where(AccountSnapshot.observed_at <= at)
    stmt = stmt.order_by(AccountSnapshot.observed_at.desc(), AccountSnapshot.id.desc())  # type: ignore[union-attr]
    return session.exec(stmt).first()


def _snapshot_at_or_after(session: Session, account_id: int, at: datetime) -> AccountSnapshot | None:
    stmt = (
        select(AccountSnapshot)
        .where(AccountSnapshot.account_id == account_id)
        .where(AccountSnapshot.observed_at >= at)
        .order_by(AccountSnapshot.observed_at.asc(), AccountSnapshot.id.asc())  # type: ignore[union-attr]
    )
    return session.exec(stmt).first()


def _reconcile_account(
    session: Session,
    account_id: int,
    trades: list[LiveTrade],
    start: datetime | None,
    end: datetime | None,
) -> AccountReconciliation:
    account = session.get(Account, account_id)
    currency = account.currency if account is not None else "USD"
    result = AccountReconciliation(account_id=account_id, currency=currency)

    account_trades = [trade for trade in trades if trade.account_id == account_id]
    ledger = _totals(account_trades)
    result.ledger_net_account_ccy = ledger.net_account_ccy

    # Equity at the window edges: the last snapshot before the start (or the
    # first one after it when history starts inside the window) and the last
    # snapshot before the end (the current value when the window is open).
    start_snapshot = _snapshot_at_or_before(session, account_id, start) if start is not None else None
    if start_snapshot is None and start is not None:
        start_snapshot = _snapshot_at_or_after(session, account_id, start)
    end_snapshot = _snapshot_at_or_before(session, account_id, end)

    if start_snapshot is not None:
        result.equity_start = start_snapshot.equity
        result.equity_start_at = start_snapshot.observed_at
        result.unrealized_start = start_snapshot.unrealized_pnl
    if end_snapshot is not None:
        result.equity_end = end_snapshot.equity
        result.equity_end_at = end_snapshot.observed_at
        result.unrealized_end = end_snapshot.unrealized_pnl
    elif account is not None and end is None:
        result.equity_end = account.equity
        result.equity_end_at = account.snapshot_at
        result.unrealized_end = account.unrealized_pnl

    if result.equity_start is not None and result.equity_end is not None:
        result.equity_delta = result.equity_end - result.equity_start
    if (
        result.equity_delta is not None
        and result.ledger_net_account_ccy is not None
        and result.equity_start_at is not None
        and result.equity_end_at is not None
        and result.equity_start_at < result.equity_end_at
    ):
        unrealized_delta = float(result.unrealized_end or 0.0) - float(result.unrealized_start or 0.0)
        result.gap = result.equity_delta - (result.ledger_net_account_ccy + unrealized_delta)
        result.status = "reconciled" if abs(result.gap) <= _RECONCILIATION_TOLERANCE else "gap"
    return result
