"""Schemas of the single performance aggregation (performance_service).

Every performance number shown anywhere (strategy card, live Performance
tab, dashboard, MCP) is an instance of these shapes, computed server-side
from the ``trades`` ledger with one scope and one window.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel

PerformanceScopeKind = Literal["account", "strategy_live", "strategy"]


class PerformanceScope(BaseModel):
    kind: PerformanceScopeKind
    account_ids: list[int] = []
    strategy_live_id: int | None = None
    strategy_id: int | None = None


class PerformanceWindow(BaseModel):
    start: datetime | None = None
    end: datetime | None = None


class PerformanceTotals(BaseModel):
    """Ledger totals over the scope+window.

    ``realized_gross`` is before costs, ``net`` = realized_gross + swap −
    commission. ``net_account_ccy`` is the same net converted to the account
    currency when every trade carried a conversion rate, else ``None``.
    ``unrealized`` is never derived from the DB: it is ``None`` unless a
    broker mark-to-market value was supplied by the caller.
    """
    realized_gross: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    net: float = 0.0
    net_account_ccy: float | None = None
    unrealized: float | None = None
    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float | None = None
    unreconciled_trades: int = 0
    currency: str | None = None
    mixed_currency: bool = False
    first_exit_at: datetime | None = None
    last_exit_at: datetime | None = None


class PerformanceDailyPoint(BaseModel):
    date: date
    realized_gross: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    net: float = 0.0
    cumulative_net: float = 0.0
    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float | None = None


class PerformanceBreakdownItem(BaseModel):
    """Per-live attribution inside an account scope.

    ``strategy_live_id`` is ``None`` for the *unattributed* bucket (manual or
    external trades on the account). The breakdown always sums to the totals.
    """
    strategy_live_id: int | None = None
    strategy_id: int | None = None
    strategy_name: str | None = None
    symbol: str | None = None
    totals: PerformanceTotals


class AccountReconciliation(BaseModel):
    """Ledger vs broker account over the window.

    gap = (equity_end − equity_start) − (ledger net in account currency +
    Δ unrealized). Deposits/withdrawals are not modeled yet, so a non-zero gap
    on a funded account is expected until ``cash_flows`` is provided.
    """
    account_id: int
    currency: str
    equity_start: float | None = None
    equity_start_at: datetime | None = None
    equity_end: float | None = None
    equity_end_at: datetime | None = None
    equity_delta: float | None = None
    unrealized_start: float | None = None
    unrealized_end: float | None = None
    ledger_net_account_ccy: float | None = None
    gap: float | None = None
    status: Literal["reconciled", "gap", "unknown"] = "unknown"


class PerformanceStats(BaseModel):
    scope: PerformanceScope
    window: PerformanceWindow
    totals: PerformanceTotals
    daily: list[PerformanceDailyPoint] = []
    breakdown: list[PerformanceBreakdownItem] = []
    reconciliation: list[AccountReconciliation] = []
    computed_at: datetime
