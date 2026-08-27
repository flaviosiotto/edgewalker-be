from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel

from app.schemas.performance import AccountReconciliation, PerformanceBreakdownItem


LiveStatus = Literal["stopped", "starting", "running", "stopping", "error"]
LiveSyncState = Literal["aligned", "stale", "missing_container", "unknown"]
PositionSide = Literal["long", "short", "flat"]


class LivePerformanceSummary(BaseModel):
    """Ledger summary of a live session (performance_service, scope strategy_live).

    ``total_pnl`` is the net realized (gross + swap − costs) of the trades
    attributed to the session. ``unrealized_pnl`` is always ``None`` here: the
    mark-to-market comes from the realtime portfolio plane only.
    """
    total_pnl: float | None = None
    realized_gross: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    net_pnl: float = 0.0
    net_pnl_account_ccy: float | None = None
    currency: str | None = None
    unrealized_pnl: float | None = None
    total_trades: int = 0
    unreconciled_trades: int = 0
    win_rate: float | None = None
    position_side: PositionSide = "flat"
    has_open_position: bool = False
    open_positions: int = 0
    last_activity_at: datetime | None = None


class LiveStrategySummaryRead(BaseModel):
    id: int
    strategy_id: int
    strategy_name: str
    status: LiveStatus
    sync_state: LiveSyncState = "unknown"
    symbol: str | None = None
    timeframe: str | None = None
    account_id: int | None = None
    account_display: str | None = None
    connection_id: int | None = None
    connection_name: str | None = None
    container_id: str | None = None
    container_name: str | None = None
    container_status: str | None = None
    container_health: str | None = None
    started_at: datetime | None = None
    stopped_at: datetime | None = None
    error_message: str | None = None
    performance_summary: LivePerformanceSummary | None = None
    created_at: datetime
    updated_at: datetime


class LiveStrategyDetailRead(LiveStrategySummaryRead):
    chat_id: int | None = None
    manager_agent_id: int | None = None
    definition: Any | None = None
    metrics: dict[str, Any] | None = None
    layout_config: dict[str, Any] | None = None


class LiveStrategyCreate(BaseModel):
    strategy_id: int
    symbol: str
    timeframe: str = "5s"
    eval_in_progress: bool = True
    debug_rules: bool = False
    # Ignored: the trading account is bound to the strategy at creation and
    # inherited by every live session. Kept optional for old clients.
    account_id: int | None = None
    # Manager agent per QUESTA sessione; None = manager della strategia.
    manager_agent_id: int | None = None


class LiveStrategyStartResponse(BaseModel):
    live_id: int
    strategy_id: int
    status: str
    container_id: str | None = None
    container_name: str | None = None
    symbol: str | None = None
    timeframe: str | None = None
    message: str | None = None


class LiveStrategyStopResponse(BaseModel):
    live_id: int
    strategy_id: int
    status: str
    message: str | None = None


class LiveDashboardDateRange(BaseModel):
    start_date: date
    end_date: date


class LiveDashboardSummaryRead(BaseModel):
    account_count: int = 0
    session_count: int = 0
    running_session_count: int = 0
    open_positions: int = 0
    active_days: int = 0
    cash_balance: float | None = None
    equity: float | None = None
    buying_power: float | None = None
    available_funds: float | None = None
    realized_pnl: float = 0.0
    unrealized_pnl: float | None = None
    net_pnl: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    net_pnl_account_ccy: float | None = None
    # Currency of realized_pnl/net_pnl/commission/swap (the trades' contract currency);
    # None when the scope mixes currencies (pnl_mixed_currency=True) or has no trades.
    pnl_currency: str | None = None
    pnl_mixed_currency: bool = False
    # Currency of cash_balance/equity/buying_power/available_funds; None when the
    # selected accounts have different base currencies (those fields are then None too).
    account_currency: str | None = None
    total_trades: int = 0
    unreconciled_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float | None = None
    last_activity_at: datetime | None = None


class LiveDashboardEquityPointRead(BaseModel):
    date: date
    realized_pnl: float = 0.0
    commission: float = 0.0
    net_pnl: float = 0.0
    cumulative_pnl: float = 0.0
    trade_count: int = 0


class LiveDashboardDailyResultRead(BaseModel):
    date: date
    realized_pnl: float = 0.0
    commission: float = 0.0
    net_pnl: float = 0.0
    trade_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float | None = None


class LiveDashboardAccountBreakdownRead(BaseModel):
    account_id: int
    account_code: str
    account_display: str
    connection_id: int
    connection_name: str
    currency: str
    cash_balance: float | None = None
    equity: float | None = None
    buying_power: float | None = None
    available_funds: float | None = None
    snapshot_at: datetime | None = None
    session_count: int = 0
    running_session_count: int = 0
    open_positions: int = 0
    realized_pnl: float = 0.0
    unrealized_pnl: float | None = None
    net_pnl: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    net_pnl_account_ccy: float | None = None
    # Currency of the pnl fields above (trades' contract currency), distinct from
    # `currency` (account base currency) e.g. MNQ trades in USD on a EUR account.
    pnl_currency: str | None = None
    pnl_mixed_currency: bool = False
    total_trades: int = 0
    unreconciled_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float | None = None
    last_activity_at: datetime | None = None
    reconciliation_status: str = "unknown"
    reconciliation_gap: float | None = None


class LiveDashboardOverviewRead(BaseModel):
    date_range: LiveDashboardDateRange
    selected_account_ids: list[int]
    summary: LiveDashboardSummaryRead
    equity_curve: list[LiveDashboardEquityPointRead]
    daily_results: list[LiveDashboardDailyResultRead]
    accounts: list[LiveDashboardAccountBreakdownRead]
    # Ledger vs broker account over the window (performance_service).
    reconciliation: list[AccountReconciliation] = []
    # Per-live attribution inside the selected accounts, unattributed bucket last.
    breakdown: list[PerformanceBreakdownItem] = []