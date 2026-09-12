from datetime import datetime, date
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from app.schemas.chat import ChatRead
from app.schemas.live_strategy import LiveStrategySummaryRead


# Live status enum values
LiveStatus = Literal["stopped", "starting", "running", "paused", "stopping", "error"]


# ─── STRATEGY LIVE (RUNTIME) SCHEMA ───


class StrategyLiveRead(BaseModel):
    """Runtime state of a live strategy session."""
    id: int
    strategy_id: int
    chat_id: Optional[int] = None
    manager_agent_id: Optional[int] = None
    status: LiveStatus = "stopped"
    container_id: Optional[str] = None
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    account_id: Optional[int] = None
    connection_id: Optional[int] = None
    definition: Optional[Any] = None
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metrics: Optional[dict[str, Any]] = None
    layout_config: Optional[dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class StrategyCreate(BaseModel):
    # Optional since 29/08/2026: the design agent names the strategy. When
    # omitted the backend picks a unique placeholder ("Nuova strategia N").
    name: Optional[str] = Field(default=None, max_length=60)
    description: Optional[str] = None
    definition: Any
    manager_agent_id: Optional[int] = None
    # Trading account the strategy belongs to (mandatory from birth). The
    # datafeed connection is derived from it server-side.
    account_id: int


class StrategyPerformanceSummary(BaseModel):
    """Strategy-scoped ledger totals (performance_service scope=strategy)."""
    total_pnl: float | None = None      # net realized (gross + swap − costs)
    realized_gross: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    net_pnl: float = 0.0
    net_pnl_account_ccy: float | None = None
    currency: str | None = None
    mixed_currency: bool = False
    total_trades: int = 0
    unreconciled_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float | None = None
    sessions: int = 0
    active_live_id: int | None = None
    first_exit_at: datetime | None = None
    last_exit_at: datetime | None = None


class StrategyRead(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    definition: Any
    created_at: datetime
    updated_at: datetime
    # Design-context chats only (``Strategy.chats`` is design-scoped at the ORM
    # level; run chats live/backtest are reached via their run entities).
    chats: list[ChatRead] = []

    # AI Agent Manager
    manager_agent_id: Optional[int] = None

    # Account binding (single source of truth) + display labels for grouping
    account_id: int
    account_display: Optional[str] = None
    connection_name: Optional[str] = None

    # Datafeed connection (derived from the account)
    connection_id: int

    # Current live session (if any)
    live: Optional[StrategyLiveRead] = None

    # Runtime summary of the current live session (container/sync state and
    # the session-scoped ledger figures). Populated only when a live session
    # exists; null otherwise.
    live_summary: Optional[LiveStrategySummaryRead] = None

    # Strategy-scoped performance from the trades ledger: every live session
    # of the strategy, lifetime. Always populated (zeros when never traded), so
    # cards and the Performance tab show the same figures whether or not the
    # strategy is live.
    performance_summary: Optional[StrategyPerformanceSummary] = None

    # UI layout persistence
    layout_config: Optional[dict[str, Any]] = None

    class Config:
        from_attributes = True


class StrategyUpdate(BaseModel):
    name: Optional[str] = Field(default=None, max_length=60)
    description: Optional[str] = None
    definition: Optional[Any] = None
    layout_config: Optional[dict[str, Any]] = None
    manager_agent_id: Optional[int] = None
    # account_id / connection_id are NOT updatable: the account is fixed at
    # creation (live history, orders and performance hang off it).


# ─── COPY / BULK ───


class StrategyCopyRequest(BaseModel):
    """Duplicate a strategy onto an account (the same one or another)."""
    target_account_id: int
    # Explicit name → 409 if taken on the target account. Omitted → original
    # name, or "<name> (copia)" when taken.
    name: Optional[str] = Field(default=None, max_length=60)


class StrategyCopyResponse(BaseModel):
    strategy: StrategyRead
    # Non-blocking symbol checks against the target datafeed.
    warnings: list[str] = []


class StrategyBulkCopyRequest(BaseModel):
    strategy_ids: list[int] = Field(min_length=1, max_length=200)
    target_account_id: int


class StrategyBulkCopyItem(BaseModel):
    strategy_id: int
    ok: bool
    new_strategy_id: Optional[int] = None
    new_name: Optional[str] = None
    warnings: list[str] = []
    # HTTP-style status + detail of the failure (409 name, 402 plan limit...).
    error_status: Optional[int] = None
    error: Optional[Any] = None


class StrategyBulkCopyResponse(BaseModel):
    items: list[StrategyBulkCopyItem]
    copied: int
    failed: int
    strategies: list[StrategyRead] = []


class StrategyBulkDeleteRequest(BaseModel):
    strategy_ids: list[int] = Field(min_length=1, max_length=200)


class StrategyBulkDeleteItem(BaseModel):
    strategy_id: int
    ok: bool
    error_status: Optional[int] = None
    error: Optional[Any] = None


class StrategyBulkDeleteResponse(BaseModel):
    items: list[StrategyBulkDeleteItem]
    deleted: int
    failed: int


# ─── BACKTEST SCHEMAS ───


class BacktestCreate(BaseModel):
    """Input parameters for creating a new backtest (status=pending).
    
    Includes all parameters for both fetch and backtest operations.
    """
    # Required parameters
    symbol: str
    start_date: date
    end_date: date
    
    # Agent to execute this backtest via n8n
    agent_id: Optional[int] = None

    # Self-learning: inject/record agent lessons in this run. Disable for the
    # baseline leg of an A/B comparison.
    use_lessons: bool = True

    # Max seconds the runner waits for each blocking agent reply before failing
    # the run. Capped at 570s: the backtest service hold-watchdog ceiling is
    # 600s and the runner adds a +15s TTL margin on top of this value.
    agent_timeout_s: Optional[float] = Field(default=None, ge=30, le=570)

    # Data source parameters (for fetch)
    source: Optional[Literal["ibkr", "yahoo", "binance", "ctrader"]] = None
    timeframe: str = "5m"  # e.g., "1m", "5m", "15m", "1h", "1d"
    asset: Literal["stock", "future", "futures", "forex", "crypto"] = "stock"
    rth: bool = True  # True = Regular Trading Hours only
    
    # IBKR-specific parameters
    ibkr_config: Optional[str] = "configs/ibkr.yaml"
    exchange: str = "SMART"
    currency: str = "USD"
    
    # Futures-specific parameters
    expiry: Optional[str] = None  # YYYYMM, YYYYMMDD, or "auto"
    
    # Backtest execution parameters
    initial_capital: float = 100000.0
    commission: float = 0.0
    
    # Additional config overrides (JSONB)
    parameters: Optional[dict[str, Any]] = None

    # Full strategy configuration snapshot (auto-captured from strategy.definition)
    config: Optional[dict[str, Any]] = None

    # UI layout config snapshot (timezone, extended hours, etc.)
    layout_config: Optional[dict[str, Any]] = None


class BacktestRead(BaseModel):
    """Full backtest representation including status and results."""
    id: int
    strategy_id: int
    chat_id: Optional[int] = None
    agent_id: Optional[int] = None
    # Data connection the run used (the owning strategy's); charts of a
    # finished run need it to subscribe history-only on the same dataset.
    connection_id: Optional[int] = None

    # Input parameters
    symbol: str
    start_date: date
    end_date: date
    
    # Data source parameters
    source: Optional[str] = None
    timeframe: Optional[str] = None
    asset: Optional[str] = None
    rth: Optional[bool] = None
    
    # IBKR-specific parameters
    ibkr_config: Optional[str] = None
    exchange: Optional[str] = None
    currency: Optional[str] = None
    expiry: Optional[str] = None
    
    # Backtest execution parameters
    initial_capital: Optional[float] = None
    commission: Optional[float] = None
    
    # Additional config overrides
    parameters: Optional[dict[str, Any]] = None

    # Full strategy configuration snapshot
    config: Optional[dict[str, Any]] = None
    
    # Status (pending | running | completed | failed | error)
    status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # Typed metrics from edgewalker (populated on completion)
    return_pct: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    win_rate_pct: Optional[float] = None
    profit_factor: Optional[float] = None
    total_trades: Optional[int] = None
    equity_final: Optional[float] = None
    equity_peak: Optional[float] = None
    
    # Extra metrics (JSONB for additional data)
    metrics: Optional[dict[str, Any]] = None
    html_report_url: Optional[str] = None

    # UI layout persistence
    layout_config: Optional[dict[str, Any]] = None
    
    created_at: datetime


class BacktestUpdate(BaseModel):
    """Schema for updating backtest status and results (used by n8n callback).
    
    The n8n workflow should populate typed metrics directly.
    """
    status: Optional[str] = None
    error_message: Optional[str] = None
    
    # Typed metrics (preferred - populate these from edgewalker results)
    return_pct: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    win_rate_pct: Optional[float] = None
    profit_factor: Optional[float] = None
    total_trades: Optional[int] = None
    equity_final: Optional[float] = None
    equity_peak: Optional[float] = None
    
    # Extra metrics JSONB (for any additional data)
    metrics: Optional[dict[str, Any]] = None
    html_report_url: Optional[str] = None

    # UI layout persistence
    layout_config: Optional[dict[str, Any]] = None


class BacktestPlaybackControl(BaseModel):
    """Runtime playback command for an active backtest replay."""
    action: Literal["pause", "paused", "resume", "play", "playing", "speed", "set_speed", "step"]
    speed_multiplier: Optional[float] = None
    bars_per_second: Optional[float] = None
    steps: Optional[int] = None


class BacktestRuntimeOrderRequest(BaseModel):
    """Manual order command for an active runtime backtest."""
    symbol: str
    side: Literal["buy", "sell"]
    order_type: Literal["market", "limit", "stop"] = "market"
    quantity: float = 1
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    order_ref: Optional[str] = None
    extra: Optional[dict[str, Any]] = None


class BacktestCreateRequest(BacktestCreate):
    """BacktestCreate plus the owning strategy, for the flat /backtests router."""
    strategy_id: int


class BacktestRuntimePositionCloseRequest(BaseModel):
    """Manual close command for a simulated position of a runtime backtest."""
    quantity: float = Field(gt=0)
    symbol: Optional[str] = None
    reason: Optional[str] = None
    extra: Optional[dict[str, Any]] = None


class BacktestSummary(BaseModel):
    """Row of the global backtests listing: scalar fields only, no JSONB blobs."""
    id: int
    strategy_id: int
    strategy_name: Optional[str] = None
    connection_id: Optional[int] = None
    chat_id: Optional[int] = None
    agent_id: Optional[int] = None
    symbol: str
    start_date: date
    end_date: date
    source: Optional[str] = None
    timeframe: Optional[str] = None
    asset: Optional[str] = None
    initial_capital: Optional[float] = None
    commission: Optional[float] = None
    status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    return_pct: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    win_rate_pct: Optional[float] = None
    profit_factor: Optional[float] = None
    total_trades: Optional[int] = None
    equity_final: Optional[float] = None
    equity_peak: Optional[float] = None
    created_at: datetime

    # Live-run enrichment (pending/running rows only, when requested):
    # phase/progress come from the coordinator; stale means the DB says
    # running but neither the coordinator nor the runner container exists.
    phase: Optional[str] = None
    progress: Optional[float] = None
    stale: Optional[bool] = None


class BacktestListPage(BaseModel):
    items: list[BacktestSummary]
    total: int
    limit: int
    offset: int



# ─── LAYOUT CONFIG SCHEMAS ───


class LayoutConfigUpdate(BaseModel):
    """Schema for updating only the layout_config field."""
    layout_config: dict[str, Any]


# ─── CHART DRAWINGS SCHEMAS ───
#
# Canonical schema of a user drawing on a strategy chart, mirrored by
# ``edgewalker.charting.drawings`` (renderer) and the FE
# ``plugins/drawings/types.ts``.  Stored in
# ``definition.strategy.charts[i].drawings``; ``t`` is epoch seconds UTC.

DrawingType = Literal[
    "trend_line", "ray", "extended_line",
    "horizontal_line", "horizontal_ray", "vertical_line",
    "rectangle", "path", "text", "callout",
]

#: type → (min anchors, max anchors | None)
DRAWING_ANCHOR_COUNTS: dict[str, tuple[int, int | None]] = {
    "trend_line": (2, 2),
    "ray": (2, 2),
    "extended_line": (2, 2),
    "horizontal_line": (1, 1),
    "horizontal_ray": (1, 1),
    "vertical_line": (1, 1),
    "rectangle": (2, 2),
    "path": (2, None),
    "text": (1, 1),
    "callout": (2, 2),
}

MAX_DRAWING_ANCHORS = 500
MAX_DRAWINGS_PER_CHART = 200


class ChartDrawingAnchor(BaseModel):
    t: float = Field(description="Epoch seconds UTC")
    p: float = Field(description="Price")


class ChartDrawingStyle(BaseModel):
    color: str = Field(default="#2962ff", max_length=32)
    width: float = Field(default=2.0, gt=0, le=20)
    dash: Literal["solid", "dashed", "dotted"] = "solid"
    fill: Optional[str] = Field(default=None, max_length=32)
    fill_opacity: Optional[float] = Field(default=None, ge=0, le=1)
    font_size: Optional[float] = Field(default=None, gt=0, le=72)
    text_color: Optional[str] = Field(default=None, max_length=32)


class ChartDrawing(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    type: DrawingType
    symbol: Optional[str] = Field(default=None, max_length=64)
    anchors: list[ChartDrawingAnchor] = Field(max_length=MAX_DRAWING_ANCHORS)
    style: ChartDrawingStyle = Field(default_factory=ChartDrawingStyle)
    props: dict[str, Any] = Field(default_factory=dict)
    locked: bool = False
    created_at: Optional[int] = None
    updated_at: Optional[int] = None

    @model_validator(mode="after")
    def _check_anchor_count(self) -> "ChartDrawing":
        lo, hi = DRAWING_ANCHOR_COUNTS[self.type]
        n = len(self.anchors)
        if n < lo or (hi is not None and n > hi):
            raise ValueError(
                f"{self.type} needs {lo}{'+' if hi is None else '' if hi == lo else f'-{hi}'} anchors, got {n}"
            )
        text = self.props.get("text")
        if text is not None and (not isinstance(text, str) or len(text) > 500):
            raise ValueError("props.text must be a string of at most 500 characters")
        return self


class ChartDrawingsUpdate(BaseModel):
    """Full replacement of the drawings of one chart (PUT semantics)."""
    drawings: list[ChartDrawing] = Field(max_length=MAX_DRAWINGS_PER_CHART)


class ChartDrawingsRead(BaseModel):
    chart_id: str
    drawings: list[ChartDrawing]


# ─── TRADE SCHEMAS ───


class TradeCreate(BaseModel):
    """Input for creating a trade record (aligned with edgewalker TradeRecord)."""
    entry_time: datetime
    exit_time: Optional[datetime] = None
    direction: str  # "long" or "short"
    size: float  # Position size
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None  # Return percentage
    session_date: Optional[date] = None  # Trading session date
    exit_reason: Optional[str] = None  # e.g., "stop_loss", "take_profit", "eod"
    extra: Optional[Any] = None  # Extra data


class TradeRead(BaseModel):
    id: int
    backtest_id: int
    strategy_id: int
    entry_time: datetime
    exit_time: Optional[datetime] = None
    direction: str
    size: float
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    session_date: Optional[date] = None
    exit_reason: Optional[str] = None
    extra: Optional[Any] = None


# ─── RULE TRIGGER SCHEMAS ───


class RuleTriggerRequest(BaseModel):
    """Request to trigger an agent from a rule during backtest."""
    agent_id: int
    chat_id: int
    rule_context: dict[str, Any]
    webhook_url: Optional[str] = None  # Optional override for backtest scenarios


class RuleTriggerResponse(BaseModel):
    """Response from agent webhook after rule trigger."""
    status: str
    agent_response: Optional[dict[str, Any]] = None
