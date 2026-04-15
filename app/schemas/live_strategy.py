from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel


LiveStatus = Literal["stopped", "starting", "running", "stopping", "error"]
LiveSyncState = Literal["aligned", "stale", "missing_container", "unknown"]
PositionSide = Literal["long", "short", "flat"]


class LivePerformanceSummary(BaseModel):
    total_pnl: float | None = None
    total_trades: int = 0
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
    account_id: int


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