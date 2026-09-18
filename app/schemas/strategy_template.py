"""Strategy templates — API contract (shared with the FE wizard).

A template never carries symbol/asset/contract/broker. Chart timeframes are
kept (multi-chart structure is strategy logic) and pre-fill the wizard.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class TemplateChartMeta(BaseModel):
    """One chart slot of the template: the wizard asks a symbol for each."""
    id: str
    role: Literal["primary", "secondary"]
    label: Optional[str] = None
    timeframe: str
    history_depth_days: Optional[int] = None
    # Names of the indicators on this chart (display only).
    indicators: list[str] = Field(default_factory=list)


class TemplateLesson(BaseModel):
    lesson: str = Field(min_length=1, max_length=2000)
    context: Optional[str] = Field(default=None, max_length=2000)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class TemplateOrigin(BaseModel):
    strategy_id: Optional[int] = None
    backtest_id: Optional[int] = None
    live_id: Optional[int] = None


class StrategyTemplateSummary(BaseModel):
    """Listing row: no definition blob."""
    id: int
    official: bool
    key: Optional[str] = None
    name: str
    description: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    charts_meta: list[TemplateChartMeta] = Field(default_factory=list)
    rules_count: int = 0
    lessons_count: int = 0
    created_at: datetime
    updated_at: datetime


class StrategyTemplateRead(StrategyTemplateSummary):
    definition: Any
    lessons: list[TemplateLesson] = Field(default_factory=list)
    origin: Optional[TemplateOrigin] = None
    # Sanitisation notes recorded at save time (e.g. studio-bound rules).
    warnings: list[str] = Field(default_factory=list)


class StrategyTemplateSource(BaseModel):
    """Exactly one of: an owned strategy, an owned backtest, an owned live
    session (its frozen definition), a raw definition."""
    strategy_id: Optional[int] = None
    backtest_id: Optional[int] = None
    live_id: Optional[int] = None
    definition: Optional[Any] = None

    @model_validator(mode="after")
    def _exactly_one(self) -> "StrategyTemplateSource":
        given = [v for v in (self.strategy_id, self.backtest_id, self.live_id, self.definition) if v is not None]
        if len(given) != 1:
            raise ValueError("Specify exactly one of strategy_id, backtest_id, live_id, definition")
        return self


class StrategyTemplateCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: Optional[str] = Field(default=None, max_length=4000)
    tags: list[str] = Field(default_factory=list, max_length=20)
    source: StrategyTemplateSource
    # Copy the strategy's active lessons into the template (strategy/backtest sources).
    include_lessons: bool = True
    # Chart labels chosen by the author, keyed by chart id ("esecuzione", "contesto").
    chart_labels: dict[str, str] = Field(default_factory=dict)


class StrategyTemplateUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=80)
    description: Optional[str] = Field(default=None, max_length=4000)
    tags: Optional[list[str]] = Field(default=None, max_length=20)
    lessons: Optional[list[TemplateLesson]] = None
    chart_labels: Optional[dict[str, str]] = None


class TemplateChartBinding(BaseModel):
    """Market chosen for one chart slot at instantiation."""
    symbol: str = Field(min_length=1, max_length=40)
    asset_type: Optional[str] = None
    timeframe: Optional[str] = None          # None → the template's timeframe
    extra_data: Optional[dict[str, Any]] = None


class StrategyTemplateInstantiate(BaseModel):
    account_id: int
    manager_agent_id: Optional[int] = None
    name: Optional[str] = Field(default=None, max_length=60)
    # chart id → binding. Every slot of charts_meta must be bound.
    charts: dict[str, TemplateChartBinding]
    include_lessons: bool = True


class StrategyTemplateInstantiateResponse(BaseModel):
    strategy_id: int
    name: str
    warnings: list[str] = Field(default_factory=list)


class TemplatePreview(BaseModel):
    """What ``POST /strategy-templates`` would save, without saving (dry run)."""
    charts_meta: list[TemplateChartMeta]
    rules_count: int
    lessons: list[TemplateLesson]
    warnings: list[str]
    custom_indicators: list[str]
