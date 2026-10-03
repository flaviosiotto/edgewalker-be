from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class AgentLessonCreate(BaseModel):
    lesson: str = Field(min_length=3, max_length=2000)
    context: Optional[str] = Field(default=None, max_length=2000)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source: str = Field(default="backtest", pattern="^(backtest|live|design|user)$")
    # The run whose playbook receives the row (and where the lesson is born).
    # None = the strategy's initial playbook (scope=strategy).
    backtest_id: Optional[int] = None
    evidence: Optional[dict[str, Any]] = None


class LessonChange(BaseModel):
    """A playbook row annotated with what the run did to it."""
    lesson: AgentLessonRead
    # new (born in the run) | changed (text/confidence differ from the input
    # row) | retired | unchanged
    change: str


class PlaybookSummary(BaseModel):
    """One selectable playbook: the output of a completed backtest (or the
    strategy's initial rows when ``backtest_id`` is None)."""
    backtest_id: Optional[int] = None
    label: str
    symbol: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    completed_at: Optional[datetime] = None
    status: Optional[str] = None
    lessons_active: int = 0
    lessons_retired: int = 0
    lessons_new: int = 0
    is_current: bool = False
    # What the run started from: none | strategy | backtest:<id>
    input_source: Optional[str] = None
    learning_mode: Optional[str] = None


class PlaybookRead(BaseModel):
    summary: PlaybookSummary
    lessons: list[LessonChange]


class PlaybookSelect(BaseModel):
    """Set (or clear) the strategy's current playbook."""
    backtest_id: Optional[int] = None


class AgentLessonUpdate(BaseModel):
    lesson: Optional[str] = Field(default=None, min_length=3, max_length=2000)
    context: Optional[str] = Field(default=None, max_length=2000)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    status: Optional[str] = Field(default=None, pattern="^(active|retired)$")
    evidence: Optional[dict[str, Any]] = None


class AgentLessonRead(BaseModel):
    id: int
    strategy_id: int
    lesson: str
    context: Optional[str] = None
    status: str
    confidence: float
    source: str
    backtest_id: Optional[int] = None
    evidence: Optional[dict[str, Any]] = None
    scope: str = "strategy"
    run_backtest_id: Optional[int] = None
    parent_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
