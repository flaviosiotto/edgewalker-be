from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class AgentLessonCreate(BaseModel):
    lesson: str = Field(min_length=3, max_length=2000)
    context: Optional[str] = Field(default=None, max_length=2000)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source: str = Field(default="backtest", pattern="^(backtest|live|design|user)$")
    backtest_id: Optional[int] = None
    evidence: Optional[dict[str, Any]] = None


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
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
