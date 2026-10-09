"""Schemas of the agent model F3: skills, memory, action requests, history search."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
SKILL_BODY_MAX_CHARS = 20_000
MEMORY_MAX_CHARS = 4_000

MemoryKind = Literal["user_profile", "market_notes", "operating_rules"]
ActionStatus = Literal["pending", "approved", "rejected", "expired", "failed"]


def validate_skill_name(value: str) -> str:
    name = str(value or "").strip().lower()
    if not SKILL_NAME_RE.match(name):
        raise ValueError("name: lowercase letters, digits and dashes, max 63 characters")
    return name


# ---------------------------------------------------------------------------
# skills
# ---------------------------------------------------------------------------


class SkillBase(BaseModel):
    description: str = Field(default="", max_length=1024)
    body: str = Field(min_length=1, max_length=SKILL_BODY_MAX_CHARS)


class SkillCreate(SkillBase):
    name: str = Field(max_length=64)

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        return validate_skill_name(value)


class SkillUpdate(BaseModel):
    description: Optional[str] = Field(default=None, max_length=1024)
    body: Optional[str] = Field(default=None, min_length=1, max_length=SKILL_BODY_MAX_CHARS)


class SkillFromPlaybook(BaseModel):
    """Promote the playbook of a strategy (current, or the one of a backtest)
    to a skill: an explicit action of the user (decision D10)."""

    name: str = Field(max_length=64)
    strategy_id: int
    backtest_id: Optional[int] = None
    description: Optional[str] = Field(default=None, max_length=1024)

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        return validate_skill_name(value)


class SkillSummary(BaseModel):
    id: int
    name: str
    description: str
    version: int
    origin: str
    source_strategy_id: Optional[int] = None
    source_backtest_id: Optional[int] = None
    updated_at: datetime

    model_config = {"from_attributes": True}


class SkillRead(SkillSummary):
    body: str
    created_at: datetime


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------


class MemoryRead(BaseModel):
    kind: MemoryKind
    content: str
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None


class MemoryUpdate(BaseModel):
    content: str = Field(max_length=MEMORY_MAX_CHARS)


# ---------------------------------------------------------------------------
# action requests ("ask first")
# ---------------------------------------------------------------------------


class ActionRequestCreate(BaseModel):
    tool_name: str = Field(max_length=64)
    args: dict[str, Any] = Field(default_factory=dict)
    rationale: Optional[str] = Field(default=None, max_length=2000)
    chat_id: Optional[int] = None
    strategy_live_id: Optional[int] = None
    account_id: Optional[int] = None
    #: minutes before the request expires (bounded by the service)
    ttl_minutes: Optional[int] = Field(default=None, ge=1, le=24 * 60)


class ActionRequestRead(BaseModel):
    id: int
    agent_id: int
    agent_name: Optional[str] = None
    chat_id: Optional[int] = None
    strategy_live_id: Optional[int] = None
    account_id: Optional[int] = None
    tool_name: str
    args: dict[str, Any]
    rationale: Optional[str] = None
    status: ActionStatus
    expires_at: datetime
    decided_at: Optional[datetime] = None
    decided_by: Optional[dict[str, Any]] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    #: one-line human summary of the action (UI, webhook)
    summary: str = ""


class ActionRequestDecision(BaseModel):
    note: Optional[str] = Field(default=None, max_length=500)


# ---------------------------------------------------------------------------
# history search
# ---------------------------------------------------------------------------


class HistoryHit(BaseModel):
    chat_id: int
    chat_name: Optional[str] = None
    row_id: int
    timestamp: Optional[datetime] = None
    type: str
    text: str
