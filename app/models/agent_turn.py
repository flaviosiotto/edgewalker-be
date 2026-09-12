"""Debug record of one agent turn, written by agent-svc (migration 057).

Read-only for the backend: the admin console browses turns (what the model
saw, the tool calls it made, the charts it was shown, tokens and timing).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import Column, DateTime, ForeignKey, Integer, LargeBinary, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


class AgentTurn(SQLModel, table=True):
    __tablename__ = "agent_turn"

    turn_id: str = Field(sa_column=Column(String(32), primary_key=True))
    session_id: str = Field(sa_column=Column(String(255), nullable=False, index=True))
    user_id: Optional[int] = Field(
        default=None, sa_column=Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=True, index=True)
    )
    agent_id: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))
    strategy_id: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))
    strategy_live_id: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))
    backtest_id: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))
    kind: Optional[str] = Field(default=None, sa_column=Column(String(160), nullable=True))
    trigger_type: Optional[str] = Field(default=None, sa_column=Column(String(64), nullable=True))
    correlation_id: Optional[str] = Field(default=None, sa_column=Column(String(100), nullable=True))
    status: str = Field(sa_column=Column(String(16), nullable=False))
    error: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    started_at: datetime = Field(sa_column=Column(DateTime(timezone=True), nullable=False))
    finished_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True), nullable=True))
    duration_ms: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))
    model: Optional[str] = Field(default=None, sa_column=Column(String(120), nullable=True))
    requests: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))
    tool_calls: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))
    tokens_input: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))
    tokens_output: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))
    tokens_reasoning: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))
    tokens_cached: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))
    prompt_chars: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))
    response_chars: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))
    user_message: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    system_prompt: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    response: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    steps: list[Any] = Field(default_factory=list, sa_column=Column(JSONB, nullable=False, server_default="[]"))
    context: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False, server_default="{}"))


class AgentTurnImage(SQLModel, table=True):
    __tablename__ = "agent_turn_image"

    id: Optional[int] = Field(default=None, primary_key=True)
    turn_id: str = Field(
        sa_column=Column(String(32), ForeignKey("agent_turn.turn_id", ondelete="CASCADE"), nullable=False, index=True)
    )
    name: str = Field(sa_column=Column(String(120), nullable=False))
    mime: str = Field(default="image/png", sa_column=Column(String(64), nullable=False))
    # DB column is "bytes"; the attribute is `size` so the builtin `bytes`
    # annotation below is not shadowed inside the class body.
    size: int = Field(sa_column=Column("bytes", Integer, nullable=False))
    data: bytes = Field(sa_column=Column(LargeBinary, nullable=False))
