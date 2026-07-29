from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import Column, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


class AgentLesson(SQLModel, table=True):
    """A lesson the manager agent distilled for one strategy.

    Lessons are the agent's persistent playbook: written during backtest
    reviews (or by the user), injected into the agent system message as the
    LEZIONI APPRESE block, and revocable at any time (status=retired).
    Every lesson carries its evidence (backtest/trade refs) so it can be
    audited and validated against later runs.
    """

    __tablename__ = "agent_lessons"

    id: Optional[int] = Field(default=None, primary_key=True)
    strategy_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("strategies.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )
    user_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("user.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )
    lesson: str = Field(sa_column=Column(Text, nullable=False))
    context: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    status: str = Field(
        default="active",
        sa_column=Column(String(16), nullable=False, server_default="active"),
    )
    confidence: float = Field(
        default=0.5,
        sa_column=Column(Float, nullable=False, server_default="0.5"),
    )
    source: str = Field(
        default="backtest",
        sa_column=Column(String(16), nullable=False, server_default="backtest"),
    )
    backtest_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("strategy_backtests.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    evidence: Optional[dict[str, Any]] = Field(
        default=None, sa_column=Column(JSONB, nullable=True)
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
