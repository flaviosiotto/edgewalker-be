from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import Column, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


class AgentLesson(SQLModel, table=True):
    """A lesson of a playbook (migration 062).

    A playbook is the set of rows of one run: a backtest starts from the
    playbook chosen at launch (copied into rows with ``scope=backtest`` and
    ``run_backtest_id`` = the run, ``parent_id`` = the input row), the agent
    edits only those rows, and the output freezes when the run ends. Live
    attaches to the output of one backtest and never writes. Rows with
    ``scope=strategy`` are the strategy's initial playbook (pre-062 rows and
    manual ones). ``backtest_id`` is the run the lesson was BORN in and
    survives copies; ``evidence`` keeps the audit trail.
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
    scope: str = Field(
        default="strategy",
        sa_column=Column(String(16), nullable=False, server_default="strategy"),
    )
    run_backtest_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("strategy_backtests.id", ondelete="CASCADE"),
            nullable=True,
            index=True,
        ),
    )
    parent_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("agent_lessons.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
