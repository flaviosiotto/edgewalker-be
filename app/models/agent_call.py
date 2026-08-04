from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import BigInteger, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


class AgentCall(SQLModel, table=True):
    """One runner→manager-agent invocation (alert, ask_agent, review, notify).

    Written by the strategy runner at dispatch time. ``bar_ts`` is the
    simulated bar timestamp (epoch ms) the call belongs to — in backtest this
    is the frozen replay clock, so the call can be placed on the chart at the
    candle where it actually happened; ``called_at`` is the wall-clock time of
    the dispatch. ``tokens_*``/``model`` are reserved for future usage
    accounting (fillable later via ``correlation_id``).
    """

    __tablename__ = "agent_call"

    id: Optional[int] = Field(default=None, primary_key=True)
    strategy_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("strategies.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )
    backtest_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("strategy_backtests.id", ondelete="CASCADE"),
            nullable=True,
            index=True,
        ),
    )
    strategy_live_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("strategy_live.id", ondelete="CASCADE"),
            nullable=True,
            index=True,
        ),
    )
    trigger_type: str = Field(sa_column=Column(String(64), nullable=False))
    trigger_name: Optional[str] = Field(default=None, sa_column=Column(String(255), nullable=True))
    correlation_id: Optional[str] = Field(default=None, sa_column=Column(String(100), nullable=True))
    session_id: Optional[str] = Field(default=None, sa_column=Column(String(100), nullable=True))
    bar_ts: Optional[int] = Field(default=None, sa_column=Column(BigInteger, nullable=True))
    called_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default="now()"),
    )
    duration_ms: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))
    status: str = Field(
        default="delivered",
        sa_column=Column(String(16), nullable=False, server_default="delivered"),
    )
    prompt_chars: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))
    response_chars: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))
    tokens_input: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))
    tokens_output: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))
    model: Optional[str] = Field(default=None, sa_column=Column(String(64), nullable=True))
    extra: Optional[Any] = Field(default=None, sa_column=Column(JSONB, nullable=True))
