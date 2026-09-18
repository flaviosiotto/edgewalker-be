from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import Column, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


class StrategyTemplate(SQLModel, table=True):
    """A strategy definition detached from any market.

    Holds rules, indicators, parameters, lessons and the chart timeframes
    (the multi-chart structure IS the strategy), but never a symbol, asset,
    contract or broker: those are chosen when the template is instantiated on
    an account (``strategy_template_service.instantiate``).

    ``user_id`` NULL = official EdgeWalker template, synced at startup from
    ``system_templates/<key>.json``. User templates are private to their owner.
    """

    __tablename__ = "strategy_templates"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("user.id", ondelete="CASCADE"),
            nullable=True,
            index=True,
        ),
    )
    # Official templates only: stable key of the source file (upsert target).
    key: Optional[str] = Field(default=None, sa_column=Column(String(64), nullable=True))
    name: str = Field(sa_column=Column(String(80), nullable=False))
    description: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    tags: list[str] = Field(default_factory=list, sa_column=Column(JSONB, nullable=False))
    # Sanitised DSL: no symbol/asset/extra_data/sources/studios/drawings.
    definition: dict[str, Any] = Field(sa_column=Column(JSONB, nullable=False))
    # [{lesson, context?, confidence}] — evidence-free seeds for the new strategy.
    lessons: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSONB, nullable=False))
    # [{id, role, label?, timeframe, history_depth_days?}] — chart slots the
    # wizard asks a symbol for (timeframe pre-filled, editable).
    charts_meta: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSONB, nullable=False))
    # Informational only: {strategy_id?, backtest_id?}. Never symbol/broker.
    origin: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSONB, nullable=True))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def official(self) -> bool:
        return self.user_id is None
