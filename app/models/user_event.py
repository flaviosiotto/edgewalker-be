"""Outcomes of long-running tasks, per user (migr. 070).

One row per *outcome* (a connection that connected or failed, a backtest
that finished, a live session that started/crashed/stopped, a studio run
that succeeded/failed), written by the webhook event sources
(``services/webhook_sources.py``) through ``services/user_events.record``.
Intermediate states and progress are deliberately not events: the FE polls
the entity for those. The closed catalogue lives in ``services/user_events``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import BigInteger, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UserEvent(SQLModel, table=True):
    __tablename__ = "user_event"

    id: Optional[int] = Field(default=None, sa_column=Column(BigInteger, primary_key=True, autoincrement=True))
    user_id: int = Field(
        sa_column=Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    #: catalogue type, e.g. ``connection.failed`` (services/user_events.UserEventType)
    type: str = Field(sa_column=Column(String(64), nullable=False))
    #: info | success | warning | error (derived from the type)
    severity: str = Field(sa_column=Column(String(16), nullable=False))
    #: connection | backtest | live | studio_run
    entity_type: str = Field(sa_column=Column(String(32), nullable=False))
    entity_id: int = Field(sa_column=Column(Integer, nullable=False))
    #: human label of the entity (connection name, strategy name, studio name)
    entity_name: Optional[str] = Field(default=None, sa_column=Column(String(200), nullable=True))
    #: the status_message / error_message / run.error, verbatim
    message: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    #: long diagnostic (runner log tail); served only by GET /events/{id}
    details: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    #: the same ``data`` the webhook envelope carries
    data: Any = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False))
    #: UI links ({"ui": "/strategies/3?backtest=210"})
    links: Any = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False))
    dedupe_key: Optional[str] = Field(default=None, sa_column=Column(String(200), nullable=True))
    created_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))
