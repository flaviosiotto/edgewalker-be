"""Outbound webhooks (migr. 067, agent bridge F2).

``WebhookSubscription`` is an endpoint a user registered for platform events;
``WebhookDelivery`` is the outbox: one row per (event, subscription), POSTed
by the dispatcher in ``services/webhook_dispatcher.py`` with retries. See
``services/webhook_service.py`` for the event catalogue and ``emit_event``.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import BigInteger, Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class WebhookSubscription(SQLModel, table=True):
    __tablename__ = "webhook_subscription"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(
        sa_column=Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    agent_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("agent.id_agent", ondelete="SET NULL"), nullable=True, index=True),
    )
    name: str = Field(sa_column=Column(String(120), nullable=False))
    url: str = Field(sa_column=Column(String(2048), nullable=False))
    secret_encrypted: str = Field(sa_column=Column(Text, nullable=False))
    events: list = Field(default_factory=lambda: ["*"], sa_column=Column(JSONB, nullable=False))
    active: bool = Field(default=True, sa_column=Column(Boolean, nullable=False, server_default="true"))
    failure_streak: int = Field(default=0, sa_column=Column(Integer, nullable=False, server_default="0"))
    disabled_reason: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    last_success_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True), nullable=True))
    last_failure_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True), nullable=True))
    created_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))
    updated_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))


class WebhookDelivery(SQLModel, table=True):
    __tablename__ = "webhook_delivery"

    id: Optional[int] = Field(default=None, sa_column=Column(BigInteger, primary_key=True, autoincrement=True))
    subscription_id: int = Field(
        sa_column=Column(
            Integer, ForeignKey("webhook_subscription.id", ondelete="CASCADE"), nullable=False, index=True
        )
    )
    event_id: uuid.UUID = Field(default_factory=uuid.uuid4, sa_column=Column(UUID(as_uuid=True), nullable=False))
    event_type: str = Field(sa_column=Column(String(64), nullable=False))
    dedupe_key: Optional[str] = Field(default=None, sa_column=Column(String(200), nullable=True))
    payload: Any = Field(sa_column=Column(JSONB, nullable=False))
    status: str = Field(default="pending", sa_column=Column(String(16), nullable=False, server_default="pending"))
    attempts: int = Field(default=0, sa_column=Column(Integer, nullable=False, server_default="0"))
    max_attempts: int = Field(default=8, sa_column=Column(Integer, nullable=False, server_default="8"))
    next_attempt_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))
    last_status_code: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))
    last_error: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    last_attempt_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True), nullable=True))
    delivered_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True), nullable=True))
    created_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))
