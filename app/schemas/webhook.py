from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator


class WebhookSubscriptionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    url: HttpUrl
    #: event names from the catalogue, or ["*"]
    events: list[str] = ["*"]
    #: optional external agent this endpoint belongs to
    agent_id: Optional[int] = None
    active: bool = True

    @field_validator("events")
    @classmethod
    def _non_empty(cls, value: list[str]) -> list[str]:
        cleaned = sorted({v.strip() for v in value if v and v.strip()})
        if not cleaned:
            raise ValueError("At least one event (or '*') is required")
        return cleaned


class WebhookSubscriptionUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    url: Optional[HttpUrl] = None
    events: Optional[list[str]] = None
    agent_id: Optional[int] = None
    active: Optional[bool] = None
    #: true = rotate the signing secret (the new one is returned once)
    rotate_secret: bool = False

    @field_validator("events")
    @classmethod
    def _non_empty(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        if value is None:
            return None
        cleaned = sorted({v.strip() for v in value if v and v.strip()})
        if not cleaned:
            raise ValueError("At least one event (or '*') is required")
        return cleaned


class WebhookSubscriptionRead(BaseModel):
    id: int
    user_id: int
    agent_id: Optional[int] = None
    name: str
    url: str
    events: list[str]
    active: bool
    failure_streak: int
    disabled_reason: Optional[str] = None
    last_success_at: Optional[datetime] = None
    last_failure_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class WebhookSubscriptionCreated(WebhookSubscriptionRead):
    #: the signing secret, returned only here (and on rotation)
    secret: str


class WebhookDeliveryRead(BaseModel):
    id: int
    subscription_id: int
    event_id: str
    event_type: str
    status: str
    attempts: int
    max_attempts: int
    next_attempt_at: Optional[datetime] = None
    last_status_code: Optional[int] = None
    last_error: Optional[str] = None
    last_attempt_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    created_at: datetime
    payload: Any = None


class WebhookEventSpec(BaseModel):
    type: str
    description: str
    data_fields: list[str]
