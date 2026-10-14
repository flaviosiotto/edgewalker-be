from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class UserEventRead(BaseModel):
    id: int
    type: str
    severity: str
    entity_type: str
    entity_id: int
    entity_name: Optional[str] = None
    message: Optional[str] = None
    #: true when GET /events/{id} has more to show (runner log tail)
    has_details: bool = False
    data: dict[str, Any] = {}
    links: dict[str, str] = {}
    created_at: datetime


class UserEventDetail(UserEventRead):
    details: Optional[str] = None


class UserEventTypeSpec(BaseModel):
    type: str
    severity: str
    entity_type: str
