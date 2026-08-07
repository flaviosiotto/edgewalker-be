from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class PatCreate(BaseModel):
    name: str
    scopes: list[str] = ["read"]
    expires_in_days: Optional[int] = None


class PatRead(BaseModel):
    id: int
    name: str
    token_prefix: str
    scopes: list[str]
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    created_at: datetime


class PatCreated(PatRead):
    #: The raw token, returned only by the create endpoint and never again.
    token: str
