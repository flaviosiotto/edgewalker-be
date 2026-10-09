from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class PatCreate(BaseModel):
    name: str
    scopes: list[str] = ["read"]
    expires_in_days: Optional[int] = None
    # The agent the token acts as (one of the user's agents). Requests are
    # attributed to it; an external agent cannot receive the 'trade' scope.
    agent_id: Optional[int] = None


class PatRead(BaseModel):
    id: int
    name: str
    token_prefix: str
    scopes: list[str]
    agent_id: Optional[int] = None
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    created_at: datetime


class PatCreated(PatRead):
    #: The raw token, returned only by the create endpoint and never again.
    token: str


class ActorRead(BaseModel):
    """Who the caller is acting as (``GET /users/me/actor``).

    Lets a machine client (the MCP server's ``whoami``) tell the user which
    agent identity its token carries and which scopes it has.
    """
    via: str
    user_id: int
    agent_id: Optional[int] = None
    agent_name: Optional[str] = None
    agent_kind: Optional[str] = None
    pat_id: Optional[int] = None
    pat_name: Optional[str] = None
    scopes: list[str] = []
