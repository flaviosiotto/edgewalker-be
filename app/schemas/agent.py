from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel

AgentKind = Literal["design", "running"]


class AgentCreate(BaseModel):
    agent_name: str
    n8n_webhook: str
    is_default: bool = False
    agent_kind: AgentKind = "running"


class AgentRead(BaseModel):
    id_agent: int
    agent_name: str
    n8n_webhook: str
    is_default: bool
    agent_kind: AgentKind = "running"


class AgentUpdate(BaseModel):
    agent_name: Optional[str] = None
    n8n_webhook: Optional[str] = None
    is_default: Optional[bool] = None
    agent_kind: Optional[AgentKind] = None


class AgentReadWithMeta(AgentRead):
    created_default_chat_id: Optional[int] = None
    created_default_chat_name: Optional[str] = None
