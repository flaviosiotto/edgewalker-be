from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.models.agent import Chat


class ChatCreate(BaseModel):
    nome: str
    descrizione: Optional[str] = None
    chat_type: Optional[Chat.ChatType] = None
    user_id: Optional[int] = None
    strategy_id: Optional[int] = None
    id_agent: Optional[int] = None


class ChatRead(BaseModel):
    id: int
    id_agent: Optional[int]
    agent_name: Optional[str] = None
    agent_webhook_url: Optional[str] = None
    user_id: Optional[int]
    strategy_id: Optional[int]
    nome: str
    descrizione: Optional[str]
    chat_type: Chat.ChatType
    created_at: datetime
    n8n_session_id: Optional[str]

    class Config:
        from_attributes = True


class ChatHistoryMessageRead(BaseModel):
    id: int
    session_id: str
    message: dict[str, Any]
    text: str
    message_type: Optional[str] = None
    sender_kind: Optional[str] = None
    sender_label: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    format: Optional[str] = None


class ChatHistoryPage(BaseModel):
    chat_id: int
    session_id: str
    items: list[ChatHistoryMessageRead]
    limit: int
    next_before: Optional[int] = None
    has_more: bool


class ChatSendMessageRequest(BaseModel):
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatSendMessageResponse(BaseModel):
    status: str
    chat_id: int
    session_id: str
    request_id: Optional[str] = None
