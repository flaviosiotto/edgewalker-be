from datetime import datetime
from typing import Optional

from pydantic import BaseModel

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
