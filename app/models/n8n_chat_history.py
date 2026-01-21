from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, ForeignKey, String
from sqlmodel import Field, SQLModel


class N8nChatHistory(SQLModel, table=True):
    __tablename__ = "n8n_chat_histories"

    id: Optional[int] = Field(default=None, primary_key=True)
    chat_id: Optional[int] = Field(
        default=None,
        sa_column=Column(ForeignKey("chat.id", ondelete="CASCADE"), nullable=False),
        description="FK alla chat",
    )

    role: Optional[str] = Field(default=None, sa_column=Column(String(50), nullable=True))
    content: Optional[str] = Field(default=None, sa_column=Column(String, nullable=True))
    created_at: datetime = Field(default_factory=datetime.utcnow)

