from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Column, DateTime, ForeignKey, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlmodel import Field, Relationship, SQLModel


class N8nChatHistory(SQLModel, table=True):
    __tablename__ = "n8n_chat_histories"
    __allow_unmapped__ = True

    id: int | None = Field(default=None, primary_key=True)

    session_id: str = Field(
        sa_column=Column(
            String(255),
            ForeignKey("chat.n8n_session_id", ondelete="CASCADE"),
            index=True,
            nullable=False,
        )
    )
    message: Any = Field(sa_column=Column(JSONB, nullable=False))
    created_at: datetime | None = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(
            DateTime(timezone=True),
            nullable=True,
            server_default=text("NOW()"),
        ),
    )

    chat: "Chat" | None = Relationship(
        sa_relationship=relationship("Chat", back_populates="chat_histories")
    )


from app.models.agent import Chat  # noqa: E402

