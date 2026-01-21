from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from sqlalchemy import Boolean, Column, Computed, Enum as SAEnum, ForeignKey, String
from sqlmodel import Field, Relationship, SQLModel


def _enum_values_callable(enum_cls: type[enum.Enum]) -> list[str]:
    return [e.value for e in enum_cls]


class Agent(SQLModel, table=True):
    __table_args__ = {"schema": "operations"}

    id_agent: Optional[int] = Field(default=None, primary_key=True)
    agent_name: str = Field(sa_column=Column(String(255), unique=True, nullable=False))
    n8n_webhook: str = Field(sa_column=Column(String(512), nullable=False))
    is_default: bool = Field(sa_column=Column(Boolean, nullable=False, server_default="false"))

    chats: list["Chat"] = Relationship(
        back_populates="agent",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class Chat(SQLModel, table=True):
    __table_args__ = {"schema": "operations"}

    id: Optional[int] = Field(default=None, primary_key=True)

    user_id: Optional[int] = Field(
        default=None,
        foreign_key="operations.user.id",
        description="ID dell'utente proprietario della chat",
    )

    id_agent: Optional[int] = Field(
        default=None,
        sa_column=Column(
            ForeignKey("operations.agent.id_agent", ondelete="CASCADE"),
            nullable=True,
        ),
        description="FK all'agente associato alla chat",
    )
    agent: Optional[Agent] = Relationship(back_populates="chats")

    nome: str = Field(max_length=255, description="Nome della chat")
    descrizione: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Descrizione della chat",
    )

    class ChatType(str, enum.Enum):
        USER = "user"
        SYSTEM = "system"

    chat_type: "Chat.ChatType" = Field(
        sa_column=Column(
            SAEnum(ChatType, native_enum=False, values_callable=_enum_values_callable),
            nullable=False,
            server_default="user",
        ),
        description="Tipo chat: user|system",
    )
    created_at: datetime = Field(default_factory=datetime.now, description="Timestamp di creazione")

    n8n_session_id: Optional[str] = Field(
        default=None,
        sa_column=Column(
            String(255),
            Computed("id::text", persisted=True),
            unique=True,
            nullable=False,
        ),
        description="ID chat in formato string (usato da n8n_chat_histories)",
    )

    chat_histories: list["N8nChatHistory"] = Relationship(
        back_populates="chat",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


if TYPE_CHECKING:
    from app.models.n8n_chat_history import N8nChatHistory
