from __future__ import annotations

import enum
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, Column, Computed, Enum as SAEnum, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from sqlmodel import Field, Relationship, SQLModel

if TYPE_CHECKING:
    from app.models.strategy import Strategy


def _enum_values_callable(enum_cls: type[enum.Enum]) -> list[str]:
    return [e.value for e in enum_cls]


class Agent(SQLModel, table=True):
    id_agent: Optional[int] = Field(default=None, primary_key=True)
    agent_name: str = Field(sa_column=Column(String(255), unique=True, nullable=False))
    n8n_webhook: str = Field(sa_column=Column(String(512), nullable=False))
    is_default: bool = Field(sa_column=Column(Boolean, nullable=False, server_default="false"))



class Chat(SQLModel, table=True):
    __allow_unmapped__ = True
    id: Optional[int] = Field(default=None, primary_key=True)

    user_id: Optional[int] = Field(
        default=None,
        foreign_key="user.id",
        description="ID dell'utente proprietario della chat",
    )

    id_agent: Optional[int] = Field(
        default=None,
        sa_column=Column(
            ForeignKey("agent.id_agent", ondelete="SET NULL"),
            nullable=True,
        ),
        description="FK all'agente associato alla chat",
    )

    strategy_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("strategies.id", ondelete="CASCADE"),
            nullable=True,
            index=True,
        ),
        description="FK alla strategia associata alla chat",
    )

    nome: str = Field(max_length=255, description="Nome della chat")
    descrizione: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Descrizione della chat",
    )

    class ChatType(str, enum.Enum):
        USER = "user"
        SYSTEM = "system"
        STRATEGY = "strategy"
        GENERIC = "generic"

    chat_type: "Chat.ChatType" = Field(
        sa_column=Column(
            SAEnum(ChatType, native_enum=False, values_callable=_enum_values_callable),
            nullable=False,
            server_default="user",
        ),
        description="Tipo chat: user|system|strategy|generic",
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

    # Relationships
    chat_histories: list["N8nChatHistory"] = Relationship(
        sa_relationship=relationship(
            "N8nChatHistory",
            back_populates="chat",
            cascade="all, delete-orphan",
        )
    )

    strategy: Optional["Strategy"] = Relationship(
        sa_relationship=relationship(
            "Strategy",
            back_populates="chats",
        )
    )

    agent: Optional["Agent"] = Relationship(
        sa_relationship=relationship(
            "Agent",
            foreign_keys="[Chat.id_agent]",
        )
    )


from app.models.n8n_chat_history import N8nChatHistory  # noqa: E402
from app.models.strategy import Strategy  # noqa: E402

