from __future__ import annotations

import enum
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, Column, Computed, Enum as SAEnum, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlmodel import Field, Relationship, SQLModel

if TYPE_CHECKING:
    from app.models.strategy import Strategy


def _enum_values_callable(enum_cls: type[enum.Enum]) -> list[str]:
    return [e.value for e in enum_cls]


class Agent(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("user_id", "agent_name", name="uq_agent_user_name"),)

    id_agent: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("user.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )
    agent_name: str = Field(sa_column=Column(String(255), nullable=False))
    n8n_webhook: str = Field(sa_column=Column(String(512), nullable=False))
    is_default: bool = Field(sa_column=Column(Boolean, nullable=False, server_default="false"))



class Chat(SQLModel, table=True):
    __allow_unmapped__ = True
    id: Optional[int] = Field(default=None, primary_key=True)

    user_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("user.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
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

    # A chat is owned by exactly ONE context, expressed as a chat-side FK:
    # design -> strategy_id, backtest -> backtest_id, live -> live_id. Only one
    # of the three is set. All ON DELETE CASCADE, so deleting a context removes
    # its chat(s). This replaces the old `chat_type`-based discrimination.
    strategy_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("strategies.id", ondelete="CASCADE"),
            nullable=True,
            index=True,
        ),
        description="FK alla strategia (SOLO chat di design)",
    )

    backtest_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("strategy_backtests.id", ondelete="CASCADE"),
            nullable=True,
            index=True,
        ),
        description="FK al backtest proprietario (chat di backtest)",
    )

    live_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("strategy_live.id", ondelete="CASCADE"),
            nullable=True,
            index=True,
        ),
        description="FK alla sessione live proprietaria (chat live per-sessione)",
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
        LIVE = "live"
        BACKTEST = "backtest"

    chat_type: "Chat.ChatType" = Field(
        sa_column=Column(
            SAEnum(ChatType, native_enum=False, values_callable=_enum_values_callable),
            nullable=False,
            server_default="user",
        ),
        description="Tipo chat: user|system|strategy|generic|live|backtest",
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

    # Owning context (exactly one is set). Design chats via strategy; run chats
    # via their run entity.
    strategy: Optional["Strategy"] = Relationship(
        sa_relationship=relationship(
            "Strategy",
            foreign_keys="[Chat.strategy_id]",
            back_populates="chats",
        )
    )

    backtest: Optional["BacktestResult"] = Relationship(
        sa_relationship=relationship(
            "BacktestResult",
            foreign_keys="[Chat.backtest_id]",
            back_populates="chat",
        )
    )

    live_session: Optional["StrategyLive"] = Relationship(
        sa_relationship=relationship(
            "StrategyLive",
            foreign_keys="[Chat.live_id]",
            back_populates="chat",
        )
    )

    agent: Optional["Agent"] = Relationship(
        sa_relationship=relationship(
            "Agent",
            foreign_keys="[Chat.id_agent]",
        )
    )

    @property
    def agent_name(self) -> Optional[str]:
        return self.agent.agent_name if self.agent else None

    @property
    def agent_webhook_url(self) -> Optional[str]:
        return self.agent.n8n_webhook if self.agent else None


from app.models.n8n_chat_history import N8nChatHistory  # noqa: E402
from app.models.strategy import Strategy  # noqa: E402

