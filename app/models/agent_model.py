"""Agent model tables of the agent bridge F3 (migr. 068):

* ``agent_skill`` — the user's skills (SKILL.md, agentskills.io format):
  procedural knowledge a hosted agent loads on demand with ``load_skill``
  and an external agent reads through MCP. Text only, never a declared
  tool (decision D9).
* ``agent_memory`` — curated, bounded text memory of a hosted agent in
  three kinds (``user_profile``, ``market_notes``, ``operating_rules``),
  written by the agent (``memory_update``) or edited by the user (D8).
* ``agent_action_request`` — the "ask first" queue: a trading action the
  hosted agent proposed under a tool policy of ``ask``; decided by the user
  (UI) or a trade-scoped PAT, executed by the backend on approval.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import Column, DateTime, ForeignKey, Integer, Numeric, String, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel

MEMORY_KINDS: tuple[str, ...] = ("user_profile", "market_notes", "operating_rules")
SKILL_ORIGINS: tuple[str, ...] = ("user", "playbook", "import")
ACTION_STATUSES: tuple[str, ...] = ("pending", "approved", "rejected", "expired", "failed")
RUN_SOURCES: tuple[str, ...] = ("api", "paperclip", "a2a", "mcp")
RUN_SCOPES: tuple[str, ...] = ("agent", "strategy")
RUN_STATUSES: tuple[str, ...] = ("queued", "running", "succeeded", "failed", "cancelled")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class AgentSkill(SQLModel, table=True):
    __tablename__ = "agent_skill"
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_agent_skill_user_name"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(sa_column=Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True))
    name: str = Field(sa_column=Column(String(64), nullable=False))
    description: str = Field(default="", sa_column=Column(String(1024), nullable=False, server_default=""))
    body: str = Field(sa_column=Column(Text, nullable=False))
    version: int = Field(default=1, sa_column=Column(Integer, nullable=False, server_default="1"))
    origin: str = Field(default="user", sa_column=Column(String(16), nullable=False, server_default="user"))
    source_strategy_id: Optional[int] = Field(
        default=None, sa_column=Column(Integer, ForeignKey("strategies.id", ondelete="SET NULL"), nullable=True)
    )
    source_backtest_id: Optional[int] = Field(
        default=None, sa_column=Column(Integer, ForeignKey("strategy_backtests.id", ondelete="SET NULL"), nullable=True)
    )
    created_at: datetime = Field(default_factory=_now, sa_column=Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()")))
    updated_at: datetime = Field(default_factory=_now, sa_column=Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()")))


class AgentMemory(SQLModel, table=True):
    __tablename__ = "agent_memory"
    __table_args__ = (UniqueConstraint("agent_id", "kind", name="uq_agent_memory_agent_kind"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    agent_id: int = Field(sa_column=Column(Integer, ForeignKey("agent.id_agent", ondelete="CASCADE"), nullable=False, index=True))
    kind: str = Field(sa_column=Column(String(32), nullable=False))
    content: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    updated_at: datetime = Field(default_factory=_now, sa_column=Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()")))
    updated_by: str = Field(default="user", sa_column=Column(String(16), nullable=False, server_default="user"))


class AgentActionRequest(SQLModel, table=True):
    __tablename__ = "agent_action_request"

    id: Optional[int] = Field(default=None, primary_key=True)
    agent_id: int = Field(sa_column=Column(Integer, ForeignKey("agent.id_agent", ondelete="CASCADE"), nullable=False, index=True))
    user_id: int = Field(sa_column=Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True))
    chat_id: Optional[int] = Field(default=None, sa_column=Column(Integer, ForeignKey("chat.id", ondelete="SET NULL"), nullable=True))
    strategy_live_id: Optional[int] = Field(
        default=None, sa_column=Column(Integer, ForeignKey("strategy_live.id", ondelete="SET NULL"), nullable=True)
    )
    account_id: Optional[int] = Field(
        default=None, sa_column=Column(Integer, ForeignKey("accounts.id", ondelete="SET NULL"), nullable=True)
    )
    tool_name: str = Field(sa_column=Column(String(64), nullable=False))
    args: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False, server_default="{}"))
    rationale: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    status: str = Field(default="pending", sa_column=Column(String(16), nullable=False, server_default="pending"))
    expires_at: datetime = Field(sa_column=Column(DateTime(timezone=True), nullable=False))
    decided_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True), nullable=True))
    decided_by: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSONB, nullable=True))
    result: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSONB, nullable=True))
    error: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    chat_row_id: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))
    created_at: datetime = Field(default_factory=_now, sa_column=Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()")))


class AgentRun(SQLModel, table=True):
    """A task handed to a hosted agent from outside (migr. 069, bridge F4):
    REST ``POST /agents/{id}/runs``, the Paperclip http adapter, A2A
    ``message/send``. Executed as one turn in a chat of the agent."""

    __tablename__ = "agent_run"

    id: Optional[int] = Field(default=None, primary_key=True)
    agent_id: int = Field(sa_column=Column(Integer, ForeignKey("agent.id_agent", ondelete="CASCADE"), nullable=False, index=True))
    user_id: int = Field(sa_column=Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True))
    source: str = Field(default="api", sa_column=Column(String(16), nullable=False, server_default="api"))
    external_run_id: Optional[str] = Field(default=None, sa_column=Column(String(128), nullable=True))
    scope: str = Field(default="agent", sa_column=Column(String(16), nullable=False, server_default="agent"))
    strategy_id: Optional[int] = Field(default=None, sa_column=Column(Integer, ForeignKey("strategies.id", ondelete="SET NULL"), nullable=True))
    chat_id: Optional[int] = Field(default=None, sa_column=Column(Integer, ForeignKey("chat.id", ondelete="SET NULL"), nullable=True))
    task: str = Field(sa_column=Column(Text, nullable=False))
    context: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False, server_default="{}"))
    status: str = Field(default="queued", sa_column=Column(String(16), nullable=False, server_default="queued"))
    request_id: Optional[str] = Field(default=None, sa_column=Column(String(64), nullable=True))
    result: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    error: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    usage: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSONB, nullable=True))
    cost_credits: Optional[float] = Field(default=None, sa_column=Column(Numeric(12, 4), nullable=True))
    callback_url: Optional[str] = Field(default=None, sa_column=Column(String(1024), nullable=True))
    callback_auth: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    callback_status: Optional[str] = Field(default=None, sa_column=Column(String(16), nullable=True))
    callback_error: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    created_by: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSONB, nullable=True))
    created_at: datetime = Field(default_factory=_now, sa_column=Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()")))
    started_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True), nullable=True))
    finished_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True), nullable=True))
