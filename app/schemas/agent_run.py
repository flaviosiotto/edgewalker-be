"""Schemas of the run endpoint (bridge F4): EdgeWalker as an agent."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.agent_model import SKILL_BODY_MAX_CHARS

RunScope = Literal["agent", "strategy"]
RunSource = Literal["api", "paperclip", "a2a", "mcp"]
RunStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]

TASK_MAX_CHARS = 20_000


class RunCreate(BaseModel):
    task: str = Field(min_length=1, max_length=TASK_MAX_CHARS)
    #: agent = the agent's own chat (briefing, creation, comparison);
    #: strategy = a design chat of ``strategy_id`` (the agent may edit it).
    scope: RunScope = "agent"
    strategy_id: Optional[int] = None
    #: free-form context echoed to the agent after the task (ids, links…)
    context: dict[str, Any] = Field(default_factory=dict)
    #: caller's id of this run (Paperclip runId, A2A task id, your own)
    external_run_id: Optional[str] = Field(default=None, max_length=128)
    #: when set, the outcome is POSTed there (see RunCallbackPayload)
    callback_url: Optional[str] = Field(default=None, max_length=1024)
    #: Authorization header value for the callback, stored encrypted
    callback_auth: Optional[str] = Field(default=None, max_length=2048)
    source: RunSource = "api"


class RunRead(BaseModel):
    id: int
    agent_id: int
    agent_name: Optional[str] = None
    source: RunSource
    external_run_id: Optional[str] = None
    scope: RunScope
    strategy_id: Optional[int] = None
    chat_id: Optional[int] = None
    task: str
    context: dict[str, Any]
    status: RunStatus
    result: Optional[str] = None
    error: Optional[str] = None
    usage: Optional[dict[str, Any]] = None
    cost_credits: Optional[float] = None
    callback_url: Optional[str] = None
    callback_status: Optional[str] = None
    callback_error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class RunCallbackPayload(BaseModel):
    """What EdgeWalker POSTs to ``callback_url`` (source api / mcp)."""

    run_id: int
    external_run_id: Optional[str] = None
    agent_id: int
    agent_name: Optional[str] = None
    status: RunStatus
    result: Optional[str] = None
    error: Optional[str] = None
    usage: Optional[dict[str, Any]] = None
    cost_credits: Optional[float] = None
    finished_at: Optional[datetime] = None


PaperclipDoneStatus = Literal["done", "in_review", "comment"]


class RunSkill(BaseModel):
    """A skill shipped with one run (Paperclip adapter: the company skills
    enabled for the agent, read from their SKILL.md at run time). Ephemeral:
    lives in ``run.context["skills"]`` and is visible to that run's turn
    only (``GET /skills?run_id=``), never copied into ``agent_skill``."""

    name: str = Field(min_length=1, max_length=64)
    description: str = Field(default="", max_length=1024)
    body: str = Field(min_length=1, max_length=SKILL_BODY_MAX_CHARS)
    version: Optional[str] = Field(default=None, max_length=64)

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not cleaned or any(ch.isspace() for ch in cleaned):
            raise ValueError("skill name must be a single lowercase token")
        return cleaned


RUN_SKILLS_MAX = 20


class PaperclipHeartbeat(BaseModel):
    """The body sent by the Paperclip ``http`` adapter
    (``server/src/adapters/http/execute.ts``): ``{...payloadTemplate, agentId,
    runId, context, connectionInstructions}``. The run context (``taskId``,
    ``issueId``, ``wakeReason`` such as ``issue_assigned`` / ``issue_commented``,
    ``commentId``…) lives inside ``context``; the same keys are accepted at the
    root too. Keys of ``payloadTemplate`` land at the root: ``task`` /
    ``instructions`` give the text, ``scope`` / ``strategy_id`` the EdgeWalker
    scope, ``paperclipApiUrl`` / ``paperclipApiKey`` where to report the
    outcome (comment + status on the issue) and ``paperclipDoneStatus`` which
    status to set when the run succeeds (``done`` default, ``in_review``, or
    ``comment`` to only comment)."""

    runId: str
    agentId: Optional[str] = None
    companyId: Optional[str] = None
    taskId: Optional[str] = None
    issueId: Optional[str] = None
    wakeReason: Optional[str] = None
    commentId: Optional[str] = None
    wakeCommentId: Optional[str] = None
    approvalId: Optional[str] = None
    approvalStatus: Optional[str] = None
    issueIds: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    task: Optional[str] = None
    instructions: Optional[str] = None
    scope: Optional[RunScope] = None
    strategy_id: Optional[int] = None
    paperclipApiUrl: Optional[str] = None
    paperclipApiKey: Optional[str] = None
    paperclipDoneStatus: Optional[PaperclipDoneStatus] = None
    #: skills enabled for the agent in Paperclip, sent with every run (≤ 20)
    skills: list[RunSkill] = Field(default_factory=list, max_length=RUN_SKILLS_MAX)

    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def _lift_context(self) -> "PaperclipHeartbeat":
        """Paperclip nests the run context under ``context``: mirror the
        well-known keys at the root when they are not already there."""
        ctx = self.context or {}
        for key in ("agentId", "companyId", "taskId", "issueId", "wakeReason", "commentId", "wakeCommentId", "approvalId", "approvalStatus"):
            if getattr(self, key) in (None, "") and isinstance(ctx.get(key), str) and ctx.get(key):
                setattr(self, key, ctx[key])
        if not self.issueIds and isinstance(ctx.get("issueIds"), list):
            self.issueIds = [i for i in ctx["issueIds"] if isinstance(i, str)]
        if not self.commentId and self.wakeCommentId:
            self.commentId = self.wakeCommentId
        return self

    @property
    def issue_ref(self) -> Optional[str]:
        """The issue this heartbeat is about (Paperclip calls it task)."""
        return self.taskId or self.issueId or None
