"""Admin console: browse the agent turns recorded by agent-svc.

What n8n's execution view used to offer — the prompt the model saw, the
chart images, every tool call with its result, tokens and timing — read from
``agent_turn`` / ``agent_turn_image`` (migration 057). Admin-only.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel
from sqlmodel import Session, select

from app.db.database import get_session
from app.models.agent_turn import AgentTurn, AgentTurnImage
from app.models.user import User
from app.utils.auth_utils import get_current_admin_user

router = APIRouter(
    prefix="/admin/agent-turns",
    tags=["Administration"],
    dependencies=[Depends(get_current_admin_user)],
)

PREVIEW_CHARS = 160


class AgentTurnImageRead(BaseModel):
    id: int
    name: str
    mime: str
    bytes: int


class AgentTurnRow(BaseModel):
    turn_id: str
    session_id: str
    user_id: Optional[int] = None
    user_email: Optional[str] = None
    agent_id: Optional[int] = None
    strategy_id: Optional[int] = None
    strategy_live_id: Optional[int] = None
    backtest_id: Optional[int] = None
    kind: Optional[str] = None
    trigger_type: Optional[str] = None
    correlation_id: Optional[str] = None
    status: str
    error: Optional[str] = None
    started_at: datetime
    duration_ms: Optional[int] = None
    model: Optional[str] = None
    requests: Optional[int] = None
    tool_calls: Optional[int] = None
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    tokens_reasoning: Optional[int] = None
    tokens_cached: Optional[int] = None
    user_message_preview: Optional[str] = None
    images: int = 0


class AgentTurnDetail(AgentTurnRow):
    prompt_chars: Optional[int] = None
    response_chars: Optional[int] = None
    user_message: Optional[str] = None
    system_prompt: Optional[str] = None
    response: Optional[str] = None
    steps: list[Any] = []
    context: dict[str, Any] = {}
    image_list: list[AgentTurnImageRead] = []


class AgentTurnPage(BaseModel):
    items: list[AgentTurnRow]
    next_before: Optional[datetime] = None


def _preview(text: Optional[str]) -> Optional[str]:
    if not text:
        return text
    flat = " ".join(text.split())
    return flat if len(flat) <= PREVIEW_CHARS else flat[:PREVIEW_CHARS] + "…"


def _row(turn: AgentTurn, *, email: Optional[str], images: int) -> dict[str, Any]:
    return {
        **{k: getattr(turn, k) for k in AgentTurnRow.model_fields if hasattr(turn, k)},
        "user_email": email,
        "user_message_preview": _preview(turn.user_message),
        "images": images,
    }


@router.get("", response_model=AgentTurnPage)
def list_agent_turns(
    session: Session = Depends(get_session),
    user_id: Optional[int] = Query(default=None),
    session_id: Optional[str] = Query(default=None, max_length=255),
    turn_status: Optional[str] = Query(default=None, alias="status", pattern="^(ok|failed)$"),
    before: Optional[datetime] = Query(default=None, description="Pagina precedente: started_at < before"),
    limit: int = Query(default=50, ge=1, le=200),
):
    stmt = select(AgentTurn)
    if user_id is not None:
        stmt = stmt.where(AgentTurn.user_id == user_id)
    if session_id:
        stmt = stmt.where(AgentTurn.session_id == session_id)
    if turn_status:
        stmt = stmt.where(AgentTurn.status == turn_status)
    if before is not None:
        stmt = stmt.where(AgentTurn.started_at < before)
    turns = session.exec(stmt.order_by(AgentTurn.started_at.desc()).limit(limit)).all()

    user_ids = {t.user_id for t in turns if t.user_id is not None}
    emails: dict[int, str] = {}
    if user_ids:
        for user in session.exec(select(User).where(User.id.in_(user_ids))).all():
            emails[user.id] = user.email
    counts: dict[str, int] = {}
    if turns:
        for image in session.exec(
            select(AgentTurnImage.turn_id).where(AgentTurnImage.turn_id.in_([t.turn_id for t in turns]))
        ).all():
            counts[image] = counts.get(image, 0) + 1

    items = [AgentTurnRow(**_row(t, email=emails.get(t.user_id or -1), images=counts.get(t.turn_id, 0))) for t in turns]
    next_before = turns[-1].started_at if len(turns) == limit else None
    return AgentTurnPage(items=items, next_before=next_before)


def _get_turn_or_404(session: Session, turn_id: str) -> AgentTurn:
    turn = session.get(AgentTurn, turn_id)
    if turn is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Turno non trovato")
    return turn


@router.get("/{turn_id}", response_model=AgentTurnDetail)
def read_agent_turn(turn_id: str, session: Session = Depends(get_session)):
    turn = _get_turn_or_404(session, turn_id)
    images = session.exec(
        select(AgentTurnImage.id, AgentTurnImage.name, AgentTurnImage.mime, AgentTurnImage.size)
        .where(AgentTurnImage.turn_id == turn_id)
        .order_by(AgentTurnImage.id)
    ).all()
    email = None
    if turn.user_id is not None:
        user = session.get(User, turn.user_id)
        email = user.email if user else None
    return AgentTurnDetail(
        **_row(turn, email=email, images=len(images)),
        prompt_chars=turn.prompt_chars,
        response_chars=turn.response_chars,
        user_message=turn.user_message,
        system_prompt=turn.system_prompt,
        response=turn.response,
        steps=turn.steps or [],
        context=turn.context or {},
        image_list=[AgentTurnImageRead(id=i, name=n, mime=m, bytes=b) for i, n, m, b in images],
    )


@router.get("/{turn_id}/images/{image_id}")
def read_agent_turn_image(turn_id: str, image_id: int, session: Session = Depends(get_session)):
    image = session.get(AgentTurnImage, image_id)
    if image is None or image.turn_id != turn_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Immagine non trovata")
    return Response(
        content=bytes(image.data),
        media_type=image.mime or "image/png",
        headers={"Cache-Control": "private, max-age=3600", "Content-Disposition": f'inline; filename="{image.name}"'},
    )
