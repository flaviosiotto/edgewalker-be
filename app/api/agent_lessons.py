"""Agent lessons and playbooks (docs/valutazione-playbook-lezioni.md).

Mounted without a router-level auth dependency so the agent's consultative
token (the same one used for /accounts/*) can read and write lessons. The
principal's purpose tells the agent apart from the trader: the agent may
write only into the playbook of a run that is (just) alive.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlmodel import Session

from app.db.database import get_session
from app.schemas.agent_lesson import (
    AgentLessonCreate,
    AgentLessonRead,
    AgentLessonUpdate,
    PlaybookRead,
    PlaybookSelect,
    PlaybookSummary,
)
from app.services import playbook_service
from app.services.agent_lesson_service import (
    ab_evaluate,
    create_lesson,
    list_lessons,
    update_lesson,
)
from app.services.strategy_service import get_strategy
from app.utils.auth_utils import AuthPrincipal, get_current_active_or_consultative_principal

router = APIRouter(tags=["agent-lessons"])


def _origin(principal: AuthPrincipal) -> str:
    # FE tokens carry purpose=ui_auth; everything else (consultative token of
    # the agent, delegated tokens) is the agent.
    return "user" if principal.claims.get("purpose") == "ui_auth" else "agent"


@router.get("/strategies/{strategy_id}/agent-lessons", response_model=list[AgentLessonRead])
def list_agent_lessons_endpoint(
    strategy_id: int,
    backtest_id: int | None = Query(default=None, description="Playbook of this run"),
    live_id: int | None = Query(default=None, description="Playbook attached to this live session"),
    status: str | None = Query(default="active", pattern="^(active|retired|all)$"),
    limit: int = Query(default=20, ge=1, le=100),
    session: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_active_or_consultative_principal),
):
    status_filter = None if status == "all" else status
    lessons = list_lessons(
        session,
        strategy_id,
        principal.user.id,
        backtest_id=backtest_id,
        live_id=live_id,
        status_filter=status_filter,
        limit=limit,
    )
    return [AgentLessonRead.model_validate(item) for item in lessons]


class AbEvaluateRequest(BaseModel):
    baseline_backtest_id: int
    lessons_backtest_id: int
    # Kept for older clients: nothing is adjusted any more.
    apply: bool = False


@router.post("/strategies/{strategy_id}/agent-lessons/ab-evaluate")
def ab_evaluate_endpoint(
    strategy_id: int,
    payload: AbEvaluateRequest,
    session: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_active_or_consultative_principal),
):
    """Compare a run launched with a playbook against a baseline launched
    without (same parameters): evidence for promoting the playbook."""
    return ab_evaluate(
        session,
        strategy_id,
        principal.user.id,
        baseline_backtest_id=payload.baseline_backtest_id,
        lessons_backtest_id=payload.lessons_backtest_id,
    )


@router.post(
    "/strategies/{strategy_id}/agent-lessons",
    response_model=AgentLessonRead,
    status_code=201,
)
def create_agent_lesson_endpoint(
    strategy_id: int,
    payload: AgentLessonCreate,
    session: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_active_or_consultative_principal),
):
    return AgentLessonRead.model_validate(
        create_lesson(session, strategy_id, principal.user.id, payload, origin=_origin(principal))
    )


@router.patch("/agent-lessons/{lesson_id}", response_model=AgentLessonRead)
def update_agent_lesson_endpoint(
    lesson_id: int,
    payload: AgentLessonUpdate,
    session: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_active_or_consultative_principal),
):
    return AgentLessonRead.model_validate(
        update_lesson(session, lesson_id, principal.user.id, payload, origin=_origin(principal))
    )


# ── playbooks ─────────────────────────────────────────────────────────────


@router.get("/strategies/{strategy_id}/playbooks", response_model=list[PlaybookSummary])
def list_playbooks_endpoint(
    strategy_id: int,
    session: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_active_or_consultative_principal),
):
    """Every selectable playbook: completed runs with lessons (newest first)
    and, last, the strategy's initial rows when they exist."""
    strategy = get_strategy(session, strategy_id, principal.user.id)
    return playbook_service.list_playbooks(session, strategy)


@router.get("/strategies/{strategy_id}/playbooks/initial", response_model=PlaybookRead)
def get_initial_playbook_endpoint(
    strategy_id: int,
    session: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_active_or_consultative_principal),
):
    strategy = get_strategy(session, strategy_id, principal.user.id)
    return playbook_service.playbook_detail(session, strategy, None)


@router.get("/strategies/{strategy_id}/playbooks/{backtest_id}", response_model=PlaybookRead)
def get_playbook_endpoint(
    strategy_id: int,
    backtest_id: int,
    session: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_active_or_consultative_principal),
):
    """The playbook of a run, each row annotated against its input row
    (new / changed / retired / unchanged)."""
    strategy = get_strategy(session, strategy_id, principal.user.id)
    return playbook_service.playbook_detail(session, strategy, backtest_id)


@router.put("/strategies/{strategy_id}/playbook", response_model=list[PlaybookSummary])
def set_playbook_endpoint(
    strategy_id: int,
    payload: PlaybookSelect,
    session: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_active_or_consultative_principal),
):
    """Promotion: the output of ``backtest_id`` becomes the strategy's current
    playbook (null = back to the initial rows, or none)."""
    strategy = get_strategy(session, strategy_id, principal.user.id)
    strategy = playbook_service.set_current_playbook(session, strategy, payload.backtest_id)
    return playbook_service.list_playbooks(session, strategy)
