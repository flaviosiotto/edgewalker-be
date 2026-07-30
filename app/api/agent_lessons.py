"""Agent lessons: the manager agent's persistent, auditable playbook.

Mounted without a router-level auth dependency so the agent's consultative
token (the same one used for /accounts/*) can read and write lessons.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlmodel import Session

from app.db.database import get_session
from app.models.user import User
from app.schemas.agent_lesson import AgentLessonCreate, AgentLessonRead, AgentLessonUpdate
from app.services.agent_lesson_service import (
    ab_evaluate,
    create_lesson,
    list_lessons,
    update_lesson,
)
from app.utils.auth_utils import get_current_active_or_consultative_user

router = APIRouter(tags=["agent-lessons"])


@router.get("/strategies/{strategy_id}/agent-lessons", response_model=list[AgentLessonRead])
def list_agent_lessons_endpoint(
    strategy_id: int,
    status: str | None = Query(default="active", pattern="^(active|retired|all)$"),
    min_confidence: float | None = Query(default=None, ge=0.0, le=1.0),
    limit: int = Query(default=20, ge=1, le=100),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    status_filter = None if status == "all" else status
    lessons = list_lessons(
        session,
        strategy_id,
        current_user.id,
        status_filter=status_filter,
        min_confidence=min_confidence,
        limit=limit,
    )
    return [AgentLessonRead.model_validate(item) for item in lessons]


class AbEvaluateRequest(BaseModel):
    baseline_backtest_id: int
    lessons_backtest_id: int
    apply: bool = True


@router.post("/strategies/{strategy_id}/agent-lessons/ab-evaluate")
def ab_evaluate_endpoint(
    strategy_id: int,
    payload: AbEvaluateRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Compare a lessons-enabled backtest vs a baseline and adjust confidences.

    Anti-drift: lessons active in the lessons leg gain confidence when the
    run beats the baseline, lose it otherwise, and are retired below the
    threshold. The full evaluation is appended to each lesson's evidence.
    """
    return ab_evaluate(
        session,
        strategy_id,
        current_user.id,
        baseline_backtest_id=payload.baseline_backtest_id,
        lessons_backtest_id=payload.lessons_backtest_id,
        apply=payload.apply,
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
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    return AgentLessonRead.model_validate(
        create_lesson(session, strategy_id, current_user.id, payload)
    )


@router.patch("/agent-lessons/{lesson_id}", response_model=AgentLessonRead)
def update_agent_lesson_endpoint(
    lesson_id: int,
    payload: AgentLessonUpdate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    return AgentLessonRead.model_validate(update_lesson(session, lesson_id, current_user.id, payload))
