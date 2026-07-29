from __future__ import annotations

from datetime import datetime, timezone

from fastapi import HTTPException, status
from sqlmodel import Session, select

from app.models.agent_lesson import AgentLesson
from app.schemas.agent_lesson import AgentLessonCreate, AgentLessonUpdate
from app.services.strategy_service import get_strategy


def list_lessons(
    session: Session,
    strategy_id: int,
    user_id: int,
    *,
    status_filter: str | None = "active",
    limit: int = 20,
) -> list[AgentLesson]:
    get_strategy(session, strategy_id, user_id)
    stmt = select(AgentLesson).where(AgentLesson.strategy_id == strategy_id)
    if status_filter:
        stmt = stmt.where(AgentLesson.status == status_filter)
    stmt = stmt.order_by(AgentLesson.confidence.desc(), AgentLesson.id.desc()).limit(limit)
    return list(session.exec(stmt).all())


def create_lesson(
    session: Session,
    strategy_id: int,
    user_id: int,
    payload: AgentLessonCreate,
) -> AgentLesson:
    strategy = get_strategy(session, strategy_id, user_id)
    lesson = AgentLesson(
        strategy_id=strategy_id,
        user_id=strategy.user_id,
        lesson=payload.lesson.strip(),
        context=(payload.context or None),
        confidence=payload.confidence,
        source=payload.source,
        backtest_id=payload.backtest_id,
        evidence=payload.evidence,
    )
    session.add(lesson)
    session.commit()
    session.refresh(lesson)
    return lesson


def update_lesson(
    session: Session,
    lesson_id: int,
    user_id: int,
    payload: AgentLessonUpdate,
) -> AgentLesson:
    lesson = session.get(AgentLesson, lesson_id)
    if lesson is None or lesson.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Lesson {lesson_id} not found",
        )
    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        setattr(lesson, key, value)
    lesson.updated_at = datetime.now(timezone.utc)
    session.add(lesson)
    session.commit()
    session.refresh(lesson)
    return lesson
