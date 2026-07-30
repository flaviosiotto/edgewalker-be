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
    min_confidence: float | None = None,
    limit: int = 20,
) -> list[AgentLesson]:
    get_strategy(session, strategy_id, user_id)
    stmt = select(AgentLesson).where(AgentLesson.strategy_id == strategy_id)
    if status_filter:
        stmt = stmt.where(AgentLesson.status == status_filter)
    if min_confidence is not None and min_confidence > 0:
        stmt = stmt.where(AgentLesson.confidence >= min_confidence)
    stmt = stmt.order_by(AgentLesson.confidence.desc(), AgentLesson.id.desc()).limit(limit)
    return list(session.exec(stmt).all())


# ── A/B evaluation ────────────────────────────────────────────────────────
# Compare a lessons-enabled backtest against a baseline (lessons disabled)
# run with the same parameters, and adjust the confidence of exactly the
# lessons that were active in the lessons leg. Coarse but honest anti-drift:
# per-lesson attribution can refine it later.

AB_CONFIDENCE_UP = 0.10
AB_CONFIDENCE_DOWN = 0.20
AB_RETIRE_BELOW = 0.20


def ab_evaluate(
    session: Session,
    strategy_id: int,
    user_id: int,
    *,
    baseline_backtest_id: int,
    lessons_backtest_id: int,
    apply: bool = True,
) -> dict:
    from app.services.strategy_service import get_backtest

    baseline = get_backtest(session, baseline_backtest_id, user_id)
    lessons_run = get_backtest(session, lessons_backtest_id, user_id)
    for run, label in ((baseline, "baseline"), (lessons_run, "lessons")):
        if run.strategy_id != strategy_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Backtest {run.id} ({label}) does not belong to strategy {strategy_id}",
            )
        if run.status != "completed":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Backtest {run.id} ({label}) is not completed (status={run.status})",
            )

    lessons_cfg = (
        lessons_run.parameters.get("lessons")
        if isinstance(lessons_run.parameters, dict)
        else None
    ) or {}
    active_ids = [int(i) for i in (lessons_cfg.get("active_ids") or [])]

    def _metric(run, name):
        value = getattr(run, name, None)
        return float(value) if value is not None else None

    deltas: dict[str, float | None] = {}
    for name in ("return_pct", "profit_factor", "win_rate_pct", "max_drawdown_pct", "total_trades"):
        b, l = _metric(baseline, name), _metric(lessons_run, name)
        deltas[name] = (l - b) if (b is not None and l is not None) else None

    # Verdict: primary = return; tie-break = profit factor. A missing metric
    # on either side yields no verdict (and no adjustment).
    verdict: str | None = None
    if deltas["return_pct"] is not None and deltas["return_pct"] != 0:
        verdict = "better" if deltas["return_pct"] > 0 else "worse"
    elif deltas["profit_factor"] is not None and deltas["profit_factor"] != 0:
        verdict = "better" if deltas["profit_factor"] > 0 else "worse"

    adjusted: list[dict] = []
    if apply and verdict is not None and active_ids:
        now = datetime.now(timezone.utc)
        for lesson_id in active_ids:
            lesson = session.get(AgentLesson, lesson_id)
            if lesson is None or lesson.user_id != user_id or lesson.status != "active":
                continue
            old_confidence = lesson.confidence
            if verdict == "better":
                lesson.confidence = min(1.0, lesson.confidence + AB_CONFIDENCE_UP)
            else:
                lesson.confidence = max(0.0, lesson.confidence - AB_CONFIDENCE_DOWN)
            new_status = lesson.status
            if lesson.confidence < AB_RETIRE_BELOW:
                new_status = "retired"
                lesson.status = new_status
            evidence = dict(lesson.evidence or {})
            ab_history = list(evidence.get("ab_evaluations") or [])
            ab_history.append({
                "baseline_backtest_id": baseline.id,
                "lessons_backtest_id": lessons_run.id,
                "verdict": verdict,
                "delta_return_pct": deltas["return_pct"],
                "delta_profit_factor": deltas["profit_factor"],
                "confidence_before": old_confidence,
                "confidence_after": lesson.confidence,
                "evaluated_at": now.isoformat(),
            })
            evidence["ab_evaluations"] = ab_history
            lesson.evidence = evidence
            lesson.updated_at = now
            session.add(lesson)
            adjusted.append({
                "id": lesson.id,
                "confidence_before": old_confidence,
                "confidence_after": lesson.confidence,
                "status": lesson.status,
            })
        session.commit()

    return {
        "strategy_id": strategy_id,
        "baseline_backtest_id": baseline.id,
        "lessons_backtest_id": lessons_run.id,
        "verdict": verdict,
        "deltas": deltas,
        "lessons_evaluated": active_ids,
        "applied": apply and verdict is not None,
        "adjusted": adjusted,
    }


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
