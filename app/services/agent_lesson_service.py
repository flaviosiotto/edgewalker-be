from __future__ import annotations

from datetime import datetime, timezone

from fastapi import HTTPException, status
from sqlmodel import Session, select

from app.models.agent_lesson import AgentLesson
from app.models.strategy import BacktestResult, StrategyLive
from app.schemas.agent_lesson import AgentLessonCreate, AgentLessonUpdate
from app.services import playbook_service
from app.services.strategy_service import get_strategy


def list_lessons(
    session: Session,
    strategy_id: int,
    user_id: int,
    *,
    backtest_id: int | None = None,
    live_id: int | None = None,
    status_filter: str | None = "active",
    limit: int = 20,
) -> list[AgentLesson]:
    """The playbook of a run (``backtest_id``), of a live session
    (``live_id``: what it attached at launch) or, by default, the strategy's
    current one. Ordered by confidence."""
    strategy = get_strategy(session, strategy_id, user_id)
    if backtest_id is not None:
        rows = playbook_service.run_rows(session, backtest_id, status_filter=status_filter)
    elif live_id is not None:
        live = session.get(StrategyLive, live_id)
        if live is None or live.strategy_id != strategy.id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Live session {live_id} not found")
        rows = playbook_service.live_rows(session, live, status_filter=status_filter)
    else:
        rows = playbook_service.current_rows(session, strategy, status_filter=status_filter)
    return rows[:limit]


# ── A/B comparison ─────────────────────────────────────────────────────────
# Two completed runs on the same parameters: the deltas are evidence for (or
# against) promoting the lessons leg's playbook. Nothing is adjusted
# automatically: confidences change only inside a run, under the agent-svc
# guard rails, or by hand.


def ab_evaluate(
    session: Session,
    strategy_id: int,
    user_id: int,
    *,
    baseline_backtest_id: int,
    lessons_backtest_id: int,
    apply: bool = False,
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

    def _metric(run, name):
        value = getattr(run, name, None)
        return float(value) if value is not None else None

    deltas: dict[str, float | None] = {}
    for name in ("return_pct", "profit_factor", "win_rate_pct", "max_drawdown_pct", "total_trades"):
        b, l = _metric(baseline, name), _metric(lessons_run, name)
        deltas[name] = (l - b) if (b is not None and l is not None) else None

    # Verdict: primary = return; tie-break = profit factor.
    verdict: str | None = None
    if deltas["return_pct"] is not None and deltas["return_pct"] != 0:
        verdict = "better" if deltas["return_pct"] > 0 else "worse"
    elif deltas["profit_factor"] is not None and deltas["profit_factor"] != 0:
        verdict = "better" if deltas["profit_factor"] > 0 else "worse"

    return {
        "strategy_id": strategy_id,
        "baseline_backtest_id": baseline.id,
        "lessons_backtest_id": lessons_run.id,
        "verdict": verdict,
        "deltas": deltas,
        "lessons_evaluated": [r.id for r in playbook_service.run_rows(session, lessons_run.id)],
        "applied": False,
        "adjusted": [],
    }


def create_lesson(
    session: Session,
    strategy_id: int,
    user_id: int,
    payload: AgentLessonCreate,
    *,
    origin: str = "user",
) -> AgentLesson:
    """A new row in the playbook of ``payload.backtest_id`` (the agent's path:
    the run must still be alive) or, without a run, in the strategy's initial
    playbook (manual rows only)."""
    strategy = get_strategy(session, strategy_id, user_id)
    if payload.backtest_id is not None:
        backtest = session.get(BacktestResult, payload.backtest_id)
        if backtest is None or backtest.strategy_id != strategy.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Backtest {payload.backtest_id} not found on strategy {strategy_id}",
            )
        if origin == "agent":
            playbook_service.assert_agent_can_write(session, backtest.id)
        scope, run_backtest_id = playbook_service.SCOPE_BACKTEST, backtest.id
    else:
        if origin == "agent":
            playbook_service.assert_agent_can_write(session, None)
        scope, run_backtest_id = playbook_service.SCOPE_STRATEGY, None
    lesson = AgentLesson(
        strategy_id=strategy_id,
        user_id=strategy.user_id,
        lesson=payload.lesson.strip(),
        context=(payload.context or None),
        confidence=payload.confidence,
        source=payload.source,
        backtest_id=payload.backtest_id,
        evidence=payload.evidence,
        scope=scope,
        run_backtest_id=run_backtest_id,
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
    *,
    origin: str = "user",
) -> AgentLesson:
    lesson = session.get(AgentLesson, lesson_id)
    if lesson is None or lesson.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Lesson {lesson_id} not found",
        )
    if origin == "agent":
        playbook_service.assert_agent_can_write(session, lesson.run_backtest_id)
    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        setattr(lesson, key, value)
    lesson.updated_at = datetime.now(timezone.utc)
    session.add(lesson)
    session.commit()
    session.refresh(lesson)
    return lesson
