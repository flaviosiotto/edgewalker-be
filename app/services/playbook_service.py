"""Playbooks: the lessons of one run (docs/valutazione-playbook-lezioni.md).

A backtest starts from the playbook chosen at launch — nothing, the
strategy's current playbook, or the output of another backtest — copied into
its own rows (``scope=backtest``, ``run_backtest_id`` = the run, ``parent_id``
= the input row). The agent edits only those rows; when the run ends the
output freezes. Live attaches to the output of one backtest and never
writes. The strategy only points at the backtest whose output is its current
playbook (``strategies.playbook_backtest_id``); rows with ``scope=strategy``
are its initial playbook (pre-062 rows), used until a run is promoted.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import HTTPException, status
from sqlmodel import Session, select

from app.models.agent_lesson import AgentLesson
from app.models.strategy import BacktestResult, BacktestStatus, LiveStatus, Strategy, StrategyLive
from app.schemas.agent_lesson import (
    AgentLessonRead,
    LessonChange,
    PlaybookRead,
    PlaybookSummary,
)

# After a run ends the agent still gets its final analysis turn: its writes
# are accepted for this long after completed_at, then the playbook is frozen
# for the agent (the trader can always edit from the Lessons tab).
AGENT_WRITE_GRACE = timedelta(hours=2)

SCOPE_STRATEGY = "strategy"
SCOPE_BACKTEST = "backtest"


# ── rows ──────────────────────────────────────────────────────────────────


def _ordered(stmt):
    return stmt.order_by(AgentLesson.confidence.desc(), AgentLesson.id.desc())


def initial_rows(session: Session, strategy_id: int, *, status_filter: str | None = "active") -> list[AgentLesson]:
    """The strategy's initial playbook (``scope=strategy``)."""
    stmt = select(AgentLesson).where(
        AgentLesson.strategy_id == strategy_id, AgentLesson.scope == SCOPE_STRATEGY
    )
    if status_filter:
        stmt = stmt.where(AgentLesson.status == status_filter)
    return list(session.exec(_ordered(stmt)).all())


def run_rows(session: Session, backtest_id: int, *, status_filter: str | None = "active") -> list[AgentLesson]:
    """The playbook of one run (input copies + rows born in the run)."""
    stmt = select(AgentLesson).where(AgentLesson.run_backtest_id == backtest_id)
    if status_filter:
        stmt = stmt.where(AgentLesson.status == status_filter)
    return list(session.exec(_ordered(stmt)).all())


def current_rows(session: Session, strategy: Strategy, *, status_filter: str | None = "active") -> list[AgentLesson]:
    """The strategy's current playbook: the promoted run's output, else the initial rows."""
    if strategy.playbook_backtest_id is not None:
        return run_rows(session, strategy.playbook_backtest_id, status_filter=status_filter)
    return initial_rows(session, strategy.id, status_filter=status_filter)


def live_rows(session: Session, live: StrategyLive, *, status_filter: str | None = "active") -> list[AgentLesson]:
    """The playbook attached to a live session (none when launched without)."""
    if live.playbook_backtest_id is None:
        return []
    return run_rows(session, live.playbook_backtest_id, status_filter=status_filter)


# ── launch ────────────────────────────────────────────────────────────────


def _owned_completed_backtest(session: Session, strategy: Strategy, backtest_id: int) -> BacktestResult:
    backtest = session.get(BacktestResult, backtest_id)
    if backtest is None or backtest.strategy_id != strategy.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Backtest {backtest_id} not found on strategy {strategy.id}",
        )
    if backtest.status != BacktestStatus.COMPLETED.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Backtest {backtest_id} is not completed (status={backtest.status}): its playbook is not final",
        )
    return backtest


def resolve_input(
    session: Session, strategy: Strategy, lessons_from: str | int | None
) -> tuple[dict[str, Any], list[AgentLesson]]:
    """Rows a new run starts from + the descriptor stored in parameters.lessons."""
    if lessons_from in (None, "strategy"):
        rows = current_rows(session, strategy)
        return {"source": "strategy", "source_backtest_id": strategy.playbook_backtest_id}, rows
    if lessons_from == "none":
        return {"source": "none", "source_backtest_id": None}, []
    backtest = _owned_completed_backtest(session, strategy, int(lessons_from))
    return {"source": f"backtest:{backtest.id}", "source_backtest_id": backtest.id}, run_rows(session, backtest.id)


def copy_into_run(session: Session, backtest: BacktestResult, rows: list[AgentLesson]) -> list[AgentLesson]:
    """Copy-on-write: the run gets its own rows, the input stays untouched."""
    now = datetime.now(timezone.utc)
    copies: list[AgentLesson] = []
    for row in rows:
        copy = AgentLesson(
            strategy_id=backtest.strategy_id,
            user_id=row.user_id,
            lesson=row.lesson,
            context=row.context,
            status=row.status,
            confidence=row.confidence,
            source=row.source,
            backtest_id=row.backtest_id,  # born run, preserved across copies
            evidence=row.evidence,
            scope=SCOPE_BACKTEST,
            run_backtest_id=backtest.id,
            parent_id=row.id,
            created_at=now,
            updated_at=now,
        )
        session.add(copy)
        copies.append(copy)
    session.flush()
    return copies


def resolve_live_playbook(
    session: Session, strategy: Strategy, playbook: str | int | None
) -> int | None:
    """Backtest id a live session attaches to: "current" (default), "none", or a run id."""
    if playbook in (None, "current"):
        return strategy.playbook_backtest_id
    if playbook == "none":
        return None
    return _owned_completed_backtest(session, strategy, int(playbook)).id


# ── agent write window ────────────────────────────────────────────────────


def assert_agent_can_write(session: Session, backtest_id: int | None) -> None:
    """The agent writes only into the playbook of a run that is (just) alive."""
    if backtest_id is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Lessons are written only inside a backtest run; there is no strategy-level pool any more",
        )
    backtest = session.get(BacktestResult, backtest_id)
    if backtest is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Backtest {backtest_id} not found")
    if backtest.status in (BacktestStatus.PENDING.value, BacktestStatus.RUNNING.value):
        return
    completed_at = backtest.completed_at
    if completed_at is not None and completed_at.tzinfo is None:
        completed_at = completed_at.replace(tzinfo=timezone.utc)
    if completed_at is not None and datetime.now(timezone.utc) - completed_at <= AGENT_WRITE_GRACE:
        return
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail=f"The playbook of backtest {backtest_id} is frozen: the run ended",
    )


# ── listing / diff / promotion ────────────────────────────────────────────


def _label(backtest: BacktestResult) -> str:
    return f"#{backtest.id} · {backtest.symbol} · {backtest.start_date} → {backtest.end_date}"


def _lessons_meta(backtest: BacktestResult) -> dict[str, Any]:
    params = backtest.parameters if isinstance(backtest.parameters, dict) else {}
    meta = params.get("lessons")
    return meta if isinstance(meta, dict) else {}


def _summary(backtest: BacktestResult, rows: list[AgentLesson], *, current_id: int | None) -> PlaybookSummary:
    meta = _lessons_meta(backtest)
    return PlaybookSummary(
        backtest_id=backtest.id,
        label=_label(backtest),
        symbol=backtest.symbol,
        start_date=str(backtest.start_date) if backtest.start_date else None,
        end_date=str(backtest.end_date) if backtest.end_date else None,
        completed_at=backtest.completed_at,
        status=backtest.status,
        lessons_active=sum(1 for r in rows if r.status == "active"),
        lessons_retired=sum(1 for r in rows if r.status == "retired"),
        lessons_new=sum(1 for r in rows if r.parent_id is None),
        is_current=backtest.id == current_id,
        input_source=meta.get("source"),
        learning_mode=meta.get("learning_mode"),
    )


def initial_summary(session: Session, strategy: Strategy) -> PlaybookSummary | None:
    rows = initial_rows(session, strategy.id, status_filter=None)
    if not rows:
        return None
    return PlaybookSummary(
        backtest_id=None,
        label="Playbook iniziale",
        lessons_active=sum(1 for r in rows if r.status == "active"),
        lessons_retired=sum(1 for r in rows if r.status == "retired"),
        lessons_new=0,
        is_current=strategy.playbook_backtest_id is None,
    )


def list_playbooks(session: Session, strategy: Strategy) -> list[PlaybookSummary]:
    """Every selectable playbook of the strategy: completed runs with rows
    (newest first) and, last, the initial rows when they exist."""
    rows = list(session.exec(
        select(AgentLesson)
        .where(AgentLesson.strategy_id == strategy.id, AgentLesson.scope == SCOPE_BACKTEST)
    ).all())
    by_run: dict[int, list[AgentLesson]] = {}
    for row in rows:
        if row.run_backtest_id is not None:
            by_run.setdefault(row.run_backtest_id, []).append(row)
    if not by_run:
        backtests: list[BacktestResult] = []
    else:
        backtests = list(session.exec(
            select(BacktestResult)
            .where(BacktestResult.id.in_(list(by_run)))
            .where(BacktestResult.status == BacktestStatus.COMPLETED.value)
            .order_by(BacktestResult.id.desc())
        ).all())
    out = [_summary(bt, by_run[bt.id], current_id=strategy.playbook_backtest_id) for bt in backtests]
    initial = initial_summary(session, strategy)
    if initial is not None:
        out.append(initial)
    return out


def _change(row: AgentLesson, parent: AgentLesson | None) -> str:
    if row.status == "retired":
        return "retired"
    if parent is None:
        return "new"
    if (row.lesson or "").strip() != (parent.lesson or "").strip() or abs(row.confidence - parent.confidence) > 1e-9:
        return "changed"
    return "unchanged"


def playbook_detail(session: Session, strategy: Strategy, backtest_id: int | None) -> PlaybookRead:
    """A playbook with every row annotated against its input row."""
    if backtest_id is None:
        summary = initial_summary(session, strategy)
        if summary is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No initial playbook")
        rows = initial_rows(session, strategy.id, status_filter=None)
    else:
        backtest = session.get(BacktestResult, backtest_id)
        if backtest is None or backtest.strategy_id != strategy.id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Backtest {backtest_id} not found")
        rows = run_rows(session, backtest.id, status_filter=None)
        summary = _summary(backtest, rows, current_id=strategy.playbook_backtest_id)
    parents = {
        p.id: p for p in session.exec(
            select(AgentLesson).where(AgentLesson.id.in_([r.parent_id for r in rows if r.parent_id is not None]))
        ).all()
    } if any(r.parent_id is not None for r in rows) else {}
    lessons = [
        LessonChange(lesson=AgentLessonRead.model_validate(r), change=_change(r, parents.get(r.parent_id)))
        for r in rows
    ]
    return PlaybookRead(summary=summary, lessons=lessons)


def set_current_playbook(session: Session, strategy: Strategy, backtest_id: int | None) -> Strategy:
    """Promotion: the output of ``backtest_id`` becomes the strategy's playbook."""
    if backtest_id is not None:
        _owned_completed_backtest(session, strategy, backtest_id)
    strategy.playbook_backtest_id = backtest_id
    strategy.updated_at = datetime.now(timezone.utc)
    session.add(strategy)
    session.commit()
    session.refresh(strategy)
    return strategy


def assert_backtest_not_referenced(session: Session, backtest: BacktestResult) -> None:
    """A run whose playbook is in use cannot be deleted (its rows would go with it)."""
    strategy = session.get(Strategy, backtest.strategy_id)
    if strategy is not None and strategy.playbook_backtest_id == backtest.id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Backtest {backtest.id} is the current playbook of the strategy: choose another playbook first",
        )
    live = session.exec(
        select(StrategyLive)
        .where(StrategyLive.playbook_backtest_id == backtest.id)
        .where(StrategyLive.status.in_(list(LiveStatus.active_values())))
    ).first()
    if live is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Backtest {backtest.id} is the playbook of live session {live.id}: stop it first",
        )
