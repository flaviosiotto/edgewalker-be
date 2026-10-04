"""Playbooks (migration 062) — service flows against an embedded Postgres.

A backtest copies the playbook chosen at launch into its own rows, the agent
edits only those (and only while the run is alive), the output is annotated
against its input, promotion points the strategy at a run, live attaches to
a run's output, and a referenced run cannot be deleted.
Run: ``venv/bin/python -m pytest tests/test_playbooks_db.py``.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest
from fastapi import HTTPException
from sqlmodel import Session

pytest.importorskip("pgserver")


@pytest.fixture
def session(app_engine):
    with Session(app_engine, expire_on_commit=False) as s:
        yield s


@pytest.fixture
def tenant(session):
    from app.models.connection import Account, Connection
    from app.models.user import User

    from sqlmodel import select

    now = datetime.now(timezone.utc)
    # The embedded Postgres outlives the session: drop a leftover of a run
    # that died before its teardown.
    stale = session.exec(select(User).where(User.email == "pb@example.com")).first()
    if stale is not None:
        session.delete(stale)
        session.commit()
    user = User(email="pb@example.com", username="user_pb", hashed_password="x")
    session.add(user)
    session.flush()
    conn = Connection(user_id=user.id, name="conn-pb", broker_type="binance", status="connected",
                      created_at=now, updated_at=now)
    session.add(conn)
    session.flush()
    acc = Account(connection_id=conn.id, account_id="acc-pb", currency="USDT", created_at=now, updated_at=now)
    session.add(acc)
    session.commit()
    yield user, conn, acc
    session.delete(user)
    session.commit()


def _definition():
    return {"strategy": {
        "name": "pb", "symbol": "BTCUSDT", "asset": "crypto", "timeframe": "5m", "timezone": "UTC", "params": {},
        "indicators": [], "charts": [{"id": "main", "symbol": "BTCUSDT", "asset": "crypto", "timeframe": "5m", "indicators": []}],
        "rules": [],
    }}


def _strategy(session, user, acc):
    from app.schemas.strategy import StrategyCreate
    from app.services.strategy_service import create_strategy

    return create_strategy(session, StrategyCreate(name="PB", definition=_definition(), account_id=acc.id), user.id)


def _backtest(session, strategy, user, **kwargs):
    from app.schemas.strategy import BacktestCreate
    from app.services.strategy_service import create_backtest

    payload = BacktestCreate(symbol="BTCUSDT", start_date=date(2026, 1, 1), end_date=date(2026, 2, 1), **kwargs)
    return create_backtest(session, strategy.id, payload, user.id)


def _complete(session, backtest, *, ago: timedelta = timedelta(0)):
    backtest.status = "completed"
    backtest.completed_at = datetime.now(timezone.utc) - ago
    session.add(backtest)
    session.commit()


def test_backtest_copies_its_input_and_live_attaches_to_a_run(session, tenant):
    from app.schemas.agent_lesson import AgentLessonCreate, AgentLessonUpdate
    from app.services import playbook_service
    from app.services.agent_lesson_service import create_lesson, list_lessons, update_lesson

    user, _, acc = tenant
    strategy = _strategy(session, user, acc)

    # initial playbook: two manual rows (scope=strategy)
    l1 = create_lesson(session, strategy.id, user.id, AgentLessonCreate(lesson="Se A allora B.", confidence=0.7, source="user"))
    l2 = create_lesson(session, strategy.id, user.id, AgentLessonCreate(lesson="Se C allora D.", confidence=0.5, source="user"))
    assert l1.scope == "strategy" and l1.run_backtest_id is None
    assert [r.id for r in list_lessons(session, strategy.id, user.id)] == [l1.id, l2.id]

    # a run launched from the current playbook gets its own copies
    bt1 = _backtest(session, strategy, user)
    meta = bt1.parameters["lessons"]
    assert meta["source"] == "strategy" and meta["learning_mode"] == "end_of_run"
    copies = playbook_service.run_rows(session, bt1.id)
    assert {c.parent_id for c in copies} == {l1.id, l2.id}
    assert all(c.scope == "backtest" and c.run_backtest_id == bt1.id for c in copies)
    assert sorted(meta["input_ids"]) == sorted(c.id for c in copies)

    # the agent edits the run's rows while it is alive...
    bt1.status = "running"
    session.add(bt1)
    session.commit()
    c1 = next(c for c in copies if c.parent_id == l1.id)
    update_lesson(session, c1.id, user.id, AgentLessonUpdate(confidence=0.3), origin="agent")
    born = create_lesson(
        session, strategy.id, user.id,
        AgentLessonCreate(lesson="Se E allora F.", confidence=0.4, backtest_id=bt1.id), origin="agent",
    )
    assert born.run_backtest_id == bt1.id and born.backtest_id == bt1.id and born.parent_id is None
    # ...never the strategy-level rows
    with pytest.raises(HTTPException) as exc:
        update_lesson(session, l1.id, user.id, AgentLessonUpdate(confidence=0.9), origin="agent")
    assert exc.value.status_code == 409
    assert session.get(type(l1), l1.id).confidence == 0.7  # the input is untouched

    # the output: annotated against the input
    c2 = next(c for c in copies if c.parent_id == l2.id)
    update_lesson(session, c2.id, user.id, AgentLessonUpdate(status="retired"), origin="agent")
    _complete(session, bt1)
    detail = playbook_service.playbook_detail(session, strategy, bt1.id)
    changes = {item.lesson.id: item.change for item in detail.lessons}
    assert changes == {c1.id: "changed", c2.id: "retired", born.id: "new"}
    assert detail.summary.lessons_active == 2 and detail.summary.lessons_retired == 1 and detail.summary.lessons_new == 1

    # frozen for the agent once the run ended long ago; the trader can still edit
    bt1.completed_at = datetime.now(timezone.utc) - timedelta(hours=3)
    session.add(bt1)
    session.commit()
    with pytest.raises(HTTPException):
        update_lesson(session, c1.id, user.id, AgentLessonUpdate(confidence=0.2), origin="agent")
    update_lesson(session, c1.id, user.id, AgentLessonUpdate(confidence=0.2), origin="user")

    # listing + promotion: the strategy points at the run, the default input follows
    summaries = playbook_service.list_playbooks(session, strategy)
    assert [s.backtest_id for s in summaries] == [bt1.id, None]
    assert summaries[1].is_current is True and summaries[0].is_current is False
    playbook_service.set_current_playbook(session, strategy, bt1.id)
    assert [r.id for r in list_lessons(session, strategy.id, user.id)] == sorted(
        [c1.id, born.id], key=lambda i: -session.get(type(l1), i).confidence
    )
    bt2 = _backtest(session, strategy, user, lessons_from=bt1.id, learning_mode="per_trade")
    assert bt2.parameters["lessons"]["source"] == f"backtest:{bt1.id}"
    assert {c.parent_id for c in playbook_service.run_rows(session, bt2.id)} == {c1.id, born.id}
    bt3 = _backtest(session, strategy, user, lessons_from="none")
    assert playbook_service.run_rows(session, bt3.id) == [] and bt3.parameters["lessons"]["source"] == "none"
    bt4 = _backtest(session, strategy, user, use_lessons=False)  # legacy flag
    assert bt4.parameters["lessons"]["source"] == "none"

    # live: "current" / "none" / a completed run; a running one is refused
    assert playbook_service.resolve_live_playbook(session, strategy, "current") == bt1.id
    assert playbook_service.resolve_live_playbook(session, strategy, "none") is None
    with pytest.raises(HTTPException) as exc:
        playbook_service.resolve_live_playbook(session, strategy, bt2.id)
    assert exc.value.status_code == 409

    # a referenced run cannot be deleted
    from app.services.strategy_service import delete_backtest

    with pytest.raises(HTTPException) as exc:
        delete_backtest(session, bt1.id, user.id)
    assert exc.value.status_code == 409
    delete_backtest(session, bt3.id, user.id)


def test_agent_evaluation_derives_the_overall_score(session, tenant):
    from pydantic import ValidationError

    from app.schemas.strategy import AgentEvaluationWrite
    from app.services.strategy_service import get_backtest, set_backtest_agent_evaluation

    user, _, acc = tenant
    strategy = _strategy(session, user, acc)
    bt = _backtest(session, strategy, user, lessons_from="none")
    scores = {"edge": 80, "risk": 60, "consistency": 50, "discipline": 90, "execution": 70, "robustness": 40}
    payload = AgentEvaluationWrite(
        scores={k: {"score": v, "rationale": f"{k} ok"} for k, v in scores.items()},
        summary="Edge reale ma concentrato.",
        hints=[{"title": "Riduci la size dopo 3 loss", "detail": "serie max 6", "category": "risk", "priority": "high"}],
        playbook_recommended=True,
    )
    out = set_backtest_agent_evaluation(session, bt.id, payload, user.id)
    # 80*.25 + 60*.20 + 50*.15 + 90*.15 + 70*.15 + 40*.10
    assert out["score_pct"] == 67.5
    stored = get_backtest(session, bt.id, user.id).agent_evaluation
    assert stored["scores"]["edge"] == {"score": 80.0, "rationale": "edge ok"}
    assert stored["hints"][0]["priority"] == "high" and stored["playbook_recommended"] is True

    # the backtests listing carries the overall score, not the whole evaluation
    from app.schemas.strategy import BacktestSummary

    assert "agent_score_pct" in BacktestSummary.model_fields

    with pytest.raises(ValidationError):
        AgentEvaluationWrite(scores={"edge": {"score": 80}})
    with pytest.raises(ValidationError):
        AgentEvaluationWrite(scores={k: {"score": 120} for k in scores})


def test_runner_token_survives_completion_for_the_final_analysis(session, tenant):
    """The final analysis turn is dispatched after the run is completed: the
    runner must still authenticate (and get the agent its tokens) for a while."""
    import asyncio

    from app.utils.auth_utils import (
        BACKTEST_RUNNER_GRACE,
        backtest_runner_window_open,
        create_user_delegated_token,
        get_current_runner_principal,
    )
    from app.core.config import settings

    user, _, acc = tenant
    strategy = _strategy(session, user, acc)
    bt = _backtest(session, strategy, user, lessons_from="none")
    token = create_user_delegated_token(
        session, user_id=user.id, audience=settings.RUNNER_TOKEN_AUDIENCE, purpose="runner_backend",
        no_expiry=True, extra_claims={"strategy_id": strategy.id, "backtest_id": bt.id},
    )

    def principal():
        return asyncio.run(get_current_runner_principal(token=token, session=session))

    assert backtest_runner_window_open(bt)
    _complete(session, bt)
    assert principal().claims["backtest_id"] == bt.id
    stored = session.get(type(bt), bt.id)
    assert stored.agent_evaluation is None

    _complete(session, bt, ago=BACKTEST_RUNNER_GRACE + timedelta(minutes=1))
    with pytest.raises(HTTPException) as exc:
        principal()
    assert exc.value.status_code == 401

    bt.status = "failed"
    bt.completed_at = datetime.now(timezone.utc)
    session.add(bt)
    session.commit()
    assert not backtest_runner_window_open(bt)
