"""F0 "Riprova" (docs/valutazione-riavvio-backtest.md §3): a finished backtest
is repeated as a NEW row with the same inputs and config snapshot, its own
chat and playbook copy, pointing back to the attempt it repeats; plus the F1
agent-timeout event handler. Run: ``venv/bin/python -m pytest tests/test_backtest_retry_db.py``.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from fastapi import HTTPException
from sqlmodel import Session, select

pytest.importorskip("pgserver")


@pytest.fixture
def session(app_engine):
    with Session(app_engine, expire_on_commit=False) as s:
        yield s


@pytest.fixture
def tenant(session):
    from app.models.connection import Account, Connection
    from app.models.user import User

    now = datetime.now(timezone.utc)
    stale = session.exec(select(User).where(User.email == "retry@example.com")).first()
    if stale is not None:
        session.delete(stale)
        session.commit()
    user = User(email="retry@example.com", username="user_retry", hashed_password="x")
    session.add(user)
    session.flush()
    conn = Connection(user_id=user.id, name="conn-retry", broker_type="binance", status="connected",
                      created_at=now, updated_at=now)
    session.add(conn)
    session.flush()
    acc = Account(connection_id=conn.id, account_id="acc-retry", currency="USDT", created_at=now, updated_at=now)
    session.add(acc)
    session.commit()
    yield user, conn, acc
    session.delete(user)
    session.commit()


def _definition(symbol: str = "BTCUSDT"):
    return {"strategy": {
        "name": "retry", "symbol": symbol, "asset": "crypto", "timeframe": "5m", "timezone": "UTC", "params": {},
        "indicators": [], "charts": [{"id": "main", "symbol": symbol, "asset": "crypto", "timeframe": "5m", "indicators": []}],
        "rules": [],
    }}


def _strategy(session, user, acc):
    from app.schemas.strategy import StrategyCreate
    from app.services.strategy_service import create_strategy

    return create_strategy(session, StrategyCreate(name="RT", definition=_definition(), account_id=acc.id), user.id)


def _failed_backtest(session, strategy, user):
    from app.schemas.strategy import BacktestCreate
    from app.services.strategy_service import create_backtest

    payload = BacktestCreate(
        symbol="BTCUSDT", start_date=date(2026, 1, 1), end_date=date(2026, 2, 1), lessons_from="none",
        learning_mode="per_trade", agent_timeout_s=300, agent_reasoning="deep", on_agent_timeout="fail",
        simulation_timeframe="1m", initial_capital=50_000, commission=1.5, rth=False,
        parameters={"rag_images": False},
        layout_config={"timezone": "Europe/Rome"},
    )
    bt = create_backtest(session, strategy.id, payload, user.id)
    bt.status = "failed"
    bt.error_message = "Playback hold not released"
    bt.failure_kind = "agent_timeout"
    bt.runner_log_tail = "tail"
    bt.agent_evaluation = {"score_pct": 10}
    bt.metrics = {"return_pct": -1.0, "agent_turns_timed_out": 2}
    session.add(bt)
    session.commit()
    return bt


def test_retry_clones_the_run_and_starts_the_clone(session, tenant, monkeypatch):
    from app.models.agent import Chat
    from app.services import strategy_service
    from app.services.strategy_service import retry_backtest

    user, _conn, acc = tenant
    strategy = _strategy(session, user, acc)
    source = _failed_backtest(session, strategy, user)
    strategy.definition = _definition("ETHUSDT")  # the strategy moved on: the retry must NOT follow it
    session.add(strategy)
    session.commit()

    started: list[int] = []

    def fake_run(sess, backtest_id, user_id=None):
        started.append(backtest_id)
        return strategy_service.get_backtest(sess, backtest_id, user_id)

    monkeypatch.setattr(strategy_service, "run_backtest", fake_run)

    clone = retry_backtest(session, source.id, user.id)

    assert clone.id != source.id and started == [clone.id]
    assert clone.retried_from_backtest_id == source.id
    assert clone.status == "pending"  # the stub did not start a container
    # same experiment: inputs and the SAME config snapshot
    assert (clone.symbol, clone.start_date, clone.end_date, clone.timeframe, clone.simulation_timeframe) == (
        source.symbol, source.start_date, source.end_date, source.timeframe, source.simulation_timeframe)
    assert clone.config == source.config and clone.config["strategy"]["symbol"] == "BTCUSDT"
    assert clone.initial_capital == 50_000 and clone.commission == 1.5 and clone.rth in (False, "False", "false")
    assert clone.layout_config == {"timezone": "Europe/Rome"}
    params = clone.parameters
    assert params["lessons"]["source"] == "none" and params["lessons"]["learning_mode"] == "per_trade"
    assert params["agent_timeout_s"] == 300 and params["agent_reasoning"] == "deep" and params["on_agent_timeout"] == "fail"
    assert params["rag_images"] is False
    # nothing of the failed attempt leaks into the clone
    assert clone.error_message is None and clone.failure_kind is None and clone.runner_log_tail is None
    assert clone.agent_evaluation is None and clone.metrics is None
    # its own chat
    chats = session.exec(select(Chat).where(Chat.backtest_id.in_([source.id, clone.id]))).all()
    assert {c.backtest_id for c in chats} == {source.id, clone.id}
    # the source row is a document: untouched
    session.refresh(source)
    assert source.status == "failed" and source.metrics["agent_turns_timed_out"] == 2

    # the listing carries lineage, failure kind and the timeout count as scalars
    rows, _total = strategy_service.list_all_backtests(session, user.id, strategy_id=strategy.id)
    by_id = {r["id"]: r for r in rows}
    assert by_id[clone.id]["retried_from_backtest_id"] == source.id
    assert by_id[source.id]["failure_kind"] == "agent_timeout" and by_id[source.id]["agent_turns_timed_out"] == 2
    assert by_id[clone.id]["agent_turns_timed_out"] is None


def test_retry_is_refused_while_the_run_is_not_finished(session, tenant):
    from app.services.strategy_service import retry_backtest

    user, _conn, acc = tenant
    strategy = _strategy(session, user, acc)
    bt = _failed_backtest(session, strategy, user)
    bt.status = "running"
    session.add(bt)
    session.commit()
    with pytest.raises(HTTPException) as exc:
        retry_backtest(session, bt.id, user.id)
    assert exc.value.status_code == 409


def test_agent_timeout_call_becomes_a_user_event_once(session, tenant):
    from app.models.agent_call import AgentCall
    from app.models.user_event import UserEvent
    from app.services.webhook_sources import on_agent_call

    user, _conn, acc = tenant
    strategy = _strategy(session, user, acc)
    bt = _failed_backtest(session, strategy, user)
    call = AgentCall(strategy_id=strategy.id, backtest_id=bt.id, trigger_type="agent_request", trigger_name="session_end",
                     correlation_id="corr-1", session_id="1", bar_ts=1790799000000,
                     called_at=datetime.now(timezone.utc), duration_ms=515000, status="timeout")
    delivered = AgentCall(strategy_id=strategy.id, backtest_id=bt.id, trigger_type="agent_request", trigger_name="x",
                          correlation_id="corr-2", session_id="1", called_at=datetime.now(timezone.utc), status="delivered")
    session.add(call)
    session.add(delivered)
    session.commit()

    on_agent_call(session, call.id)
    on_agent_call(session, call.id)  # the trigger fires on INSERT and on UPDATE OF status: one event
    on_agent_call(session, delivered.id)

    events = session.exec(select(UserEvent).where(UserEvent.user_id == user.id, UserEvent.type == "backtest.agent_timeout")).all()
    assert len(events) == 1
    ev = events[0]
    assert ev.entity_id == bt.id and ev.severity == "warning" and "session_end" in (ev.message or "")
    assert ev.data["correlation_id"] == "corr-1" and ev.data["bar_ts"] == 1790799000000
    for e in events:
        session.delete(e)
    session.commit()
