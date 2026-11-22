"""Bulk backtests (``mode = "bulk"``, docs/valutazione-backtest-massivo.md):
no agent and no lessons on the row, a definition the bulk engine cannot run
is refused at creation, a retry keeps the mode, and ``/run`` hands the run to
the coordinator without starting a runner container.
Run: ``venv/bin/python -m pytest tests/test_backtest_bulk_db.py``.
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
    stale = session.exec(select(User).where(User.email == "bulk@example.com")).first()
    if stale is not None:
        session.delete(stale)
        session.commit()
    user = User(email="bulk@example.com", username="user_bulk", hashed_password="x")
    session.add(user)
    session.flush()
    conn = Connection(user_id=user.id, name="conn-bulk", broker_type="binance", status="connected",
                      created_at=now, updated_at=now)
    session.add(conn)
    session.flush()
    acc = Account(connection_id=conn.id, account_id="acc-bulk", currency="USDT", created_at=now, updated_at=now)
    session.add(acc)
    session.commit()
    yield user, conn, acc
    session.delete(user)
    session.commit()


def _definition(rules: list[dict] | None = None):
    return {"strategy": {
        "name": "bulk", "symbol": "BTCUSDT", "asset": "crypto", "timeframe": "1h", "timezone": "UTC", "params": {},
        "indicators": [],
        "charts": [
            {"id": "main", "symbol": "BTCUSDT", "asset": "crypto", "timeframe": "1h", "indicators": []},
            {"id": "htf", "symbol": "BTCUSDT", "asset": "crypto", "timeframe": "4h", "indicators": []},
        ],
        "rules": rules or [],
    }}


def _strategy(session, user, acc, rules: list[dict] | None = None):
    from app.schemas.strategy import StrategyCreate
    from app.services.strategy_service import create_strategy

    return create_strategy(session, StrategyCreate(name="BK", definition=_definition(rules), account_id=acc.id), user.id)


def _no_multiplier_lookup(monkeypatch):
    """The multiplier lookup reads ``trades``, a migration-managed table the
    embedded test schema does not have."""
    import app.services.contract_multiplier as contract_multiplier

    monkeypatch.setattr(contract_multiplier, "resolve_contract_multiplier", lambda *a, **k: (1.0, "default"))


def _payload(**kw):
    from app.schemas.strategy import BacktestCreate

    base = dict(symbol="BTCUSDT", start_date=date(2020, 1, 1), end_date=date(2026, 1, 1), timeframe="1h",
                asset="crypto", mode="bulk")
    base.update(kw)
    return BacktestCreate(**base)


def test_bulk_backtest_has_no_agent_and_no_lessons(session, tenant):
    from app.services.strategy_service import create_backtest, list_all_backtests

    user, _conn, acc = tenant
    strategy = _strategy(session, user, acc)

    # An agent id that does not exist would be refused (404) in replay mode:
    # in bulk mode the agent is not even looked up.
    bt = create_backtest(
        session, strategy.id,
        _payload(agent_id=987_654_321, learning_mode="per_trade", agent_timeout_s=300,
                 agent_reasoning="deep", on_agent_timeout="fail", simulation_timeframe="1m"),
        user.id,
    )

    assert bt.mode == "bulk" and bt.status == "pending"
    assert bt.agent_id is None  # asked for an agent: a bulk run never calls one
    assert bt.simulation_timeframe == "1m"  # the driver is kept: it times the TP/SL touches
    lessons = bt.parameters["lessons"]
    assert lessons["learning_mode"] == "off" and lessons["source"] == "none"
    assert not {"agent_timeout_s", "agent_reasoning", "on_agent_timeout"} & set(bt.parameters)

    rows, _total = list_all_backtests(session, user.id, strategy_id=strategy.id)
    assert rows[0]["mode"] == "bulk"

    replay = create_backtest(session, strategy.id, _payload(mode="replay", lessons_from="none"), user.id)
    assert replay.mode == "replay"


def test_definition_the_bulk_engine_cannot_run_is_refused_at_creation(session, tenant):
    from app.services.strategy_service import create_backtest

    user, _conn, acc = tenant
    rules = [{"name": "legacy", "action": "buy", "evaluate_on": "htf",
              "conditions": [{"field": "price.close", "op": ">", "value": 1}]}]
    strategy = _strategy(session, user, acc, rules)

    with pytest.raises(HTTPException) as exc:
        create_backtest(session, strategy.id, _payload(), user.id)
    assert exc.value.status_code == 400 and "legacy" in exc.value.detail and "evaluate_on" in exc.value.detail

    # the same strategy still runs as a replay
    session.rollback()
    assert create_backtest(session, strategy.id, _payload(mode="replay", lessons_from="none"), user.id).mode == "replay"


def test_ask_agent_rules_do_not_block_a_bulk_backtest(session, tenant):
    from app.services.strategy_service import create_backtest

    user, _conn, acc = tenant
    rules = [{"name": "ask", "action": "ask_agent", "conditions": [{"field": "price.close", "op": ">", "value": 1}]}]
    strategy = _strategy(session, user, acc, rules)
    assert create_backtest(session, strategy.id, _payload(), user.id).mode == "bulk"


def test_run_hands_a_bulk_backtest_to_the_coordinator(session, tenant, monkeypatch):
    from app.services import strategy_service
    from app.services.backtest_runner_service import backtest_runner_service
    from app.services.strategy_service import create_backtest, run_backtest

    user, conn, acc = tenant
    strategy = _strategy(session, user, acc)
    bt = create_backtest(session, strategy.id, _payload(), user.id)

    calls: list[dict] = []

    def fake_bulk(**kw):
        calls.append(kw)
        return {"status": "accepted"}

    def no_container(**_kw):
        raise AssertionError("a bulk run must not start a runner container")

    monkeypatch.setattr(backtest_runner_service, "start_bulk_backtest", fake_bulk)
    monkeypatch.setattr(backtest_runner_service, "start_backtest", no_container)
    monkeypatch.setattr(strategy_service, "assert_within", lambda *a, **k: None)
    _no_multiplier_lookup(monkeypatch)

    started = run_backtest(session, bt.id, user.id)

    assert started.status == "running" and started.started_at is not None
    assert len(calls) == 1
    call = calls[0]
    assert call["backtest_id"] == bt.id and call["connection_id"] == conn.id
    assert call["broker_type"] == "binance" and call["position_accounting_mode"]
    assert call["strategy_config"]["strategy"]["symbol"] == "BTCUSDT"


def test_coordinator_refusal_marks_the_bulk_backtest_as_error(session, tenant, monkeypatch):
    from app.services import strategy_service
    from app.services.backtest_runner_service import backtest_runner_service
    from app.services.strategy_service import create_backtest, run_backtest

    user, _conn, acc = tenant
    strategy = _strategy(session, user, acc)
    bt = create_backtest(session, strategy.id, _payload(), user.id)

    def unreachable(**_kw):
        raise RuntimeError("Backtest coordinator is not reachable")

    monkeypatch.setattr(backtest_runner_service, "start_bulk_backtest", unreachable)
    monkeypatch.setattr(strategy_service, "assert_within", lambda *a, **k: None)
    _no_multiplier_lookup(monkeypatch)

    with pytest.raises(HTTPException) as exc:
        run_backtest(session, bt.id, user.id)
    assert exc.value.status_code == 500
    session.refresh(bt)
    assert bt.status == "error" and bt.failure_kind == "start" and "not reachable" in bt.error_message


def test_retry_keeps_the_mode(session, tenant, monkeypatch):
    from app.services import strategy_service
    from app.services.strategy_service import create_backtest, retry_backtest

    user, _conn, acc = tenant
    strategy = _strategy(session, user, acc)
    bt = create_backtest(session, strategy.id, _payload(simulation_timeframe="1m"), user.id)
    bt.status = "failed"
    session.add(bt)
    session.commit()
    monkeypatch.setattr(strategy_service, "run_backtest", lambda sess, bid, uid=None: strategy_service.get_backtest(sess, bid, uid))

    clone = retry_backtest(session, bt.id, user.id)
    assert clone.mode == "bulk" and clone.agent_id is None and clone.simulation_timeframe == "1m"
