"""Global backtests listing — service flow against an embedded Postgres.

``list_all_backtests`` projects only the scalar columns of the row: the
``metrics`` JSONB embeds the run ledger (one equity snapshot per replayed
bar, tens of MB per run), and a page of 50 full rows used to take seconds to
produce a 6 kB response. The listing must carry ``chat_id`` and the agent's
``score_pct`` without loading the blobs.
Run: ``venv/bin/python -m pytest tests/test_backtest_listing_db.py``.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
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
    stale = session.exec(select(User).where(User.email == "bl@example.com")).first()
    if stale is not None:
        session.delete(stale)
        session.commit()
    user = User(email="bl@example.com", username="user_bl", hashed_password="x")
    session.add(user)
    session.flush()
    conn = Connection(user_id=user.id, name="conn-bl", broker_type="binance", status="connected",
                      created_at=now, updated_at=now)
    session.add(conn)
    session.flush()
    acc = Account(connection_id=conn.id, account_id="acc-bl", currency="USDT", created_at=now, updated_at=now)
    session.add(acc)
    session.commit()
    yield user, conn, acc
    session.delete(user)
    session.commit()


def _definition(symbol: str):
    return {"strategy": {
        "name": "bl", "symbol": symbol, "asset": "crypto", "timeframe": "5m", "timezone": "UTC", "params": {},
        "indicators": [], "charts": [{"id": "main", "symbol": symbol, "asset": "crypto", "timeframe": "5m", "indicators": []}],
        "rules": [],
    }}


def _strategy(session, user, acc, name: str, symbol: str = "BTCUSDT"):
    from app.schemas.strategy import StrategyCreate
    from app.services.strategy_service import create_strategy

    return create_strategy(session, StrategyCreate(name=name, definition=_definition(symbol), account_id=acc.id), user.id)


def _backtest(session, strategy, user, symbol: str = "BTCUSDT"):
    from app.schemas.strategy import BacktestCreate
    from app.services.strategy_service import create_backtest

    payload = BacktestCreate(symbol=symbol, start_date=date(2026, 1, 1), end_date=date(2026, 2, 1), lessons_from="none")
    return create_backtest(session, strategy.id, payload, user.id)


def test_listing_projects_scalars_only_and_carries_chat_and_score(session, tenant):
    from app.schemas.strategy import AgentEvaluationWrite, BacktestSummary
    from app.services.strategy_service import list_all_backtests, set_backtest_agent_evaluation

    user, conn, acc = tenant
    s1 = _strategy(session, user, acc, "BL-1")
    s2 = _strategy(session, user, acc, "BL-2", symbol="ETHUSDT")
    bt1 = _backtest(session, s1, user)
    bt2 = _backtest(session, s2, user, symbol="ETHUSDT")
    bt3 = _backtest(session, s1, user)

    # a finished run with a heavy ledger embedded in metrics, like the coordinator writes it
    bt1.status = "completed"
    bt1.return_pct = 12.5
    bt1.metrics = {"return_pct": 12.5, "ledger": {"orders": [], "fills": [],
                   "equity_snapshots": [{"ts": i, "equity": 100.0 + i, "positions": []} for i in range(20_000)]}}
    bt2.status = "failed"
    session.add(bt1)
    session.add(bt2)
    session.commit()

    scores = {"edge": 80, "risk": 60, "consistency": 50, "discipline": 90, "execution": 70, "robustness": 40}
    set_backtest_agent_evaluation(
        session, bt1.id,
        AgentEvaluationWrite(scores={k: {"score": v, "rationale": k} for k, v in scores.items()}, summary="ok"),
        user.id,
    )
    session.expunge_all()

    rows, total = list_all_backtests(session, user.id)
    assert total == 3
    assert [r["id"] for r in rows] == [bt3.id, bt2.id, bt1.id]  # newest first
    for r in rows:
        assert isinstance(r, dict)
        assert not {"metrics", "config", "parameters", "agent_evaluation"} & set(r)

    by_id = {r["id"]: r for r in rows}
    assert by_id[bt1.id]["agent_score_pct"] == 67.5
    assert by_id[bt2.id]["agent_score_pct"] is None
    assert by_id[bt1.id]["strategy_name"] == "BL-1" and by_id[bt2.id]["strategy_name"] == "BL-2"
    assert by_id[bt1.id]["connection_id"] == conn.id
    assert by_id[bt1.id]["return_pct"] == 12.5 and by_id[bt1.id]["status"] == "completed"
    # create_backtest opens the run's chat: its id travels with the row
    assert all(r["chat_id"] is not None for r in rows)

    # the rows validate straight into the API schema, enrichment fields untouched
    items = [BacktestSummary.model_validate(r) for r in rows]
    assert items[0].phase is None and items[0].progress is None and items[0].stale is None
    assert items[2].agent_score_pct == 67.5 and items[2].chat_id == by_id[bt1.id]["chat_id"]

    # filters and pagination
    rows, total = list_all_backtests(session, user.id, statuses=["completed", "failed"])
    assert total == 2 and {r["id"] for r in rows} == {bt1.id, bt2.id}
    rows, total = list_all_backtests(session, user.id, strategy_id=s1.id)
    assert total == 2 and {r["id"] for r in rows} == {bt1.id, bt3.id}
    rows, total = list_all_backtests(session, user.id, symbol="eth")
    assert total == 1 and rows[0]["id"] == bt2.id
    rows, total = list_all_backtests(session, user.id, limit=1, offset=1)
    assert total == 3 and [r["id"] for r in rows] == [bt2.id]
    # another user sees nothing
    rows, total = list_all_backtests(session, user.id + 1_000_000)
    assert (rows, total) == ([], 0)
