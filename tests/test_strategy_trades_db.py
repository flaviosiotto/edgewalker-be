"""``/strategies/{id}/trades`` (workspace Performance tab, live/design): the
ledger rows of every live session of the strategy, oldest exit first, same
population as ``compute_strategy_performance``; rows of other strategies and
unattributed rows of the same account are left out.
Run: ``venv/bin/python -m pytest tests/test_strategy_trades_db.py``.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import Session, select

import app.models.agent_turn  # noqa: E402,F401
import app.models.connection  # noqa: E402,F401
import app.models.live_trading  # noqa: E402,F401
import app.models.strategy  # noqa: E402,F401
import app.models.strategy_template  # noqa: E402,F401
import app.models.user_event  # noqa: E402,F401
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
    stale = session.exec(select(User).where(User.email == "st@example.com")).first()
    if stale is not None:
        session.delete(stale)
        session.commit()
    user = User(email="st@example.com", username="user_st", hashed_password="x")
    session.add(user)
    session.flush()
    conn = Connection(user_id=user.id, name="OKX demo", broker_type="okx", status="connected", created_at=now, updated_at=now)
    session.add(conn)
    session.flush()
    acc = Account(connection_id=conn.id, account_id="acc-st", currency="USDT", created_at=now, updated_at=now)
    session.add(acc)
    session.commit()
    yield user, conn, acc
    session.delete(user)
    session.commit()


def _strategy(session, user, acc, name):
    from app.schemas.strategy import StrategyCreate
    from app.services.strategy_service import create_strategy

    definition = {"strategy": {
        "name": name, "symbol": "BTCUSDT", "asset": "crypto", "timeframe": "5m", "timezone": "UTC", "params": {},
        "indicators": [], "charts": [{"id": "main", "symbol": "BTCUSDT", "asset": "crypto", "timeframe": "5m", "indicators": []}], "rules": [],
    }}
    return create_strategy(session, StrategyCreate(name=name, definition=definition, account_id=acc.id), user.id)


def _trade(acc, live_id, exit_at, net):
    from app.models.live_trading import LiveTrade

    return LiveTrade(
        strategy_live_id=live_id, account_id=acc.id, symbol="BTCUSDT", direction="long", quantity=1.0,
        entry_price=100.0, exit_price=100.0 + net, realized_pnl=net, commission=0.0, net_pnl=net, currency="USDT",
        entry_time=exit_at - timedelta(minutes=30), exit_time=exit_at,
    )


def test_strategy_trades_span_every_session_oldest_first(session, tenant):
    from app.models.live_trading import LiveTrade
    from app.models.strategy import StrategyLive
    from app.services.performance_service import compute_strategy_performance, list_strategy_trades

    # The trades table is migration-managed: create_all skips it.
    LiveTrade.__table__.create(session.get_bind(), checkfirst=True)
    user, conn, acc = tenant
    mine = _strategy(session, user, acc, "mine")
    other = _strategy(session, user, acc, "other")
    live_a = StrategyLive(strategy_id=mine.id, status="stopped", symbol="BTCUSDT", timeframe="5m", account_id=acc.id, connection_id=conn.id)
    live_b = StrategyLive(strategy_id=mine.id, status="running", symbol="BTCUSDT", timeframe="5m", account_id=acc.id, connection_id=conn.id)
    live_other = StrategyLive(strategy_id=other.id, status="running", symbol="BTCUSDT", timeframe="5m", account_id=acc.id, connection_id=conn.id)
    session.add_all([live_a, live_b, live_other])
    session.commit()

    t0 = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
    session.add_all([
        _trade(acc, live_b.id, t0 + timedelta(days=2), 30.0),   # newest, second session
        _trade(acc, live_a.id, t0, -10.0),                      # oldest, first session
        _trade(acc, live_a.id, t0 + timedelta(days=1), 20.0),
        _trade(acc, live_other.id, t0 + timedelta(hours=1), 99.0),   # other strategy
        _trade(acc, None, t0 + timedelta(hours=2), 77.0),         # manual trade on the account
    ])
    session.commit()

    rows = list_strategy_trades(session, strategy_id=mine.id)
    assert [r.net_pnl for r in rows] == [-10.0, 20.0, 30.0]
    assert {r.strategy_live_id for r in rows} == {live_a.id, live_b.id}
    # Same population as the aggregate behind the strategy card / Performance tab.
    stats = compute_strategy_performance(session, strategy_id=mine.id)
    assert stats.totals.trades == len(rows) == 3
    assert stats.totals.net == pytest.approx(sum(r.net_pnl for r in rows))

    # Window and limit follow the aggregate's semantics (exit_time, end exclusive).
    windowed = list_strategy_trades(session, strategy_id=mine.id, start=t0 + timedelta(days=1), end=t0 + timedelta(days=2))
    assert [r.net_pnl for r in windowed] == [20.0]
    assert [r.net_pnl for r in list_strategy_trades(session, strategy_id=mine.id, limit=2)] == [-10.0, 20.0]

    # No live session yet → empty, no query on the ledger.
    fresh = _strategy(session, user, acc, "fresh")
    assert list_strategy_trades(session, strategy_id=fresh.id) == []
