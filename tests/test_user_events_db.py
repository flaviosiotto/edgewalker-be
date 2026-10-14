"""Task outcomes against an embedded Postgres (migr. 070): the webhook
sources write ``user_event`` rows for every user (no subscription needed),
dedupe, the studio_run raw handler, listing/detail, retention.
Run: ``venv/bin/python -m pytest tests/test_user_events_db.py``.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import text
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


@pytest.fixture(autouse=True)
def _no_redis(monkeypatch):
    from app.services import user_events

    monkeypatch.setattr(user_events, "get_redis", lambda: None)


@pytest.fixture
def tenant(session):
    from app.models.connection import Account, Connection
    from app.models.user import User

    now = datetime.now(timezone.utc)
    stale = session.exec(select(User).where(User.email == "ue@example.com")).first()
    if stale is not None:
        session.delete(stale)
        session.commit()
    user = User(email="ue@example.com", username="user_ue", hashed_password="x")
    session.add(user)
    session.flush()
    conn = Connection(user_id=user.id, name="OKX demo", broker_type="okx", status="connecting",
                      status_message="Spawning gateway container…", created_at=now, updated_at=now)
    session.add(conn)
    session.flush()
    acc = Account(connection_id=conn.id, account_id="acc-ue", currency="USDT", created_at=now, updated_at=now)
    session.add(acc)
    session.commit()
    yield user, conn, acc
    session.delete(user)
    session.commit()


def _events(session, user_id):
    from app.services.user_events import list_events

    return list_events(session, user_id)


def test_connection_outcomes_without_subscription(session, tenant):
    from app.services import webhook_sources as src

    user, conn, _ = tenant
    # connecting → error: a failed attempt, kept even though the user has no webhook
    conn.status, conn.status_message = "error", "Gateway did not connect in time: not connected yet; gateway logs: OKX connect failed"
    session.add(conn)
    session.commit()
    assert src.handle_notification({"table": "connections", "op": "UPDATE", "id": conn.id, "old_status": "connecting"}) == 0
    rows = _events(session, user.id)
    assert [r.type for r in rows] == ["connection.failed"]
    assert rows[0].severity == "error" and rows[0].entity_type == "connection" and rows[0].entity_id == conn.id
    assert rows[0].entity_name == "OKX demo" and "gateway logs" in rows[0].message
    assert rows[0].links == {"ui": "/settings?tab=connections"}

    # health-loop cleanup error → disconnected: no second event
    conn.status = "disconnected"
    session.add(conn)
    session.commit()
    src.handle_notification({"table": "connections", "op": "UPDATE", "id": conn.id, "old_status": "error"})
    assert len(_events(session, user.id)) == 1

    # a later successful connect, then a fall
    conn.status, conn.status_message = "connected", None
    session.add(conn)
    session.commit()
    src.handle_notification({"table": "connections", "op": "UPDATE", "id": conn.id, "old_status": "connecting"})
    conn.status, conn.status_message = "disconnected", "Gateway container terminated"
    session.add(conn)
    session.commit()
    src.handle_notification({"table": "connections", "op": "UPDATE", "id": conn.id, "old_status": "connected"})
    assert [r.type for r in _events(session, user.id)] == ["connection.lost", "connection.connected", "connection.failed"]


def test_live_and_backtest_outcomes_dedupe(session, tenant):
    from app.models.strategy import BacktestResult, StrategyLive
    from app.schemas.strategy import StrategyCreate
    from app.services import webhook_sources as src
    from app.services.strategy_service import create_strategy

    user, conn, acc = tenant
    definition = {"strategy": {
        "name": "ue", "symbol": "BTCUSDT", "asset": "crypto", "timeframe": "5m", "timezone": "UTC", "params": {},
        "indicators": [], "charts": [{"id": "main", "symbol": "BTCUSDT", "asset": "crypto", "timeframe": "5m", "indicators": []}], "rules": [],
    }}
    strategy = create_strategy(session, StrategyCreate(name="UE strat", definition=definition, account_id=acc.id), user.id)

    live = StrategyLive(strategy_id=strategy.id, status="running", symbol="BTCUSDT", timeframe="5m", account_id=acc.id, connection_id=conn.id)
    session.add(live)
    session.commit()
    src.handle_notification({"table": "strategy_live", "op": "UPDATE", "id": live.id, "old_status": "starting"})
    src.handle_notification({"table": "strategy_live", "op": "UPDATE", "id": live.id, "old_status": "starting"})  # same minute: deduped
    live.status, live.error_message = "error", "Runner container terminated unexpectedly"
    session.add(live)
    session.commit()
    src.handle_notification({"table": "strategy_live", "op": "UPDATE", "id": live.id, "old_status": "running"})
    # paused → running is not a start
    live.status = "running"
    session.add(live)
    session.commit()
    src.handle_notification({"table": "strategy_live", "op": "UPDATE", "id": live.id, "old_status": "paused"})

    bt = BacktestResult(strategy_id=strategy.id, symbol="BTCUSDT", start_date=date(2026, 1, 1), end_date=date(2026, 2, 1),
                        status="failed", error_message="ds-hist timed out", runner_log_tail="[runner exit_code=1 last 3 lines]\nboom",
                        completed_at=datetime.now(timezone.utc))
    session.add(bt)
    session.commit()
    src.handle_notification({"table": "strategy_backtests", "op": "UPDATE", "id": bt.id, "old_status": "running"})
    src.handle_notification({"table": "strategy_backtests", "op": "UPDATE", "id": bt.id})  # no old_status: still deduped

    rows = _events(session, user.id)
    assert [r.type for r in rows] == ["backtest.failed", "live.failed", "live.started"]
    failed = rows[0]
    assert failed.message == "ds-hist timed out" and failed.details.startswith("[runner")
    assert failed.links["ui"] == f"/strategies/{strategy.id}/backtests/{bt.id}"
    assert failed.entity_name == "UE strat"
    assert rows[1].message == "Runner container terminated unexpectedly"
    assert rows[1].links["ui"] == f"/live/instances/{live.id}"


def test_studio_run_raw_handler(session, tenant):
    from app.services import webhook_sources as src

    user, _, _ = tenant
    # studio-svc owns these tables: a minimal copy of its schema for the test
    session.execute(text("""
        CREATE TABLE IF NOT EXISTS studio (id SERIAL PRIMARY KEY, user_id INTEGER NOT NULL, name VARCHAR NOT NULL);
        CREATE TABLE IF NOT EXISTS studio_trigger (id SERIAL PRIMARY KEY, studio_id INTEGER NOT NULL, type VARCHAR NOT NULL);
        CREATE TABLE IF NOT EXISTS studio_run (
            id SERIAL PRIMARY KEY, studio_id INTEGER NOT NULL, trigger_id INTEGER NULL, status VARCHAR NOT NULL,
            error TEXT NULL, queued_at TIMESTAMPTZ NOT NULL DEFAULT now(), started_at TIMESTAMPTZ NULL, finished_at TIMESTAMPTZ NULL);
    """))
    studio_id = session.execute(text("INSERT INTO studio (user_id, name) VALUES (:u, 'COT report') RETURNING id"), {"u": user.id}).scalar()
    trigger_id = session.execute(text("INSERT INTO studio_trigger (studio_id, type) VALUES (:s, 'youtube') RETURNING id"), {"s": studio_id}).scalar()
    run_id = session.execute(
        text("INSERT INTO studio_run (studio_id, trigger_id, status, error) VALUES (:s, :t, 'failed', 'Execution timed out after 600s') RETURNING id"),
        {"s": studio_id, "t": trigger_id},
    ).scalar()
    ok_id = session.execute(text("INSERT INTO studio_run (studio_id, status) VALUES (:s, 'succeeded') RETURNING id"), {"s": studio_id}).scalar()
    session.commit()

    assert src.handle_notification({"table": "studio_run", "op": "UPDATE", "id": run_id, "old_status": "running"}) == 0
    src.handle_notification({"table": "studio_run", "op": "UPDATE", "id": ok_id, "old_status": "running"})
    src.handle_notification({"table": "studio_run", "op": "UPDATE", "id": 999999, "old_status": "running"})
    rows = _events(session, user.id)
    assert [r.type for r in rows] == ["studio.run.succeeded", "studio.run.failed"]
    assert rows[1].message == "Execution timed out after 600s" and rows[1].entity_name == "COT report"
    assert rows[1].data["origin"] == "trigger" and rows[1].data["trigger_type"] == "youtube"
    assert rows[0].data["origin"] == "manual"
    assert rows[1].links["ui"] == f"/studi/{studio_id}?tab=runs"


def test_list_filters_detail_and_retention(session, tenant):
    from app.models.user_event import UserEvent
    from app.services import user_events as ue

    user, conn, _ = tenant
    for i, (t, msg) in enumerate([
        (ue.UserEventType.CONNECTION_FAILED, "bad key"),
        (ue.UserEventType.CONNECTION_CONNECTED, None),
        (ue.UserEventType.CONNECTION_LOST, "gone"),
    ]):
        ue.record(session, user_id=user.id, event_type=t, entity_id=conn.id, entity_name=conn.name, message=msg,
                  details="tail" if msg else None, dedupe_key=f"t:{i}")
    assert ue.record(session, user_id=user.id, event_type=ue.UserEventType.CONNECTION_LOST, entity_id=conn.id, dedupe_key="t:2") is None

    rows = ue.list_events(session, user.id)
    assert [r.type for r in rows] == ["connection.lost", "connection.connected", "connection.failed"]
    assert [r.type for r in ue.list_events(session, user.id, severity=["error", "warning"])] == ["connection.lost", "connection.failed"]
    assert [r.type for r in ue.list_events(session, user.id, since_id=rows[-1].id)] == ["connection.lost", "connection.connected"]
    assert [r.type for r in ue.list_events(session, user.id, before_id=rows[0].id, limit=1)] == ["connection.connected"]
    assert ue.get_event(session, user.id, rows[0].id).details == "tail"
    assert ue.get_event(session, user.id + 1000, rows[0].id) is None

    old = session.get(UserEvent, rows[-1].id)
    old.created_at = datetime.now(timezone.utc) - timedelta(days=ue.RETENTION_DAYS + 1)
    session.add(old)
    session.commit()
    assert ue.purge_old(session) == 1
    assert len(ue.list_events(session, user.id)) == 2
