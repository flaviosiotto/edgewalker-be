"""Outbound webhooks against an embedded Postgres (migr. 067): subscriptions,
emit + dedupe, claim with SKIP LOCKED, attempt bookkeeping and auto-disable,
event sources handlers from real rows.
Run: ``venv/bin/python -m pytest tests/test_webhooks_db.py``.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest
from fastapi import HTTPException
from sqlmodel import Session, select

# The ORM registry configures every mapper at once: import the whole
# model tree, as the app startup does, before instantiating any row.
import app.models.agent_turn  # noqa: E402,F401
import app.models.connection  # noqa: E402,F401
import app.models.live_trading  # noqa: E402,F401
import app.models.strategy  # noqa: E402,F401
import app.models.strategy_template  # noqa: E402,F401
pytest.importorskip("pgserver")


@pytest.fixture
def session(app_engine):
    with Session(app_engine, expire_on_commit=False) as s:
        yield s


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "SECRETS_ENCRYPTION_KEY", None)
    monkeypatch.setattr(settings, "SECRET_KEY", "unit-test-secret-key-for-webhooks")


@pytest.fixture
def tenant(session):
    from app.models.connection import Account, Connection
    from app.models.user import User

    now = datetime.now(timezone.utc)
    stale = session.exec(select(User).where(User.email == "wh@example.com")).first()
    if stale is not None:
        session.delete(stale)
        session.commit()
    user = User(email="wh@example.com", username="user_wh", hashed_password="x")
    session.add(user)
    session.flush()
    conn = Connection(user_id=user.id, name="conn-wh", broker_type="binance", status="connected", created_at=now, updated_at=now)
    session.add(conn)
    session.flush()
    acc = Account(connection_id=conn.id, account_id="acc-wh", currency="USDT", created_at=now, updated_at=now)
    session.add(acc)
    session.commit()
    yield user, conn, acc
    session.delete(user)
    session.commit()


def _sub(session, user, events=("*",), **kw):
    from app.services.webhook_service import create_subscription

    return create_subscription(session, user_id=user.id, name=kw.pop("name", "hook"), url="https://example.com/hook", events=list(events), **kw)


def test_subscription_crud_and_secret_rotation(session, tenant):
    from app.services import webhook_service as ws

    user, _, _ = tenant
    row, secret = _sub(session, user, events=["backtest.completed"])
    assert secret.startswith("whsec_") and ws.decrypt_secret(row) == secret
    assert row.events == ["backtest.completed"] and row.active

    with pytest.raises(HTTPException) as exc:
        _sub(session, user, events=["nope"])
    assert exc.value.status_code == 400
    with pytest.raises(HTTPException):
        ws.create_subscription(session, user_id=user.id, name="x", url="ftp://x", events=["*"])

    updated, new_secret = ws.update_subscription(session, user_id=user.id, subscription_id=row.id, rotate_secret=True, active=False)
    assert new_secret and new_secret != secret and ws.decrypt_secret(updated) == new_secret and not updated.active

    ws.delete_subscription(session, user.id, row.id)
    assert ws.list_subscriptions(session, user.id) == []


def test_emit_matches_filters_and_dedupes(session, tenant):
    from app.models.webhook import WebhookDelivery
    from app.services import webhook_service as ws

    user, _, _ = tenant
    all_events, _ = _sub(session, user, name="all")
    only_bt, _ = _sub(session, user, name="bt", events=["backtest.completed"])
    off, _ = _sub(session, user, name="off", active=False)

    rows = ws.emit_event(session, user_id=user.id, event_type="live.status.changed", data={"live_id": 1}, dedupe_key="live:1:running", commit=True)
    assert {r.subscription_id for r in rows} == {all_events.id}
    again = ws.emit_event(session, user_id=user.id, event_type="live.status.changed", data={"live_id": 1}, dedupe_key="live:1:running", commit=True)
    assert again == []
    rows = ws.emit_event(session, user_id=user.id, event_type="backtest.completed", data={"backtest_id": 9}, commit=True)
    assert {r.subscription_id for r in rows} == {all_events.id, only_bt.id}
    assert all(r.payload["type"] == "backtest.completed" and r.status == "pending" for r in rows)
    assert ws.emit_event(session, user_id=user.id, event_type="unknown.event", data={}, commit=True) == []
    assert session.exec(select(WebhookDelivery).where(WebhookDelivery.subscription_id == off.id)).all() == []


def test_claim_and_record_attempts(session, tenant):
    from app.models.webhook import WebhookDelivery, WebhookSubscription
    from app.services import webhook_service as ws
    from app.services.webhook_dispatcher import claim_due_deliveries

    user, _, _ = tenant
    sub, _ = _sub(session, user)
    ping = ws.enqueue_ping(session, user.id, sub.id)
    later = ws.emit_event(session, user_id=user.id, event_type="backtest.failed", data={}, commit=True)[0]
    later.next_attempt_at = datetime.now(timezone.utc) + timedelta(hours=1)
    session.add(later)
    session.commit()

    claimed = claim_due_deliveries(session)
    assert claimed == [ping.id]
    assert claim_due_deliveries(session) == []  # already delivering
    session.refresh(ping)
    assert ping.status == "delivering"

    ws.record_attempt(session, ping, ok=False, status_code=503, error="busy")
    session.refresh(ping)
    assert ping.status == "pending" and ping.attempts == 1 and ping.next_attempt_at > datetime.now(timezone.utc)
    sub_row = session.get(WebhookSubscription, sub.id)
    assert sub_row.failure_streak == 1 and sub_row.last_failure_at is not None

    ws.record_attempt(session, ping, ok=True, status_code=200, error=None)
    session.refresh(ping)
    assert ping.status == "succeeded" and ping.delivered_at is not None
    session.refresh(sub_row)
    assert sub_row.failure_streak == 0 and sub_row.last_success_at is not None

    # exhaustion → failed; streak → auto-disable
    ws.AUTO_DISABLE_AFTER_FAILURES_backup = ws.AUTO_DISABLE_AFTER_FAILURES
    ws.AUTO_DISABLE_AFTER_FAILURES = 3
    try:
        for _ in range(ws.MAX_ATTEMPTS):
            ws.record_attempt(session, later, ok=False, status_code=500, error="x")
        session.refresh(later)
        assert later.status == "failed" and later.attempts == ws.MAX_ATTEMPTS
        session.refresh(sub_row)
        assert not sub_row.active and "Disabled after" in (sub_row.disabled_reason or "")
    finally:
        ws.AUTO_DISABLE_AFTER_FAILURES = ws.AUTO_DISABLE_AFTER_FAILURES_backup
    assert session.exec(select(WebhookDelivery).where(WebhookDelivery.id == later.id)).one().status == "failed"


def test_sources_from_rows(session, tenant):
    from app.models.live_trading import LiveTrade
    from app.models.strategy import BacktestResult, LiveAlert, Strategy, StrategyLive
    from app.schemas.strategy import StrategyCreate
    from app.services import webhook_sources as src
    from app.services.strategy_service import create_strategy

    user, conn, acc = tenant
    sub, _ = _sub(session, user)
    definition = {"strategy": {
        "name": "wh", "symbol": "BTCUSDT", "asset": "crypto", "timeframe": "5m", "timezone": "UTC", "params": {},
        "indicators": [], "charts": [{"id": "main", "symbol": "BTCUSDT", "asset": "crypto", "timeframe": "5m", "indicators": []}], "rules": [],
    }}
    strategy = create_strategy(session, StrategyCreate(name="WH", definition=definition, account_id=acc.id), user.id)

    # backtest completed
    bt = BacktestResult(strategy_id=strategy.id, symbol="BTCUSDT", start_date=date(2026, 1, 1), end_date=date(2026, 2, 1),
                        status="completed", completed_at=datetime.now(timezone.utc), metrics={"net_pnl": 12.5, "ledger": [1, 2]})
    session.add(bt)
    session.commit()
    assert src.handle_notification({"table": "strategy_backtests", "op": "UPDATE", "id": bt.id}) == 1
    assert src.handle_notification({"table": "strategy_backtests", "op": "UPDATE", "id": bt.id}) == 0  # dedupe
    assert src.handle_notification({"table": "strategy_backtests", "op": "UPDATE", "id": 999999}) == 0
    assert src.handle_notification({"table": "nope", "id": 1}) == 0

    # live status + alert
    live = StrategyLive(strategy_id=strategy.id, status="running", symbol="BTCUSDT", timeframe="5s", account_id=acc.id, connection_id=conn.id)
    session.add(live)
    session.commit()
    assert src.handle_notification({"table": "strategy_live", "op": "UPDATE", "id": live.id}) == 1
    alert = LiveAlert(strategy_live_id=live.id, name="break 60k", trigger_type="price_level", trigger={"levels": [60000]},
                      last_triggered_at=datetime.now(timezone.utc), trigger_count=1, last_triggered_price=60001.0)
    session.add(alert)
    session.commit()
    assert src.handle_notification({"table": "live_alert", "op": "UPDATE", "id": alert.id}) == 1

    # trades: fresh counts, stale does not, re-materialised duplicate deduped
    # (the trades table is migration-managed: create_all skips it)
    LiveTrade.__table__.create(session.get_bind(), checkfirst=True)
    fresh = LiveTrade(strategy_live_id=live.id, account_id=acc.id, symbol="BTCUSDT", direction="long", quantity=0.1,
                      entry_price=59000.0, exit_price=60000.0, realized_pnl=100.0, exit_fill_id=77,
                      exit_time=datetime.now(timezone.utc) - timedelta(minutes=1))
    stale = LiveTrade(strategy_live_id=live.id, account_id=acc.id, symbol="BTCUSDT", direction="short", quantity=0.1,
                      entry_price=59000.0, exit_price=60000.0, realized_pnl=-100.0, exit_fill_id=12,
                      exit_time=datetime.now(timezone.utc) - timedelta(days=3))
    session.add(fresh)
    session.add(stale)
    session.commit()
    assert src.handle_notification({"table": "trades", "op": "INSERT", "id": fresh.id}) == 1
    assert src.handle_notification({"table": "trades", "op": "INSERT", "id": stale.id}) == 0
    rematerialised = LiveTrade(strategy_live_id=live.id, account_id=acc.id, symbol="BTCUSDT", direction="long", quantity=0.1,
                               entry_price=59000.0, exit_price=60000.0, realized_pnl=100.0, exit_fill_id=77,
                               exit_time=fresh.exit_time)
    session.add(rematerialised)
    session.commit()
    assert src.handle_notification({"table": "trades", "op": "INSERT", "id": rematerialised.id}) == 0

    # connection: only alarming statuses
    assert src.handle_notification({"table": "connections", "op": "UPDATE", "id": conn.id}) == 0
    conn.status = "disconnected"
    session.add(conn)
    session.commit()
    assert src.handle_notification({"table": "connections", "op": "UPDATE", "id": conn.id}) == 1

    from app.models.webhook import WebhookDelivery

    types = sorted(r.event_type for r in session.exec(select(WebhookDelivery).where(WebhookDelivery.subscription_id == sub.id)).all())
    assert types == ["backtest.completed", "connection.stale", "live.alert.triggered", "live.status.changed", "live.trade.closed"]
    payload = next(r.payload for r in session.exec(select(WebhookDelivery)).all() if r.event_type == "backtest.completed")
    assert payload["data"]["metrics"] == {"net_pnl": 12.5}  # ledger stripped
