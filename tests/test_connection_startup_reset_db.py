"""Startup reconcile of broker connections: probe first, then reset only the
connected/degraded rows the probe did not stamp. A gateway that survived the
backend redeploy keeps its status (no false "connection lost"/"connected"
pair at every deploy); an unverified one is reset.
Run: ``venv/bin/python -m pytest tests/test_connection_startup_reset_db.py``.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import Session, select

import app.models.connection  # noqa: E402,F401
import app.models.user_event  # noqa: E402,F401

pytest.importorskip("pgserver")


@pytest.fixture
def session(app_engine):
    with Session(app_engine, expire_on_commit=False) as s:
        yield s


@pytest.fixture
def owner(session):
    from app.models.user import User

    email = "startup-reset@example.com"
    stale = session.exec(select(User).where(User.email == email)).first()
    if stale is not None:
        session.delete(stale)
        session.commit()
    user = User(email=email, username="user_startup_reset", hashed_password="x")
    session.add(user)
    session.commit()
    return user


def _connection(session, owner, name, status, checked_at):
    from app.models.connection import Connection

    conn = Connection(
        user_id=owner.id,
        name=name,
        broker_type="binance",
        config={},
        status=status,
        last_checked_at=checked_at,
    )
    session.add(conn)
    session.commit()
    return conn


def test_only_unverified_rows_are_reset(session, owner):
    from app.models.connection import Connection
    from app.services.connection_manager import reset_unverified_connections

    since = datetime.now(timezone.utc)
    before = since - timedelta(minutes=5)
    after = since + timedelta(milliseconds=200)

    probed = _connection(session, owner, "probed", "connected", after)
    degraded_probed = _connection(session, owner, "degraded-probed", "degraded", after)
    stale = _connection(session, owner, "stale", "connected", before)
    never = _connection(session, owner, "never", "degraded", None)
    idle = _connection(session, owner, "idle", "disconnected", before)
    pending = _connection(session, owner, "pending", "awaiting_auth", before)

    assert reset_unverified_connections(since) == 2

    # The helper writes through its own session: drop the identity map first.
    session.expire_all()
    rows = {c.id: c for c in session.exec(select(Connection)).all()}
    assert rows[probed.id].status == "connected"
    assert rows[probed.id].status_message is None
    assert rows[degraded_probed.id].status == "degraded"
    for conn in (stale, never):
        row = rows[conn.id]
        assert row.status == "disconnected"
        assert row.status_message == "Reset on server restart"
        assert row.last_checked_at is not None and row.last_checked_at >= since
    assert rows[idle.id].status == "disconnected"
    assert rows[pending.id].status == "awaiting_auth"


def test_naive_since_is_read_as_utc(session, owner):
    from app.models.connection import Connection
    from app.services.connection_manager import reset_unverified_connections

    now = datetime.now(timezone.utc)
    fresh = _connection(session, owner, "fresh", "connected", now + timedelta(seconds=1))
    assert reset_unverified_connections(now.replace(tzinfo=None)) == 0
    session.expire_all()
    assert session.get(Connection, fresh.id).status == "connected"
