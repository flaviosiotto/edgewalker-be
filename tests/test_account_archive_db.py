"""Accounts the gateway stops listing are archived, not deleted (migr. 078).

An account under a connection is one that can be traded through it now. The
connect-time sync used to add and update only, so an account closed at the
broker, or no longer the one a cTrader connection is connected on, stayed in
every picker and a live start failed at the gateway.
Run: ``venv/bin/python -m pytest tests/test_account_archive_db.py``.
"""
from __future__ import annotations

import pytest
from sqlmodel import Session, select

import app.models.connection  # noqa: E402,F401

pytest.importorskip("pgserver")


@pytest.fixture
def session(app_engine):
    with Session(app_engine, expire_on_commit=False) as s:
        yield s


@pytest.fixture
def connection(session):
    from app.models.connection import Connection
    from app.models.user import User

    email = "account-archive@example.com"
    stale = session.exec(select(User).where(User.email == email)).first()
    if stale is not None:
        session.delete(stale)
        session.commit()
    user = User(email=email, username="user_account_archive", hashed_password="x")
    session.add(user)
    session.commit()
    conn = Connection(user_id=user.id, name="FTMO", broker_type="ctrader", config={}, status="connected")
    session.add(conn)
    session.commit()
    yield conn
    # The embedded Postgres is shared: leave no connected row behind for the
    # tests that count them.
    session.rollback()
    session.delete(session.get(Connection, conn.id))
    session.commit()


def _discovered(account_id: str, login: str):
    from app.services.broker_connectors.base import DiscoveredAccount

    return DiscoveredAccount(account_id=account_id, display_name=login, account_type="live", currency="USD")


def _accounts(session, connection):
    from app.models.connection import Account

    session.expire_all()
    rows = session.exec(select(Account).where(Account.connection_id == connection.id)).all()
    return {row.account_id: row for row in rows}


def test_missing_account_is_archived_and_comes_back(session, connection):
    from app.models.connection import Account
    from app.services.connection_manager import _sync_accounts

    _sync_accounts(session, connection.id, [_discovered("47494755", "4255142"), _discovered("48013697", "17160282")])
    assert all(row.archived_at is None for row in _accounts(session, connection).values())

    # The onboarding placeholder is not a broker account: never archived.
    session.add(Account(
        connection_id=connection.id, account_id="onboarding", display_name="Conto da collegare",
        account_type="provisional", currency="USD", extra={"onboarding_provisional": True},
    ))
    session.commit()

    _sync_accounts(session, connection.id, [_discovered("49713817", "17258923")])
    rows = _accounts(session, connection)
    assert rows["49713817"].archived_at is None
    assert rows["47494755"].archived_at is not None and rows["48013697"].archived_at is not None
    assert rows["onboarding"].archived_at is None
    first_archival = rows["47494755"].archived_at

    # Still missing: the first archival moment is kept. Back: active again.
    _sync_accounts(session, connection.id, [_discovered("49713817", "17258923")])
    assert _accounts(session, connection)["47494755"].archived_at == first_archival
    _sync_accounts(session, connection.id, [_discovered("47494755", "4255142")])
    rows = _accounts(session, connection)
    assert rows["47494755"].archived_at is None and rows["49713817"].archived_at is not None


def test_archived_account_stops_a_live_start_with_the_login(session, connection):
    from app.services.connection_manager import _sync_accounts
    from app.services.live_trading_service import validate_account_for_live

    _sync_accounts(session, connection.id, [_discovered("47494755", "4255142")])
    _sync_accounts(session, connection.id, [_discovered("49713817", "17258923")])
    rows = _accounts(session, connection)

    account, _conn = validate_account_for_live(session, rows["49713817"].id, connection.user_id)
    assert account.account_id == "49713817"
    with pytest.raises(ValueError) as info:
        validate_account_for_live(session, rows["47494755"].id, connection.user_id)
    assert "Account 4255142 is no longer available on connection 'FTMO'" in str(info.value)
