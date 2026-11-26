"""Watchlist (migr. 079): add is idempotent, items are the owner's only, a
connection of another user is refused, deleting the connection deletes its
items. Run: ``venv/bin/python -m pytest tests/test_watchlist_db.py``.
"""
from __future__ import annotations

from datetime import datetime, timezone

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
    from app.models.connection import Connection
    from app.models.user import User

    now = datetime.now(timezone.utc)
    users = []
    for tag in ("wl", "wl-other"):
        stale = session.exec(select(User).where(User.email == f"{tag}@example.com")).first()
        if stale is not None:
            session.delete(stale)
            session.commit()
        user = User(email=f"{tag}@example.com", username=f"user_{tag}", hashed_password="x")
        session.add(user)
        session.flush()
        conn = Connection(user_id=user.id, name=f"conn-{tag}", broker_type="okx", status="connected", created_at=now, updated_at=now)
        session.add(conn)
        session.flush()
        users.append((user, conn))
    session.commit()
    yield users
    for user, _ in users:
        session.delete(user)
    session.commit()


def _payload(conn, symbol="BTC-USDT-SWAP"):
    from app.schemas.watchlist import WatchlistItemCreate

    return WatchlistItemCreate(connection_id=conn.id, symbol=f"  {symbol} ", name="BTC", asset_type="crypto", info={"tick_size": 0.1})


def test_add_is_idempotent_and_keeps_the_broker_spelling(session, tenant):
    from app.services.watchlist_service import add_item, list_items

    (user, conn), _ = tenant
    item, created = add_item(session, _payload(conn, "US500.cash"), user.id)
    again, created_again = add_item(session, _payload(conn, "US500.cash"), user.id)
    assert created is True and created_again is False
    assert again.id == item.id
    assert item.symbol == "US500.cash" and item.info == {"tick_size": 0.1}
    assert [i.symbol for i in list_items(session, user.id)] == ["US500.cash"]


def test_items_are_private_and_foreign_connections_are_refused(session, tenant):
    from app.services.watchlist_service import add_item, list_items, remove_item

    (user, conn), (other, other_conn) = tenant
    item, _ = add_item(session, _payload(conn), user.id)

    assert list_items(session, other.id) == []
    with pytest.raises(HTTPException) as refused:
        add_item(session, _payload(conn), other.id)
    assert refused.value.status_code == 404
    with pytest.raises(HTTPException) as hidden:
        remove_item(session, item.id, other.id)
    assert hidden.value.status_code == 404

    remove_item(session, item.id, user.id)
    assert list_items(session, user.id) == []


def test_list_filters_by_connection_and_connection_delete_cascades(session, tenant):
    from app.models.connection import Connection
    from app.services.watchlist_service import add_item, list_items

    (user, conn), _ = tenant
    now = datetime.now(timezone.utc)
    second = Connection(user_id=user.id, name="conn-wl-2", broker_type="ctrader", status="connected", created_at=now, updated_at=now)
    session.add(second)
    session.commit()
    add_item(session, _payload(conn, "BTC-USDT-SWAP"), user.id)
    add_item(session, _payload(second, "US500.cash"), user.id)

    assert [i.symbol for i in list_items(session, user.id, second.id)] == ["US500.cash"]
    session.delete(second)
    session.commit()
    session.expire_all()
    assert [i.symbol for i in list_items(session, user.id)] == ["BTC-USDT-SWAP"]
