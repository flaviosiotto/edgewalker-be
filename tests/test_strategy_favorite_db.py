"""``PUT /strategies/{id}/favorite`` (migr. 071): star/unstar is a UI
preference, so it must not bump ``updated_at`` nor leak to other users; the
list carries the flag and a copy starts unstarred.
Run: ``venv/bin/python -m pytest tests/test_strategy_favorite_db.py``.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi import HTTPException
from sqlmodel import Session, select

import app.models.agent_lesson  # noqa: E402,F401
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
    for email in ("fav@example.com", "fav-other@example.com"):
        stale = session.exec(select(User).where(User.email == email)).first()
        if stale is not None:
            session.delete(stale)
            session.commit()
    user = User(email="fav@example.com", username="user_fav", hashed_password="x")
    other = User(email="fav-other@example.com", username="user_fav_other", hashed_password="x")
    session.add(user)
    session.add(other)
    session.flush()
    conn = Connection(user_id=user.id, name="OKX demo", broker_type="okx", status="connected", created_at=now, updated_at=now)
    session.add(conn)
    session.flush()
    acc = Account(connection_id=conn.id, account_id="acc-fav", currency="USDT", created_at=now, updated_at=now)
    session.add(acc)
    session.commit()
    yield user, other, acc
    session.delete(user)
    session.delete(other)
    session.commit()


def _strategy(session, user, acc, name):
    from app.schemas.strategy import StrategyCreate
    from app.services.strategy_service import create_strategy

    definition = {"strategy": {
        "name": name, "symbol": "BTCUSDT", "asset": "crypto", "timeframe": "5m", "timezone": "UTC", "params": {},
        "indicators": [], "charts": [{"id": "main", "symbol": "BTCUSDT", "asset": "crypto", "timeframe": "5m", "indicators": []}], "rules": [],
    }}
    return create_strategy(session, StrategyCreate(name=name, definition=definition, account_id=acc.id), user.id)


def test_favorite_toggle_is_idempotent_and_does_not_touch_updated_at(session, tenant):
    from app.schemas.strategy import StrategyRead
    from app.services.strategy_service import list_strategies, set_strategy_favorite

    user, _other, acc = tenant
    s = _strategy(session, user, acc, "fav-a")
    assert s.is_favorite is False
    stamp = s.updated_at

    s = set_strategy_favorite(session, s.id, True, user.id)
    assert s.is_favorite is True
    assert s.updated_at == stamp
    assert s.updated_by is None

    # Idempotent: a second star is a no-op.
    s = set_strategy_favorite(session, s.id, True, user.id)
    assert s.is_favorite is True
    assert s.updated_at == stamp

    # The list carries the flag (the FE filters on it).
    listed = {x.id: StrategyRead.model_validate(x).is_favorite for x in list_strategies(session, user.id)}
    assert listed[s.id] is True

    s = set_strategy_favorite(session, s.id, False, user.id)
    assert s.is_favorite is False
    assert s.updated_at == stamp


def test_favorite_is_scoped_to_the_owner(session, tenant):
    from app.services.strategy_service import set_strategy_favorite

    user, other, acc = tenant
    s = _strategy(session, user, acc, "fav-b")
    with pytest.raises(HTTPException) as exc:
        set_strategy_favorite(session, s.id, True, other.id)
    assert exc.value.status_code == 404
    session.refresh(s)
    assert s.is_favorite is False


def test_copy_starts_unstarred(session, tenant):
    from app.services.strategy_service import copy_strategy, set_strategy_favorite

    user, _other, acc = tenant
    s = _strategy(session, user, acc, "fav-c")
    set_strategy_favorite(session, s.id, True, user.id)
    copied, _warnings = copy_strategy(session, s.id, user.id, target_account_id=acc.id)
    assert copied.id != s.id
    assert copied.is_favorite is False
