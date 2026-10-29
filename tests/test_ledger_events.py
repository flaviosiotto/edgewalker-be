"""Ledger push relay (``events:projection`` → ``ledger_changed`` SSE).

Pure parts (parsing, coalescing, publish fan-out) run without a database;
``test_resolver_*`` need the embedded Postgres (``pip install pgserver``).
Run: ``venv/bin/python -m pytest tests/test_ledger_events.py``.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest

from app.services import ledger_events as le


def _fields(event_type: str, payload: dict, connection_id: str = "4") -> dict[str, str]:
    return {
        "event_id": "e1",
        "event_type": event_type,
        "occurred_at": "1",
        "request_id": "r",
        "connection_id": connection_id,
        "payload": json.dumps(payload),
    }


def test_parse_ledger_event_reads_kind_account_and_symbol():
    event = le.parse_ledger_event(_fields(le.LEDGER_PROJECTED_EVENT, {
        "kind": "fill", "account_id": 2, "symbol": "mnqz6",
        "broker_account": "U16002214", "connection_id": "4",
        "source_event": "broker.fill.sync",
    }))
    assert event == le.LedgerEvent(kind="fill", account_id=2, broker_account="U16002214",
                                   connection_id=4, symbol="MNQZ6")


def test_parse_position_applied_is_accepted_without_account_id():
    """The old aggregator event (no account_id) still drives the push."""
    event = le.parse_ledger_event(_fields(le.POSITION_PROJECTED_EVENT, {
        "symbol": "MNQZ6", "broker_account": "U16002214", "connection_id": 4,
        "source_event": "broker.positions.snapshot",
    }))
    assert event is not None
    assert event.kind == "position" and event.account_id is None and event.connection_id == 4


def test_parse_rejects_foreign_events_and_bad_payloads():
    assert le.parse_ledger_event(_fields("projection.something.else", {"kind": "fill"})) is None
    assert le.parse_ledger_event(_fields(le.LEDGER_PROJECTED_EVENT, {"kind": "bogus"})) is None
    bad = _fields(le.LEDGER_PROJECTED_EVENT, {})
    bad["payload"] = "{not json"
    assert le.parse_ledger_event(bad) is None


def test_coalesce_merges_one_batch_into_one_message_per_account():
    ev = lambda kind, sym: le.LedgerEvent(kind=kind, account_id=2, broker_account="", connection_id=4, symbol=sym)  # noqa: E731
    changes = le.coalesce_changes([
        (ev("order", "MNQZ6"), (2, 7)),
        (ev("fill", "MNQZ6"), (2, 7)),
        (ev("position", "MNQZ6"), (2, 7)),
        (ev("fill", "ESZ6"), (3, 7)),
        (ev("fill", "ESZ6"), (3, 9)),
    ])
    by_key = {(c.user_id, c.account_id): c for c in changes}
    assert set(by_key) == {(7, 2), (7, 3), (9, 3)}
    assert by_key[(7, 2)].payload() == {
        "v": 1, "type": "ledger_changed", "account_id": 2,
        "symbols": ["MNQZ6"], "kinds": ["fill", "order", "position"],
    }


class _FakeRedis:
    def __init__(self, batches):
        self._batches = list(batches)
        self.published: list[tuple[str, dict]] = []

    async def xreadgroup(self, group, consumer, streams, count, block, noack):
        assert noack is True and streams == {le.PROJECTION_STREAM: ">"}
        return self._batches.pop(0) if self._batches else []

    async def publish(self, channel, data):
        self.published.append((channel, json.loads(data)))


class _StaticResolver(le.AccountResolver):
    def __init__(self, mapping):
        super().__init__()
        self._mapping = mapping

    async def resolve(self, event):
        key = event.account_id if event.account_id is not None else (event.connection_id, event.broker_account)
        return self._mapping.get(key)


def test_relay_once_publishes_on_the_user_channel_bytes_fields_included():
    batch = [(b"events:projection", [
        (b"1-0", {k.encode(): v.encode() for k, v in _fields(le.LEDGER_PROJECTED_EVENT, {
            "kind": "fill", "account_id": 2, "symbol": "MNQZ6", "broker_account": "U1", "connection_id": 4,
        }).items()}),
        (b"2-0", _fields(le.POSITION_PROJECTED_EVENT, {
            "symbol": "MNQZ6", "broker_account": "U1", "connection_id": 4, "source_event": "broker.fill.sync",
        })),
        (b"3-0", _fields(le.LEDGER_PROJECTED_EVENT, {"kind": "order", "account_id": 99, "symbol": "X"})),
    ])]
    redis = _FakeRedis([batch, []])
    resolver = _StaticResolver({2: (2, 7), (4, "U1"): (2, 7)})

    published = asyncio.run(le.relay_once(redis, resolver, "c1"))

    assert published == 1
    assert redis.published == [(f"{le.CONNECTION_EVENTS_CHANNEL_PREFIX}7", {
        "v": 1, "type": "ledger_changed", "account_id": 2, "symbols": ["MNQZ6"], "kinds": ["fill"],
    })]
    assert asyncio.run(le.relay_once(redis, resolver, "c1")) == 0


# ── resolver against the embedded Postgres ──────────────────────────

pytest.importorskip("pgserver")


@pytest.fixture
def tenant(app_engine):
    from sqlmodel import Session, select

    import app.models.agent_turn  # noqa: F401
    import app.models.live_trading  # noqa: F401
    import app.models.strategy  # noqa: F401
    import app.models.strategy_template  # noqa: F401
    import app.models.user_event  # noqa: F401
    from app.models.connection import Account, Connection
    from app.models.user import User

    with Session(app_engine, expire_on_commit=False) as session:
        now = datetime.now(timezone.utc)
        stale = session.exec(select(User).where(User.email == "ledger@example.com")).first()
        if stale is not None:
            session.delete(stale)
            session.commit()
        user = User(email="ledger@example.com", username="user_ledger", hashed_password="x")
        session.add(user)
        session.flush()
        conn = Connection(user_id=user.id, name="IBKR ledger", broker_type="ibkr", status="connected",
                          created_at=now, updated_at=now)
        session.add(conn)
        session.flush()
        account = Account(connection_id=conn.id, account_id="U16002214", display_name="U16002214",
                          account_type="live", currency="EUR", created_at=now, updated_at=now)
        session.add(account)
        session.commit()
        yield {"user_id": user.id, "connection_id": conn.id, "account_id": account.id}


def test_resolver_by_internal_account_id_and_by_broker_account(tenant, app_engine, monkeypatch):
    import app.db.database as database

    monkeypatch.setattr(database, "engine", app_engine)
    resolver = le.AccountResolver()
    by_id = le.LedgerEvent(kind="fill", account_id=tenant["account_id"], broker_account="",
                           connection_id=None, symbol="MNQZ6")
    by_broker = le.LedgerEvent(kind="position", account_id=None, broker_account="U16002214",
                               connection_id=tenant["connection_id"], symbol="MNQZ6")
    unknown = le.LedgerEvent(kind="fill", account_id=None, broker_account="NOPE",
                             connection_id=tenant["connection_id"], symbol="MNQZ6")
    orphan = le.LedgerEvent(kind="fill", account_id=None, broker_account="", connection_id=None, symbol="")

    expected = (tenant["account_id"], tenant["user_id"])
    assert asyncio.run(resolver.resolve(by_id)) == expected
    assert asyncio.run(resolver.resolve(by_broker)) == expected
    assert asyncio.run(resolver.resolve(unknown)) is None
    assert asyncio.run(resolver.resolve(orphan)) is None
    # Second call is served by the cache (no DB): same answer.
    assert asyncio.run(resolver.resolve(by_id)) == expected
