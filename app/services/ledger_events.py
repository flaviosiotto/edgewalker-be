"""Ledger push: ``events:projection`` → per-user SSE ``ledger_changed``.

The order-aggregator is the only writer of orders/fills/positions/trades and
publishes ``projection.ledger.applied`` on the Redis stream ``events:projection``
*after* each commit. Until now the live workspace learned about a fill only
through its 5 s fallback poll (6 GET, ~1 MB per round): a fill written in the
DB within 1 s of the broker execution could show up on the chart 5-10 s
later. This relay turns the post-commit event into a ``ledger_changed``
message on ``connection-events:{user_id}`` (the one SSE stream the FE already
holds per session, see ``connection_events``), carrying only *which* account
changed: the client refetches, so the push never serves a stale snapshot.

``projection.position.applied`` (fills + position snapshots, consumed by the
strategy-runner) is accepted too, so the relay works even before the
aggregator is redeployed with the new event; only order submit/cancel
notifications need the new aggregator.

Best-effort by design: Redis down or an unresolved account just means no
push, the FE keeps its (slower) fallback poll.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from sqlmodel import select

from app.services.connection_events import CONNECTION_EVENTS_CHANNEL_PREFIX
from app.utils.redis_async import get_async_redis

logger = logging.getLogger(__name__)

PROJECTION_STREAM = "events:projection"
CONSUMER_GROUP = "cg:backend:ledger-events"
LEDGER_PROJECTED_EVENT = "projection.ledger.applied"
POSITION_PROJECTED_EVENT = "projection.position.applied"
LEDGER_CHANGED_TYPE = "ledger_changed"

READ_COUNT = 200
READ_BLOCK_MS = 1000
RESOLVE_TTL_S = 300.0
NEGATIVE_TTL_S = 30.0
ACCEPTED_EVENTS = frozenset({LEDGER_PROJECTED_EVENT, POSITION_PROJECTED_EVENT})
_KINDS = ("order", "fill", "position")


@dataclass(frozen=True, slots=True)
class LedgerEvent:
    """One accepted stream entry, before the account is resolved."""

    kind: str
    account_id: int | None
    broker_account: str
    connection_id: int | None
    symbol: str


@dataclass(slots=True)
class LedgerChange:
    """What the FE receives: one message per (user, account) per batch."""

    user_id: int
    account_id: int
    symbols: set[str] = field(default_factory=set)
    kinds: set[str] = field(default_factory=set)

    def payload(self) -> dict[str, Any]:
        return {
            "v": 1,
            "type": LEDGER_CHANGED_TYPE,
            "account_id": self.account_id,
            "symbols": sorted(self.symbols),
            "kinds": sorted(self.kinds),
        }


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_ledger_event(fields: dict[str, Any]) -> LedgerEvent | None:
    """Decode one ``events:projection`` entry; ``None`` for foreign/invalid ones."""
    event_type = str(fields.get("event_type") or "")
    if event_type not in ACCEPTED_EVENTS:
        return None
    raw_payload = fields.get("payload") or "{}"
    try:
        payload = json.loads(raw_payload) if isinstance(raw_payload, str) else dict(raw_payload)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if event_type == LEDGER_PROJECTED_EVENT:
        kind = str(payload.get("kind") or "")
        if kind not in _KINDS:
            return None
    else:
        # projection.position.applied: fills and snapshots both move positions.
        kind = "fill" if str(payload.get("source_event") or "").endswith("fill.sync") else "position"
    connection_id = _coerce_int(payload.get("connection_id"))
    if connection_id is None:
        connection_id = _coerce_int(fields.get("connection_id"))
    return LedgerEvent(
        kind=kind,
        account_id=_coerce_int(payload.get("account_id")),
        broker_account=str(payload.get("broker_account") or "").strip(),
        connection_id=connection_id,
        symbol=str(payload.get("symbol") or "").strip().upper(),
    )


def coalesce_changes(resolved: Iterable[tuple[LedgerEvent, tuple[int, int]]]) -> list[LedgerChange]:
    """Merge events of one read batch into one message per (user, account).

    A fill produces an order event, a fill event and a positions snapshot in
    a row; the FE refetches the whole ledger anyway, so one push is enough.
    """
    merged: dict[tuple[int, int], LedgerChange] = {}
    for event, (account_id, user_id) in resolved:
        change = merged.get((user_id, account_id))
        if change is None:
            change = LedgerChange(user_id=user_id, account_id=account_id)
            merged[(user_id, account_id)] = change
        if event.symbol:
            change.symbols.add(event.symbol)
        change.kinds.add(event.kind)
    return list(merged.values())


class AccountResolver:
    """``LedgerEvent`` → ``(Account.id, Connection.user_id)`` with a small TTL cache.

    Keyed by internal account id when the event carries it, otherwise by
    (connection id, broker account) like the aggregator's own resolver.
    Lookups run in a worker thread: the backend ORM is synchronous.
    """

    def __init__(self, *, ttl_s: float = RESOLVE_TTL_S, negative_ttl_s: float = NEGATIVE_TTL_S) -> None:
        self._ttl_s = ttl_s
        self._negative_ttl_s = negative_ttl_s
        self._cache: dict[Any, tuple[float, tuple[int, int] | None]] = {}

    @staticmethod
    def _key(event: LedgerEvent) -> Any:
        if event.account_id is not None:
            return ("account", event.account_id)
        if event.broker_account and event.connection_id is not None:
            return ("broker", event.connection_id, event.broker_account)
        return None

    async def resolve(self, event: LedgerEvent) -> tuple[int, int] | None:
        key = self._key(event)
        if key is None:
            return None
        now = time.monotonic()
        cached = self._cache.get(key)
        if cached is not None and cached[0] > now:
            return cached[1]
        result = await asyncio.to_thread(self.lookup, event)
        ttl = self._ttl_s if result is not None else self._negative_ttl_s
        self._cache[key] = (now + ttl, result)
        if len(self._cache) > 5000:
            self._cache = {k: v for k, v in self._cache.items() if v[0] > now}
        return result

    @staticmethod
    def lookup(event: LedgerEvent) -> tuple[int, int] | None:
        from app.db.database import get_session_context
        from app.models.connection import Account, Connection

        with get_session_context() as session:
            stmt = (
                select(Account.id, Connection.user_id)
                .join(Connection, Connection.id == Account.connection_id)  # type: ignore[arg-type]
            )
            if event.account_id is not None:
                stmt = stmt.where(Account.id == event.account_id)
            else:
                stmt = stmt.where(
                    Account.connection_id == event.connection_id,
                    Account.account_id == event.broker_account,
                )
            row = session.exec(stmt.order_by(Account.id.desc())).first()  # type: ignore[union-attr]
            if row is None:
                return None
            account_id, user_id = row
            if account_id is None or user_id is None:
                return None
            return int(account_id), int(user_id)


def _consumer_name() -> str:
    return f"{socket.gethostname()}-{os.getpid()}"


async def publish_changes(redis_client: Any, changes: Iterable[LedgerChange]) -> int:
    published = 0
    for change in changes:
        try:
            await redis_client.publish(
                f"{CONNECTION_EVENTS_CHANNEL_PREFIX}{change.user_id}",
                json.dumps(change.payload(), default=str),
            )
            published += 1
        except Exception as exc:  # noqa: BLE001 - notification only
            logger.debug("ledger_changed publish failed (non-critical): %s", exc)
    return published


async def _ensure_group(redis_client: Any) -> None:
    try:
        await redis_client.xgroup_create(PROJECTION_STREAM, CONSUMER_GROUP, id="$", mkstream=True)
        logger.info("Created consumer group %s on %s", CONSUMER_GROUP, PROJECTION_STREAM)
    except Exception as exc:  # noqa: BLE001
        if "BUSYGROUP" not in str(exc):
            raise


def _decode_fields(fields: dict[Any, Any]) -> dict[str, Any]:
    return {
        (k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
        for k, v in fields.items()
    }


async def relay_once(redis_client: Any, resolver: AccountResolver, consumer: str) -> int:
    """Read one batch, resolve, coalesce, publish. Returns messages published."""
    results = await redis_client.xreadgroup(
        CONSUMER_GROUP,
        consumer,
        {PROJECTION_STREAM: ">"},
        count=READ_COUNT,
        block=READ_BLOCK_MS,
        noack=True,
    )
    if not results:
        return 0
    resolved: list[tuple[LedgerEvent, tuple[int, int]]] = []
    for _stream, messages in results:
        for _msg_id, fields in messages:
            event = parse_ledger_event(_decode_fields(fields))
            if event is None:
                continue
            target = await resolver.resolve(event)
            if target is None:
                logger.debug("ledger event without a resolvable account: %s", event)
                continue
            resolved.append((event, target))
    if not resolved:
        return 0
    return await publish_changes(redis_client, coalesce_changes(resolved))


async def ledger_relay_loop(stop: asyncio.Event) -> None:
    """The lifespan task. Never raises: a bad tick is logged and retried."""
    resolver = AccountResolver()
    consumer = _consumer_name()
    group_ready = False
    logger.info("ledger events relay started (stream %s, group %s)", PROJECTION_STREAM, CONSUMER_GROUP)
    while not stop.is_set():
        redis_client = get_async_redis()
        if redis_client is None:
            await _sleep(stop, 5.0)
            continue
        try:
            if not group_ready:
                await _ensure_group(redis_client)
                group_ready = True
            await relay_once(redis_client, resolver, consumer)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - keep the loop alive
            group_ready = False
            logger.warning("ledger events relay tick failed: %s", exc)
            await _sleep(stop, 2.0)
    logger.info("ledger events relay stopped")


async def _sleep(stop: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass
