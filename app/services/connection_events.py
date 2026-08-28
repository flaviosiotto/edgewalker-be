"""Push notifications for connection/account changes (Redis pub/sub → SSE).

Connection status is written from many places (explicit connect/disconnect,
the health loop, the TWS auth reconcile loop, the startup reset), so instead
of instrumenting each write site the ORM session itself is the choke point:
a before_flush hook records which users' connections are about to change
(status, message, name, is_active, creation/deletion, account discovery)
and after_commit publishes one ``connections_changed`` message per user on
``connection-events:{user_id}``.  The FE relays it through
``GET /connections/events`` and refetches its connection list.

Best-effort by design: Redis down just means no live notification (the FE
keeps a slow fallback poll).
"""
from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import event, inspect as sa_inspect
from sqlalchemy.orm import Session as SASession

from app.models.connection import Account, Connection
from app.utils.redis_client import get_redis

logger = logging.getLogger(__name__)

CONNECTION_EVENTS_CHANNEL_PREFIX = "connection-events:"
CONNECTION_EVENT_HEARTBEAT_SECONDS = 15.0

# Connection attributes whose change is worth a notification. last_checked_at /
# last_ok_at are bumped by every health-loop probe and are deliberately excluded.
_CONNECTION_WATCHED_ATTRS = ("status", "status_message", "name", "is_active", "config")

_INFO_KEY = "connection_events.pending"
_installed = False


def _pending(session: SASession) -> dict[int, set[int]]:
    return session.info.setdefault(_INFO_KEY, {})


def _mark(session: SASession, user_id: int | None, connection_id: int | None) -> None:
    if user_id is None:
        return
    bucket = _pending(session).setdefault(int(user_id), set())
    if connection_id is not None:
        bucket.add(int(connection_id))


def _connection_changed(conn: Connection) -> bool:
    state = sa_inspect(conn)
    for attr in _CONNECTION_WATCHED_ATTRS:
        if state.attrs[attr].history.has_changes():
            return True
    return False


def _user_id_for_account(session: SASession, account: Account) -> int | None:
    conn = session.get(Connection, account.connection_id)
    return conn.user_id if conn is not None else None


def _collect_pending(session: SASession, flush_context: Any, instances: Any) -> None:
    for obj in list(session.new) + list(session.deleted):
        if isinstance(obj, Connection):
            _mark(session, obj.user_id, obj.id)
        elif isinstance(obj, Account):
            _mark(session, _user_id_for_account(session, obj), obj.connection_id)
    for obj in session.dirty:
        if isinstance(obj, Connection) and session.is_modified(obj) and _connection_changed(obj):
            _mark(session, obj.user_id, obj.id)


def _publish_pending(session: SASession) -> None:
    pending = session.info.pop(_INFO_KEY, None)
    if not pending:
        return
    client = get_redis()
    if client is None:
        return
    for user_id, connection_ids in pending.items():
        payload = {
            "v": 1,
            "type": "connections_changed",
            "connection_ids": sorted(connection_ids),
        }
        try:
            client.publish(
                f"{CONNECTION_EVENTS_CHANNEL_PREFIX}{user_id}",
                json.dumps(payload, default=str),
            )
        except Exception as exc:  # noqa: BLE001 - never fail the commit path on Redis errors
            logger.debug("connection-events publish failed (non-critical): %s", exc)


def _discard_pending(session: SASession) -> None:
    session.info.pop(_INFO_KEY, None)


def install_connection_event_listeners() -> None:
    """Register the session hooks once per process (idempotent)."""
    global _installed
    if _installed:
        return
    event.listen(SASession, "before_flush", _collect_pending)
    event.listen(SASession, "after_commit", _publish_pending)
    event.listen(SASession, "after_rollback", _discard_pending)
    _installed = True
