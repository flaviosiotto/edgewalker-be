"""User-facing outcomes of long-running tasks (migr. 070).

The closed catalogue (:class:`UserEventType`), the pure mapping from a
status transition to a catalogue type (:func:`classify`), and
:func:`record`, which inserts the row and pokes the user's push channel.

Design (docs/valutazione-notifiche-task.md): the webhook event sources
already observe the state transitions that matter (NOTIFY triggers of migr.
067). They call :func:`record` *before* ``emit_event``, so the outcome is
kept for every user, not only for the ones with a webhook subscription.
Only transitions to a final state (or the fall of a healthy process) are
events; intermediate states and progress never are.

The push reuses the existing per-user channel ``connection-events:{user_id}``
with a second message type (``user_event``): the FE already holds one SSE
stream on it and refetches on either message.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

from sqlalchemy import text
from sqlmodel import Session, select

from app.models.user_event import UserEvent
from app.services.connection_events import CONNECTION_EVENTS_CHANNEL_PREFIX
from app.utils.redis_client import get_redis

logger = logging.getLogger(__name__)

RETENTION_DAYS = 90


class UserEventType(str, Enum):
    CONNECTION_CONNECTED = "connection.connected"
    CONNECTION_FAILED = "connection.failed"
    CONNECTION_LOST = "connection.lost"
    BACKTEST_COMPLETED = "backtest.completed"
    BACKTEST_FAILED = "backtest.failed"
    LIVE_STARTED = "live.started"
    LIVE_FAILED = "live.failed"
    LIVE_STOPPED = "live.stopped"
    STUDIO_RUN_SUCCEEDED = "studio.run.succeeded"
    STUDIO_RUN_FAILED = "studio.run.failed"


SEVERITY: dict[UserEventType, str] = {
    UserEventType.CONNECTION_CONNECTED: "success",
    UserEventType.CONNECTION_FAILED: "error",
    UserEventType.CONNECTION_LOST: "warning",
    UserEventType.BACKTEST_COMPLETED: "success",
    UserEventType.BACKTEST_FAILED: "error",
    UserEventType.LIVE_STARTED: "success",
    UserEventType.LIVE_FAILED: "error",
    UserEventType.LIVE_STOPPED: "info",
    UserEventType.STUDIO_RUN_SUCCEEDED: "success",
    UserEventType.STUDIO_RUN_FAILED: "error",
}

ENTITY_OF: dict[UserEventType, str] = {
    UserEventType.CONNECTION_CONNECTED: "connection",
    UserEventType.CONNECTION_FAILED: "connection",
    UserEventType.CONNECTION_LOST: "connection",
    UserEventType.BACKTEST_COMPLETED: "backtest",
    UserEventType.BACKTEST_FAILED: "backtest",
    UserEventType.LIVE_STARTED: "live",
    UserEventType.LIVE_FAILED: "live",
    UserEventType.LIVE_STOPPED: "live",
    UserEventType.STUDIO_RUN_SUCCEEDED: "studio_run",
    UserEventType.STUDIO_RUN_FAILED: "studio_run",
}


def catalog() -> list[dict[str, str]]:
    return [{"type": t.value, "severity": SEVERITY[t], "entity_type": ENTITY_OF[t]} for t in UserEventType]


# ── classification: (table, old status, new status) → type | None ──────────
#
# ``old_status`` comes from the NOTIFY payload (``to_jsonb(OLD)->>'status'``,
# migr. 070). It is None on INSERT, on payloads from the 067 trigger function
# and in tests: then the new status alone decides, which is right for every
# final state and only loses the "was it healthy before" nuance of a
# connection.

_CONNECTION_HEALTHY = {"connected", "degraded"}
_CONNECTION_ATTEMPTING = {"connecting", "awaiting_auth"}
_CONNECTION_DOWN = {"disconnected", "error", "degraded"}


def _norm(value: Optional[str]) -> Optional[str]:
    return str(value).strip().lower() if value else None


def classify(table: str, old_status: Optional[str], new_status: Optional[str]) -> Optional[UserEventType]:
    old, new = _norm(old_status), _norm(new_status)
    if new is None or old == new:
        return None
    if table == "connections":
        if new == "connected":
            return UserEventType.CONNECTION_CONNECTED
        if old in _CONNECTION_ATTEMPTING and new in _CONNECTION_DOWN:
            return UserEventType.CONNECTION_FAILED
        if old in _CONNECTION_HEALTHY and new in _CONNECTION_DOWN:
            return UserEventType.CONNECTION_LOST
        if old is None:
            # no transition info: error reads as a failed attempt, the rest as a fall
            if new == "error":
                return UserEventType.CONNECTION_FAILED
            if new in _CONNECTION_DOWN:
                return UserEventType.CONNECTION_LOST
        # error → disconnected (health-loop cleanup), disconnected → error, … : already reported
        return None
    if table == "strategy_live":
        if new == "running" and old in {None, "starting"}:
            return UserEventType.LIVE_STARTED
        if new == "error":
            return UserEventType.LIVE_FAILED
        if new == "stopped" and old != "starting":
            return UserEventType.LIVE_STOPPED
        return None
    if table == "strategy_backtests":
        if new == "completed":
            return UserEventType.BACKTEST_COMPLETED
        if new in {"failed", "error"}:
            return UserEventType.BACKTEST_FAILED
        return None
    if table == "studio_run":
        if new == "succeeded":
            return UserEventType.STUDIO_RUN_SUCCEEDED
        if new in {"failed", "cancelled"}:
            return UserEventType.STUDIO_RUN_FAILED
        return None
    return None


# ── record ─────────────────────────────────────────────────────────────────


def _publish(user_id: int, row: UserEvent) -> None:
    client = get_redis()
    if client is None:
        return
    payload = {
        "v": 1,
        "type": "user_event",
        "id": row.id,
        "event_type": row.type,
        "severity": row.severity,
    }
    try:
        client.publish(f"{CONNECTION_EVENTS_CHANNEL_PREFIX}{user_id}", json.dumps(payload, default=str))
    except Exception as exc:  # noqa: BLE001 - push is best effort
        logger.debug("user_event publish failed (non-critical): %s", exc)


def record(
    session: Session,
    *,
    user_id: int,
    event_type: UserEventType,
    entity_id: int,
    entity_name: Optional[str] = None,
    message: Optional[str] = None,
    details: Optional[str] = None,
    data: Optional[dict[str, Any]] = None,
    links: Optional[dict[str, str]] = None,
    dedupe_key: Optional[str] = None,
    commit: bool = True,
) -> Optional[UserEvent]:
    """Insert one outcome. Returns the row, or None when deduped or on
    failure. Never raises: an event must not break the business write."""
    try:
        if dedupe_key is not None:
            existing = session.exec(
                select(UserEvent.id).where(UserEvent.user_id == user_id, UserEvent.dedupe_key == dedupe_key)
            ).first()
            if existing is not None:
                return None
        row = UserEvent(
            user_id=user_id,
            type=event_type.value,
            severity=SEVERITY[event_type],
            entity_type=ENTITY_OF[event_type],
            entity_id=entity_id,
            entity_name=(entity_name or None) and str(entity_name)[:200],
            message=message or None,
            details=details or None,
            data=_json_safe(data or {}),
            links=dict(links or {}),
            dedupe_key=dedupe_key,
        )
        session.add(row)
        if commit:
            session.commit()
            session.refresh(row)
        else:
            session.flush()
        _publish(user_id, row)
        return row
    except Exception:  # noqa: BLE001
        logger.exception("user_event record failed: user=%s type=%s", user_id, event_type.value)
        try:
            session.rollback()
        except Exception:  # noqa: BLE001
            pass
        return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


# ── read ───────────────────────────────────────────────────────────────────


def list_events(
    session: Session,
    user_id: int,
    *,
    since_id: Optional[int] = None,
    before_id: Optional[int] = None,
    severity: Optional[list[str]] = None,
    entity_type: Optional[list[str]] = None,
    limit: int = 100,
) -> list[UserEvent]:
    stmt = select(UserEvent).where(UserEvent.user_id == user_id)
    if since_id is not None:
        stmt = stmt.where(UserEvent.id > since_id)
    if before_id is not None:
        stmt = stmt.where(UserEvent.id < before_id)
    if severity:
        stmt = stmt.where(UserEvent.severity.in_(severity))  # type: ignore[attr-defined]
    if entity_type:
        stmt = stmt.where(UserEvent.entity_type.in_(entity_type))  # type: ignore[attr-defined]
    stmt = stmt.order_by(UserEvent.id.desc()).limit(max(1, min(limit, 500)))  # type: ignore[attr-defined]
    return list(session.exec(stmt).all())


def get_event(session: Session, user_id: int, event_id: int) -> Optional[UserEvent]:
    row = session.get(UserEvent, event_id)
    if row is None or row.user_id != user_id:
        return None
    return row


def purge_old(session: Session, *, days: int = RETENTION_DAYS) -> int:
    result = session.execute(
        text("DELETE FROM user_event WHERE created_at < :cutoff"),
        {"cutoff": datetime.now(timezone.utc) - timedelta(days=days)},
    )
    session.commit()
    return int(result.rowcount or 0)
