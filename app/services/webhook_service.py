"""Outbound webhooks: catalogue, subscriptions, signing, enqueue (migr. 067).

Flow: a service that knows a state change calls :func:`emit_event` inside
its own session/transaction; the function expands the event into one
``webhook_delivery`` row per matching active subscription of the user
(outbox pattern, committed together with the change). The dispatcher
(``webhook_dispatcher.py``) POSTs the rows with retries.

Payload envelope (what the endpoint receives)::

    {"id": "<event uuid>", "type": "backtest.completed", "created_at": "...",
     "user_id": 7, "data": {...event fields...}, "links": {...}}

Headers::

    X-EdgeWalker-Event: backtest.completed
    X-EdgeWalker-Delivery: <delivery id>
    X-EdgeWalker-Timestamp: <unix seconds>
    X-EdgeWalker-Signature: v1=<hex HMAC-SHA256(secret, "<timestamp>.<body>")>

The signing secret is stored Fernet-encrypted (same key as user secrets);
it is shown to the user once, at creation or rotation.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

from cryptography.fernet import Fernet
from fastapi import HTTPException, status
from sqlmodel import Session, select

from app.core.config import settings
from app.models.agent import Agent
from app.models.webhook import WebhookDelivery, WebhookSubscription

logger = logging.getLogger(__name__)

#: Endpoints per user (plan limits can refine this later: D14 of the study).
MAX_SUBSCRIPTIONS_PER_USER = 10
#: Consecutive failed deliveries after which an endpoint is switched off.
AUTO_DISABLE_AFTER_FAILURES = 50
#: Retry schedule (seconds) by attempt number; after the last one the
#: delivery is marked failed. ~23 h in total.
RETRY_BACKOFF_SECONDS = (10, 60, 300, 900, 3600, 10800, 21600, 43200)
MAX_ATTEMPTS = len(RETRY_BACKOFF_SECONDS)

#: The event catalogue: name -> (description, data fields). The FE builds
#: the subscription form from it and the docs quote it.
EVENT_CATALOG: dict[str, tuple[str, list[str]]] = {
    "ping": ("Test event sent on demand from the settings page.", ["message"]),
    "live.status.changed": (
        "A live session changed status (starting, running, paused, stopped, error).",
        ["live_id", "strategy_id", "strategy_name", "status", "previous_status", "error_message"],
    ),
    "live.alert.triggered": (
        "An alert of a live session fired (price level or condition group).",
        ["live_id", "strategy_id", "alert_id", "alert_name", "alert_type", "symbol", "price", "recipient"],
    ),
    "live.trade.closed": (
        "A realized trade was recorded on a live session.",
        ["live_id", "strategy_id", "account_id", "trade_id", "symbol", "side", "quantity", "entry_price", "exit_price", "realized_pnl", "pnl_currency", "exit_reason"],
    ),
    "backtest.completed": (
        "A backtest finished successfully (metrics available).",
        ["backtest_id", "strategy_id", "strategy_name", "symbol", "timeframe", "start_date", "end_date", "metrics"],
    ),
    "backtest.failed": (
        "A backtest ended with an error.",
        ["backtest_id", "strategy_id", "strategy_name", "error_message"],
    ),
    "agent.turn.completed": (
        "A hosted agent finished a turn in a chat (reply available).",
        ["chat_id", "agent_id", "agent_name", "strategy_id", "live_id", "backtest_id", "message_type", "summary"],
    ),
    "credits.exhausted": (
        "The AI credits of the period ran out: agent turns are refused until a top-up or the next period.",
        ["period_key", "limit", "used"],
    ),
    "connection.stale": (
        "A broker connection stopped reporting (health check stale) or disconnected.",
        ["connection_id", "connection_name", "broker_type", "status", "last_checked_at"],
    ),
}

_DELIVERY_STATUS_PENDING = "pending"
_DELIVERY_STATUS_DELIVERING = "delivering"
_DELIVERY_STATUS_SUCCEEDED = "succeeded"
_DELIVERY_STATUS_FAILED = "failed"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── secrets ────────────────────────────────────────────────────────────────


def _fernet() -> Fernet:
    """SECRETS_ENCRYPTION_KEY when configured (shared with user secrets),
    otherwise the key derived from SECRET_KEY that MFA secrets use: webhooks
    must work on an installation that never configured user secrets."""
    if settings.SECRETS_ENCRYPTION_KEY:
        return Fernet(settings.SECRETS_ENCRYPTION_KEY.encode())
    from app.services.totp_service import _fernet as mfa_fernet

    return mfa_fernet()


def _encrypt(value: str) -> str:
    return _fernet().encrypt(value.encode("utf-8")).decode("ascii")


def decrypt_secret(subscription: WebhookSubscription) -> str:
    return _fernet().decrypt(subscription.secret_encrypted.encode("ascii")).decode("utf-8")


def new_secret() -> str:
    return "whsec_" + secrets.token_urlsafe(32)


def sign_payload(secret: str, timestamp: int, body: bytes) -> str:
    """``v1=<hex>`` over ``"<timestamp>.<body>"`` (Stripe-style, replay-safe
    when the receiver checks the timestamp)."""
    mac = hmac.new(secret.encode("utf-8"), f"{timestamp}.".encode("utf-8") + body, hashlib.sha256)
    return "v1=" + mac.hexdigest()


def verify_signature(secret: str, timestamp: int, body: bytes, header: str, *, tolerance_s: int = 300) -> bool:
    """Receiver-side check, exported for tests and for the docs."""
    if abs(int(datetime.now(timezone.utc).timestamp()) - timestamp) > tolerance_s:
        return False
    expected = sign_payload(secret, timestamp, body)
    return hmac.compare_digest(expected, header or "")


# ── subscriptions ──────────────────────────────────────────────────────────


def event_catalog() -> list[dict[str, Any]]:
    return [{"type": k, "description": v[0], "data_fields": v[1]} for k, v in EVENT_CATALOG.items()]


def _validate_events(events: Iterable[str]) -> list[str]:
    cleaned = sorted({e for e in events})
    if cleaned == ["*"]:
        return cleaned
    unknown = [e for e in cleaned if e not in EVENT_CATALOG or e == "ping"]
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown events: {', '.join(unknown)}. Valid: * or {', '.join(k for k in EVENT_CATALOG if k != 'ping')}",
        )
    return cleaned


def _validate_url(url: str) -> str:
    cleaned = str(url).strip()
    if not cleaned.lower().startswith(("https://", "http://")):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="url must be http(s)")
    if len(cleaned) > 2048:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="url too long")
    return cleaned


def _validate_agent(session: Session, user_id: int, agent_id: Optional[int]) -> Optional[int]:
    if agent_id is None:
        return None
    agent = session.get(Agent, agent_id)
    if agent is None or agent.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
    return agent.id_agent


def list_subscriptions(session: Session, user_id: int) -> list[WebhookSubscription]:
    return list(
        session.exec(
            select(WebhookSubscription)
            .where(WebhookSubscription.user_id == user_id)
            .order_by(WebhookSubscription.created_at.desc())
        ).all()
    )


def get_subscription(session: Session, user_id: int, subscription_id: int) -> WebhookSubscription:
    row = session.get(WebhookSubscription, subscription_id)
    if row is None or row.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Webhook not found")
    return row


def create_subscription(
    session: Session,
    *,
    user_id: int,
    name: str,
    url: str,
    events: list[str],
    agent_id: Optional[int] = None,
    active: bool = True,
) -> tuple[WebhookSubscription, str]:
    """Returns ``(row, secret)``; the secret exists in clear only here."""
    count = len(list_subscriptions(session, user_id))
    if count >= MAX_SUBSCRIPTIONS_PER_USER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"At most {MAX_SUBSCRIPTIONS_PER_USER} webhooks per user",
        )
    secret = new_secret()
    row = WebhookSubscription(
        user_id=user_id,
        agent_id=_validate_agent(session, user_id, agent_id),
        name=name.strip(),
        url=_validate_url(url),
        secret_encrypted=_encrypt(secret),
        events=_validate_events(events),
        active=active,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row, secret


def update_subscription(
    session: Session,
    *,
    user_id: int,
    subscription_id: int,
    name: Optional[str] = None,
    url: Optional[str] = None,
    events: Optional[list[str]] = None,
    agent_id: Optional[int] = None,
    agent_id_set: bool = False,
    active: Optional[bool] = None,
    rotate_secret: bool = False,
) -> tuple[WebhookSubscription, Optional[str]]:
    row = get_subscription(session, user_id, subscription_id)
    if name is not None:
        row.name = name.strip()
    if url is not None:
        row.url = _validate_url(url)
    if events is not None:
        row.events = _validate_events(events)
    if agent_id_set:
        row.agent_id = _validate_agent(session, user_id, agent_id)
    if active is not None:
        row.active = active
        if active:
            # re-enabling clears the auto-disable state
            row.failure_streak = 0
            row.disabled_reason = None
    secret: Optional[str] = None
    if rotate_secret:
        secret = new_secret()
        row.secret_encrypted = _encrypt(secret)
    row.updated_at = _utcnow()
    session.add(row)
    session.commit()
    session.refresh(row)
    return row, secret


def delete_subscription(session: Session, user_id: int, subscription_id: int) -> None:
    row = get_subscription(session, user_id, subscription_id)
    session.delete(row)
    session.commit()


def list_deliveries(session: Session, user_id: int, subscription_id: int, *, limit: int = 50) -> list[WebhookDelivery]:
    get_subscription(session, user_id, subscription_id)
    return list(
        session.exec(
            select(WebhookDelivery)
            .where(WebhookDelivery.subscription_id == subscription_id)
            .order_by(WebhookDelivery.created_at.desc())
            .limit(max(1, min(limit, 200)))
        ).all()
    )


# ── emit ───────────────────────────────────────────────────────────────────


def _matches(subscription: WebhookSubscription, event_type: str) -> bool:
    events = subscription.events or []
    return "*" in events or event_type in events


def _json_safe(value: Any) -> Any:
    """Make a payload JSON-serialisable (datetimes, Decimals, UUIDs)."""
    return json.loads(json.dumps(value, default=str))


def build_envelope(
    *,
    event_id: uuid.UUID,
    event_type: str,
    user_id: int,
    data: dict[str, Any],
    links: Optional[dict[str, str]] = None,
    created_at: Optional[datetime] = None,
) -> dict[str, Any]:
    return _json_safe(
        {
            "id": str(event_id),
            "type": event_type,
            "created_at": (created_at or _utcnow()).isoformat(),
            "user_id": user_id,
            "data": data,
            "links": links or {},
        }
    )


def emit_event(
    session: Session,
    *,
    user_id: int,
    event_type: str,
    data: dict[str, Any],
    links: Optional[dict[str, str]] = None,
    dedupe_key: Optional[str] = None,
    commit: bool = False,
) -> list[WebhookDelivery]:
    """Enqueue ``event_type`` for every active subscription of ``user_id``
    that listens to it. Adds rows to ``session`` and, unless ``commit``,
    leaves the commit to the caller so the deliveries ride the same
    transaction as the state change they describe. Never raises: a webhook
    problem must not break the business operation.
    """
    if event_type not in EVENT_CATALOG:
        logger.warning("emit_event: unknown event type %s", event_type)
        return []
    try:
        subscriptions = [
            s
            for s in session.exec(
                select(WebhookSubscription).where(
                    WebhookSubscription.user_id == user_id, WebhookSubscription.active.is_(True)  # type: ignore[attr-defined]
                )
            ).all()
            if _matches(s, event_type)
        ]
    except Exception:  # pragma: no cover - defensive, see docstring
        logger.exception("emit_event: subscription lookup failed")
        return []
    if not subscriptions:
        return []

    event_id = uuid.uuid4()
    envelope = build_envelope(event_id=event_id, event_type=event_type, user_id=user_id, data=data, links=links)
    rows: list[WebhookDelivery] = []
    for sub in subscriptions:
        if dedupe_key is not None:
            existing = session.exec(
                select(WebhookDelivery.id).where(
                    WebhookDelivery.subscription_id == sub.id, WebhookDelivery.dedupe_key == dedupe_key
                )
            ).first()
            if existing is not None:
                continue
        row = WebhookDelivery(
            subscription_id=sub.id,
            event_id=event_id,
            event_type=event_type,
            dedupe_key=dedupe_key,
            payload=envelope,
            max_attempts=MAX_ATTEMPTS,
        )
        session.add(row)
        rows.append(row)
    if commit and rows:
        try:
            session.commit()
        except Exception:  # pragma: no cover
            logger.exception("emit_event: commit failed")
            session.rollback()
            return []
    return rows


def enqueue_ping(session: Session, user_id: int, subscription_id: int) -> WebhookDelivery:
    """A test delivery for one endpoint, regardless of its event filter."""
    sub = get_subscription(session, user_id, subscription_id)
    event_id = uuid.uuid4()
    row = WebhookDelivery(
        subscription_id=sub.id,
        event_id=event_id,
        event_type="ping",
        payload=build_envelope(
            event_id=event_id,
            event_type="ping",
            user_id=user_id,
            data={"message": f"Hello from EdgeWalker: webhook '{sub.name}' is configured."},
        ),
        max_attempts=MAX_ATTEMPTS,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


# ── delivery bookkeeping (used by the dispatcher) ──────────────────────────


def next_backoff(attempts: int) -> Optional[timedelta]:
    """Delay before attempt number ``attempts + 1``; ``None`` when exhausted."""
    if attempts >= MAX_ATTEMPTS:
        return None
    return timedelta(seconds=RETRY_BACKOFF_SECONDS[min(attempts, MAX_ATTEMPTS - 1)])


def record_attempt(
    session: Session,
    delivery: WebhookDelivery,
    *,
    ok: bool,
    status_code: Optional[int],
    error: Optional[str],
) -> None:
    """Update the delivery and its subscription after one HTTP attempt.
    Commits."""
    now = _utcnow()
    delivery.attempts += 1
    delivery.last_attempt_at = now
    delivery.last_status_code = status_code
    delivery.last_error = (error or "")[:2000] or None
    sub = session.get(WebhookSubscription, delivery.subscription_id)
    if ok:
        delivery.status = _DELIVERY_STATUS_SUCCEEDED
        delivery.delivered_at = now
        if sub is not None:
            sub.last_success_at = now
            sub.failure_streak = 0
    else:
        delay = next_backoff(delivery.attempts)
        if delay is None:
            delivery.status = _DELIVERY_STATUS_FAILED
        else:
            delivery.status = _DELIVERY_STATUS_PENDING
            delivery.next_attempt_at = now + delay
        if sub is not None:
            sub.last_failure_at = now
            sub.failure_streak += 1
            if sub.failure_streak >= AUTO_DISABLE_AFTER_FAILURES and sub.active:
                sub.active = False
                sub.disabled_reason = f"Disabled after {sub.failure_streak} consecutive failed deliveries ({now.isoformat()})"
                logger.warning("webhook %s auto-disabled after %s failures", sub.id, sub.failure_streak)
    session.add(delivery)
    if sub is not None:
        session.add(sub)
    session.commit()
