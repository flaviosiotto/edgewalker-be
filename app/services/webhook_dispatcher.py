"""Background dispatcher for outbound webhooks (migr. 067).

Runs as one asyncio task in the backend lifespan. Every tick it claims a
batch of due deliveries with ``FOR UPDATE SKIP LOCKED`` (safe with several
backend replicas: a row is delivered by one of them), POSTs each one with
the HMAC signature and records the outcome through
``webhook_service.record_attempt`` (success, retry with backoff, or
exhaustion). DB work runs in a worker thread (the ORM is synchronous), the
HTTP calls are async and run concurrently within the batch.

Receivers must answer 2xx within ``WEBHOOK_TIMEOUT_S``; anything else is a
failed attempt. Redirects are not followed (a webhook URL must be final).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from sqlalchemy import text
from sqlmodel import Session

from app.db.database import engine, get_session_context
from app.models.webhook import WebhookDelivery, WebhookSubscription
from app.services import webhook_service

logger = logging.getLogger(__name__)

TICK_SECONDS = 5.0
BATCH_SIZE = 20
WEBHOOK_TIMEOUT_S = 10.0
USER_AGENT = "EdgeWalker-Webhooks/1.0"
MAX_RESPONSE_SNIPPET = 500

_CLAIM_SQL = text(
    """
    UPDATE webhook_delivery
       SET status = 'delivering', last_attempt_at = now()
     WHERE id IN (
           SELECT id FROM webhook_delivery
            WHERE status = 'pending' AND next_attempt_at <= now()
            ORDER BY next_attempt_at
            LIMIT :batch
              FOR UPDATE SKIP LOCKED)
    RETURNING id
    """
)


def claim_due_deliveries(session: Session, batch: int = BATCH_SIZE) -> list[int]:
    ids = [row[0] for row in session.execute(_CLAIM_SQL, {"batch": batch}).fetchall()]
    session.commit()
    return ids


def _load(session: Session, delivery_id: int) -> tuple[Optional[WebhookDelivery], Optional[WebhookSubscription]]:
    delivery = session.get(WebhookDelivery, delivery_id)
    if delivery is None:
        return None, None
    return delivery, session.get(WebhookSubscription, delivery.subscription_id)


def build_request(delivery: WebhookDelivery, secret: str, *, now_ts: Optional[int] = None) -> tuple[bytes, dict[str, str]]:
    """Body bytes and headers for one delivery (pure, used by tests)."""
    body = json.dumps(delivery.payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ts = now_ts if now_ts is not None else int(time.time())
    headers = {
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
        "X-EdgeWalker-Event": delivery.event_type,
        "X-EdgeWalker-Delivery": str(delivery.id),
        "X-EdgeWalker-Event-Id": str(delivery.event_id),
        "X-EdgeWalker-Timestamp": str(ts),
        "X-EdgeWalker-Signature": webhook_service.sign_payload(secret, ts, body),
    }
    return body, headers


async def _post(client: httpx.AsyncClient, url: str, body: bytes, headers: dict[str, str]) -> tuple[bool, Optional[int], Optional[str]]:
    try:
        response = await client.post(url, content=body, headers=headers)
    except httpx.HTTPError as exc:
        return False, None, f"{type(exc).__name__}: {exc}"[:MAX_RESPONSE_SNIPPET]
    ok = 200 <= response.status_code < 300
    snippet = None if ok else (response.text or "")[:MAX_RESPONSE_SNIPPET]
    return ok, response.status_code, snippet


async def deliver_one(client: httpx.AsyncClient, delivery_id: int) -> None:
    """Load, POST, record. Any unexpected error counts as a failed attempt."""
    def _prepare() -> Optional[tuple[str, bytes, dict[str, str]]]:
        with get_session_context() as session:
            delivery, sub = _load(session, delivery_id)
            if delivery is None or sub is None:
                return None
            if not sub.active:
                # Switched off while queued: drop silently (no attempt).
                delivery.status = "failed"
                delivery.last_error = "subscription inactive"
                session.add(delivery)
                session.commit()
                return None
            body, headers = build_request(delivery, webhook_service.decrypt_secret(sub))
            return sub.url, body, headers

    prepared = await asyncio.to_thread(_prepare)
    if prepared is None:
        return
    url, body, headers = prepared
    ok, status_code, error = await _post(client, url, body, headers)

    def _record() -> None:
        with get_session_context() as session:
            delivery, _ = _load(session, delivery_id)
            if delivery is None:
                return
            webhook_service.record_attempt(session, delivery, ok=ok, status_code=status_code, error=error)

    await asyncio.to_thread(_record)
    if not ok:
        logger.info("webhook delivery %s failed: status=%s error=%s", delivery_id, status_code, error)


async def run_once(client: httpx.AsyncClient) -> int:
    """One tick: claim due rows and deliver them concurrently. Returns the
    number of claimed deliveries."""
    def _claim() -> list[int]:
        with get_session_context() as session:
            return claim_due_deliveries(session)

    ids = await asyncio.to_thread(_claim)
    if not ids:
        return 0
    await asyncio.gather(*(deliver_one(client, i) for i in ids), return_exceptions=True)
    return len(ids)


async def requeue_stuck(max_age_s: float = 10 * WEBHOOK_TIMEOUT_S) -> int:
    """Rows left in ``delivering`` by a crashed replica go back to pending."""
    def _run() -> int:
        with get_session_context() as session:
            result = session.execute(
                text(
                    "UPDATE webhook_delivery SET status = 'pending' "
                    "WHERE status = 'delivering' AND last_attempt_at < now() - make_interval(secs => :age)"
                ),
                {"age": max_age_s},
            )
            session.commit()
            return int(result.rowcount or 0)

    return await asyncio.to_thread(_run)


async def dispatcher_loop(stop: asyncio.Event) -> None:
    """The lifespan task. Never raises: a bad tick is logged and retried."""
    logger.info("webhook dispatcher started (tick %.0fs, batch %s)", TICK_SECONDS, BATCH_SIZE)
    async with httpx.AsyncClient(timeout=httpx.Timeout(WEBHOOK_TIMEOUT_S, connect=5.0), follow_redirects=False) as client:
        last_requeue = 0.0
        while not stop.is_set():
            try:
                if time.monotonic() - last_requeue > 60:
                    stuck = await requeue_stuck()
                    if stuck:
                        logger.warning("webhook dispatcher: requeued %s stuck deliveries", stuck)
                    last_requeue = time.monotonic()
                claimed = await run_once(client)
                if claimed >= BATCH_SIZE:
                    continue  # more may be due: no sleep
            except Exception:  # pragma: no cover - keep the loop alive
                logger.exception("webhook dispatcher tick failed")
            try:
                await asyncio.wait_for(stop.wait(), timeout=TICK_SECONDS)
            except asyncio.TimeoutError:
                pass
    logger.info("webhook dispatcher stopped")


def pending_count() -> dict[str, Any]:
    """For /system or admin diagnostics."""
    with Session(engine) as session:
        rows = session.execute(
            text("SELECT status, count(*) FROM webhook_delivery GROUP BY status")
        ).fetchall()
    return {status: int(n) for status, n in rows} | {"checked_at": datetime.now(timezone.utc).isoformat()}
