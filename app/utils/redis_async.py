"""Shared async Redis client for SSE fan-out (chat stream relay, strategy events).

Mirror of :mod:`app.utils.redis_client` for async consumers (pub/sub relays
inside SSE endpoints). Callers must tolerate ``None``: Redis being down should
degrade a feature (no live relay), never fail the request.

No ``socket_timeout`` is set on purpose: the client is used for blocking
pub/sub reads, where an idle-read timeout would kill healthy subscriptions.
"""

import logging
import threading
from typing import Optional

import redis.asyncio as aioredis

from app.core.config import settings

logger = logging.getLogger(__name__)

_client: Optional["aioredis.Redis"] = None
_client_lock = threading.Lock()


def get_async_redis() -> Optional["aioredis.Redis"]:
    global _client
    if _client is not None:
        return _client

    with _client_lock:
        if _client is None:
            try:
                if settings.REDIS_URL:
                    _client = aioredis.from_url(
                        settings.REDIS_URL,
                        decode_responses=True,
                        socket_connect_timeout=2,
                    )
                else:
                    _client = aioredis.Redis(
                        host=settings.REDIS_HOST,
                        port=settings.REDIS_PORT,
                        username=settings.REDIS_USERNAME or None,
                        password=settings.REDIS_PASSWORD or None,
                        decode_responses=True,
                        socket_connect_timeout=2,
                    )
            except Exception as exc:  # noqa: BLE001 - never break a request
                logger.warning("Could not build an async Redis client: %s", exc)
                return None
    return _client


async def close_pubsub(pubsub) -> None:
    """Best-effort cleanup compatible with redis-py 5.x (close) and 8.x (aclose)."""
    try:
        await pubsub.unsubscribe()
    except Exception:  # noqa: BLE001
        pass
    try:
        closer = getattr(pubsub, "aclose", None) or getattr(pubsub, "close")
        await closer()
    except Exception:  # noqa: BLE001
        pass


def reset_async_redis_client() -> None:
    """Drop the cached client. Only used by tests."""
    global _client
    with _client_lock:
        _client = None
