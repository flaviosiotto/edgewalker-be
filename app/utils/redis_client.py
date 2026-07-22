"""Shared Redis client for short-lived auth state.

One lazily-built client for everything auth-related (rate limit counters, OAuth
state, one-time exchange codes) so the backend does not open a separate pool per
feature. Callers must tolerate ``None``: Redis being down should degrade a
feature, not take the process with it.
"""

import logging
import threading
from typing import Optional

import redis as sync_redis

from app.core.config import settings

logger = logging.getLogger(__name__)

_client: Optional["sync_redis.Redis"] = None
_client_lock = threading.Lock()


def get_redis() -> Optional["sync_redis.Redis"]:
    global _client
    if _client is not None:
        return _client

    with _client_lock:
        if _client is None:
            try:
                _client = sync_redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    decode_responses=True,
                    socket_timeout=2,
                    socket_connect_timeout=2,
                )
            except Exception as exc:  # noqa: BLE001 - never break a request
                logger.warning("Could not build a Redis client: %s", exc)
                return None
    return _client


def reset_redis_client() -> None:
    """Drop the cached client. Only used by tests."""
    global _client
    with _client_lock:
        _client = None
