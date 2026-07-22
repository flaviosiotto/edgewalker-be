"""Fixed-window rate limiting backed by Redis.

Used to put a ceiling on unauthenticated endpoints that trigger outbound email,
which would otherwise let anyone use the backend to flood a third party's inbox.

The limiter fails open: if Redis is unreachable the request is allowed and a
warning is logged. Locking users out of password recovery because a cache node
is down is a worse outcome than the abuse window it would close.
"""

import logging
from dataclasses import dataclass

from app.utils.redis_client import get_redis

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int = 0


def _get_client():
    return get_redis()


def check_rate_limit(key: str, *, limit: int, window_seconds: int) -> RateLimitDecision:
    """Count one hit against ``key`` and report whether it stays under ``limit``."""
    if limit <= 0:
        return RateLimitDecision(allowed=True)

    client = _get_client()
    if client is None:
        return RateLimitDecision(allowed=True)

    namespaced_key = f"ratelimit:{key}"
    try:
        pipeline = client.pipeline()
        pipeline.incr(namespaced_key)
        pipeline.ttl(namespaced_key)
        count, ttl = pipeline.execute()

        # A fresh counter (or one that somehow lost its TTL) gets the window
        # stamped on it, so the key cannot survive as a permanent block.
        if ttl is None or ttl < 0:
            client.expire(namespaced_key, window_seconds)
            ttl = window_seconds

        if int(count) > limit:
            return RateLimitDecision(allowed=False, retry_after_seconds=int(ttl))
        return RateLimitDecision(allowed=True)
    except Exception as exc:  # noqa: BLE001 - fail open, see module docstring
        logger.warning("Rate limit check failed for %s, allowing request: %s", key, exc)
        return RateLimitDecision(allowed=True)
