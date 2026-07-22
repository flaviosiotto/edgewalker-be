"""Brute-force protection for the password login.

Two independent counters, because they stop different attacks:

* per identifier - stops someone grinding a password list against one account
* per client IP  - stops one source spraying one password across many accounts

Both live in Redis with a sliding expiry and **fail open**. A Redis outage
degrading to "no brute-force protection" is bad; a Redis outage locking every
user out of a trading product is worse, and the failure is logged loudly.

Note the deliberate trade-off in the per-identifier counter: anyone who knows an
address can burn its budget and lock that user out for the window. The window is
therefore short, and the per-IP counter is what actually carries the load
against distributed guessing.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from app.core.config import settings
from app.utils.redis_client import get_redis

logger = logging.getLogger(__name__)

_IDENTIFIER_PREFIX = "login:fail:id:"
_IP_PREFIX = "login:fail:ip:"


@dataclass(frozen=True)
class LockoutState:
    locked: bool
    retry_after_seconds: int = 0


def _normalize(identifier: str) -> str:
    return identifier.strip().lower()


def _read_counter(client, key: str) -> tuple[int, int]:
    pipeline = client.pipeline()
    pipeline.get(key)
    pipeline.ttl(key)
    value, ttl = pipeline.execute()
    return int(value or 0), int(ttl or 0)


def check_lockout(identifier: str, client_ip: str) -> LockoutState:
    """Report whether this identifier or caller has exhausted its budget."""
    client = get_redis()
    if client is None:
        return LockoutState(locked=False)

    checks = (
        (f"{_IDENTIFIER_PREFIX}{_normalize(identifier)}", settings.LOGIN_MAX_FAILURES_PER_IDENTIFIER),
        (f"{_IP_PREFIX}{client_ip}", settings.LOGIN_MAX_FAILURES_PER_IP),
    )

    try:
        for key, limit in checks:
            if limit <= 0:
                continue
            count, ttl = _read_counter(client, key)
            if count >= limit:
                return LockoutState(
                    locked=True,
                    retry_after_seconds=max(ttl, 1),
                )
    except Exception as exc:  # noqa: BLE001
        # Deliberately broader than RedisError: this path exists to keep login
        # working, so no failure of the limiter may ever reach the user as a 500.
        logger.warning("Login lockout check failed, allowing attempt: %s", exc)
        return LockoutState(locked=False)

    return LockoutState(locked=False)


def register_failure(identifier: str, client_ip: str) -> None:
    """Count one failed attempt against both budgets."""
    client = get_redis()
    if client is None:
        return

    window = settings.LOGIN_LOCKOUT_WINDOW_SECONDS
    try:
        pipeline = client.pipeline()
        for key in (
            f"{_IDENTIFIER_PREFIX}{_normalize(identifier)}",
            f"{_IP_PREFIX}{client_ip}",
        ):
            pipeline.incr(key)
            # Refreshing the expiry on every failure makes the window slide, so
            # a slow drip of guesses cannot outlast a fixed window.
            pipeline.expire(key, window)
        pipeline.execute()
    except Exception as exc:  # noqa: BLE001 - see check_lockout
        logger.warning("Could not record a failed login attempt: %s", exc)


def clear_failures(identifier: str, client_ip: Optional[str] = None) -> None:
    """Reset the identifier budget after a genuine success.

    The per-IP counter is intentionally left alone: a successful login from a
    host that has just failed against many accounts is exactly the pattern worth
    keeping under a cap.
    """
    client = get_redis()
    if client is None:
        return
    try:
        client.delete(f"{_IDENTIFIER_PREFIX}{_normalize(identifier)}")
    except Exception as exc:  # noqa: BLE001 - see check_lockout
        logger.warning("Could not clear login failure counters: %s", exc)
