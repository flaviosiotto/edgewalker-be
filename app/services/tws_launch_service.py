"""Launch-session service for the interactive IB Gateway (TWS) login.

The interactive login runs inside a per-connection ``twsgw-{id}`` container
exposed through Traefik under a stable path prefix (noVNC). The backend hands
the browser a short-lived launch URL; the launch endpoint sets a scoped cookie
that the Traefik forwardAuth gate validates before forwarding the browser to
the container.

NOTE: public URL paths (``/client-portal/launch``, ``/client-portal/access-check``)
and the legacy ``CLIENT_PORTAL_*`` env names are kept for Traefik/Dokploy
routing continuity, mirroring how the ``datasource-realtime`` router label was
retained after that service was renamed.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import redis.asyncio as aioredis
from fastapi import HTTPException

from app.core.config import settings


logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_USERNAME = os.getenv("REDIS_USERNAME", "")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

# Cookie name is stable so an in-flight login survives a backend deploy.
TWS_LAUNCH_COOKIE_NAME = "edgewalker_client_portal_launch"
TWS_RUNTIME_SESSION_ID_KEY = "_tws_runtime_session_id"


def _create_async_redis_client() -> aioredis.Redis:
    if REDIS_URL:
        return aioredis.from_url(REDIS_URL, decode_responses=True)
    return aioredis.Redis(
        host=REDIS_HOST,
        port=int(REDIS_PORT),
        username=REDIS_USERNAME or None,
        password=REDIS_PASSWORD or None,
        decode_responses=True,
    )


def _launch_session_key(token: str) -> str:
    return f"client-portal:launch:{token}"


def _connection_launch_session_key(connection_id: int, user_id: int) -> str:
    return f"client-portal:launch-connection:{user_id}:{connection_id}"


def _decode_launch_session_payload(payload: str | None) -> dict[str, Any] | None:
    if not payload:
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _runtime_session_id_from_config(config: dict[str, Any] | None) -> str:
    if not isinstance(config, dict):
        return ""
    value = config.get(TWS_RUNTIME_SESSION_ID_KEY)
    return value.strip() if isinstance(value, str) else ""


def _launch_session_matches(
    launch_session: dict[str, Any],
    *,
    connection_id: int,
    user_id: int,
    runtime_session_id: str,
) -> bool:
    try:
        payload_connection_id = int(launch_session.get("connection_id"))
        payload_user_id = int(launch_session.get("user_id"))
    except (TypeError, ValueError):
        return False
    if payload_connection_id != connection_id or payload_user_id != user_id:
        return False
    return str(launch_session.get("runtime_session_id") or "").strip() == runtime_session_id


async def _refresh_launch_session_ttl(
    redis: aioredis.Redis,
    launch_token: str,
    launch_session: dict[str, Any] | None = None,
) -> None:
    ttl_seconds = get_tws_launch_cookie_ttl_seconds()
    await redis.expire(_launch_session_key(launch_token), ttl_seconds)

    if launch_session is None:
        return
    try:
        connection_id = int(launch_session.get("connection_id"))
        user_id = int(launch_session.get("user_id"))
    except (TypeError, ValueError):
        return
    await redis.expire(_connection_launch_session_key(connection_id, user_id), ttl_seconds)


def _normalized_routing_base_url() -> str | None:
    raw = (settings.TWS_ROUTING_BASE_URL or "").strip()
    if not raw:
        return None

    parts = urlsplit(raw)
    scheme = (parts.scheme or "https").lower()
    netloc = parts.netloc or parts.path
    if not netloc:
        raise HTTPException(status_code=500, detail="TWS_ROUTING_BASE_URL is invalid")

    return urlunsplit((scheme, netloc, "", "", "")).rstrip("/")


def _launch_base_url() -> str:
    """Base URL serving the launch endpoint and hosting the launch cookie.

    The browser is sent straight to the per-connection container under the
    routing host, so the launch cookie must be set on that same host for the
    forwardAuth gate to see it.
    """
    routing_base_url = _normalized_routing_base_url()
    if not routing_base_url:
        raise HTTPException(
            status_code=500,
            detail="TWS_ROUTING_BASE_URL is required for the interactive IB Gateway login",
        )
    return routing_base_url


def get_tws_launch_cookie_name() -> str:
    return TWS_LAUNCH_COOKIE_NAME


def get_tws_launch_cookie_ttl_seconds() -> int:
    return max(60, int(settings.TWS_LAUNCH_TTL_SECONDS))


def is_tws_path_routing_enabled() -> bool:
    return bool(settings.TWS_PATH_ROUTING_ENABLED)


def tws_routing_base_url() -> str | None:
    """Public accessor for the normalized routing base URL (scheme + host)."""
    return _normalized_routing_base_url()


def _tws_path_prefix_base() -> str:
    base = (settings.TWS_PATH_PREFIX_BASE or "/ib-access").strip().rstrip("/")
    if not base:
        return "/ib-access"
    return base if base.startswith("/") else f"/{base}"


def tws_path_prefix(connection_id: int) -> str:
    return f"{_tws_path_prefix_base()}/{int(connection_id)}"


def normalize_launch_path_prefix(value: str | None) -> str | None:
    if value is None:
        return None
    prefix = str(value).strip().rstrip("/")
    if not prefix:
        return None
    return prefix if prefix.startswith("/") else f"/{prefix}"


async def validate_tws_launch_access(launch_token: str, connection_id: int) -> bool:
    """forwardAuth gate: confirm the launch cookie maps to a live launch session
    owning *connection_id*.
    """
    token = (launch_token or "").strip()
    if not token:
        return False

    launch_session = await get_tws_launch_session(token)
    if launch_session is None:
        return False

    try:
        return int(launch_session.get("connection_id")) == int(connection_id)
    except (TypeError, ValueError):
        return False


async def create_tws_launch_url(
    *,
    connection_id: int,
    user_id: int,
    config: dict[str, Any] | None,
    force_new: bool = False,
    path_prefix: str | None = None,
) -> str:
    launch_base_url = _launch_base_url()

    ttl_seconds = get_tws_launch_cookie_ttl_seconds()
    mapping_key = _connection_launch_session_key(connection_id, user_id)
    runtime_session_id = _runtime_session_id_from_config(config)
    normalized_path_prefix = normalize_launch_path_prefix(path_prefix)

    redis = _create_async_redis_client()
    try:
        existing_token = str(await redis.get(mapping_key) or "").strip()
        existing_session = None
        if existing_token:
            existing_session = _decode_launch_session_payload(
                await redis.get(_launch_session_key(existing_token))
            )

        if (
            not force_new
            and existing_token
            and existing_session is not None
            and _launch_session_matches(
                existing_session,
                connection_id=connection_id,
                user_id=user_id,
                runtime_session_id=runtime_session_id,
            )
            and normalize_launch_path_prefix(existing_session.get("path_prefix")) == normalized_path_prefix
        ):
            await _refresh_launch_session_ttl(redis, existing_token, existing_session)
            return f"{launch_base_url}/client-portal/launch/{existing_token}"

        token = secrets.token_urlsafe(32)
        payload = {
            "connection_id": connection_id,
            "user_id": user_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "runtime_session_id": runtime_session_id,
        }
        if normalized_path_prefix:
            payload["path_prefix"] = normalized_path_prefix

        await redis.setex(_launch_session_key(token), ttl_seconds, json.dumps(payload))
        await redis.setex(mapping_key, ttl_seconds, token)

        if existing_token and existing_token != token:
            await redis.delete(_launch_session_key(existing_token))
    finally:
        await redis.aclose()

    return f"{launch_base_url}/client-portal/launch/{token}"


async def clear_tws_launch_session(*, connection_id: int, user_id: int) -> None:
    redis = _create_async_redis_client()
    mapping_key = _connection_launch_session_key(connection_id, user_id)
    try:
        launch_token = str(await redis.get(mapping_key) or "").strip()
        keys = [mapping_key]
        if launch_token:
            keys.append(_launch_session_key(launch_token))
        await redis.delete(*keys)
    finally:
        await redis.aclose()


async def get_tws_launch_session(launch_token: str) -> dict[str, Any] | None:
    if not launch_token.strip():
        return None

    payload = None
    redis = _create_async_redis_client()
    try:
        payload = await redis.get(_launch_session_key(launch_token))
        launch_session = _decode_launch_session_payload(payload)
        if launch_session is not None:
            await _refresh_launch_session_ttl(redis, launch_token, launch_session)
    finally:
        await redis.aclose()

    return _decode_launch_session_payload(payload)
