from __future__ import annotations

from http.cookies import CookieError, SimpleCookie
import json
import os
import secrets
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
import redis.asyncio as aioredis
from fastapi import HTTPException, Request

from app.core.config import settings
from app.db.database import get_session_context
from app.models.connection import Connection
from app.services.client_portal_service import (
    is_client_portal_transport,
    resolve_client_portal_base_url,
    resolve_client_portal_verify_ssl,
)


REDIS_URL = os.getenv("REDIS_URL", "")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_USERNAME = os.getenv("REDIS_USERNAME", "")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

CLIENT_PORTAL_LAUNCH_COOKIE_NAME = "edgewalker_client_portal_launch"

_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


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


def _launch_session_cookie_key(token: str) -> str:
    return f"client-portal:launch-cookies:{token}"


def _normalized_access_base_url() -> str | None:
    raw = settings.CLIENT_PORTAL_ACCESS_BASE_URL.strip()
    if not raw:
        return None

    parts = urlsplit(raw)
    scheme = (parts.scheme or "https").lower()
    netloc = parts.netloc or parts.path
    if not netloc:
        raise HTTPException(status_code=500, detail="CLIENT_PORTAL_ACCESS_BASE_URL is invalid")

    return urlunsplit((scheme, netloc, "", "", "")).rstrip("/")


def _normalized_access_host() -> str | None:
    access_base_url = _normalized_access_base_url()
    if not access_base_url:
        return None
    return urlsplit(access_base_url).hostname or None


def get_client_portal_launch_cookie_name() -> str:
    return CLIENT_PORTAL_LAUNCH_COOKIE_NAME


def get_client_portal_launch_cookie_ttl_seconds() -> int:
    return max(60, int(settings.CLIENT_PORTAL_LAUNCH_TTL_SECONDS))


def is_client_portal_access_request(request: Request) -> bool:
    expected_host = _normalized_access_host()
    if not expected_host:
        return False

    host_header = (request.headers.get("host") or "").split(":", 1)[0].strip().lower()
    return host_header == expected_host.lower()


async def create_client_portal_launch_url(
    *,
    connection_id: int,
    user_id: int,
    config: dict[str, Any] | None,
) -> str:
    access_base_url = _normalized_access_base_url()
    if not access_base_url:
        raise HTTPException(
            status_code=500,
            detail="CLIENT_PORTAL_ACCESS_BASE_URL is required for on-demand private Client Portal launch",
        )

    token = secrets.token_urlsafe(32)
    payload = {
        "connection_id": connection_id,
        "user_id": user_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    redis = _create_async_redis_client()
    try:
        await redis.setex(
            _launch_session_key(token),
            get_client_portal_launch_cookie_ttl_seconds(),
            json.dumps(payload),
        )
    finally:
        await redis.aclose()

    return f"{access_base_url}/client-portal/launch/{token}"


async def get_client_portal_launch_session(launch_token: str) -> dict[str, Any] | None:
    if not launch_token.strip():
        return None

    redis = _create_async_redis_client()
    try:
        payload = await redis.get(_launch_session_key(launch_token))
        if payload:
            ttl_seconds = get_client_portal_launch_cookie_ttl_seconds()
            await redis.expire(_launch_session_key(launch_token), ttl_seconds)
            await redis.expire(_launch_session_cookie_key(launch_token), ttl_seconds)
    finally:
        await redis.aclose()

    if not payload:
        return None

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


async def _get_client_portal_launch_cookies(launch_token: str) -> dict[str, str]:
    if not launch_token.strip():
        return {}

    redis = _create_async_redis_client()
    try:
        payload = await redis.get(_launch_session_cookie_key(launch_token))
        if payload:
            await redis.expire(_launch_session_cookie_key(launch_token), get_client_portal_launch_cookie_ttl_seconds())
    finally:
        await redis.aclose()

    if not payload:
        return {}

    try:
        cookies = json.loads(payload)
    except json.JSONDecodeError:
        return {}

    if not isinstance(cookies, dict):
        return {}

    return {
        str(name): str(value)
        for name, value in cookies.items()
        if isinstance(name, str) and isinstance(value, str) and name and value
    }


async def _set_client_portal_launch_cookies(launch_token: str, cookies: dict[str, str]) -> None:
    redis = _create_async_redis_client()
    try:
        if cookies:
            await redis.setex(
                _launch_session_cookie_key(launch_token),
                get_client_portal_launch_cookie_ttl_seconds(),
                json.dumps(cookies),
            )
        else:
            await redis.delete(_launch_session_cookie_key(launch_token))
    finally:
        await redis.aclose()


def _merge_set_cookie_headers(current: dict[str, str], set_cookie_headers: list[str]) -> dict[str, str]:
    updated = dict(current)
    for raw_cookie in set_cookie_headers:
        parsed = SimpleCookie()
        try:
            parsed.load(raw_cookie)
        except CookieError:
            continue

        for morsel in parsed.values():
            if morsel.value:
                updated[morsel.key] = morsel.value
            else:
                updated.pop(morsel.key, None)

    return updated


async def _capture_client_portal_launch_cookies(launch_token: str, upstream_headers) -> None:
    if not launch_token.strip():
        return

    try:
        set_cookie_headers = upstream_headers.get_list("set-cookie")
    except AttributeError:
        set_cookie_headers = [
            value
            for key, value in upstream_headers.multi_items()
            if str(key).lower() == "set-cookie"
        ]

    if not set_cookie_headers:
        return

    current = await _get_client_portal_launch_cookies(launch_token)
    updated = _merge_set_cookie_headers(current, set_cookie_headers)
    await _set_client_portal_launch_cookies(launch_token, updated)


def _filtered_proxy_headers(headers, session_cookies: dict[str, str] | None = None) -> dict[str, str]:
    filtered: dict[str, str] = {}
    for key, value in headers.items():
        lower = key.lower()
        if lower in _HOP_BY_HOP_HEADERS or lower in {"host", "cookie"}:
            continue
        filtered[key] = value
    filtered["Accept-Encoding"] = "identity"
    if session_cookies:
        filtered["Cookie"] = "; ".join(
            f"{name}={value}"
            for name, value in sorted(session_cookies.items())
        )
    return filtered


async def resolve_client_portal_proxy_target(request: Request) -> tuple[str, str, bool]:
    launch_token = request.cookies.get(get_client_portal_launch_cookie_name(), "").strip()
    launch_session = await get_client_portal_launch_session(launch_token)
    if launch_session is None:
        raise HTTPException(status_code=404, detail="Launch session not found or expired")

    try:
        connection_id = int(launch_session.get("connection_id"))
        user_id = int(launch_session.get("user_id"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=404, detail="Launch session is invalid")

    with get_session_context() as session:
        conn = session.get(Connection, connection_id)
        if conn is None or conn.user_id != user_id:
            raise HTTPException(status_code=404, detail="Connection not found")
        if conn.broker_type != "ibkr" or not is_client_portal_transport(conn.config or {}):
            raise HTTPException(status_code=400, detail="Connection is not configured for IBKR Client Portal")
        config = dict(conn.config or {})

    return launch_token, resolve_client_portal_base_url(config), resolve_client_portal_verify_ssl(config)


async def proxy_http_request(
    request: Request,
    *,
    launch_token: str,
    upstream_base_url: str,
    verify_ssl: bool,
) -> httpx.Response:
    path = str(request.scope.get("path") or "/")
    upstream_url = f"{upstream_base_url}{path}"

    query_string = request.scope.get("query_string", b"")
    if query_string:
        upstream_url = f"{upstream_url}?{query_string.decode('utf-8')}"

    body = await request.body()
    session_cookies = await _get_client_portal_launch_cookies(launch_token)

    async with httpx.AsyncClient(verify=verify_ssl, follow_redirects=False, timeout=120.0) as client:
        response = await client.request(
            request.method,
            upstream_url,
            headers=_filtered_proxy_headers(request.headers, session_cookies),
            content=body if body else None,
        )
    await _capture_client_portal_launch_cookies(launch_token, response.headers)
    if path == "/v1/api/logout":
        await _set_client_portal_launch_cookies(launch_token, {})
    return response