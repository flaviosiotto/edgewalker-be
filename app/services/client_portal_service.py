from __future__ import annotations

import logging
import os
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import HTTPException, status

from app.core.config import settings


logger = logging.getLogger(__name__)

_CLIENT_PORTAL_DISPATCHER_RECEIVED_AT_KEY = "_client_portal_dispatcher_received_at"
_CLIENT_PORTAL_PROXY_BRIDGE_HEADER = "X-Edgewalker-Client-Portal-Bridge"

_CLIENT_PORTAL_CONFIG_KEYS = frozenset(
    {
        "client_portal_base_url",
        "client_portal_browser_url",
        "client_portal_verify_ssl",
        "client_portal_enabled",
        "transport",
    }
)

_CLIENT_PORTAL_SESSION_POST_FALLBACK_STATUS_CODES = frozenset({404, 405})
_CLIENT_PORTAL_ACCOUNTS_PROBE_INTERVAL_SECONDS = 5.0
_CLIENT_PORTAL_SESSION_INIT_INTERVAL_SECONDS = 5.0

_client_portal_accounts_probe_state: dict[str, dict[str, Any]] = {}
_client_portal_session_init_attempted_at: dict[str, float] = {}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _client_portal_proxy_headers() -> dict[str, str]:
    token = settings.CLIENT_PORTAL_PROXY_BRIDGE_TOKEN.strip()
    if not token:
        return {}
    return {_CLIENT_PORTAL_PROXY_BRIDGE_HEADER: token}


def _normalize_client_portal_url(
    raw_url: str | None,
    *,
    default_url: str,
    include_trailing_slash: bool,
    field_name: str,
) -> str:
    candidate = (raw_url or default_url or "").strip()
    if not candidate:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{field_name} is required")

    parts = urlsplit(candidate)
    scheme = (parts.scheme or "https").lower()
    netloc = parts.netloc or parts.path
    if not netloc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{field_name} is invalid")
    if scheme != "https":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} must use https",
        )
    if parts.username or parts.password or "@" in netloc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} must not include credentials",
        )

    return urlunsplit((scheme, netloc, "/" if include_trailing_slash else "", "", ""))


def _allowed_client_portal_urls(raw_allowed: str, *, default_url: str, include_trailing_slash: bool) -> set[str]:
    allowed = {
        _normalize_client_portal_url(
            default_url,
            default_url=default_url,
            include_trailing_slash=include_trailing_slash,
            field_name="Client Portal URL",
        )
    }
    for url in _split_csv(raw_allowed):
        allowed.add(
            _normalize_client_portal_url(
                url,
                default_url=default_url,
                include_trailing_slash=include_trailing_slash,
                field_name="Client Portal URL",
            )
        )
    return allowed


def _validated_client_portal_url(
    raw_url: str | None,
    *,
    default_url: str,
    allowed_urls: str,
    include_trailing_slash: bool,
    field_name: str,
) -> str:
    normalized = _normalize_client_portal_url(
        raw_url,
        default_url=default_url,
        include_trailing_slash=include_trailing_slash,
        field_name=field_name,
    )
    allowed = _allowed_client_portal_urls(
        allowed_urls,
        default_url=default_url,
        include_trailing_slash=include_trailing_slash,
    )
    if normalized not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} is not allowed",
        )
    return normalized


def _needs_client_portal_sanitization(config: dict[str, Any] | None) -> bool:
    if not config:
        return False
    return bool(_CLIENT_PORTAL_CONFIG_KEYS.intersection(config.keys()))


def sanitize_connection_config(
    broker_type: str,
    config: dict[str, Any] | None,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    normalized = dict(config or {})
    if str(broker_type).strip().lower() != "ibkr":
        return normalized
    if not _needs_client_portal_sanitization(normalized):
        return normalized

    try:
        normalized["client_portal_base_url"] = _validated_client_portal_url(
            normalized.get("client_portal_base_url"),
            default_url=settings.CLIENT_PORTAL_BASE_URL,
            allowed_urls=settings.CLIENT_PORTAL_ALLOWED_BASE_URLS,
            include_trailing_slash=False,
            field_name="client_portal_base_url",
        )
        normalized["client_portal_browser_url"] = _validated_client_portal_url(
            normalized.get("client_portal_browser_url"),
            default_url=settings.CLIENT_PORTAL_BROWSER_URL,
            allowed_urls=settings.CLIENT_PORTAL_ALLOWED_BROWSER_URLS,
            include_trailing_slash=True,
            field_name="client_portal_browser_url",
        )
    except HTTPException:
        if strict:
            raise
        logger.warning("Falling back to trusted Client Portal defaults for IBKR connection config")
        normalized["client_portal_base_url"] = _normalize_client_portal_url(
            settings.CLIENT_PORTAL_BASE_URL,
            default_url=settings.CLIENT_PORTAL_BASE_URL,
            include_trailing_slash=False,
            field_name="client_portal_base_url",
        )
        normalized["client_portal_browser_url"] = _normalize_client_portal_url(
            settings.CLIENT_PORTAL_BROWSER_URL,
            default_url=settings.CLIENT_PORTAL_BROWSER_URL,
            include_trailing_slash=True,
            field_name="client_portal_browser_url",
        )

    normalized["client_portal_verify_ssl"] = settings.CLIENT_PORTAL_VERIFY_SSL
    return normalized


def resolve_client_portal_base_url(config: dict[str, Any] | None = None) -> str:
    config = sanitize_connection_config("ibkr", config, strict=False)
    return str(config.get("client_portal_base_url", settings.CLIENT_PORTAL_BASE_URL)).rstrip("/")


def resolve_client_portal_browser_url(config: dict[str, Any] | None = None) -> str:
    config = sanitize_connection_config("ibkr", config, strict=False)
    browser_url = str(config.get("client_portal_browser_url", settings.CLIENT_PORTAL_BROWSER_URL))
    normalized = browser_url.rstrip("/")
    if (
        not normalized
        or normalized.endswith("/ibkr-client-portal")
        or normalized.endswith("/sso/Login?forwardTo=22&RL=1&ip2loc=US")
    ):
        parts = urlsplit(browser_url)
        return urlunsplit(
            (
                parts.scheme or "https",
                parts.netloc or "localhost:5000",
                "/",
                "",
                "",
            )
        )
    return browser_url


def resolve_client_portal_verify_ssl(config: dict[str, Any] | None = None) -> bool:
    config = sanitize_connection_config("ibkr", config, strict=False)
    return _as_bool(config.get("client_portal_verify_ssl", settings.CLIENT_PORTAL_VERIFY_SSL), default=False)


def is_client_portal_transport(config: dict[str, Any] | None = None) -> bool:
    config = config or {}
    transport = str(config.get("transport", "legacy")).strip().lower()
    enabled = _as_bool(config.get("client_portal_enabled"), default=transport == "client_portal")
    return transport == "client_portal" or enabled


def _unwrap_client_portal_payload(payload: Any) -> Any:
    current = payload
    for _ in range(3):
        if not isinstance(current, dict):
            break

        success = current.get("success")
        if isinstance(success, dict) and "value" in success:
            current = success["value"]
            continue

        fail = current.get("fail")
        if isinstance(fail, dict) and "value" in fail:
            current = fail["value"]
            continue

        break

    return current


def _extract_message(payload: Any) -> str | None:
    source = _unwrap_client_portal_payload(payload)
    if not isinstance(source, dict):
        return None

    message = source.get("message") or source.get("error") or source.get("text")
    if not message:
        return None
    return str(message)


def _clear_client_portal_probe_state(base_url: str) -> None:
    _client_portal_accounts_probe_state.pop(base_url, None)
    _client_portal_session_init_attempted_at.pop(base_url, None)


def _get_cached_client_portal_accounts(base_url: str) -> list[dict[str, Any]]:
    entry = _client_portal_accounts_probe_state.get(base_url)
    if not isinstance(entry, dict):
        return []

    last_successful = entry.get("last_successful")
    if not isinstance(last_successful, (int, float)):
        return []
    if time.monotonic() - float(last_successful) > _CLIENT_PORTAL_ACCOUNTS_PROBE_INTERVAL_SECONDS:
        return []

    accounts = entry.get("accounts")
    if not isinstance(accounts, list):
        return []

    return [item for item in accounts if isinstance(item, dict)]


def _should_probe_client_portal_accounts(base_url: str) -> bool:
    now = time.monotonic()
    entry = _client_portal_accounts_probe_state.setdefault(base_url, {})
    last_attempted = entry.get("last_attempted")
    if isinstance(last_attempted, (int, float)) and now - float(last_attempted) < _CLIENT_PORTAL_ACCOUNTS_PROBE_INTERVAL_SECONDS:
        return False

    entry["last_attempted"] = now
    return True


def _store_client_portal_accounts(base_url: str, accounts: list[dict[str, Any]]) -> None:
    entry = _client_portal_accounts_probe_state.setdefault(base_url, {})
    entry["accounts"] = accounts
    entry["last_successful"] = time.monotonic()


def _should_initialize_client_portal_session(base_url: str) -> bool:
    now = time.monotonic()
    last_attempted = _client_portal_session_init_attempted_at.get(base_url)
    if isinstance(last_attempted, (int, float)) and now - float(last_attempted) < _CLIENT_PORTAL_SESSION_INIT_INTERVAL_SECONDS:
        return False

    _client_portal_session_init_attempted_at[base_url] = now
    return True


async def _post_with_get_fallback(
    client: httpx.AsyncClient,
    path: str,
    *,
    json_body: Any | None = None,
) -> httpx.Response:
    post_kwargs: dict[str, Any]
    if json_body is None:
        post_kwargs = {"headers": {"Content-Length": "0"}}
    else:
        post_kwargs = {"json": json_body}

    response = await client.post(path, **post_kwargs)
    if response.status_code not in _CLIENT_PORTAL_SESSION_POST_FALLBACK_STATUS_CODES:
        return response

    return await client.get(path)


async def _initialize_client_portal_brokerage_session(
    client: httpx.AsyncClient,
    *,
    base_url: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "attempted": False,
        "supported": True,
        "initialized": False,
        "message": None,
    }

    if not _should_initialize_client_portal_session(base_url):
        return result

    result["attempted"] = True

    try:
        response = await client.post(
            "/v1/api/iserver/auth/ssodh/init",
            json={"publish": True, "compete": False},
        )
        if response.status_code in _CLIENT_PORTAL_SESSION_POST_FALLBACK_STATUS_CODES:
            result["supported"] = False
            return result
        if response.status_code == 401:
            return result
        if response.status_code == 400:
            try:
                result["message"] = _extract_message(response.json()) or response.text
            except ValueError:
                result["message"] = response.text
            return result

        response.raise_for_status()
        result["initialized"] = True
        if response.content:
            try:
                result["message"] = _extract_message(response.json())
            except ValueError:
                result["message"] = None
    except Exception as exc:
        logger.warning("Client Portal ssodh/init failed for %s: %s", base_url, exc)
        result["message"] = str(exc)

    return result


def _extract_authenticated(payload: Any) -> bool:
    payload = _unwrap_client_portal_payload(payload)
    if not isinstance(payload, dict):
        return False

    if isinstance(payload.get("authenticated"), bool):
        return payload["authenticated"]

    nested = payload.get("iserver")
    if isinstance(nested, dict):
        auth_status = nested.get("authStatus")
        if isinstance(auth_status, dict) and isinstance(auth_status.get("authenticated"), bool):
            return auth_status["authenticated"]

    return False


def _extract_status_flags(payload: Any) -> dict[str, bool]:
    flags = {
        "connected": False,
        "authenticated": False,
        "established": False,
        "competing": False,
    }

    payload = _unwrap_client_portal_payload(payload)
    if not isinstance(payload, dict):
        return flags

    source = payload
    nested = payload.get("iserver")
    if isinstance(nested, dict):
        auth_status = nested.get("authStatus")
        if isinstance(auth_status, dict):
            source = auth_status

    for key in flags:
        if isinstance(source.get(key), bool):
            flags[key] = source[key]

    return flags


def _extract_gateway_session_ready(payload: Any) -> bool:
    payload = _unwrap_client_portal_payload(payload)
    if not isinstance(payload, dict):
        return False

    session = payload.get("session")
    sso_expires = payload.get("ssoExpires")
    user_id = payload.get("userId")
    return bool(session or sso_expires or user_id)


def _extract_competing(payload: Any) -> bool:
    payload = _unwrap_client_portal_payload(payload)
    if not isinstance(payload, dict):
        return False
    return bool(payload.get("competing"))


def _extract_accounts(payload: Any) -> list[dict[str, Any]]:
    payload = _unwrap_client_portal_payload(payload)
    if not isinstance(payload, list):
        return []

    accounts: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        account_id = item.get("accountId") or item.get("id") or item.get("accountVan")
        if not account_id:
            continue
        accounts.append(item)
    return accounts


def _extract_bridge_ready(payload: Any) -> bool:
    payload = _unwrap_client_portal_payload(payload)
    if isinstance(payload, list):
        return len(payload) > 0
    if isinstance(payload, dict):
        accounts = payload.get("accounts")
        return isinstance(accounts, list) and len(accounts) > 0
    return False


def _has_dispatcher_marker(config: dict[str, Any] | None) -> bool:
    if not isinstance(config, dict):
        return False
    value = config.get(_CLIENT_PORTAL_DISPATCHER_RECEIVED_AT_KEY)
    return isinstance(value, str) and bool(value.strip())


async def get_client_portal_auth_status(config: dict[str, Any] | None = None) -> dict[str, Any]:
    base_url = resolve_client_portal_base_url(config)
    browser_url = resolve_client_portal_browser_url(config)
    verify_ssl = resolve_client_portal_verify_ssl(config)
    dispatcher_marker_present = _has_dispatcher_marker(config)

    result: dict[str, Any] = {
        "service_ready": False,
        "gateway_session_ready": False,
        "connected": False,
        "session_authenticated": False,
        "authenticated": False,
        "established": False,
        "competing": False,
        "bridge_ready": False,
        "ready_to_connect": False,
        "accounts": [],
        "auth_url": browser_url,
        "base_url": base_url,
        "message": None,
        "payload": None,
        "tickle_payload": None,
    }

    try:
        async with httpx.AsyncClient(
            base_url=base_url,
            verify=verify_ssl,
            timeout=10.0,
            follow_redirects=True,
            headers=_client_portal_proxy_headers(),
        ) as client:
            payload = None
            tickle_payload = None
            session_authenticated = False
            competing = False
            response = await _post_with_get_fallback(client, "/v1/api/iserver/auth/status")
            result["service_ready"] = True

            if response.status_code == 401:
                # Gateway SSO session not established yet.  Return immediately
                # WITHOUT hitting tickle / portfolio / iserver endpoints.
                # The gateway is a single-session proxy: every API request it
                # forwards to api.ibkr.com carries (and overwrites) the same
                # server-side cookie jar used by the browser's SSO flow.
                # Sending extra 401-bound requests during login corrupts the
                # x-sess-uuid cookies and prevents the 2FA handshake from
                # completing — the Authenticator loop never reaches Dispatcher.
                if not dispatcher_marker_present:
                    result["message"] = "Sessione Client Portal non autenticata. Completa il login nel popup."
                    return result

                # After Dispatcher completion the browser flow has finished, so
                # auth/status may legitimately lag behind the session becoming
                # usable. In that phase use account/bridge endpoints as the real
                # readiness signal instead of getting stuck on repeated 401 here.
                result["message"] = "Autorizzazione 2FA ricevuta. Verifico l'apertura della brokerage session IBKR."
            else:
                response.raise_for_status()
                payload = response.json()
                result["payload"] = payload

                try:
                    tickle_response = await _post_with_get_fallback(client, "/v1/api/tickle")
                    if tickle_response.status_code != 401:
                        tickle_response.raise_for_status()
                        tickle_payload = tickle_response.json()
                except httpx.HTTPStatusError as tickle_exc:
                    if tickle_exc.response.status_code != 401:
                        raise

                flags = _extract_status_flags(tickle_payload if tickle_payload is not None else payload)
                session_authenticated = flags["authenticated"]
                competing = flags["competing"]
                result["tickle_payload"] = tickle_payload
                # A 200 from auth/status means the browser SSO session is established
                # even if tickle has not started returning session markers yet.
                result["gateway_session_ready"] = bool(
                    _extract_gateway_session_ready(tickle_payload if tickle_payload is not None else payload)
                    or response.status_code != 401
                )
                result["connected"] = flags["connected"]
                result["session_authenticated"] = session_authenticated
                result["authenticated"] = flags["authenticated"]
                result["established"] = flags["established"]
                result["competing"] = competing
                result["message"] = _extract_message(payload)

            if competing:
                result["message"] = (
                    "Client Portal segnala una sessione concorrente. Chiudi eventuali altre sessioni IBKR/TWS/Client Portal e riprova."
                )

            if dispatcher_marker_present and not session_authenticated and not competing:
                init_result = await _initialize_client_portal_brokerage_session(client, base_url=base_url)
                if init_result.get("initialized"):
                    result["gateway_session_ready"] = True
                    if not result["message"]:
                        result["message"] = "Autorizzazione 2FA ricevuta. Inizializzo la brokerage session IBKR."
                elif init_result.get("message") and not result["message"]:
                    result["message"] = str(init_result["message"])

            # Before Dispatcher completion, avoid probing heavier endpoints if
            # auth/status still reports unauthenticated: 401-bound requests can
            # overwrite the gateway's shared SSO cookie jar. After Dispatcher we
            # can safely use portfolio/accounts as the readiness signal because
            # auth/status may lag behind the real CP session state.
            should_probe_accounts = session_authenticated or dispatcher_marker_present
            if should_probe_accounts:
                cached_accounts = _get_cached_client_portal_accounts(base_url)
                if not _should_probe_client_portal_accounts(base_url):
                    if cached_accounts:
                        result["accounts"] = cached_accounts
                        session_authenticated = True
                        result["gateway_session_ready"] = True
                        result["session_authenticated"] = True
                        result["authenticated"] = True
                else:
                    accounts_response = await client.get("/v1/api/portfolio/accounts")
                    if accounts_response.status_code != 401:
                        accounts_response.raise_for_status()
                        accounts = _extract_accounts(accounts_response.json())
                        result["accounts"] = accounts
                        _store_client_portal_accounts(base_url, accounts)
                        session_authenticated = True
                        result["gateway_session_ready"] = True
                        result["session_authenticated"] = True
                        result["authenticated"] = True
                    elif cached_accounts:
                        result["accounts"] = cached_accounts
                        session_authenticated = True
                        result["gateway_session_ready"] = True
                        result["session_authenticated"] = True
                        result["authenticated"] = True

            if session_authenticated:
                try:
                    bridge_response = await client.get("/v1/api/iserver/accounts")
                    if bridge_response.status_code != 401:
                        bridge_response.raise_for_status()
                        result["bridge_ready"] = _extract_bridge_ready(bridge_response.json())
                        if result["bridge_ready"]:
                            result["session_authenticated"] = True
                            result["authenticated"] = True
                except httpx.HTTPStatusError as bridge_exc:
                    if bridge_exc.response.status_code == 400 and "no bridge" in bridge_exc.response.text.lower():
                        result["bridge_ready"] = False
                        if not competing:
                            result["message"] = "Sessione Client Portal autenticata ma bridge brokerage non pronto. Attendi il completamento del login IBKR."
                    elif bridge_exc.response.status_code != 401:
                        raise

            result["ready_to_connect"] = bool(result["gateway_session_ready"] and result["bridge_ready"])

            if not session_authenticated and not result["message"]:
                if result["gateway_session_ready"]:
                    result["message"] = "Login Client Portal completato. In attesa dell'apertura della brokerage session IBKR."
                else:
                    result["message"] = "Sessione Client Portal non autenticata. Completa il login nel popup."
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            result["service_ready"] = True
            result["gateway_session_ready"] = False
            result["connected"] = False
            result["session_authenticated"] = False
            result["authenticated"] = False
            result["established"] = False
            result["competing"] = False
            result["ready_to_connect"] = False
            result["message"] = "Sessione Client Portal non autenticata. Completa il login nel popup."
        else:
            result["message"] = str(exc)
    except Exception as exc:
        result["message"] = str(exc)

    return result


async def logout_client_portal_session(config: dict[str, Any] | None = None) -> dict[str, Any]:
    base_url = resolve_client_portal_base_url(config)
    verify_ssl = resolve_client_portal_verify_ssl(config)

    result: dict[str, Any] = {
        "service_ready": False,
        "logged_out": False,
        "message": None,
    }

    try:
        async with httpx.AsyncClient(
            base_url=base_url,
            verify=verify_ssl,
            timeout=10.0,
            follow_redirects=True,
            headers=_client_portal_proxy_headers(),
        ) as client:
            response = await client.post("/v1/api/logout", headers={"Content-Length": "0"})
            if response.status_code not in (200, 204, 401):
                response.raise_for_status()
            result["service_ready"] = True
            result["logged_out"] = response.status_code in (200, 204, 401)
            if result["logged_out"]:
                _clear_client_portal_probe_state(base_url)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            result["service_ready"] = True
            result["logged_out"] = True
            _clear_client_portal_probe_state(base_url)
        else:
            result["message"] = str(exc)
    except Exception as exc:
        result["message"] = str(exc)

    return result