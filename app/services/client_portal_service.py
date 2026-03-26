from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def resolve_client_portal_base_url(config: dict[str, Any] | None = None) -> str:
    config = config or {}
    return str(
        config.get(
            "client_portal_base_url",
            os.getenv("CLIENT_PORTAL_BASE_URL", "https://ibkr-client-portal-gw:5000"),
        )
    ).rstrip("/")


def resolve_client_portal_browser_url(config: dict[str, Any] | None = None) -> str:
    config = config or {}
    browser_url = str(
        config.get(
            "client_portal_browser_url",
            os.getenv("CLIENT_PORTAL_BROWSER_URL", "https://localhost:5000/"),
        )
    )
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
    config = config or {}
    return _as_bool(
        config.get("client_portal_verify_ssl", os.getenv("CLIENT_PORTAL_VERIFY_SSL", "false")),
        default=False,
    )


def is_client_portal_transport(config: dict[str, Any] | None = None) -> bool:
    config = config or {}
    transport = str(config.get("transport", "legacy")).strip().lower()
    enabled = _as_bool(config.get("client_portal_enabled"), default=transport == "client_portal")
    return transport == "client_portal" or enabled


def _extract_authenticated(payload: Any) -> bool:
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
    if not isinstance(payload, dict):
        return False

    session = payload.get("session")
    sso_expires = payload.get("ssoExpires")
    user_id = payload.get("userId")
    return bool(session or sso_expires or user_id)


def _extract_competing(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return bool(payload.get("competing"))


def _extract_accounts(payload: Any) -> list[dict[str, Any]]:
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
    if isinstance(payload, list):
        return len(payload) > 0
    if isinstance(payload, dict):
        accounts = payload.get("accounts")
        return isinstance(accounts, list) and len(accounts) > 0
    return False


async def get_client_portal_auth_status(config: dict[str, Any] | None = None) -> dict[str, Any]:
    base_url = resolve_client_portal_base_url(config)
    browser_url = resolve_client_portal_browser_url(config)
    verify_ssl = resolve_client_portal_verify_ssl(config)

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
        async with httpx.AsyncClient(base_url=base_url, verify=verify_ssl, timeout=10.0, follow_redirects=True) as client:
            payload = None
            tickle_payload = None
            response = await client.get("/v1/api/iserver/auth/status")
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
                result["message"] = "Sessione Client Portal non autenticata. Completa il login nel popup."
                return result

            response.raise_for_status()
            payload = response.json()
            result["payload"] = payload

            try:
                tickle_response = await client.get("/v1/api/tickle")
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
            result["gateway_session_ready"] = _extract_gateway_session_ready(tickle_payload)
            result["connected"] = flags["connected"]
            result["session_authenticated"] = session_authenticated
            result["authenticated"] = flags["authenticated"]
            result["established"] = flags["established"]
            result["competing"] = competing
            result["message"] = payload.get("message") if isinstance(payload, dict) else None

            if competing:
                result["message"] = (
                    "Client Portal segnala una sessione concorrente. Chiudi eventuali altre sessioni IBKR/TWS/Client Portal e riprova."
                )

            # Only query heavy endpoints once authenticated — during login
            # these would all return 401 and further pollute the cookie jar.
            if session_authenticated:
                accounts_response = await client.get("/v1/api/portfolio/accounts")
                if accounts_response.status_code != 401:
                    accounts_response.raise_for_status()
                    result["accounts"] = _extract_accounts(accounts_response.json())

                try:
                    bridge_response = await client.get("/v1/api/iserver/accounts")
                    if bridge_response.status_code != 401:
                        bridge_response.raise_for_status()
                        result["bridge_ready"] = _extract_bridge_ready(bridge_response.json())
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
        async with httpx.AsyncClient(base_url=base_url, verify=verify_ssl, timeout=10.0, follow_redirects=True) as client:
            response = await client.post("/v1/api/logout", headers={"Content-Length": "0"})
            if response.status_code not in (200, 204, 401):
                response.raise_for_status()
            result["service_ready"] = True
            result["logged_out"] = response.status_code in (200, 204, 401)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            result["service_ready"] = True
            result["logged_out"] = True
        else:
            result["message"] = str(exc)
    except Exception as exc:
        result["message"] = str(exc)

    return result