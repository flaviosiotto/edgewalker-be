"""
Connection Manager – async background service that manages broker
connect / disconnect lifecycle and auto-discovers accounts.

**Architecture (gateway-per-connection):**

When the user clicks "Connect" the manager spawns a dedicated
``gateway`` Docker container for that Connection.  The container
maintains the persistent broker connection(s) and exposes a REST API.
All further operations (search, streaming, historical fetch, orders)
are routed through that container via ``GatewayClient``.

One container per Connection = one Gateway = complete user segregation.
New brokers are registered in ``GATEWAY_REGISTRY`` — no code changes
required in ConnectionManager itself.
"""
from __future__ import annotations

import asyncio
import httpx
import json
import logging
import os
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import docker
from docker.errors import NotFound, APIError, ContainerError
from sqlmodel import Session, select

from app.db.database import get_session_context
from app.models.connection import Account, Connection, ConnectionStatus, AccountSnapshot
from app.services.broker_connectors.base import ConnectorResult, DiscoveredAccount
from app.services.tws_launch_service import (
    clear_tws_launch_session,
    create_tws_launch_url,
)
from app.services.gateway_client import GatewayClient

logger = logging.getLogger(__name__)


def _env_first(*names: str, default: str = "") -> str:
    """Read the first non-empty env var among *names* (legacy-name fallbacks)."""
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip() != "":
            return value
    return default


# ── Docker settings ──────────────────────────────────────────────────
DOCKER_NETWORK = os.getenv("DOCKER_NETWORK", "edgewalker-devops_default")

# Paths for volume mounts — these are HOST paths passed to Docker bind mounts,
# so they MUST be absolute host paths (not container-relative).
# Do NOT use os.path.abspath() here: it resolves against the container's CWD.
EDGEWALKER_PATH = os.getenv("EDGEWALKER_PATH", "/home/flavio/playground/edgewalker")
RUNTIME_PATH = os.getenv("RUNTIME_PATH", "/home/flavio/playground/edgewalker-runtime")

if not os.path.isabs(EDGEWALKER_PATH) or not os.path.isabs(RUNTIME_PATH):
    logger.warning(
        "EDGEWALKER_PATH and RUNTIME_PATH must be absolute host paths for Docker "
        "bind mounts. Current values: EDGEWALKER_PATH=%s, RUNTIME_PATH=%s",
        EDGEWALKER_PATH, RUNTIME_PATH,
    )

# Optional container user override for dynamically spawned gateways.
# In local dev we often set PUID/PGID to keep host-written files owned by the
# workstation user. In production bind mounts may be owned by root or another
# service account, so forcing 1000:1000 by default can break parquet writes.
HOST_PUID = os.getenv("PUID", "").strip()
HOST_PGID = os.getenv("PGID", "").strip()
SPAWN_CONTAINER_USER = os.getenv("SPAWN_CONTAINER_USER", "").strip()
SPAWN_CODE_MOUNTS = os.getenv("SPAWN_CODE_MOUNTS", "false").lower() == "true"
# Background connection-health loop: probes active connections off the request
# path so GET /connections can read straight from the DB. A short status-probe
# timeout keeps a single hung gateway from stalling the whole reconciliation.
CONNECTION_HEALTH_INTERVAL_SECONDS = max(
    5.0,
    float(os.getenv("CONNECTION_HEALTH_INTERVAL_SECONDS", "20")),
)
GATEWAY_STATUS_PROBE_TIMEOUT_SECONDS = max(
    2.0,
    float(os.getenv("GATEWAY_STATUS_PROBE_TIMEOUT_SECONDS", "8")),
)

# ── Interactive IB Gateway (TWS) runtime lifecycle ─────────────────
# Legacy CLIENT_PORTAL_* env names are accepted as fallbacks for deployment
# continuity (same convention as the retained Traefik router names).
TWS_RUNTIME_IDLE_TIMEOUT_SECONDS = max(
    0,
    int(_env_first(
        "TWS_RUNTIME_IDLE_TIMEOUT_SECONDS",
        "CLIENT_PORTAL_RUNTIME_IDLE_TIMEOUT_SECONDS",
        "TWS_LAUNCH_TTL_SECONDS",
        "CLIENT_PORTAL_LAUNCH_TTL_SECONDS",
        default="900",
    )),
)
# A login still in `awaiting_auth` is an in-progress interaction (the user may
# be entering credentials / 2FA in the popup). Give it a dedicated, longer
# grace so the idle reaper can't kill the runtime mid-login.
TWS_AUTH_RUNTIME_IDLE_TIMEOUT_SECONDS = max(
    TWS_RUNTIME_IDLE_TIMEOUT_SECONDS,
    int(_env_first(
        "TWS_AUTH_RUNTIME_IDLE_TIMEOUT_SECONDS",
        "CLIENT_PORTAL_AUTH_RUNTIME_IDLE_TIMEOUT_SECONDS",
        default="1800",
    )),
)
TWS_RUNTIME_CLEANUP_INTERVAL_SECONDS = max(
    5.0,
    float(_env_first(
        "TWS_RUNTIME_CLEANUP_INTERVAL_SECONDS",
        "CLIENT_PORTAL_RUNTIME_CLEANUP_INTERVAL_SECONDS",
        default="30",
    )),
)

# ── Path-based browser routing for the noVNC login popup ────────────
# The browser reaches each spawned IB Gateway container through Traefik under
# a stable per-connection path prefix (e.g. https://app.../ib-access/{id}/).
TWS_PATH_ROUTING_ENABLED = (
    _env_first("TWS_PATH_ROUTING_ENABLED", "CLIENT_PORTAL_PATH_ROUTING_ENABLED", default="false").lower()
    == "true"
)
TWS_ROUTING_HOST = _env_first("TWS_ROUTING_HOST", "CLIENT_PORTAL_ROUTING_HOST").strip()
TWS_TRAEFIK_NETWORK = _env_first(
    "TWS_TRAEFIK_NETWORK", "CLIENT_PORTAL_TRAEFIK_NETWORK", "DOCKER_NETWORK"
).strip()
TWS_TRAEFIK_ENTRYPOINT = _env_first(
    "TWS_TRAEFIK_ENTRYPOINT", "CLIENT_PORTAL_TRAEFIK_ENTRYPOINT", default="websecure"
).strip()
TWS_TRAEFIK_CERTRESOLVER = _env_first(
    "TWS_TRAEFIK_CERTRESOLVER", "CLIENT_PORTAL_TRAEFIK_CERTRESOLVER", default="letsencrypt"
).strip()
# Whether the per-connection twsgw router terminates TLS. Set to "false" for
# local development where Traefik only exposes a plain-HTTP entrypoint.
TWS_TRAEFIK_TLS = (
    _env_first("TWS_TRAEFIK_TLS", "CLIENT_PORTAL_TRAEFIK_TLS", default="true").strip().lower() != "false"
)
TWS_TRAEFIK_ROUTER_PRIORITY = _env_first(
    "TWS_TRAEFIK_ROUTER_PRIORITY", "CLIENT_PORTAL_TRAEFIK_ROUTER_PRIORITY", default="1000"
).strip()
# forwardAuth gate: Traefik calls this backend URL before forwarding the
# browser to the container; the backend validates the short-lived launch
# cookie and that the requesting user owns the connection.
TWS_FORWARD_AUTH_URL_TEMPLATE = _env_first(
    "TWS_FORWARD_AUTH_URL_TEMPLATE", "CLIENT_PORTAL_FORWARD_AUTH_URL_TEMPLATE"
).strip()
# Traefik API base URL used to confirm the per-connection router has actually
# been discovered before the backend hands the launch URL to the browser.
# Empty disables the router-readiness wait.
TWS_TRAEFIK_API_URL = _env_first(
    "TWS_TRAEFIK_API_URL", "CLIENT_PORTAL_TRAEFIK_API_URL"
).strip().rstrip("/")
TWS_ROUTER_READY_TIMEOUT_SECONDS = max(
    0.0,
    float(_env_first(
        "TWS_ROUTER_READY_TIMEOUT_SECONDS", "CLIENT_PORTAL_ROUTER_READY_TIMEOUT_SECONDS", default="15"
    )),
)
TWS_ROUTER_READY_POLL_INTERVAL_SECONDS = max(
    0.25,
    float(_env_first(
        "TWS_ROUTER_READY_POLL_INTERVAL_SECONDS",
        "CLIENT_PORTAL_ROUTER_READY_POLL_INTERVAL_SECONDS",
        default="0.5",
    )),
)

# ── IB Gateway / TWS interactive runtime ───────────────────────────
# This runtime is the vendor IB Gateway UI/API process exposed through noVNC.
# IB Gateway only completes API handshakes for localhost clients in this image,
# so twsgw-{id} exposes a local TCP proxy on API port + offset. The Edgewalker
# broker gateway remains a separate gw-{id} container and connects to the proxy.
TWS_GATEWAY_IMAGE = os.getenv(
    "TWS_GATEWAY_IMAGE",
    "edgewalker-devops-ib-gateway-tws:latest",
).strip()
TWS_GATEWAY_PREFIX = os.getenv("TWS_GATEWAY_PREFIX", "twsgw-").strip() or "twsgw-"
TWS_GATEWAY_API_PORT = max(1, int(os.getenv("TWS_GATEWAY_API_PORT", "4002")))
TWS_GATEWAY_LIVE_API_PORT = max(1, int(os.getenv("TWS_GATEWAY_LIVE_API_PORT", "4001")))
TWS_GATEWAY_PAPER_API_PORT = max(1, int(os.getenv("TWS_GATEWAY_PAPER_API_PORT", str(TWS_GATEWAY_API_PORT))))
TWS_GATEWAY_API_PROXY_OFFSET = max(1, int(os.getenv("TWS_GATEWAY_API_PROXY_OFFSET", "100")))
TWS_NOVNC_PORT = max(1, int(os.getenv("TWS_NOVNC_PORT", "6080")))
TWS_READY_POLL_INTERVAL_SECONDS = max(0.5, float(os.getenv("TWS_READY_POLL_INTERVAL_SECONDS", "2")))
TWS_READY_TIMEOUT_SECONDS = max(5.0, float(os.getenv("TWS_READY_TIMEOUT_SECONDS", "90")))
TWS_PATH_PREFIX_BASE = _env_first(
    "TWS_PATH_PREFIX_BASE", "CLIENT_PORTAL_PATH_PREFIX_BASE", default="/ib-access"
).strip().rstrip("/") or "/ib-access"
TWS_API_PROBE_TIMEOUT_SECONDS = max(3.0, float(os.getenv("TWS_API_PROBE_TIMEOUT_SECONDS", "15")))
TWS_API_PROBE_CACHE_SECONDS = max(0.0, float(os.getenv("TWS_API_PROBE_CACHE_SECONDS", "8")))
# Internal retries inside the probe so a transient handshake hiccup (or accounts
# not yet published right after login) doesn't read as "not ready".
TWS_API_PROBE_RETRIES = max(1, int(os.getenv("TWS_API_PROBE_RETRIES", "2")))
# The readonly readiness probe must NOT reuse the data gateway's clientId. The
# gateway connects RT with ``client_id`` (default 100) and HIST with ``client_id+1``;
# if the probe grabbed the same id, IB Gateway would either reject the probe or,
# worse, still hold the id for a few seconds after the probe disconnects so the
# gateway's own RT connect fails with error 326 ("client id already in use").
# Reserve a high, dedicated offset so probe and gateway never compete.
TWS_API_PROBE_CLIENT_ID_OFFSET = max(2, int(os.getenv("TWS_API_PROBE_CLIENT_ID_OFFSET", "90")))
# When enabled, a dedicated background loop is the single authority that finishes
# an interactive IB Gateway login (probe → complete). The frontend is a pure
# observer, so a login completes even if the user reloaded the page, closed the
# tab, or the popup flow already timed out. Disable to fall back to manual-only.
TWS_AUTO_COMPLETE_ENABLED = os.getenv("TWS_AUTO_COMPLETE_ENABLED", "true").lower() == "true"
# Fast cadence (independent of the heavier all-connections health loop) so a
# finished login is promoted to "connected" within a few seconds.
TWS_AUTH_RECONCILE_INTERVAL_SECONDS = max(
    1.0,
    float(os.getenv("TWS_AUTH_RECONCILE_INTERVAL_SECONDS", "4")),
)


_TWS_API_PROBE_SCRIPT = r"""
import asyncio
import json
import os
import sys

from ib_async import IB


def emit(payload, *, failed=False):
    print(json.dumps(payload, separators=(",", ":")), file=sys.stderr if failed else sys.stdout, flush=True)


async def attempt(host, port, client_id, timeout):
    # Returns (ready, accounts_count, server_version) or raises.
    ib = IB()
    try:
        await asyncio.wait_for(
            ib.connectAsync(host=host, port=port, clientId=client_id, readonly=True),
            timeout=timeout,
        )
        # The API server accepts clients as soon as the gateway process is up,
        # which can be BEFORE the login fully settles and managed accounts are
        # published. Wait briefly for at least one account so "ready" means the
        # brokerage session is actually usable, not just that the socket answers.
        accounts = list(ib.managedAccounts() or [])
        deadline = asyncio.get_event_loop().time() + min(5.0, timeout)
        while not accounts and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.5)
            accounts = list(ib.managedAccounts() or [])
        return (bool(accounts), len(accounts), ib.client.serverVersion())
    finally:
        if ib.isConnected():
            ib.disconnect()


async def main():
    host = os.environ["IBKR_HOST"]
    port = int(os.environ["IBKR_PORT"])
    client_id = int(os.environ.get("IBKR_CLIENT_ID", "100"))
    timeout = float(os.environ.get("IBKR_PROBE_TIMEOUT", "15"))
    retries = max(1, int(os.environ.get("IBKR_PROBE_RETRIES", "3")))

    last_error_type = None
    last_error = None
    accounts_count = 0
    server_version = None
    per_attempt_timeout = max(3.0, timeout / retries)

    for i in range(retries):
        try:
            ready, accounts_count, server_version = await attempt(
                host, port, client_id, per_attempt_timeout
            )
            if ready:
                emit({
                    "ready": True,
                    "host": host,
                    "port": port,
                    "client_id": client_id,
                    "server_version": server_version,
                    "accounts_count": accounts_count,
                })
                return 0
            last_error_type = "NoManagedAccounts"
            last_error = "API handshake OK but no managed accounts yet (login not fully settled)"
        except asyncio.TimeoutError:
            last_error_type = "TimeoutError"
            last_error = f"timeout after {per_attempt_timeout:.1f}s waiting for IB API handshake"
        except Exception as exc:
            last_error_type = type(exc).__name__
            last_error = str(exc) or repr(exc)
        if i < retries - 1:
            await asyncio.sleep(1.0)

    emit({
        "ready": False,
        "host": host,
        "port": port,
        "client_id": client_id,
        "accounts_count": accounts_count,
        "error_type": last_error_type,
        "error": last_error,
    }, failed=True)
    return 1


raise SystemExit(asyncio.run(main()))
"""


def _tws_path_prefix(connection_id: int) -> str:
    return f"{TWS_PATH_PREFIX_BASE}/{connection_id}"


# Shared Redis settings propagated to dynamically spawned gateway containers.
REDIS_URL = os.getenv("REDIS_URL", "")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_USERNAME = os.getenv("REDIS_USERNAME", "")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
BROKER_SYNC_STREAM = os.getenv("BROKER_SYNC_STREAM", "events:broker-sync")
# Broker account state is projected by order-aggregator (broker.account.sync
# -> accounts/account_snapshots); the backend only discovers accounts at connect.
TWS_RUNTIME_HOST_KEY = "_tws_runtime_host"
TWS_RUNTIME_PORT_KEY = "_tws_runtime_port"
TWS_RUNTIME_CONTAINER_NAME_KEY = "_tws_runtime_container_name"
TWS_RUNTIME_SESSION_ID_KEY = "_tws_runtime_session_id"


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, "", "N/A"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_snapshot_at(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _with_tws_runtime_state(
    config: dict[str, Any] | None,
    *,
    host: str,
    port: int,
    container_name: str,
    runtime_session_id: str,
) -> dict[str, Any]:
    updated = dict(config or {})
    updated[TWS_RUNTIME_HOST_KEY] = host.strip()
    updated[TWS_RUNTIME_PORT_KEY] = int(port)
    updated[TWS_RUNTIME_CONTAINER_NAME_KEY] = container_name
    updated[TWS_RUNTIME_SESSION_ID_KEY] = runtime_session_id.strip()
    return updated


def _clear_tws_runtime_state(config: dict[str, Any] | None) -> dict[str, Any]:
    updated = dict(config or {})
    updated.pop(TWS_RUNTIME_HOST_KEY, None)
    updated.pop(TWS_RUNTIME_PORT_KEY, None)
    updated.pop(TWS_RUNTIME_CONTAINER_NAME_KEY, None)
    updated.pop(TWS_RUNTIME_SESSION_ID_KEY, None)
    return updated


def _has_tws_runtime_state(config: dict[str, Any] | None) -> bool:
    if not isinstance(config, dict):
        return False
    host = config.get(TWS_RUNTIME_HOST_KEY)
    return isinstance(host, str) and bool(host.strip())


def _tws_api_port_for_config(config: dict[str, Any] | None) -> int:
    if isinstance(config, dict):
        for key in ("tws_api_port", "api_port"):
            value = config.get(key)
            if value is None:
                continue
            try:
                return max(1, int(value))
            except (TypeError, ValueError):
                continue
        if str(config.get("trading_mode", "paper")).strip().lower() == "live":
            return TWS_GATEWAY_LIVE_API_PORT
    return TWS_GATEWAY_PAPER_API_PORT


def _tws_api_proxy_port_for_config(config: dict[str, Any] | None) -> int:
    if isinstance(config, dict):
        for key in ("tws_api_proxy_port", "api_proxy_port"):
            value = config.get(key)
            if value is None:
                continue
            try:
                return max(1, int(value))
            except (TypeError, ValueError):
                continue
    return _tws_api_port_for_config(config) + TWS_GATEWAY_API_PROXY_OFFSET


def _first_non_empty_config_value(config: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = config.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _docker_runtime_requirements() -> str:
    return (
        "backend must have Docker Engine access (for example via /var/run/docker.sock or DOCKER_HOST), "
        f"the Docker network '{DOCKER_NETWORK}' must exist, and host paths EDGEWALKER_PATH={EDGEWALKER_PATH} "
        f"and RUNTIME_PATH={RUNTIME_PATH} must be valid absolute paths on the Docker host; "
        "if bind-mounted data paths are not writable by the spawned container user, parquet writes will fail"
    )


def _spawn_container_user() -> str | None:
    if SPAWN_CONTAINER_USER:
        return SPAWN_CONTAINER_USER
    if HOST_PUID and HOST_PGID:
        return f"{HOST_PUID}:{HOST_PGID}"
    return None


# ── Gateway Registry ─────────────────────────────────────────────────
# Each broker type maps to its Docker image, container prefix, label,
# and a function that builds the environment dict from Connection.config.
# To add a new broker, add an entry here — no other code changes needed
# in ConnectionManager.

def _ibkr_env(config: dict[str, Any]) -> dict[str, str]:
    """Build env vars for a gateway IBKR container.

    The data gateway always targets the interactive ``twsgw-{id}`` runtime
    (its API proxy) recorded in the connection config at login time.
    """
    ibkr_host = str(config.get(TWS_RUNTIME_HOST_KEY) or "")
    ibkr_port = str(config.get(TWS_RUNTIME_PORT_KEY) or _tws_api_proxy_port_for_config(config))
    return {
        "IBKR_HOST": ibkr_host,
        "IBKR_PORT": ibkr_port,
        "IBKR_CLIENT_ID": str(config.get("client_id", 100)),
    }


def _binance_env(config: dict[str, Any]) -> dict[str, str]:
    """Build env vars for a gateway Binance container."""
    market_type = str(config.get("market_type", "futures") or "futures").strip().lower()
    if market_type not in {"spot", "futures"}:
        raise ValueError("Binance market_type must be 'spot' or 'futures'")
    return {
        "BINANCE_API_KEY": str(config.get("api_key", "")),
        "BINANCE_API_SECRET": str(config.get("api_secret", "")),
        "BINANCE_TESTNET": str(config.get("testnet", False)).lower(),
        "BINANCE_MARKET_TYPE": market_type,
    }


def _ctrader_env(config: dict[str, Any]) -> dict[str, str]:
    """Build env vars for a gateway cTrader Open API container."""
    environment = str(config.get("environment", "demo") or "demo").strip().lower()
    if environment not in {"demo", "live"}:
        raise ValueError("cTrader environment must be 'demo' or 'live'")
    client_id = str(config.get("client_id") or os.getenv("CTRADER_OAUTH_CLIENT_ID", ""))
    client_secret = str(config.get("client_secret") or os.getenv("CTRADER_OAUTH_CLIENT_SECRET", ""))

    return {
        "CTRADER_CLIENT_ID": client_id,
        "CTRADER_CLIENT_SECRET": client_secret,
        "CTRADER_ACCESS_TOKEN": str(config.get("access_token", "")),
        "CTRADER_ACCOUNT_ID": str(config.get("account_id", "")),
        "CTRADER_ENVIRONMENT": environment,
        "CTRADER_HOST": str(config.get("host", "")),
        "CTRADER_PORT": str(config.get("port", 5035)),
        "CTRADER_VOLUME_SCALE": str(config.get("volume_scale", 100)),
    }


def _ctrader_config_error(config: dict[str, Any]) -> str | None:
    """Return a user-facing cTrader configuration error before spawning a gateway."""
    environment = str(config.get("environment", "demo") or "demo").strip().lower()
    if environment not in {"demo", "live"}:
        return "cTrader environment must be 'demo' or 'live'"
    client_id = str(config.get("client_id") or os.getenv("CTRADER_OAUTH_CLIENT_ID", "")).strip()
    client_secret = str(config.get("client_secret") or os.getenv("CTRADER_OAUTH_CLIENT_SECRET", "")).strip()
    if not client_id or not client_secret:
        return "Applicazione OAuth cTrader non configurata sul backend"
    if not str(config.get("access_token") or "").strip():
        return "Access token cTrader mancante. Completa Autorizza e Scambia nella configurazione cTrader, poi salva la connessione."
    return None


def resolve_order_history_lookback_days(
    config: dict[str, Any] | None,
    *,
    default_days: int = 1,
) -> int:
    config = config or {}
    raw_days = config.get("order_history_lookback_days")
    if raw_days not in (None, ""):
        try:
            return max(int(raw_days), 1)
        except (TypeError, ValueError):
            return default_days

    raw_hours = config.get("order_history_lookback_hours")
    if raw_hours in (None, ""):
        return default_days

    try:
        parsed_hours = int(raw_hours)
    except (TypeError, ValueError):
        return default_days

    return max((max(parsed_hours, 1) + 23) // 24, 1)


def _broker_sync_env(config: dict[str, Any]) -> dict[str, str]:
    lookback_days = resolve_order_history_lookback_days(config)
    return {
        "BROKER_SYNC_HISTORY_LOOKBACK_S": str(lookback_days * 86400),
    }


@dataclass
class GatewaySpec:
    """Specification for a broker gateway Docker container."""
    image: str
    prefix: str
    label: str
    env_mapper: Any  # Callable[[dict], dict[str, str]]
    # Dev-overlay volume: <broker>-gateway/app → /app/app
    app_dir: str
    # Extra Docker host entries (e.g. host.docker.internal)
    extra_hosts: dict[str, str] | None = None


_DEFAULT_GATEWAY_IMAGE = os.getenv("GATEWAY_IMAGE", "edgewalker-devops-gateway:latest")
_IBKR_GATEWAY_IMAGE = os.getenv("IBKR_GATEWAY_IMAGE", _DEFAULT_GATEWAY_IMAGE)
_BINANCE_GATEWAY_IMAGE = os.getenv("BINANCE_GATEWAY_IMAGE", _DEFAULT_GATEWAY_IMAGE)
_CTRADER_GATEWAY_IMAGE = os.getenv("CTRADER_GATEWAY_IMAGE", _DEFAULT_GATEWAY_IMAGE)

GATEWAY_REGISTRY: dict[str, GatewaySpec] = {
    "ibkr": GatewaySpec(
        image=_IBKR_GATEWAY_IMAGE,
        prefix="gw-",
        label="gateway",
        env_mapper=_ibkr_env,
        app_dir="gateway/app",
        extra_hosts={"host.docker.internal": "host-gateway"},
    ),
    "binance": GatewaySpec(
        image=_BINANCE_GATEWAY_IMAGE,
        prefix="gw-",
        label="gateway",
        env_mapper=_binance_env,
        app_dir="gateway/app",
    ),
    "ctrader": GatewaySpec(
        image=_CTRADER_GATEWAY_IMAGE,
        prefix="gw-",
        label="gateway",
        env_mapper=_ctrader_env,
        app_dir="gateway/app",
    ),
}


def get_gateway_spec(broker_type: str) -> GatewaySpec | None:
    """Look up the gateway specification for a broker type."""
    return GATEWAY_REGISTRY.get(broker_type)


# Active gateway clients — keyed by connection_id
_gateway_clients: dict[int, GatewayClient] = {}


def _sync_accounts_from_gateway(
    session: Session,
    connection_id: int,
    accounts: list[dict[str, Any]],
) -> None:
    """Sync accounts discovered by a gateway container.

    ``accounts`` is a list of normalized account-state dicts returned by
    the gateway's ``/accounts`` endpoint.
    """
    discovered = [
        DiscoveredAccount(
            account_id=a["account_id"],
            display_name=a.get("display_name") or a["account_id"],
            account_type=a.get("account_type", "unknown"),
            currency=a.get("currency") or "USD",
            cash_balance=_safe_float(a.get("cash_balance")),
            equity=_safe_float(a.get("equity")),
            buying_power=_safe_float(a.get("buying_power")),
            available_funds=_safe_float(a.get("available_funds")),
            unrealized_pnl=_safe_float(a.get("unrealized_pnl")),
            margin_used=_safe_float(a.get("margin_used")),
            maintenance_margin=_safe_float(a.get("maintenance_margin")),
            init_margin=_safe_float(a.get("init_margin")),
            snapshot_at=_parse_snapshot_at(a.get("snapshot_at")),
            extra=a.get("extra") if isinstance(a.get("extra"), dict) else None,
        )
        for a in accounts
        if a.get("account_id")
    ]
    _sync_accounts(session, connection_id, discovered)


def _upsert_account_snapshot(
    session: Session,
    connection_id: int,
    discovered: DiscoveredAccount,
    *,
    now: datetime | None = None,
) -> Account:
    now = now or datetime.now(timezone.utc)
    stmt = (
        select(Account)
        .where(Account.connection_id == connection_id)
        .where(Account.account_id == discovered.account_id)
    )
    acct = session.exec(stmt).first()
    before = (acct.cash_balance, acct.equity, acct.unrealized_pnl) if acct is not None else (None, None, None)

    if acct is None:
        acct = Account(
            connection_id=connection_id,
            account_id=discovered.account_id,
            display_name=discovered.display_name,
            account_type=discovered.account_type,
            currency=discovered.currency,
            cash_balance=discovered.cash_balance,
            equity=discovered.equity,
            buying_power=discovered.buying_power,
            available_funds=discovered.available_funds,
            unrealized_pnl=discovered.unrealized_pnl,
            margin_used=discovered.margin_used,
            maintenance_margin=discovered.maintenance_margin,
            init_margin=discovered.init_margin,
            snapshot_at=discovered.snapshot_at,
            extra=discovered.extra,
        )
        session.add(acct)
        logger.info(
            "Auto-discovered new account %s for connection %s",
            discovered.account_id,
            connection_id,
        )
        _append_account_history(session, acct, discovered, before=before, now=now)
        return acct

    if discovered.display_name:
        acct.display_name = discovered.display_name
    if discovered.account_type:
        acct.account_type = discovered.account_type
    if discovered.currency:
        acct.currency = discovered.currency
    if discovered.cash_balance is not None:
        acct.cash_balance = discovered.cash_balance
    if discovered.equity is not None:
        acct.equity = discovered.equity
    if discovered.buying_power is not None:
        acct.buying_power = discovered.buying_power
    if discovered.available_funds is not None:
        acct.available_funds = discovered.available_funds
    if discovered.unrealized_pnl is not None:
        acct.unrealized_pnl = discovered.unrealized_pnl
    if discovered.margin_used is not None:
        acct.margin_used = discovered.margin_used
    if discovered.maintenance_margin is not None:
        acct.maintenance_margin = discovered.maintenance_margin
    if discovered.init_margin is not None:
        acct.init_margin = discovered.init_margin
    if discovered.snapshot_at is not None:
        acct.snapshot_at = discovered.snapshot_at
    if discovered.extra is not None:
        acct.extra = discovered.extra
    acct.updated_at = now
    _append_account_history(session, acct, discovered, before=before, now=now)
    return acct


def _append_account_history(
    session: Session,
    acct: Account,
    discovered: DiscoveredAccount,
    *,
    before: tuple[float | None, float | None, float | None],
    now: datetime,
) -> None:
    """Append an account_snapshots row when the broker state actually changed.

    The history feeds the real equity curve and the ledger reconciliation
    (performance_service). Only observed changes are recorded so an idle
    account does not grow the table at poll cadence.
    """
    after = (acct.cash_balance, acct.equity, acct.unrealized_pnl)
    if after == before or all(value is None for value in after):
        return
    if acct.id is None:
        session.flush()
    session.add(
        AccountSnapshot(
            account_id=acct.id,
            observed_at=discovered.snapshot_at or now,
            currency=acct.currency,
            cash_balance=acct.cash_balance,
            equity=acct.equity,
            unrealized_pnl=acct.unrealized_pnl,
            margin_used=acct.margin_used,
            source="broker_sync",
        )
    )


def _sync_accounts(session: Session, connection_id: int, discovered: list[DiscoveredAccount]) -> None:
    """Upsert the accounts discovered on the gateway."""
    now = datetime.now(timezone.utc)

    existing_stmt = select(Account).where(Account.connection_id == connection_id)
    existing: list[Account] = list(session.exec(existing_stmt).all())
    existing_map = {a.account_id: a for a in existing}

    discovered_ids = {d.account_id for d in discovered}

    # Upsert
    for d in discovered:
        acct = _upsert_account_snapshot(
            session,
            connection_id,
            d,
            now=now,
        )
        existing_map[d.account_id] = acct

    session.commit()


def _update_connection_status(
    session: Session,
    connection_id: int,
    status: ConnectionStatus,
    message: str | None = None,
) -> None:
    """Persist connection status in DB."""
    conn = session.get(Connection, connection_id)
    if conn is None:
        return
    conn.status = status.value
    conn.status_message = message
    now = datetime.now(timezone.utc)
    conn.updated_at = now
    conn.last_checked_at = now
    if status == ConnectionStatus.CONNECTED:
        conn.last_connected_at = now
        conn.last_ok_at = now
    session.commit()


def _gateway_status_is_degraded(status: dict[str, Any]) -> bool:
    gateway_state = str(status.get("status") or "").lower()
    return gateway_state == "degraded" or bool(
        status.get("critical_error")
        or status.get("broker_critical_error")
    )


def _gateway_degraded_message(status: dict[str, Any]) -> str:
    for key in ("critical_error", "broker_critical_error", "error", "message"):
        value = status.get(key)
        if value:
            return f"Gateway degraded: {value}"
    return "Gateway is running but the broker connection is degraded"


# ── Public async API (called by route handlers) ─────────────────────

class ConnectionManager:
    """Manages broker connections via per-connection gateway Docker containers.

    For each Connection the manager:
    1. Spawns a dedicated ``<broker>-gw-{id}`` container on connect
    2. Waits for the gateway to report "connected"
    3. Syncs discovered accounts into the DB
    4. Provides a ``GatewayClient`` for the rest of the backend
    5. Destroys the container on disconnect

    Supported brokers are defined in ``GATEWAY_REGISTRY``.
    """

    def __init__(self) -> None:
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._tws_api_probe_cache: dict[int, tuple[float, dict[str, Any]]] = {}
        # Serialize interactive TWS completion per connection so the frontend
        # polling loop and the background health loop can't both run connect()
        # concurrently (double spawn / clientId races).
        self._tws_connect_locks: dict[int, asyncio.Lock] = {}
        try:
            self._docker = docker.from_env()
        except Exception as e:
            logger.warning(
                "Docker not available: %s — gateway management disabled (%s)",
                e,
                _docker_runtime_requirements(),
            )
            self._docker = None

    def _tws_container_name(self, connection_id: int) -> str:
        return f"{TWS_GATEWAY_PREFIX}{connection_id}"

    def _get_tws_container(self, connection_id: int):
        if not self._docker:
            return None
        try:
            return self._docker.containers.get(self._tws_container_name(connection_id))
        except NotFound:
            return None
        except Exception as e:
            logger.warning("Docker error checking TWS container: %s", e)
            return None

    def _tws_traefik_labels(self, connection_id: int, prefix: str) -> dict[str, str]:
        """Build Traefik Docker-provider labels that route the public host's
        ``<prefix>`` to this container's noVNC web port. A stripPrefix
        middleware removes ``<prefix>``; a forwardAuth middleware (when
        configured) gates the browser behind the backend launch session.
        """
        if not TWS_ROUTING_HOST:
            raise RuntimeError(
                "TWS_ROUTING_HOST (or legacy CLIENT_PORTAL_ROUTING_HOST) is required when TWS path routing is enabled"
            )

        router = f"twsgw-{connection_id}"
        service = f"twsgw-{connection_id}"
        strip_middleware = f"twsgw-strip-{connection_id}"
        rule = f"Host(`{TWS_ROUTING_HOST}`) && PathPrefix(`{prefix}`)"
        labels: dict[str, str] = {
            "traefik.enable": "true",
            f"traefik.http.routers.{router}.rule": rule,
            f"traefik.http.routers.{router}.entrypoints": TWS_TRAEFIK_ENTRYPOINT,
            f"traefik.http.routers.{router}.priority": TWS_TRAEFIK_ROUTER_PRIORITY,
            f"traefik.http.routers.{router}.service": service,
            f"traefik.http.services.{service}.loadbalancer.server.port": str(TWS_NOVNC_PORT),
            f"traefik.http.middlewares.{strip_middleware}.stripprefix.prefixes": prefix,
        }
        middlewares = [f"{strip_middleware}@docker"]

        if TWS_TRAEFIK_TLS:
            if TWS_TRAEFIK_CERTRESOLVER:
                labels[f"traefik.http.routers.{router}.tls.certresolver"] = TWS_TRAEFIK_CERTRESOLVER
            else:
                labels[f"traefik.http.routers.{router}.tls"] = "true"

        if TWS_TRAEFIK_NETWORK:
            labels["traefik.docker.network"] = TWS_TRAEFIK_NETWORK

        if TWS_FORWARD_AUTH_URL_TEMPLATE:
            middleware = f"twsgw-auth-{connection_id}"
            auth_url = TWS_FORWARD_AUTH_URL_TEMPLATE.format(connection_id=connection_id)
            labels[f"traefik.http.middlewares.{middleware}.forwardauth.address"] = auth_url
            labels[f"traefik.http.middlewares.{middleware}.forwardauth.trustForwardHeader"] = "true"
            # forwardAuth runs BEFORE stripPrefix so the gate sees the full
            # /ib-access/{id} path the launch cookie was scoped to.
            middlewares.insert(0, f"{middleware}@docker")

        labels[f"traefik.http.routers.{router}.middlewares"] = ",".join(middlewares)
        return labels

    def _spawn_tws_gateway(
        self,
        connection_id: int,
        config: dict[str, Any],
        *,
        allow_recreate: bool = True,
    ) -> tuple[str, int, str]:
        if not self._docker:
            raise RuntimeError(f"Docker is not available; {_docker_runtime_requirements()}")
        if not TWS_GATEWAY_IMAGE:
            raise RuntimeError("TWS_GATEWAY_IMAGE is not configured")

        container_name = self._tws_container_name(connection_id)
        existing = self._get_tws_container(connection_id)
        api_port = _tws_api_port_for_config(config)
        api_proxy_port = _tws_api_proxy_port_for_config(config)
        if existing:
            if existing.status == "running":
                try:
                    stored_port = int(config.get(TWS_RUNTIME_PORT_KEY) or 0)
                except (TypeError, ValueError):
                    stored_port = 0
                if stored_port == api_proxy_port:
                    return container_name, api_proxy_port, existing.id
                if not allow_recreate:
                    # Never tear down a live IB Gateway just to read status: the
                    # user may be mid-login (typing credentials / 2FA) in the popup.
                    logger.debug(
                        "Reusing running TWS container %s despite port mismatch (stored=%s, computed=%s); "
                        "recreate suppressed",
                        container_name,
                        stored_port or "unset",
                        api_proxy_port,
                    )
                    return container_name, stored_port or api_proxy_port, existing.id
                logger.info(
                    "Restarting TWS container %s because runtime API port changed from %s to proxy port %s",
                    container_name,
                    stored_port or "unset",
                    api_proxy_port,
                )
            elif not allow_recreate:
                # Container exists but isn't running and we were asked not to
                # recreate (status-only path): surface the current state instead.
                return container_name, api_proxy_port, existing.id
            existing.remove(force=True)

        self._tws_api_probe_cache.pop(connection_id, None)

        env = {
            "TWS_API_PORT": str(api_port),
            "IB_GATEWAY_API_PORT": str(api_port),
            "TWS_API_PROXY_PORT": str(api_proxy_port),
            "TWS_NOVNC_PORT": str(TWS_NOVNC_PORT),
            "TWS_TRADING_MODE": str(config.get("trading_mode") or config.get("environment") or "paper"),
            "TWS_READ_ONLY_API": str(config.get("read_only_api", False)).lower(),
            "TWS_READ_ONLY_LOGIN": str(config.get("read_only_login", False)).lower(),
            "LOG_LEVEL": "INFO",
        }
        username = _first_non_empty_config_value(config, "username", "tws_username", "ib_username", "ib_login_id")
        password = _first_non_empty_config_value(config, "password", "tws_password", "ib_password", "ib_login_password")
        second_factor_device = _first_non_empty_config_value(config, "second_factor_device", "tws_second_factor_device")
        if username:
            env["TWS_USERNAME"] = username
        if password:
            env["TWS_PASSWORD"] = password
        if second_factor_device:
            env["TWS_SECOND_FACTOR_DEVICE"] = second_factor_device
        if "relogin_after_2fa_timeout" in config:
            env["TWS_RELOGIN_AFTER_2FA_TIMEOUT"] = str(config.get("relogin_after_2fa_timeout", False)).lower()

        labels = {
            "edgewalker.type": "ibkr-tws",
            "edgewalker.connection_id": str(connection_id),
        }

        if TWS_PATH_ROUTING_ENABLED:
            prefix = _tws_path_prefix(connection_id)
            env["TWS_BASE_PATH"] = prefix
            labels.update(self._tws_traefik_labels(connection_id, prefix))

        try:
            user = _spawn_container_user()
            container = self._docker.containers.run(
                image=TWS_GATEWAY_IMAGE,
                name=container_name,
                environment=env,
                labels=labels,
                network=DOCKER_NETWORK,
                user=user,
                detach=True,
                restart_policy={"Name": "no"},
            )
            logger.info("Spawned TWS container %s", container_name)
        except APIError as e:
            logger.error("Failed to spawn TWS container: %s", e)
            raise

        return container_name, api_proxy_port, container.id

    def _destroy_tws_gateway(self, connection_id: int) -> None:
        self._tws_api_probe_cache.pop(connection_id, None)
        container = self._get_tws_container(connection_id)
        if container:
            try:
                container.stop(timeout=10)
                container.remove(force=True)
                logger.info("Destroyed TWS container %s", self._tws_container_name(connection_id))
            except Exception as e:
                logger.warning("Error destroying TWS container: %s", e)

    def _tws_logs_tail(self, container_name: str, *, tail: int = 80) -> str:
        if not self._docker:
            return ""
        try:
            container = self._docker.containers.get(container_name)
            raw = container.logs(tail=tail, stdout=True, stderr=True)
        except Exception as exc:
            logger.warning("Unable to read TWS container logs for %s: %s", container_name, exc)
            return ""
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ""
        return " | ".join(lines[-10:])[:1600]

    async def _wait_for_tws_novnc(self, container_name: str) -> None:
        deadline = asyncio.get_running_loop().time() + TWS_READY_TIMEOUT_SECONDS
        last_error = ""
        base_url = f"http://{container_name}:{TWS_NOVNC_PORT}"
        while asyncio.get_running_loop().time() < deadline:
            if self._docker:
                try:
                    container = self._docker.containers.get(container_name)
                    container.reload()
                    if container.status in {"exited", "dead"}:
                        logs = self._tws_logs_tail(container_name)
                        detail = f"container {container.status}"
                        if logs:
                            detail = f"{detail}; logs: {logs}"
                        raise RuntimeError(detail)
                except NotFound:
                    raise RuntimeError(f"container {container_name} not found")
                except RuntimeError:
                    raise
                except Exception as exc:
                    logger.warning("Unable to inspect TWS container %s while waiting for noVNC: %s", container_name, exc)
            try:
                async with httpx.AsyncClient(base_url=base_url, timeout=5.0, follow_redirects=False) as client:
                    response = await client.get("/")
                if response.status_code < 500:
                    return
                last_error = f"HTTP {response.status_code}"
            except Exception as exc:
                last_error = str(exc)
                if "Name or service not known" in last_error and TWS_PATH_ROUTING_ENABLED:
                    logger.warning(
                        "TWS noVNC container %s is running but not resolvable from backend Docker DNS (%s); "
                        "continuing with Traefik-routed popup readiness",
                        container_name,
                        last_error,
                    )
                    return
            await asyncio.sleep(TWS_READY_POLL_INTERVAL_SECONDS)
        logs = self._tws_logs_tail(container_name)
        detail = f"TWS noVNC runtime did not become ready in time: {last_error or 'timeout'}"
        if logs:
            detail = f"{detail}; logs: {logs}"
        raise RuntimeError(detail)

    def _tws_api_target(self, connection_id: int, config: dict[str, Any]) -> tuple[str, int, int]:
        host = str(config.get(TWS_RUNTIME_HOST_KEY) or self._tws_container_name(connection_id))
        port = int(config.get(TWS_RUNTIME_PORT_KEY) or TWS_GATEWAY_API_PORT)
        try:
            base_client_id = int(config.get("client_id") or 100)
        except (TypeError, ValueError):
            base_client_id = 100
        # Dedicated probe clientId: never the gateway's RT (base) or HIST (base+1).
        client_id = base_client_id + TWS_API_PROBE_CLIENT_ID_OFFSET
        return host, port, client_id

    async def _tws_tcp_ready(self, host: str, port: int) -> bool:
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=3.0)
            writer.close()
            await writer.wait_closed()
            _ = reader
            return True
        except Exception:
            return False

    @staticmethod
    def _parse_tws_api_probe_output(raw: Any) -> dict[str, Any] | None:
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw or "")
        for line in reversed(text.splitlines()):
            candidate = line.strip()
            if not candidate.startswith("{"):
                continue
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        return None

    def _run_tws_api_probe_container(self, connection_id: int, host: str, port: int, client_id: int) -> dict[str, Any]:
        if not self._docker:
            return {
                "ready": False,
                "message": f"Docker non disponibile per il probe TWS API ({_docker_runtime_requirements()})",
            }
        spec = get_gateway_spec("ibkr")
        if spec is None:
            return {"ready": False, "message": "Gateway IBKR non registrato per il probe TWS API"}

        environment = {
            "IBKR_HOST": host,
            "IBKR_PORT": str(port),
            "IBKR_CLIENT_ID": str(client_id),
            "IBKR_PROBE_TIMEOUT": str(TWS_API_PROBE_TIMEOUT_SECONDS),
            "IBKR_PROBE_RETRIES": str(TWS_API_PROBE_RETRIES),
        }
        labels = {
            "edgewalker.type": "tws-api-probe",
            "edgewalker.connection_id": str(connection_id),
        }
        try:
            raw = self._docker.containers.run(
                image=spec.image,
                command=["python", "-c", _TWS_API_PROBE_SCRIPT],
                environment=environment,
                labels=labels,
                network=DOCKER_NETWORK,
                user=_spawn_container_user(),
                detach=False,
                remove=True,
                stdout=True,
                stderr=True,
            )
        except ContainerError as exc:
            payload = self._parse_tws_api_probe_output(getattr(exc, "stderr", b""))
            if payload is None:
                payload = {
                    "ready": False,
                    "error_type": "ContainerError",
                    "error": str(exc),
                }
        except Exception as exc:
            payload = {
                "ready": False,
                "error_type": type(exc).__name__,
                "error": str(exc) or repr(exc),
            }
        else:
            payload = self._parse_tws_api_probe_output(raw) or {
                "ready": False,
                "error_type": "ProbeOutputError",
                "error": "probe did not return JSON output",
            }

        ready = bool(payload.get("ready"))
        if ready:
            server_version = payload.get("server_version") or "unknown"
            accounts_count = payload.get("accounts_count")
            account_text = f", accounts={accounts_count}" if accounts_count is not None else ""
            payload["message"] = f"IB Gateway API pronta (server_version={server_version}{account_text})."
        else:
            error = payload.get("error") or payload.get("message") or "unknown error"
            error_type = payload.get("error_type")
            prefix = f"{error_type}: " if error_type else ""
            payload["message"] = f"IB Gateway API non pronta: {prefix}{error}"
        return payload

    async def _probe_tws_api(
        self,
        connection_id: int,
        config: dict[str, Any],
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        host, port, client_id = self._tws_api_target(connection_id, config)
        now = asyncio.get_running_loop().time()
        cached = self._tws_api_probe_cache.get(connection_id)
        if not force and cached and now - cached[0] <= TWS_API_PROBE_CACHE_SECONDS:
            return dict(cached[1])

        if not await self._tws_tcp_ready(host, port):
            payload = {
                "ready": False,
                "host": host,
                "port": port,
                "client_id": client_id,
                "tcp_ready": False,
                "message": "IB Gateway API socket non ancora raggiungibile. Completa il login nel popup.",
            }
            self._tws_api_probe_cache[connection_id] = (now, payload)
            return dict(payload)

        payload = await asyncio.to_thread(
            self._run_tws_api_probe_container,
            connection_id,
            host,
            port,
            client_id,
        )
        payload["tcp_ready"] = True
        self._tws_api_probe_cache[connection_id] = (asyncio.get_running_loop().time(), payload)
        return dict(payload)

    async def _expire_idle_tws_runtime(
        self,
        *,
        connection_id: int,
        user_id: int,
        config: dict[str, Any],
        status_value: str,
        age_seconds: float,
    ) -> None:
        _ = config
        logger.info(
            "Stopping idle TWS container for connection %s after %.0fs (status=%s)",
            connection_id,
            age_seconds,
            status_value,
        )

        # The data gateway is useless without its login runtime — destroy it
        # too, otherwise it lingers forever retrying a dead twsgw host.
        self._destroy_gateway(connection_id, "ibkr")
        self._destroy_tws_gateway(connection_id)

        try:
            await clear_tws_launch_session(connection_id=connection_id, user_id=user_id)
        except Exception as exc:
            logger.warning(
                "Failed to clear stale TWS launch session for connection %s: %s",
                connection_id,
                exc,
            )

        with get_session_context() as session:
            conn = session.get(Connection, connection_id)
            if conn is None:
                return

            conn.status = ConnectionStatus.DISCONNECTED.value
            conn.status_message = "Sessione IB Gateway arrestata automaticamente per inattivita'. Riapri il login se vuoi riprovare."
            conn.updated_at = datetime.now(timezone.utc)
            conn.config = _clear_tws_runtime_state(conn.config)
            session.commit()

    async def _cleanup_idle_tws_runtimes(self) -> None:
        if TWS_RUNTIME_IDLE_TIMEOUT_SECONDS <= 0:
            return

        now = datetime.now(timezone.utc)
        candidates: list[dict[str, Any]] = []

        with get_session_context() as session:
            connections = list(session.exec(select(Connection).where(Connection.broker_type == "ibkr")).all())

        for conn in connections:
            config = dict(conn.config or {})
            if not _has_tws_runtime_state(config):
                continue

            if conn.status not in {
                ConnectionStatus.AWAITING_AUTH.value,
                ConnectionStatus.DISCONNECTED.value,
                ConnectionStatus.ERROR.value,
            }:
                continue

            updated_at = conn.updated_at
            if updated_at is None:
                continue
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=timezone.utc)
            else:
                updated_at = updated_at.astimezone(timezone.utc)

            idle_timeout = (
                TWS_AUTH_RUNTIME_IDLE_TIMEOUT_SECONDS
                if conn.status == ConnectionStatus.AWAITING_AUTH.value
                else TWS_RUNTIME_IDLE_TIMEOUT_SECONDS
            )
            age_seconds = (now - updated_at).total_seconds()
            if age_seconds < idle_timeout:
                continue

            candidates.append(
                {
                    "connection_id": conn.id,
                    "user_id": conn.user_id,
                    "config": config,
                    "status_value": conn.status,
                    "age_seconds": age_seconds,
                }
            )

        for candidate in candidates:
            await self._expire_idle_tws_runtime(**candidate)

    async def _run_tws_runtime_cleanup(self) -> None:
        while self._running:
            try:
                await self._cleanup_idle_tws_runtimes()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("TWS runtime cleanup iteration failed: %s", exc, exc_info=True)

            await asyncio.sleep(TWS_RUNTIME_CLEANUP_INTERVAL_SECONDS)

    async def _wait_for_tws_router(self, connection_id: int) -> None:
        """Wait until Traefik has discovered the per-connection ``twsgw-{id}``
        router. The Docker provider picks up the new container's labels
        asynchronously, so the browser popup can open before the router exists
        and fall through to the lower-priority frontend SPA router.
        """
        if not TWS_TRAEFIK_API_URL or TWS_ROUTER_READY_TIMEOUT_SECONDS <= 0:
            return

        router_name = f"twsgw-{connection_id}"
        deadline = asyncio.get_running_loop().time() + TWS_ROUTER_READY_TIMEOUT_SECONDS
        last_error = ""

        while asyncio.get_running_loop().time() < deadline:
            try:
                async with httpx.AsyncClient(
                    base_url=TWS_TRAEFIK_API_URL,
                    timeout=5.0,
                    follow_redirects=False,
                ) as client:
                    response = await client.get("/api/rawdata")
                if response.status_code < 500:
                    routers = (response.json() or {}).get("routers", {})
                    for name, router in routers.items():
                        if name.split("@", 1)[0] != router_name:
                            continue
                        if str(router.get("status", "")).lower() in ("", "enabled"):
                            logger.info(
                                "Traefik router %s discovered for TWS connection %s",
                                name,
                                connection_id,
                            )
                            return
                    last_error = "router not yet present"
                else:
                    last_error = f"HTTP {response.status_code}"
            except Exception as exc:
                last_error = str(exc)

            await asyncio.sleep(TWS_ROUTER_READY_POLL_INTERVAL_SECONDS)

        logger.warning(
            "Traefik TWS router %s not confirmed within %.0fs for connection %s (%s); proceeding anyway",
            router_name,
            TWS_ROUTER_READY_TIMEOUT_SECONDS,
            connection_id,
            last_error or "timeout",
        )

    async def _ensure_tws_runtime(
        self,
        connection_id: int,
        config: dict[str, Any],
        *,
        allow_recreate: bool = True,
    ) -> dict[str, Any]:
        if not TWS_PATH_ROUTING_ENABLED:
            raise RuntimeError(
                "TWS interactive login requires TWS_PATH_ROUTING_ENABLED=true "
                "(legacy env name: CLIENT_PORTAL_PATH_ROUTING_ENABLED)"
            )

        existing = self._get_tws_container(connection_id)
        runtime_already_running = bool(existing and existing.status == "running")
        container_name, api_port, runtime_session_id = self._spawn_tws_gateway(
            connection_id, config, allow_recreate=allow_recreate
        )
        if not runtime_already_running:
            try:
                await self._wait_for_tws_novnc(container_name)
            except Exception:
                self._destroy_tws_gateway(connection_id)
                raise
        else:
            logger.info(
                "Reusing running TWS container %s for connection %s",
                container_name,
                connection_id,
            )

        if TWS_PATH_ROUTING_ENABLED:
            await self._wait_for_tws_router(connection_id)

        updated_config = _with_tws_runtime_state(
            config,
            host=container_name,
            port=api_port,
            container_name=container_name,
            runtime_session_id=runtime_session_id,
        )
        with get_session_context() as session:
            conn = session.get(Connection, connection_id)
            if conn is not None:
                conn.config = updated_config
                conn.updated_at = datetime.now(timezone.utc)
                session.commit()
        return updated_config

    # ── Container management ─────────────────────────────────────────

    def _container_name(self, connection_id: int, broker_type: str) -> str:
        spec = get_gateway_spec(broker_type)
        prefix = spec.prefix if spec else f"{broker_type}-gw-"
        return f"{prefix}{connection_id}"

    def _get_container(self, connection_id: int, broker_type: str):
        """Get existing gateway container for a connection, if any."""
        if not self._docker:
            return None
        try:
            return self._docker.containers.get(self._container_name(connection_id, broker_type))
        except NotFound:
            return None
        except Exception as e:
            logger.warning("Docker error checking container: %s", e)
            return None

    def _spawn_gateway(self, connection_id: int, broker_type: str, config: dict[str, Any]) -> None:
        """Spawn a gateway container for the given Connection.

        The container image, env vars, and volumes are determined by the
        ``GATEWAY_REGISTRY`` entry for *broker_type*.
        """
        if not self._docker:
            raise RuntimeError(f"Docker is not available; {_docker_runtime_requirements()}")

        spec = get_gateway_spec(broker_type)
        if spec is None:
            raise ValueError(f"No gateway registered for broker type '{broker_type}'")

        container_name = self._container_name(connection_id, broker_type)

        # Remove any existing stopped container
        existing = self._get_container(connection_id, broker_type)
        if existing:
            if existing.status == "running":
                logger.info("Gateway container %s already running", container_name)
                return
            existing.remove(force=True)

        # Common env vars (shared by all gateways)
        env = {
            "BROKER_TYPE": broker_type,
            "REDIS_HOST": REDIS_HOST,
            "REDIS_PORT": REDIS_PORT,
            "CONNECTION_ID": str(connection_id),
            "DATA_DIR": "/opt/edgewalker/data",
            "LOG_LEVEL": "INFO",
            # Forward OTel endpoint so dynamically spawned containers
            # export metrics/traces to the collector.
            "OTEL_EXPORTER_OTLP_ENDPOINT": os.getenv(
                "OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317"
            ),
            "OTEL_SERVICE_NAME": f"{spec.label}-{connection_id}",
        }
        if REDIS_URL:
            env["REDIS_URL"] = REDIS_URL
        if REDIS_USERNAME:
            env["REDIS_USERNAME"] = REDIS_USERNAME
        if REDIS_PASSWORD:
            env["REDIS_PASSWORD"] = REDIS_PASSWORD
        env.update(_broker_sync_env(config))
        # Broker-specific env vars (from registry)
        env.update(spec.env_mapper(config))

        volumes = {
            # Edgewalker configs
            f"{EDGEWALKER_PATH}/configs": {
                "bind": "/opt/edgewalker/configs",
                "mode": "ro",
            },
            # Data directory (read/write — fetch writes parquet)
            f"{EDGEWALKER_PATH}/data": {
                "bind": "/opt/edgewalker/data",
                "mode": "rw",
            },
        }

        if SPAWN_CODE_MOUNTS:
            volumes.update({
                # Shared module (constants, schemas) — local dev overlay
                f"{RUNTIME_PATH}/shared": {
                    "bind": "/app/shared",
                    "mode": "ro",
                },
                # Application code — local dev overlay (live reload)
                f"{RUNTIME_PATH}/{spec.app_dir}": {
                    "bind": "/app/app",
                    "mode": "ro",
                },
            })

        labels = {
            "edgewalker.type": spec.label,
            "edgewalker.broker_type": broker_type,
            "edgewalker.connection_id": str(connection_id),
        }

        extra_hosts = spec.extra_hosts or {}

        try:
            user = _spawn_container_user()
            self._docker.containers.run(
                image=spec.image,
                name=container_name,
                environment=env,
                volumes=volumes,
                labels=labels,
                extra_hosts=extra_hosts or None,
                network=DOCKER_NETWORK,
                user=user,
                detach=True,
                restart_policy={"Name": "no"},
            )
            logger.info("Spawned %s gateway container %s", broker_type, container_name)
        except APIError as e:
            logger.error("Failed to spawn gateway container: %s", e)
            raise

    def _destroy_gateway(self, connection_id: int, broker_type: str) -> None:
        """Stop and remove the gateway container for a Connection."""
        container = self._get_container(connection_id, broker_type)
        if container:
            try:
                container.stop(timeout=10)
                container.remove(force=True)
                logger.info("Destroyed gateway container %s", self._container_name(connection_id, broker_type))
            except Exception as e:
                logger.warning("Error destroying container: %s", e)
        _gateway_clients.pop(connection_id, None)

    def _gateway_logs_tail(self, connection_id: int, broker_type: str, *, tail: int = 80) -> str:
        container = self._get_container(connection_id, broker_type)
        if container is None:
            return ""
        try:
            raw = container.logs(tail=tail, stdout=True, stderr=True)
        except Exception as exc:
            logger.warning("Unable to read gateway logs for connection %s: %s", connection_id, exc)
            return ""
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        # Drop uvicorn access-log noise from the backend's own /status and
        # /health polling so the tail surfaces real broker errors instead.
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
            and '"GET /status' not in line
            and '"GET /health' not in line
        ]
        if not lines:
            return ""
        return " | ".join(lines[-8:])[:1200]

    # ── Gateway client ───────────────────────────────────────────────

    def get_gateway_client(self, connection_id: int, broker_type: str) -> GatewayClient:
        """Get (or create) the HTTP client for a connection's gateway."""
        if connection_id not in _gateway_clients:
            _gateway_clients[connection_id] = GatewayClient(connection_id, broker_type=broker_type)
        return _gateway_clients[connection_id]

    async def begin_tws_auth(self, connection_id: int, *, user_id: int) -> dict[str, Any]:
        with get_session_context() as session:
            conn = session.get(Connection, connection_id)
            if conn is None:
                raise ValueError("Connection not found")
            config = dict(conn.config or {})

        try:
            config = await self._ensure_tws_runtime(connection_id, config)
        except Exception as exc:
            message = f"IB Gateway non raggiungibile: {exc}"
            logger.warning("TWS auth start failed for connection %s: %s", connection_id, exc)
            with get_session_context() as session:
                conn = session.get(Connection, connection_id)
                if conn is not None:
                    conn.status = ConnectionStatus.ERROR.value
                    conn.status_message = message
                    conn.updated_at = datetime.now(timezone.utc)
                    session.commit()
            return {
                "service_ready": False,
                "authenticated": False,
                "ready_to_connect": False,
                "gateway_started": False,
                "connection_status": ConnectionStatus.ERROR.value,
                "launch_url": None,
                "message": message,
            }

        message = "Autenticazione IB Gateway richiesta nel popup."
        with get_session_context() as session:
            conn = session.get(Connection, connection_id)
            if conn is not None:
                conn.status = ConnectionStatus.AWAITING_AUTH.value
                conn.status_message = message
                conn.updated_at = datetime.now(timezone.utc)
                session.commit()

        launch_url = await create_tws_launch_url(
            connection_id=connection_id,
            user_id=user_id,
            config=config,
            force_new=True,
            path_prefix=_tws_path_prefix(connection_id),
        )
        return {
            "service_ready": True,
            "authenticated": False,
            "ready_to_connect": False,
            "gateway_started": True,
            "connection_status": ConnectionStatus.AWAITING_AUTH.value,
            "launch_url": launch_url,
            "message": message,
        }

    async def tws_auth_status(self, connection_id: int, *, user_id: int | None = None) -> dict[str, Any]:
        with get_session_context() as session:
            conn = session.get(Connection, connection_id)
            if conn is None:
                raise ValueError("Connection not found")
            config = dict(conn.config or {})
            status_value = conn.status
            status_message = conn.status_message

        # Status read: never recreate a live runtime (the user may be mid-login).
        config = await self._ensure_tws_runtime(connection_id, config, allow_recreate=False)
        launch_url = None
        if user_id is not None and status_value == ConnectionStatus.AWAITING_AUTH.value:
            launch_url = await create_tws_launch_url(
                connection_id=connection_id,
                user_id=user_id,
                config=config,
                path_prefix=_tws_path_prefix(connection_id),
            )

        container = self._get_tws_container(connection_id)
        gateway_started = bool(container and container.status == "running")
        probe = await self._probe_tws_api(connection_id, config) if gateway_started else {
            "ready": False,
            "message": status_message or "IB Gateway runtime non avviato.",
        }
        ready = gateway_started and bool(probe.get("ready"))
        return {
            "service_ready": gateway_started,
            "authenticated": ready,
            "ready_to_connect": ready,
            "gateway_started": gateway_started,
            "connection_status": status_value,
            "launch_url": launch_url,
            "message": probe.get("message") or status_message or "Completa il login IB Gateway nel popup.",
        }

    def _tws_connect_lock(self, connection_id: int) -> asyncio.Lock:
        lock = self._tws_connect_locks.get(connection_id)
        if lock is None:
            lock = asyncio.Lock()
            self._tws_connect_locks[connection_id] = lock
        return lock

    async def complete_tws_connect(self, connection_id: int) -> ConnectorResult:
        async with self._tws_connect_lock(connection_id):
            with get_session_context() as session:
                conn = session.get(Connection, connection_id)
                if conn is None:
                    return ConnectorResult(success=False, message="Connection not found")
                # A concurrent caller (frontend poll vs. health loop) may have
                # already finished while we waited for the lock.
                if conn.status == ConnectionStatus.CONNECTED.value:
                    return ConnectorResult(success=True, message="IB Gateway già connesso")
                config = dict(conn.config or {})
                user_id = conn.user_id

            try:
                config = await self._ensure_tws_runtime(connection_id, config)
            except Exception as exc:
                return ConnectorResult(success=False, message=f"IB Gateway non raggiungibile: {exc}")

            probe = await self._probe_tws_api(connection_id, config, force=True)
            if not probe.get("ready"):
                message = probe.get("message") or "IB Gateway API non ancora pronta"
                with get_session_context() as session:
                    conn = session.get(Connection, connection_id)
                    if conn is not None:
                        conn.status = ConnectionStatus.AWAITING_AUTH.value
                        conn.status_message = message
                        conn.updated_at = datetime.now(timezone.utc)
                        session.commit()
                return ConnectorResult(success=False, message=message)

            result = await self.connect(connection_id)
            if result.success:
                await clear_tws_launch_session(connection_id=connection_id, user_id=user_id)
            return result

    async def _try_autocomplete_tws_connection(
        self, connection_id: int, config: dict[str, Any]
    ) -> bool:
        """Probe an interactive IB Gateway login and finish it autonomously.

        The actual login happens inside the noVNC popup, which the backend cannot
        observe. Once the TWS API answers we can complete the connection here so
        completion no longer depends on the frontend polling loop being alive
        (page reloaded, popup flow timed out, browser closed, ...).
        """
        container = self._get_tws_container(connection_id)
        if container is None or container.status != "running":
            return False

        probe = await self._probe_tws_api(connection_id, config)
        if not probe.get("ready"):
            return False

        logger.info(
            "TWS API ready for connection %s; auto-completing login from health loop",
            connection_id,
        )
        result = await self.complete_tws_connect(connection_id)
        if not result.success:
            logger.warning(
                "Auto-complete of TWS connection %s failed: %s",
                connection_id,
                result.message,
            )
        return result.success

    # ── Connect ──────────────────────────────────────────────────────

    async def connect(self, connection_id: int) -> ConnectorResult:
        """Connect to a broker by spawning the appropriate gateway container.

        Steps:
        1. Look up the broker type from ``GATEWAY_REGISTRY``
        2. Spawn ``<broker>-gw-{connection_id}`` container
        3. Poll ``/health`` until the gateway is connected (up to 60 s)
        4. Fetch discovered accounts from the gateway
        5. Sync accounts into the DB
        """
        # Load connection from DB
        with get_session_context() as session:
            conn = session.get(Connection, connection_id)
            if conn is None:
                return ConnectorResult(success=False, message="Connection not found")

            broker_type = conn.broker_type
            config = dict(conn.config or {})

            if broker_type == "ctrader":
                ctrader_config_error = _ctrader_config_error(config)
                if ctrader_config_error:
                    conn.status = ConnectionStatus.ERROR.value
                    conn.status_message = ctrader_config_error
                    conn.updated_at = datetime.now(timezone.utc)
                    session.commit()
                    return ConnectorResult(success=False, message=ctrader_config_error)

            if get_gateway_spec(broker_type) is None:
                return ConnectorResult(
                    success=False,
                    message=f"Unsupported broker type: {broker_type}. "
                            f"Registered types: {', '.join(GATEWAY_REGISTRY.keys())}",
                )

            # Mark as "connecting" immediately
            conn.status = ConnectionStatus.CONNECTING.value
            conn.status_message = "Spawning gateway container…"
            session.commit()

        # 1. Spawn the gateway container
        try:
            self._spawn_gateway(connection_id, broker_type, config)
        except Exception as e:
            with get_session_context() as session:
                _update_connection_status(
                    session, connection_id, ConnectionStatus.ERROR,
                    f"Failed to spawn gateway: {e}",
                )
            return ConnectorResult(success=False, message=f"Failed to spawn gateway: {e}")

        # 2. Wait for the gateway to become healthy / connected
        client = self.get_gateway_client(connection_id, broker_type)
        connected = False
        last_error = ""
        for attempt in range(30):  # 30 × 2 s = 60 s max
            await asyncio.sleep(2)
            try:
                status = await client.get_status()
                if status.get("connected"):
                    connected = True
                    break
                last_error = status.get("error", "not connected yet")
            except Exception as e:
                last_error = str(e)

        if not connected:
            gateway_logs = self._gateway_logs_tail(connection_id, broker_type)
            detail = last_error or "not connected yet"
            if gateway_logs:
                detail = f"{detail}; gateway logs: {gateway_logs}"
            # Teardown on failure
            self._destroy_gateway(connection_id, broker_type)
            with get_session_context() as session:
                _update_connection_status(
                    session, connection_id, ConnectionStatus.ERROR,
                    f"Gateway did not connect in time: {detail}",
                )
            return ConnectorResult(
                success=False,
                message=f"Gateway did not connect: {detail}",
            )

        # 3. Fetch discovered accounts
        try:
            accounts = await client.get_accounts()
        except Exception as e:
            accounts = []
            logger.warning("Could not fetch accounts from gateway: %s", e)

        # 4. Persist outcome
        with get_session_context() as session:
            _update_connection_status(session, connection_id, ConnectionStatus.CONNECTED)
            if accounts:
                _sync_accounts_from_gateway(session, connection_id, accounts)

        discovered = [
            DiscoveredAccount(
                account_id=a.get("account_id", ""),
                display_name=a.get("display_name") or a.get("account_id", ""),
                account_type=a.get("account_type", "unknown"),
                currency=a.get("currency") or "USD",
                cash_balance=_safe_float(a.get("cash_balance")),
                equity=_safe_float(a.get("equity")),
                buying_power=_safe_float(a.get("buying_power")),
                available_funds=_safe_float(a.get("available_funds")),
                snapshot_at=_parse_snapshot_at(a.get("snapshot_at")),
                extra=a.get("extra") if isinstance(a.get("extra"), dict) else None,
            )
            for a in accounts
            if a.get("account_id")
        ]

        return ConnectorResult(success=True, accounts=discovered)

    # ── Disconnect ───────────────────────────────────────────────────

    async def disconnect(self, connection_id: int) -> ConnectorResult:
        """Disconnect by destroying the gateway container."""
        with get_session_context() as session:
            conn = session.get(Connection, connection_id)
            if conn is None:
                return ConnectorResult(success=False, message="Connection not found")
            broker_type = conn.broker_type

        # Gracefully tell the gateway to disconnect first
        try:
            client = self.get_gateway_client(connection_id, broker_type)
            await client.disconnect()
        except Exception:
            pass  # Container may already be gone

        if broker_type == "ibkr":
            self._destroy_tws_gateway(connection_id)

        # Destroy the container
        self._destroy_gateway(connection_id, broker_type)

        with get_session_context() as session:
            _update_connection_status(session, connection_id, ConnectionStatus.DISCONNECTED, "Disconnected")
            conn = session.get(Connection, connection_id)
            if conn is not None:
                conn.config = _clear_tws_runtime_state(conn.config)
                conn.updated_at = datetime.now(timezone.utc)
                session.commit()

        return ConnectorResult(success=True, message="Disconnected")

    # ── Health check ─────────────────────────────────────────────────

    async def check_connection_status(self, connection_id: int) -> str:
        """Probe a gateway container's real status and update DB if changed."""
        with get_session_context() as session:
            conn = session.get(Connection, connection_id)
            if conn is None:
                return ConnectionStatus.DISCONNECTED.value
            stored_status = conn.status
            broker_type = conn.broker_type

        # Interactive IB Gateway logins are driven to completion by the
        # dedicated reconcile loop, not here. The health loop only monitors
        # already-established connections.
        if broker_type == "ibkr" and stored_status == ConnectionStatus.AWAITING_AUTH.value:
            return stored_status

        # For an interactive IB Gateway connection that is not actively
        # connected, the data gateway (gw-N) is absent BY DESIGN — it is only
        # spawned after login completes. The TWS login runtime (twsgw-N) is owned
        # by the auth flow (begin/complete) and the idle-cleanup loop, so the
        # data-gateway health probe must NOT tear it down here. Doing so races
        # with begin_tws_auth respawning it and surfaces to the user as
        # "container twsgw-N not found".
        #
        # EXCEPTION: if a data gateway container is still running (e.g. the
        # broker feed dropped transiently and the gateway self-reconnected),
        # fall through and probe it so the DB status can flip back to
        # connected without requiring a manual reconnect.
        if (
            broker_type == "ibkr"
            and stored_status not in {
                ConnectionStatus.CONNECTED.value,
                ConnectionStatus.CONNECTING.value,
            }
        ):
            existing = self._get_container(connection_id, broker_type)
            if not existing or existing.status != "running":
                return stored_status

        actual_message: str | None = None

        # Check if the container exists and is running
        container = self._get_container(connection_id, broker_type)
        if not container or container.status != "running":
            actual = ConnectionStatus.DISCONNECTED
            if broker_type == "ibkr":
                actual_message = "Gateway container terminated; IB Gateway runtime stopped"
                self._destroy_gateway(connection_id, broker_type)
                self._destroy_tws_gateway(connection_id)
        else:
            # Ask the gateway if it's still connected to the broker
            try:
                client = self.get_gateway_client(connection_id, broker_type)
                gateway_status = await client.get_status(timeout=GATEWAY_STATUS_PROBE_TIMEOUT_SECONDS)
                if gateway_status.get("connected"):
                    actual = ConnectionStatus.CONNECTED
                elif _gateway_status_is_degraded(gateway_status):
                    actual = ConnectionStatus.DEGRADED
                    actual_message = _gateway_degraded_message(gateway_status)
                elif stored_status == ConnectionStatus.CONNECTING.value:
                    # Preserve an in-flight connect attempt while the gateway
                    # container is still alive; the explicit connect() flow
                    # will promote to CONNECTED or fail to ERROR on timeout.
                    actual = ConnectionStatus.CONNECTING
                else:
                    actual = ConnectionStatus.DISCONNECTED
            except Exception:
                actual = (
                    ConnectionStatus.CONNECTING
                    if stored_status == ConnectionStatus.CONNECTING.value
                    else ConnectionStatus.DISCONNECTED
                )

        status_changed = actual.value != stored_status
        message_changed = False
        if actual == ConnectionStatus.DEGRADED:
            with get_session_context() as session:
                conn = session.get(Connection, connection_id)
                message_changed = conn is not None and conn.status_message != actual_message

        if status_changed or message_changed:
            with get_session_context() as session:
                _update_connection_status(session, connection_id, actual, actual_message)
            logger.info(
                "Connection %s status changed: %s → %s",
                connection_id, stored_status, actual.value,
            )
        else:
            # Status unchanged: still record that we probed it (and when it was
            # last seen healthy) so the API can detect stale "connected" values.
            now = datetime.now(timezone.utc)
            with get_session_context() as session:
                conn = session.get(Connection, connection_id)
                if conn is not None:
                    conn.last_checked_at = now
                    if actual == ConnectionStatus.CONNECTED:
                        conn.last_ok_at = now
                    session.commit()

        # Account state (balances) is kept fresh by the order-aggregator
        # projection of broker.account.sync; the health loop only tracks
        # the connection status.

        return actual.value

    async def check_all_connected(self) -> None:
        """Probe every connection currently marked as 'connected'."""
        with get_session_context() as session:
            stmt = select(Connection).where(
                Connection.status == ConnectionStatus.CONNECTED.value
            )
            connected_ids = [c.id for c in session.exec(stmt).all()]

        if not connected_ids:
            return

        tasks = [self.check_connection_status(cid) for cid in connected_ids]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def probe_active_connections(self) -> None:
        """Probe ALL active connections regardless of stored status.

        Called after startup reset to reconcile DB state with running
        gateway containers (e.g. a container that survived a backend restart).
        """
        with get_session_context() as session:
            stmt = select(Connection).where(
                Connection.is_active == True  # noqa: E712
            )
            active_ids = [c.id for c in session.exec(stmt).all()]

        if not active_ids:
            return

        logger.debug("Probing %d active connection(s) for running gateways...", len(active_ids))
        tasks = [self.check_connection_status(cid) for cid in active_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        reconnected = sum(
            1 for r in results
            if isinstance(r, str) and r == ConnectionStatus.CONNECTED.value
        )
        if reconnected:
            logger.debug("Reconciled %d connection(s) back to 'connected'", reconnected)

    async def _run_connection_health_loop(self) -> None:
        """Periodically reconcile DB connection status with the live gateways.

        This moves health probing off the request path: GET /connections can
        then read the DB directly, and ``last_checked_at`` stays fresh so the
        API can flag stale "connected" statuses.
        """
        while self._running:
            await asyncio.sleep(CONNECTION_HEALTH_INTERVAL_SECONDS)
            try:
                await self.probe_active_connections()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Connection health loop iteration failed")

    async def reconcile_interactive_auth(self) -> None:
        """Single completion authority for interactive IB Gateway/TWS logins.

        The actual login happens inside the noVNC popup, which the backend cannot
        observe directly. This finds connections parked in ``awaiting_auth`` and,
        once their TWS API answers, completes them — so completion never depends
        on the frontend (it is a pure observer).
        """
        with get_session_context() as session:
            stmt = select(Connection).where(
                Connection.is_active == True,  # noqa: E712
                Connection.status == ConnectionStatus.AWAITING_AUTH.value,
                Connection.broker_type == "ibkr",
            )
            candidates = [
                (c.id, dict(c.config or {}))
                for c in session.exec(stmt).all()
            ]

        if not candidates:
            return

        tasks = [
            self._try_autocomplete_tws_connection(cid, config)
            for cid, config in candidates
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_interactive_auth_reconcile_loop(self) -> None:
        while self._running:
            await asyncio.sleep(TWS_AUTH_RECONCILE_INTERVAL_SECONDS)
            try:
                await self.reconcile_interactive_auth()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Interactive auth reconcile loop iteration failed")

    # ── Lifecycle ────────────────────────────────────────────────────

    def _reset_connected_on_startup(self) -> None:
        """Reset all 'connected' statuses to 'disconnected' on boot.

        Also cleans up any orphan gateway containers that were left
        running from a previous backend instance.
        """
        with get_session_context() as session:
            now = datetime.now(timezone.utc)

            # Reset stale "connected" connections
            stmt = select(Connection).where(
                Connection.status == ConnectionStatus.CONNECTED.value
            )
            stale = list(session.exec(stmt).all())
            for conn in stale:
                conn.status = ConnectionStatus.DISCONNECTED.value
                conn.status_message = "Reset on server restart"
                conn.updated_at = now
                logger.info(
                    "Reset stale connection %s (%s) from 'connected' to 'disconnected'",
                    conn.id, conn.name,
                )

            if stale:
                session.commit()

        # NOTE: We do NOT clean up gateway containers here — they are
        # independent microservices that should survive backend restarts.
        # probe_active_connections() (called after this) will reconcile
        # DB status with actually running containers.

    def _cleanup_stale_containers(self) -> None:
        """Remove any gateway containers left from a previous run."""
        if not self._docker:
            return
        for spec in GATEWAY_REGISTRY.values():
            try:
                containers = self._docker.containers.list(
                    all=True,
                    filters={"label": f"edgewalker.type={spec.label}"},
                )
                for c in containers:
                    try:
                        c.stop(timeout=5)
                        c.remove(force=True)
                        logger.info("Cleaned up stale gateway container %s", c.name)
                    except Exception as e:
                        logger.warning("Failed to clean up container %s: %s", c.name, e)
            except Exception as e:
                logger.warning("Failed to list gateway containers: %s", e)

    async def start(self) -> None:
        """Start the connection manager."""
        if self._running:
            return
        self._running = True

        if TWS_RUNTIME_IDLE_TIMEOUT_SECONDS > 0:
            self._tasks.append(asyncio.create_task(self._run_tws_runtime_cleanup()))

        # On (re)start, no broker connections can be alive
        self._reset_connected_on_startup()

        # Reconcile: check if any gateway containers survived the restart
        await self.probe_active_connections()

        # Keep DB connection status fresh off the request path
        self._tasks.append(asyncio.create_task(self._run_connection_health_loop()))

        # Single authority that drives interactive logins to completion
        if TWS_AUTO_COMPLETE_ENABLED:
            self._tasks.append(
                asyncio.create_task(self._run_interactive_auth_reconcile_loop())
            )

        logger.info("Connection manager started")

    async def stop(self) -> None:
        """Stop the connection manager and clean up."""
        if not self._running:
            return
        self._running = False

        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()

        # NOTE: We do NOT destroy gateway containers on stop() — they are
        # independent microservices that should survive backend restarts.
        # Only explicit disconnect (user action) destroys the container.
        _gateway_clients.clear()

        logger.info("Connection manager stopped")


# ── Global singleton ─────────────────────────────────────────────────

_connection_manager: ConnectionManager | None = None


def get_connection_manager() -> ConnectionManager:
    """Return the global ``ConnectionManager`` instance."""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    return _connection_manager


async def start_connection_manager() -> None:
    """Initialize and start the connection manager (called from lifespan)."""
    manager = get_connection_manager()
    await manager.start()


async def stop_connection_manager() -> None:
    """Stop the connection manager (called from lifespan)."""
    manager = get_connection_manager()
    await manager.stop()
