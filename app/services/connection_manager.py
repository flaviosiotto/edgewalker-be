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
import json
import logging
import os
import socket
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import docker
import redis.asyncio as aioredis
from docker.errors import NotFound, APIError
from sqlmodel import Session, select

from app.db.database import get_session_context
from app.models.connection import Account, Connection, ConnectionStatus
from app.services.broker_connectors.base import ConnectorResult, DiscoveredAccount
from app.services.client_portal_service import (
    get_client_portal_auth_status,
    is_client_portal_transport,
    logout_client_portal_session,
    resolve_client_portal_base_url,
    resolve_client_portal_browser_url,
    resolve_client_portal_verify_ssl,
)
from app.services.gateway_client import GatewayClient

logger = logging.getLogger(__name__)

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

# Shared Redis settings propagated to dynamically spawned gateway containers.
REDIS_URL = os.getenv("REDIS_URL", "")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_USERNAME = os.getenv("REDIS_USERNAME", "")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
BROKER_SYNC_STREAM = os.getenv("BROKER_SYNC_STREAM", "events:broker-sync")
BROKER_ACCOUNT_SYNC_EVENT = "broker.account.sync"
BROKER_ACCOUNT_SYNC_GROUP = os.getenv("BROKER_ACCOUNT_SYNC_GROUP", "backend-account-sync")
BROKER_ACCOUNT_SYNC_ENABLED = os.getenv("BROKER_ACCOUNT_SYNC_ENABLED", "true").lower() == "true"
BROKER_ACCOUNT_SYNC_BLOCK_MS = max(100, int(os.getenv("BROKER_ACCOUNT_SYNC_BLOCK_MS", "1000")))
BROKER_ACCOUNT_SYNC_COUNT = max(1, int(os.getenv("BROKER_ACCOUNT_SYNC_COUNT", "50")))
CLIENT_PORTAL_DISPATCHER_GRACE_PERIOD_SECONDS = max(
    0,
    int(os.getenv("CLIENT_PORTAL_DISPATCHER_GRACE_PERIOD_SECONDS", "30")),
)
CLIENT_PORTAL_DISPATCHER_WAIT_MESSAGE = (
    "Autorizzazione 2FA ricevuta. Attendo che IBKR apra la brokerage session."
)
CLIENT_PORTAL_DISPATCHER_ACK_MESSAGE = (
    "Dispatcher ricevuto. Attendo l'apertura della brokerage session IBKR."
)
CLIENT_PORTAL_DISPATCHER_RECEIVED_AT_KEY = "_client_portal_dispatcher_received_at"


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


def _shared_client_portal_base_url() -> str:
    return resolve_client_portal_base_url()


def _parse_client_portal_dispatcher_received_at(config: dict[str, Any] | None) -> datetime | None:
    if not config:
        return None

    raw_value = config.get(CLIENT_PORTAL_DISPATCHER_RECEIVED_AT_KEY)
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None

    try:
        parsed = datetime.fromisoformat(raw_value)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _with_client_portal_dispatcher_received_at(
    config: dict[str, Any] | None,
    received_at: datetime | None,
) -> dict[str, Any]:
    updated = dict(config or {})
    if received_at is None:
        updated.pop(CLIENT_PORTAL_DISPATCHER_RECEIVED_AT_KEY, None)
    else:
        updated[CLIENT_PORTAL_DISPATCHER_RECEIVED_AT_KEY] = received_at.astimezone(timezone.utc).isoformat()
    return updated


# ── Gateway Registry ─────────────────────────────────────────────────
# Each broker type maps to its Docker image, container prefix, label,
# and a function that builds the environment dict from Connection.config.
# To add a new broker, add an entry here — no other code changes needed
# in ConnectionManager.

def _ibkr_env(config: dict[str, Any]) -> dict[str, str]:
    """Build env vars for a gateway IBKR container."""
    transport = str(config.get("transport", "legacy"))
    return {
        "IBKR_HOST": str(config.get("host", "host.docker.internal")),
        "IBKR_PORT": str(config.get("port", 4001)),
        "IBKR_CLIENT_ID": str(config.get("client_id", 100)),
        "IBKR_TRANSPORT": transport,
        "CLIENT_PORTAL_ENABLED": str(config.get("client_portal_enabled", transport == "client_portal")).lower(),
        "CLIENT_PORTAL_BASE_URL": resolve_client_portal_base_url(config),
        "CLIENT_PORTAL_BROWSER_URL": resolve_client_portal_browser_url(config),
        "CLIENT_PORTAL_VERIFY_SSL": str(resolve_client_portal_verify_ssl(config)).lower(),
    }


def _binance_env(config: dict[str, Any]) -> dict[str, str]:
    """Build env vars for a gateway Binance container."""
    return {
        "BINANCE_API_KEY": str(config.get("api_key", "")),
        "BINANCE_API_SECRET": str(config.get("api_secret", "")),
        "BINANCE_TESTNET": str(config.get("testnet", False)).lower(),
        "BINANCE_MARKET_TYPE": str(config.get("market_type", "futures")),
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
            snapshot_at=discovered.snapshot_at,
            is_active=True,
            extra=discovered.extra,
        )
        session.add(acct)
        logger.info(
            "Auto-discovered new account %s for connection %s",
            discovered.account_id,
            connection_id,
        )
        return acct

    acct.is_active = True
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
    if discovered.snapshot_at is not None:
        acct.snapshot_at = discovered.snapshot_at
    if discovered.extra is not None:
        acct.extra = discovered.extra
    acct.updated_at = now
    return acct


def _sync_accounts(session: Session, connection_id: int, discovered: list[DiscoveredAccount]) -> None:
    """Upsert discovered accounts and deactivate stale ones."""
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

    # Deactivate accounts no longer present
    for acct_id, acct in existing_map.items():
        if acct_id not in discovered_ids:
            acct.is_active = False
            acct.updated_at = now
            logger.info("Deactivated stale account %s for connection %s", acct_id, connection_id)

    session.commit()


def _update_connection_status(
    session: Session,
    connection_id: int,
    status: ConnectionStatus,
    message: str | None = None,
) -> None:
    """Persist connection status in DB.

    When transitioning away from *connected*, all associated accounts
    are deactivated.  They will be re-activated on the next successful
    connect via ``_sync_accounts``.
    """
    conn = session.get(Connection, connection_id)
    if conn is None:
        return
    conn.status = status.value
    conn.status_message = message
    now = datetime.now(timezone.utc)
    conn.updated_at = now
    if status == ConnectionStatus.CONNECTED:
        conn.last_connected_at = now
    else:
        # Deactivate accounts when connection is no longer alive
        acct_stmt = select(Account).where(Account.connection_id == connection_id, Account.is_active == True)  # noqa: E712
        for acct in session.exec(acct_stmt).all():
            acct.is_active = False
            acct.updated_at = now
    session.commit()


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
        self._broker_account_sync_consumer = (
            f"{BROKER_ACCOUNT_SYNC_GROUP}-{socket.gethostname()}-{os.getpid()}"
        )
        self._client_portal_dispatcher_received_at: dict[int, datetime] = {}
        try:
            self._docker = docker.from_env()
        except Exception as e:
            logger.warning(
                "Docker not available: %s — gateway management disabled (%s)",
                e,
                _docker_runtime_requirements(),
            )
            self._docker = None

    def _create_async_redis_client(self) -> aioredis.Redis:
        if REDIS_URL:
            return aioredis.from_url(REDIS_URL, decode_responses=True)
        return aioredis.Redis(
            host=REDIS_HOST,
            port=int(REDIS_PORT),
            username=REDIS_USERNAME or None,
            password=REDIS_PASSWORD or None,
            decode_responses=True,
        )

    async def _ensure_broker_account_sync_group(
        self,
        redis: aioredis.Redis,
    ) -> None:
        try:
            await redis.xgroup_create(
                BROKER_SYNC_STREAM,
                BROKER_ACCOUNT_SYNC_GROUP,
                id="$",
                mkstream=True,
            )
            logger.info(
                "Created broker account-sync consumer group '%s' on %s",
                BROKER_ACCOUNT_SYNC_GROUP,
                BROKER_SYNC_STREAM,
            )
        except aioredis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    @staticmethod
    def _build_discovered_account_from_sync_payload(
        payload: dict[str, Any],
    ) -> DiscoveredAccount | None:
        account_id = str(payload.get("account") or payload.get("account_id") or "").strip()
        if not account_id:
            return None
        return DiscoveredAccount(
            account_id=account_id,
            display_name=str(payload.get("display_name") or account_id),
            account_type=str(payload.get("account_type") or "unknown"),
            currency=str(payload.get("currency") or "USD"),
            cash_balance=_safe_float(payload.get("cash_balance")),
            equity=_safe_float(payload.get("equity")),
            buying_power=_safe_float(payload.get("buying_power")),
            available_funds=_safe_float(payload.get("available_funds")),
            snapshot_at=_parse_snapshot_at(payload.get("snapshot_at")),
            extra=payload.get("extra") if isinstance(payload.get("extra"), dict) else None,
        )

    async def _handle_broker_account_sync_payload(
        self,
        *,
        connection_id: int,
        payload: dict[str, Any],
    ) -> None:
        discovered = self._build_discovered_account_from_sync_payload(payload)
        if discovered is None:
            return

        with get_session_context() as session:
            conn = session.get(Connection, connection_id)
            if conn is None:
                logger.debug(
                    "Ignoring broker account sync for unknown connection_id=%s account=%s",
                    connection_id,
                    discovered.account_id,
                )
                return

            _upsert_account_snapshot(session, connection_id, discovered)
            session.commit()

    async def _run_broker_account_sync_consumer(self) -> None:
        redis = self._create_async_redis_client()
        try:
            await self._ensure_broker_account_sync_group(redis)
            logger.info(
                "Broker account-sync consumer started (group=%s consumer=%s)",
                BROKER_ACCOUNT_SYNC_GROUP,
                self._broker_account_sync_consumer,
            )
            while self._running:
                try:
                    results = await redis.xreadgroup(
                        BROKER_ACCOUNT_SYNC_GROUP,
                        self._broker_account_sync_consumer,
                        {BROKER_SYNC_STREAM: ">"},
                        count=BROKER_ACCOUNT_SYNC_COUNT,
                        block=BROKER_ACCOUNT_SYNC_BLOCK_MS,
                    )
                    if not results:
                        continue

                    ack_ids: list[str] = []
                    for _stream_name, messages in results:
                        for msg_id, fields in messages:
                            ack_ids.append(str(msg_id))
                            if str(fields.get("event_type") or "") != BROKER_ACCOUNT_SYNC_EVENT:
                                continue
                            payload_raw = str(fields.get("payload") or "{}")
                            try:
                                payload = json.loads(payload_raw)
                            except json.JSONDecodeError:
                                logger.warning(
                                    "Invalid broker account-sync payload JSON: connection_id=%s payload=%s",
                                    fields.get("connection_id"),
                                    payload_raw,
                                )
                                continue

                            raw_connection_id = fields.get("connection_id") or payload.get("connection_id")
                            try:
                                connection_id = int(str(raw_connection_id))
                            except (TypeError, ValueError):
                                logger.warning(
                                    "Ignoring broker account sync with invalid connection_id=%s",
                                    raw_connection_id,
                                )
                                continue

                            await self._handle_broker_account_sync_payload(
                                connection_id=connection_id,
                                payload=payload if isinstance(payload, dict) else {},
                            )

                    if ack_ids:
                        await redis.xack(
                            BROKER_SYNC_STREAM,
                            BROKER_ACCOUNT_SYNC_GROUP,
                            *ack_ids,
                        )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "Broker account-sync consumer iteration failed: %s",
                        exc,
                        exc_info=True,
                    )
                    await asyncio.sleep(1.0)
        finally:
            await redis.aclose()
            logger.info("Broker account-sync consumer stopped")

    def _clear_client_portal_dispatcher_grace(self, connection_id: int) -> None:
        self._client_portal_dispatcher_received_at.pop(connection_id, None)

    def _client_portal_dispatcher_grace_deadline(
        self,
        connection_id: int,
        *,
        dispatcher_received_at: datetime | None,
        connection_status: str,
        status_message: str | None,
        updated_at: datetime | None,
    ) -> datetime | None:
        if CLIENT_PORTAL_DISPATCHER_GRACE_PERIOD_SECONDS <= 0:
            self._clear_client_portal_dispatcher_grace(connection_id)
            return None

        received_at = self._client_portal_dispatcher_received_at.get(connection_id)
        if received_at is None:
            received_at = dispatcher_received_at

        if received_at is None:
            if connection_status != ConnectionStatus.AWAITING_AUTH.value:
                return None
            if status_message != CLIENT_PORTAL_DISPATCHER_WAIT_MESSAGE:
                return None
            received_at = updated_at
            if received_at is None:
                return None

        deadline = received_at + timedelta(seconds=CLIENT_PORTAL_DISPATCHER_GRACE_PERIOD_SECONDS)
        if deadline <= datetime.now(timezone.utc):
            self._clear_client_portal_dispatcher_grace(connection_id)
            return None

        return deadline

    def _client_portal_dispatcher_grace_payload(
        self,
        connection_id: int,
        *,
        config: dict[str, Any],
        gateway_started: bool,
        connection_status: str,
        status_message: str | None,
        updated_at: datetime | None,
    ) -> dict[str, Any] | None:
        dispatcher_received_at = _parse_client_portal_dispatcher_received_at(config)
        deadline = self._client_portal_dispatcher_grace_deadline(
            connection_id,
            dispatcher_received_at=dispatcher_received_at,
            connection_status=connection_status,
            status_message=status_message,
            updated_at=updated_at,
        )
        if deadline is None:
            return None

        return {
            "service_ready": True,
            "gateway_session_ready": True,
            "connected": False,
            "session_authenticated": False,
            "authenticated": False,
            "established": False,
            "competing": False,
            "bridge_ready": False,
            "ready_to_connect": False,
            "gateway_started": gateway_started,
            "connection_status": connection_status,
            "auth_url": resolve_client_portal_browser_url(config),
            "message": CLIENT_PORTAL_DISPATCHER_WAIT_MESSAGE,
        }

    async def mark_client_portal_dispatcher_received(self, connection_id: int) -> dict[str, Any]:
        with get_session_context() as session:
            conn = session.get(Connection, connection_id)
            if conn is None:
                raise ValueError("Connection not found")

            dispatcher_received_at = datetime.now(timezone.utc)
            conn.status = ConnectionStatus.AWAITING_AUTH.value
            conn.status_message = CLIENT_PORTAL_DISPATCHER_WAIT_MESSAGE
            conn.updated_at = dispatcher_received_at
            conn.config = _with_client_portal_dispatcher_received_at(conn.config, dispatcher_received_at)
            session.commit()

        self._client_portal_dispatcher_received_at[connection_id] = dispatcher_received_at
        return {
            "success": True,
            "message": CLIENT_PORTAL_DISPATCHER_ACK_MESSAGE,
        }

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

    def _restart_client_portal_gateway(self) -> None:
        """Restart the shared Client Portal gateway to clear sticky auth state."""
        if not self._docker:
            return

        candidates = []
        try:
            candidates = self._docker.containers.list(
                all=True,
                filters={"label": "com.docker.compose.service=ibkr-client-portal-gw"},
            )
        except Exception as e:
            logger.warning("Docker error listing Client Portal gateway containers: %s", e)
            return

        if not candidates:
            try:
                candidates = [
                    container
                    for container in self._docker.containers.list(all=True)
                    if "ibkr-client-portal-gw" in container.name
                ]
            except Exception as e:
                logger.warning("Docker error scanning Client Portal gateway containers: %s", e)
                return

        for container in candidates:
            try:
                logger.info("Restarting shared Client Portal gateway container %s", container.name)
                container.restart(timeout=10)
            except Exception as e:
                logger.warning("Could not restart Client Portal gateway container %s: %s", container.name, e)

    def _has_other_active_client_portal_connections(self, connection_id: int, base_url: str) -> bool:
        active_statuses = {
            ConnectionStatus.CONNECTING.value,
            ConnectionStatus.CONNECTED.value,
            ConnectionStatus.AWAITING_AUTH.value,
        }

        with get_session_context() as session:
            candidates = list(
                session.exec(
                    select(Connection).where(Connection.id != connection_id).where(Connection.broker_type == "ibkr")
                ).all()
            )

        for candidate in candidates:
            candidate_config = dict(candidate.config or {})
            if not is_client_portal_transport(candidate_config):
                continue
            if candidate.status not in active_statuses:
                continue
            if resolve_client_portal_base_url(candidate_config) == base_url:
                return True

        return False

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
                restart_policy={"Name": "unless-stopped"},
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

    # ── Gateway client ───────────────────────────────────────────────

    def get_gateway_client(self, connection_id: int, broker_type: str) -> GatewayClient:
        """Get (or create) the HTTP client for a connection's gateway."""
        if connection_id not in _gateway_clients:
            _gateway_clients[connection_id] = GatewayClient(connection_id, broker_type=broker_type)
        return _gateway_clients[connection_id]

    async def begin_client_portal_auth(self, connection_id: int) -> dict[str, Any]:
        self._clear_client_portal_dispatcher_grace(connection_id)
        with get_session_context() as session:
            conn = session.get(Connection, connection_id)
            if conn is None:
                raise ValueError("Connection not found")
            config = _with_client_portal_dispatcher_received_at(conn.config, None)
            conn.config = config
            session.commit()

        auth = await get_client_portal_auth_status(config)
        message = "Autenticazione IBKR richiesta nel popup Client Portal."
        if auth["ready_to_connect"]:
            message = "Sessione Client Portal pronta per il gateway."
        elif auth["gateway_session_ready"] and not auth["authenticated"]:
            message = "Login Client Portal completato. In attesa dell'apertura della brokerage session IBKR."
        elif auth["service_ready"] and auth.get("session_authenticated") and not auth.get("bridge_ready"):
            message = "Sessione Client Portal autenticata ma bridge brokerage non pronto. Attendi il completamento del login IBKR."
        elif not auth["service_ready"]:
            message = f"Client Portal non raggiungibile: {auth['message']}"

        with get_session_context() as session:
            conn = session.get(Connection, connection_id)
            if conn is not None:
                conn.status = ConnectionStatus.AWAITING_AUTH.value
                conn.status_message = message
                conn.updated_at = datetime.now(timezone.utc)
                session.commit()

        return {
            "service_ready": auth["service_ready"],
            "authenticated": auth["authenticated"],
            "ready_to_connect": auth["ready_to_connect"],
            "auth_url": resolve_client_portal_browser_url(config),
            "message": message,
        }

    async def complete_client_portal_connect(self, connection_id: int) -> ConnectorResult:
        with get_session_context() as session:
            conn = session.get(Connection, connection_id)
            if conn is None:
                return ConnectorResult(success=False, message="Connection not found")
            config = dict(conn.config or {})

        auth = await get_client_portal_auth_status(config)
        if not auth["service_ready"]:
            return ConnectorResult(success=False, message=f"Client Portal non raggiungibile: {auth['message']}")
        if not auth["ready_to_connect"]:
            wait_message = auth.get("message") or "Client Portal non autenticato"
            if auth.get("gateway_session_ready") and not auth.get("authenticated"):
                wait_message = "Login Client Portal completato, ma la brokerage session IBKR non e' ancora pronta. Attendi qualche secondo e riprova."
            elif auth.get("session_authenticated") and not auth.get("bridge_ready"):
                wait_message = "Autorizzazione 2FA ricevuta, ma il bridge brokerage IBKR non e' ancora pronto. Attendi qualche secondo e riprova."
            with get_session_context() as session:
                conn = session.get(Connection, connection_id)
                if conn is not None:
                    conn.status = ConnectionStatus.AWAITING_AUTH.value
                    conn.status_message = wait_message
                    conn.updated_at = datetime.now(timezone.utc)
                    session.commit()
            return ConnectorResult(success=False, message=wait_message)

        result = await self.connect(connection_id)
        if result.success:
            self._clear_client_portal_dispatcher_grace(connection_id)
            with get_session_context() as session:
                conn = session.get(Connection, connection_id)
                if conn is not None:
                    conn.config = _with_client_portal_dispatcher_received_at(conn.config, None)
                    session.commit()
        return result

    async def client_portal_auth_status(self, connection_id: int) -> dict[str, Any]:
        with get_session_context() as session:
            conn = session.get(Connection, connection_id)
            if conn is None:
                raise ValueError("Connection not found")
            broker_type = conn.broker_type
            config = dict(conn.config or {})
            status_value = conn.status
            status_message = conn.status_message
            updated_at = conn.updated_at

        container = self._get_container(connection_id, broker_type)
        gateway_started = bool(container and container.status == "running")

        grace_payload = self._client_portal_dispatcher_grace_payload(
            connection_id,
            config=config,
            gateway_started=gateway_started,
            connection_status=status_value,
            status_message=status_message,
            updated_at=updated_at,
        )
        if grace_payload is not None:
            return grace_payload

        auth = await get_client_portal_auth_status(config)
        if auth.get("authenticated") or auth.get("session_authenticated") or auth.get("bridge_ready") or auth.get("ready_to_connect"):
            self._clear_client_portal_dispatcher_grace(connection_id)

        return {
            "service_ready": auth["service_ready"],
            "gateway_session_ready": auth.get("gateway_session_ready", False),
            "connected": auth.get("connected", False),
            "session_authenticated": auth.get("session_authenticated", False),
            "authenticated": auth["authenticated"],
            "established": auth.get("established", False),
            "competing": auth.get("competing", False),
            "bridge_ready": auth.get("bridge_ready", False),
            "ready_to_connect": auth.get("ready_to_connect", False),
            "gateway_started": gateway_started,
            "connection_status": status_value,
            "auth_url": resolve_client_portal_browser_url(config),
            "message": auth["message"],
        }

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
            # Teardown on failure
            self._destroy_gateway(connection_id, broker_type)
            with get_session_context() as session:
                _update_connection_status(
                    session, connection_id, ConnectionStatus.ERROR,
                    f"Gateway did not connect in time: {last_error}",
                )
            return ConnectorResult(
                success=False,
                message=f"Gateway did not connect: {last_error}",
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
        self._clear_client_portal_dispatcher_grace(connection_id)
        with get_session_context() as session:
            conn = session.get(Connection, connection_id)
            if conn is None:
                return ConnectorResult(success=False, message="Connection not found")
            broker_type = conn.broker_type
            conn.config = _with_client_portal_dispatcher_received_at(conn.config, None)
            config = dict(conn.config or {})
            session.commit()

        # Gracefully tell the gateway to disconnect first
        try:
            client = self.get_gateway_client(connection_id, broker_type)
            await client.disconnect()
        except Exception:
            pass  # Container may already be gone

        if is_client_portal_transport(config):
            client_portal_base_url = resolve_client_portal_base_url(config)
            has_peers = self._has_other_active_client_portal_connections(connection_id, client_portal_base_url)

            if not has_peers:
                try:
                    await logout_client_portal_session(config)
                except Exception:
                    pass

                if client_portal_base_url == _shared_client_portal_base_url():
                    self._restart_client_portal_gateway()
            else:
                logger.info(
                    "Skipping Client Portal logout/reset for connection %s because another active connection uses %s",
                    connection_id,
                    client_portal_base_url,
                )

        # Destroy the container
        self._destroy_gateway(connection_id, broker_type)

        with get_session_context() as session:
            _update_connection_status(
                session, connection_id, ConnectionStatus.DISCONNECTED,
                "Disconnected",
            )

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
            config = dict(conn.config or {})

        if is_client_portal_transport(config) and stored_status == ConnectionStatus.AWAITING_AUTH.value:
            return stored_status

        # Check if the container exists and is running
        container = self._get_container(connection_id, broker_type)
        if not container or container.status != "running":
            actual = ConnectionStatus.DISCONNECTED
        else:
            # Ask the gateway if it's still connected to the broker
            try:
                client = self.get_gateway_client(connection_id, broker_type)
                is_alive = await client.is_connected()
                actual = ConnectionStatus.CONNECTED if is_alive else ConnectionStatus.DISCONNECTED
            except Exception:
                actual = ConnectionStatus.DISCONNECTED

        status_changed = actual.value != stored_status

        if status_changed:
            with get_session_context() as session:
                _update_connection_status(session, connection_id, actual)
            logger.info(
                "Connection %s status changed: %s → %s",
                connection_id, stored_status, actual.value,
            )

        # Always re-sync accounts when connection is alive so that any
        # stale is_active=False accounts (left over from a previous
        # disconnect cycle) are re-activated.
        if actual == ConnectionStatus.CONNECTED:
            try:
                client = self.get_gateway_client(connection_id, broker_type)
                accounts = await client.get_accounts()
                if accounts:
                    with get_session_context() as session:
                        _sync_accounts_from_gateway(session, connection_id, accounts)
            except Exception as e:
                logger.warning(
                    "Could not re-sync accounts for connection %s: %s",
                    connection_id, e,
                )

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

        logger.info("Probing %d active connection(s) for running gateways...", len(active_ids))
        tasks = [self.check_connection_status(cid) for cid in active_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        reconnected = sum(
            1 for r in results
            if isinstance(r, str) and r == ConnectionStatus.CONNECTED.value
        )
        if reconnected:
            logger.info("Reconciled %d connection(s) back to 'connected'", reconnected)

    # ── Lifecycle ────────────────────────────────────────────────────

    def _reset_connected_on_startup(self) -> None:
        """Reset all 'connected' statuses to 'disconnected' on boot.

        Also cleans up any orphan gateway containers that were left
        running from a previous backend instance.
        """
        with get_session_context() as session:
            now = datetime.now(timezone.utc)

            # 1) Reset stale "connected" connections
            stmt = select(Connection).where(
                Connection.status == ConnectionStatus.CONNECTED.value
            )
            stale = list(session.exec(stmt).all())
            for conn in stale:
                conn.status = ConnectionStatus.DISCONNECTED.value
                conn.status_message = "Reset on server restart"
                conn.updated_at = now
                for acct in conn.accounts:
                    if acct.is_active:
                        acct.is_active = False
                        acct.updated_at = now
                logger.info(
                    "Reset stale connection %s (%s) from 'connected' to 'disconnected'",
                    conn.id, conn.name,
                )

            # 2) Deactivate orphan active accounts on non-connected connections
            orphan_stmt = (
                select(Account)
                .join(Connection, Account.connection_id == Connection.id)
                .where(
                    Account.is_active == True,  # noqa: E712
                    Connection.status != ConnectionStatus.CONNECTED.value,
                )
            )
            orphans = list(session.exec(orphan_stmt).all())
            for acct in orphans:
                acct.is_active = False
                acct.updated_at = now
            if orphans:
                logger.info(
                    "Deactivated %d orphan active account(s) on disconnected connections",
                    len(orphans),
                )

            if stale or orphans:
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

        if BROKER_ACCOUNT_SYNC_ENABLED:
            self._tasks.append(asyncio.create_task(self._run_broker_account_sync_consumer()))

        # On (re)start, no broker connections can be alive
        self._reset_connected_on_startup()

        # Reconcile: check if any gateway containers survived the restart
        await self.probe_active_connections()

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
