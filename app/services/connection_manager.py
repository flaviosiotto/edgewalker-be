"""
Connection Manager – async background service that manages broker
connect / disconnect lifecycle and auto-discovers accounts.

**Architecture (v2 — ibkr-gateway):**

When the user clicks "Connect" the manager spawns a dedicated
``ibkr-gateway`` Docker container for that Connection.  The container
maintains the persistent IBKR connection(s) and exposes a REST API.
All further operations (search, streaming, historical fetch, orders)
are routed through that container via ``IBKRGatewayClient``.

One container per Connection = one IBKR Gateway = complete user
segregation.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any

import docker
from docker.errors import NotFound, APIError
from sqlmodel import Session, select

from app.db.database import get_session_context
from app.models.connection import Account, Connection, ConnectionStatus
from app.services.broker_connectors.base import ConnectorResult, DiscoveredAccount
from app.services.ibkr_gateway_client import IBKRGatewayClient

logger = logging.getLogger(__name__)

# ── Docker settings ──────────────────────────────────────────────────
DOCKER_NETWORK = os.getenv("DOCKER_NETWORK", "edgewalker-devops_default")
GATEWAY_IMAGE = os.getenv("GATEWAY_IMAGE", "edgewalker-devops-ibkr-gateway:latest")
GATEWAY_CONTAINER_PREFIX = "ibkr-gw-"

# Paths for volume mounts (same as docker-compose)
EDGEWALKER_PATH = os.getenv("EDGEWALKER_PATH", "/home/flavio/playground/edgewalker")

# Host UID/GID — gateway containers run as this user so written files
# on mounted volumes match the host user (avoids root-owned data).
HOST_PUID = os.getenv("PUID", "1000")
HOST_PGID = os.getenv("PGID", "1000")

# Active gateway clients — keyed by connection_id
_gateway_clients: dict[int, IBKRGatewayClient] = {}


def _sync_accounts_from_gateway(
    session: Session,
    connection_id: int,
    accounts: list[dict[str, str]],
) -> None:
    """Sync accounts discovered by the ibkr-gateway container.

    ``accounts`` is a list of ``{"account_id": "...", "account_type": "..."}``
    dicts returned by the gateway's ``/accounts`` endpoint.
    """
    discovered = [
        DiscoveredAccount(
            account_id=a["account_id"],
            display_name=a["account_id"],
            account_type=a.get("account_type", "unknown"),
            currency="USD",
        )
        for a in accounts
    ]
    _sync_accounts(session, connection_id, discovered)


def _sync_accounts(session: Session, connection_id: int, discovered: list[DiscoveredAccount]) -> None:
    """Upsert discovered accounts and deactivate stale ones."""
    now = datetime.now(timezone.utc)

    existing_stmt = select(Account).where(Account.connection_id == connection_id)
    existing: list[Account] = list(session.exec(existing_stmt).all())
    existing_map = {a.account_id: a for a in existing}

    discovered_ids = {d.account_id for d in discovered}

    # Upsert
    for d in discovered:
        acct = existing_map.get(d.account_id)
        if acct is None:
            acct = Account(
                connection_id=connection_id,
                account_id=d.account_id,
                display_name=d.display_name,
                account_type=d.account_type,
                currency=d.currency,
                is_active=True,
                extra=d.extra,
            )
            session.add(acct)
            logger.info("Auto-discovered new account %s for connection %s", d.account_id, connection_id)
        else:
            # Re-activate if it was previously deactivated
            acct.is_active = True
            if d.display_name:
                acct.display_name = d.display_name
            if d.account_type:
                acct.account_type = d.account_type
            if d.currency:
                acct.currency = d.currency
            acct.updated_at = now

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
    """Manages broker connections via ``ibkr-gateway`` Docker containers.

    For each IBKR Connection the manager:
    1. Spawns a dedicated ``ibkr-gateway`` container on connect
    2. Waits for the gateway to report "connected"
    3. Syncs discovered accounts into the DB
    4. Provides an ``IBKRGatewayClient`` for the rest of the backend
    5. Destroys the container on disconnect
    """

    def __init__(self) -> None:
        self._running = False
        self._tasks: list[asyncio.Task] = []
        try:
            self._docker = docker.from_env()
        except Exception as e:
            logger.warning("Docker not available: %s — gateway management disabled", e)
            self._docker = None

    # ── Container management ─────────────────────────────────────────

    def _container_name(self, connection_id: int) -> str:
        return f"{GATEWAY_CONTAINER_PREFIX}{connection_id}"

    def _get_container(self, connection_id: int):
        """Get existing gateway container for a connection, if any."""
        if not self._docker:
            return None
        try:
            return self._docker.containers.get(self._container_name(connection_id))
        except NotFound:
            return None
        except Exception as e:
            logger.warning("Docker error checking container: %s", e)
            return None

    def _spawn_gateway(self, connection_id: int, config: dict[str, Any]) -> None:
        """Spawn a new ibkr-gateway container for the given Connection."""
        if not self._docker:
            raise RuntimeError("Docker is not available")

        container_name = self._container_name(connection_id)

        # Remove any existing stopped container
        existing = self._get_container(connection_id)
        if existing:
            if existing.status == "running":
                logger.info("Gateway container %s already running", container_name)
                return
            existing.remove(force=True)

        env = {
            "IBKR_HOST": str(config.get("host", "host.docker.internal")),
            "IBKR_PORT": str(config.get("port", 4001)),
            "IBKR_CLIENT_ID": str(config.get("client_id", 100)),
            "REDIS_HOST": "redis",
            "REDIS_PORT": "6379",
            "CONNECTION_ID": str(connection_id),
            "DATA_DIR": "/opt/edgewalker/data",
            "LOG_LEVEL": "INFO",
            "PYTHONPATH": "/app:/opt/edgewalker",
        }

        volumes = {
            # Edgewalker library (for historical fetch)
            f"{EDGEWALKER_PATH}/edgewalker": {
                "bind": "/opt/edgewalker/edgewalker",
                "mode": "ro",
            },
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

        labels = {
            "edgewalker.type": "ibkr-gateway",
            "edgewalker.connection_id": str(connection_id),
        }

        extra_hosts = {"host.docker.internal": "host-gateway"}

        try:
            self._docker.containers.run(
                image=GATEWAY_IMAGE,
                name=container_name,
                environment=env,
                volumes=volumes,
                labels=labels,
                extra_hosts=extra_hosts,
                network=DOCKER_NETWORK,
                user=f"{HOST_PUID}:{HOST_PGID}",
                detach=True,
                restart_policy={"Name": "unless-stopped"},
            )
            logger.info("Spawned gateway container %s", container_name)
        except APIError as e:
            logger.error("Failed to spawn gateway container: %s", e)
            raise

    def _destroy_gateway(self, connection_id: int) -> None:
        """Stop and remove the gateway container for a Connection."""
        container = self._get_container(connection_id)
        if container:
            try:
                container.stop(timeout=10)
                container.remove(force=True)
                logger.info("Destroyed gateway container %s", self._container_name(connection_id))
            except Exception as e:
                logger.warning("Error destroying container: %s", e)
        _gateway_clients.pop(connection_id, None)

    # ── Gateway client ───────────────────────────────────────────────

    def get_gateway_client(self, connection_id: int) -> IBKRGatewayClient:
        """Get (or create) the HTTP client for a connection's gateway."""
        if connection_id not in _gateway_clients:
            _gateway_clients[connection_id] = IBKRGatewayClient(connection_id)
        return _gateway_clients[connection_id]

    # ── Connect ──────────────────────────────────────────────────────

    async def connect(self, connection_id: int) -> ConnectorResult:
        """Connect to IBKR by spawning an ibkr-gateway container.

        Steps:
        1. Spawn ``ibkr-gw-{connection_id}`` container
        2. Poll ``/health`` until the gateway is connected (up to 60 s)
        3. Fetch discovered accounts from the gateway
        4. Sync accounts into the DB
        """
        # Load connection from DB
        with get_session_context() as session:
            conn = session.get(Connection, connection_id)
            if conn is None:
                return ConnectorResult(success=False, message="Connection not found")

            broker_type = conn.broker_type
            config = dict(conn.config or {})

            if broker_type != "ibkr":
                return ConnectorResult(
                    success=False,
                    message=f"Unsupported broker type: {broker_type}",
                )

            # Mark as "connecting" immediately
            conn.status = "connecting"
            conn.status_message = "Spawning gateway container…"
            session.commit()

        # 1. Spawn the gateway container
        try:
            self._spawn_gateway(connection_id, config)
        except Exception as e:
            with get_session_context() as session:
                _update_connection_status(
                    session, connection_id, ConnectionStatus.ERROR,
                    f"Failed to spawn gateway: {e}",
                )
            return ConnectorResult(success=False, message=f"Failed to spawn gateway: {e}")

        # 2. Wait for the gateway to become healthy / connected
        client = self.get_gateway_client(connection_id)
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
            self._destroy_gateway(connection_id)
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
                display_name=a.get("account_id", ""),
                account_type=a.get("account_type", "unknown"),
                currency="USD",
            )
            for a in accounts
        ]

        return ConnectorResult(success=True, accounts=discovered)

    # ── Disconnect ───────────────────────────────────────────────────

    async def disconnect(self, connection_id: int) -> ConnectorResult:
        """Disconnect by destroying the ibkr-gateway container."""
        with get_session_context() as session:
            conn = session.get(Connection, connection_id)
            if conn is None:
                return ConnectorResult(success=False, message="Connection not found")

        # Gracefully tell the gateway to disconnect IBKR first
        try:
            client = self.get_gateway_client(connection_id)
            await client.disconnect()
        except Exception:
            pass  # Container may already be gone

        # Destroy the container
        self._destroy_gateway(connection_id)

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

        # Check if the container exists and is running
        container = self._get_container(connection_id)
        if not container or container.status != "running":
            actual = ConnectionStatus.DISCONNECTED
        else:
            # Ask the gateway if it's still connected to IBKR
            try:
                client = self.get_gateway_client(connection_id)
                is_alive = await client.is_connected()
                actual = ConnectionStatus.CONNECTED if is_alive else ConnectionStatus.DISCONNECTED
            except Exception:
                actual = ConnectionStatus.DISCONNECTED

        if actual.value != stored_status:
            with get_session_context() as session:
                _update_connection_status(session, connection_id, actual)
            logger.info(
                "Connection %s status changed: %s → %s",
                connection_id, stored_status, actual.value,
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

        Also cleans up any orphan ibkr-gateway containers that were left
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
        """Remove any ibkr-gateway containers left from a previous run."""
        if not self._docker:
            return
        try:
            containers = self._docker.containers.list(
                all=True,
                filters={"label": "edgewalker.type=ibkr-gateway"},
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
