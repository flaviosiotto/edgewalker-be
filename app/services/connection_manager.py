"""
Connection Manager – async background service that manages broker
connect / disconnect lifecycle and auto-discovers accounts.

Replaces the old ``SyncManager`` for connection-level operations.
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from typing import Any

from sqlmodel import Session, select

from app.db.database import get_session_context
from app.models.connection import Account, Connection, ConnectionStatus
from app.services.broker_connectors.base import BrokerConnector, ConnectorResult, DiscoveredAccount
from app.services.broker_connectors.ibkr import IBKRConnector

logger = logging.getLogger(__name__)

# ── Registry of broker connectors ────────────────────────────────────
_CONNECTORS: dict[str, BrokerConnector] = {}


def _get_connector(broker_type: str) -> BrokerConnector | None:
    """Get (or lazily create) the connector for a broker type."""
    if broker_type not in _CONNECTORS:
        if broker_type == "ibkr":
            _CONNECTORS[broker_type] = IBKRConnector()
        else:
            logger.warning("No connector registered for broker_type=%s", broker_type)
            return None
    return _CONNECTORS[broker_type]


# ── Helpers ──────────────────────────────────────────────────────────

async def _run_blocking(executor: ThreadPoolExecutor, func, *args, **kwargs) -> Any:
    """Run a blocking function in a thread-pool executor."""
    loop = asyncio.get_event_loop()
    if kwargs:
        func = partial(func, **kwargs)
    return await loop.run_in_executor(executor, func, *args)


def _sync_accounts(session: Session, connection_id: int, discovered: list[DiscoveredAccount]) -> None:
    """Upsert discovered accounts and deactivate stale ones.

    - New accounts are created.
    - Existing accounts are kept (updated if metadata changed).
    - Accounts no longer reported by the broker get ``is_active=False``.
    """
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
    """Manages broker connections asynchronously."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="connmgr_")
        self._running = False
        self._tasks: list[asyncio.Task] = []

    # ── Connect ──────────────────────────────────────────────────────

    async def connect(self, connection_id: int) -> ConnectorResult:
        """Connect to a broker and auto-discover accounts.

        Called from the API endpoint.  The blocking broker call runs in a
        thread-pool so the event loop stays free.
        """
        # Load connection from DB
        with get_session_context() as session:
            conn = session.get(Connection, connection_id)
            if conn is None:
                return ConnectorResult(success=False, message="Connection not found")

            broker_type = conn.broker_type
            config = dict(conn.config or {})

            # Mark as "connecting" immediately
            conn.status = "connecting"
            conn.status_message = None
            session.commit()

        connector = _get_connector(broker_type)
        if connector is None:
            with get_session_context() as session:
                _update_connection_status(
                    session, connection_id, ConnectionStatus.ERROR,
                    f"No connector available for broker type '{broker_type}'",
                )
            return ConnectorResult(success=False, message=f"Unsupported broker type: {broker_type}")

        # Run blocking connect in thread pool
        try:
            result: ConnectorResult = await asyncio.wait_for(
                _run_blocking(self._executor, connector.connect, config),
                timeout=30,
            )
        except asyncio.TimeoutError:
            with get_session_context() as session:
                _update_connection_status(
                    session, connection_id, ConnectionStatus.ERROR,
                    "Connection timed out",
                )
            return ConnectorResult(success=False, message="Connection timed out")
        except Exception as e:
            with get_session_context() as session:
                _update_connection_status(
                    session, connection_id, ConnectionStatus.ERROR, str(e),
                )
            return ConnectorResult(success=False, message=str(e))

        # Persist outcome
        with get_session_context() as session:
            if result.success:
                _update_connection_status(session, connection_id, ConnectionStatus.CONNECTED)
                # Sync discovered accounts
                if result.accounts:
                    _sync_accounts(session, connection_id, result.accounts)
            else:
                _update_connection_status(
                    session, connection_id, ConnectionStatus.ERROR, result.message,
                )

        return result

    # ── Disconnect ───────────────────────────────────────────────────

    async def disconnect(self, connection_id: int) -> ConnectorResult:
        """Disconnect from a broker."""
        with get_session_context() as session:
            conn = session.get(Connection, connection_id)
            if conn is None:
                return ConnectorResult(success=False, message="Connection not found")

            broker_type = conn.broker_type
            config = dict(conn.config or {})

        connector = _get_connector(broker_type)
        if connector is None:
            return ConnectorResult(success=False, message=f"Unsupported broker type: {broker_type}")

        try:
            result: ConnectorResult = await asyncio.wait_for(
                _run_blocking(self._executor, connector.disconnect, config),
                timeout=15,
            )
        except Exception as e:
            result = ConnectorResult(success=False, message=str(e))

        with get_session_context() as session:
            _update_connection_status(
                session, connection_id, ConnectionStatus.DISCONNECTED,
                result.message,
            )

        return result

    # ── Health check ─────────────────────────────────────────────────

    async def check_connection_status(self, connection_id: int) -> str:
        """Probe a connection's real status and update DB if changed.

        Returns the actual status string.
        """
        with get_session_context() as session:
            conn = session.get(Connection, connection_id)
            if conn is None:
                return ConnectionStatus.DISCONNECTED.value

            broker_type = conn.broker_type
            config = dict(conn.config or {})
            stored_status = conn.status

        connector = _get_connector(broker_type)
        if connector is None:
            return stored_status

        try:
            is_alive = await asyncio.wait_for(
                _run_blocking(self._executor, connector.is_connected, config),
                timeout=10,
            )
        except Exception:
            is_alive = False

        actual = ConnectionStatus.CONNECTED if is_alive else ConnectionStatus.DISCONNECTED

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

    # ── Lifecycle ────────────────────────────────────────────────────

    def _reset_connected_on_startup(self) -> None:
        """Reset all 'connected' statuses to 'disconnected' on boot.

        When the backend restarts, no broker connections are active.
        Accounts are also deactivated — they will be re-discovered on
        the next connect.
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

    async def start(self) -> None:
        """Start the connection manager."""
        if self._running:
            return
        self._running = True

        # On (re)start, no broker connections can be alive
        self._reset_connected_on_startup()

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

        self._executor.shutdown(wait=False)
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
