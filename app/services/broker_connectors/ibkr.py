"""
IBKR Broker Connector – connect / disconnect / discover accounts via ib_async.

All methods are **blocking** because ``ib_async`` expects to run in its own
synchronous context.  The connection manager calls them inside a
``ThreadPoolExecutor``.
"""
from __future__ import annotations

import logging
from typing import Any

from app.services.broker_connectors.base import (
    BrokerConnector,
    ConnectorResult,
    DiscoveredAccount,
)

logger = logging.getLogger(__name__)


class IBKRConnector(BrokerConnector):
    """Interactive Brokers connector via ``ib_async``."""

    @property
    def broker_type(self) -> str:
        return "ibkr"

    # ── connect ───────────────────────────────────────────────────────
    def connect(self, config: dict[str, Any]) -> ConnectorResult:
        """Connect to TWS / IB Gateway, discover managed accounts."""
        host = config.get("host", "127.0.0.1")
        port = config.get("port", 4001)
        client_id = config.get("client_id", 101)
        timeout = config.get("timeout_s", 10)

        try:
            from ib_async import IB

            ib = IB()
            ib.connect(host, port, clientId=client_id, timeout=timeout)

            if not ib.isConnected():
                return ConnectorResult(
                    success=False,
                    message=f"IBKR gateway at {host}:{port} did not respond",
                )

            # Discover accounts
            accounts: list[DiscoveredAccount] = []
            try:
                managed = ib.managedAccounts()
                for acct_id in managed:
                    acct_type = "paper" if acct_id.startswith("D") else "live"
                    accounts.append(
                        DiscoveredAccount(
                            account_id=acct_id,
                            display_name=acct_id,
                            account_type=acct_type,
                            currency="USD",
                        )
                    )
            except Exception as e:
                logger.warning("Could not list managed accounts: %s", e)

            ib.disconnect()

            logger.info(
                "IBKR connected at %s:%s – %d account(s) discovered",
                host, port, len(accounts),
            )
            return ConnectorResult(success=True, accounts=accounts)

        except Exception as e:
            logger.error("IBKR connect failed: %s", e)
            return ConnectorResult(success=False, message=str(e))

    # ── disconnect ────────────────────────────────────────────────────
    def disconnect(self, config: dict[str, Any]) -> ConnectorResult:
        """Disconnect is a no-op for IBKR (connections are short-lived)."""
        return ConnectorResult(success=True, message="Disconnected")

    # ── is_connected ──────────────────────────────────────────────────
    def is_connected(self, config: dict[str, Any]) -> bool:
        """Quick liveness probe — open / close a lightweight connection."""
        host = config.get("host", "127.0.0.1")
        port = config.get("port", 4001)
        client_id = config.get("client_id", 101)
        timeout = config.get("timeout_s", 5)

        try:
            from ib_async import IB

            ib = IB()
            ib.connect(host, port, clientId=client_id, timeout=timeout)
            ok = ib.isConnected()
            ib.disconnect()
            return ok
        except Exception:
            return False
