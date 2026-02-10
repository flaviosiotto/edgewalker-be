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
    SymbolMatch,
    SymbolSearchResult,
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
        """Quick liveness probe — open / close a lightweight connection.

        Uses ``client_id + 1000`` to avoid kicking existing sessions.
        """
        host = config.get("host", "127.0.0.1")
        port = config.get("port", 4001)
        client_id = config.get("client_id", 101) + 1000  # probe-only id
        timeout = config.get("timeout_s", 3)

        try:
            from ib_async import IB

            ib = IB()
            ib.connect(host, port, clientId=client_id, timeout=timeout)
            ok = ib.isConnected()
            ib.disconnect()
            return ok
        except Exception:
            return False

    # ── search_symbols ────────────────────────────────────────────────

    # Map IBKR secType codes to our asset_type strings
    _SECTYPE_MAP: dict[str, str] = {
        "STK": "stock",
        "FUT": "futures",
        "IND": "index",
        "OPT": "option",
        "FOP": "futures_option",
        "CASH": "forex",
        "BOND": "bond",
        "WAR": "warrant",
        "FUND": "fund",
        "CMDTY": "commodity",
        "CFD": "cfd",
        "CRYPTO": "crypto",
    }

    # Reverse map: our asset_type → IBKR secType code (for derivative matching)
    _ASSET_TO_SECTYPE: dict[str, str] = {v: k for k, v in _SECTYPE_MAP.items()}

    def search_symbols(
        self,
        config: dict[str, Any],
        query: str,
        asset_type: str | None = None,
    ) -> SymbolSearchResult:
        """Search IBKR for matching symbols via ``reqMatchingSymbols``.

        Uses ``client_id + 2000`` to avoid collisions with other sessions.
        """
        host = config.get("host", "127.0.0.1")
        port = config.get("port", 4001)
        client_id = config.get("client_id", 101) + 2000  # search-only id
        timeout = config.get("timeout_s", 10)

        try:
            from ib_async import IB

            ib = IB()
            ib.connect(host, port, clientId=client_id, timeout=timeout)

            if not ib.isConnected():
                return SymbolSearchResult(
                    success=False,
                    message=f"IBKR gateway at {host}:{port} did not respond",
                )

            try:
                matches = ib.reqMatchingSymbols(query)
                symbols: list[SymbolMatch] = []

                for m in matches or []:
                    contract = m.contract
                    sec_type = getattr(contract, "secType", "")
                    mapped_type = self._SECTYPE_MAP.get(sec_type, sec_type.lower())

                    # Collect derivative sec types for extra info
                    derivative_types = []
                    if hasattr(m, "derivativeSecTypes") and m.derivativeSecTypes:
                        derivative_types = list(m.derivativeSecTypes)

                    # Apply asset_type filter: match the contract's own type
                    # OR any of its derivative types (e.g. NQ is IND→"index"
                    # but has FUT in derivativeSecTypes, so it should match
                    # when filtering for "futures").
                    if asset_type and mapped_type != asset_type:
                        target_sec = self._ASSET_TO_SECTYPE.get(asset_type, "")
                        if target_sec not in derivative_types:
                            continue

                    symbols.append(
                        SymbolMatch(
                            symbol=getattr(contract, "symbol", ""),
                            name=getattr(m, "description", None) or getattr(contract, "description", None),
                            asset_type=mapped_type,
                            exchange=getattr(contract, "primaryExchange", None) or getattr(contract, "exchange", None),
                            currency=getattr(contract, "currency", "USD"),
                            con_id=getattr(contract, "conId", None),
                            extra={
                                "sec_type": sec_type,
                                "derivative_sec_types": derivative_types,
                            } if derivative_types else {
                                "sec_type": sec_type,
                            },
                        )
                    )

                logger.info(
                    "IBKR symbol search for '%s' returned %d result(s)",
                    query, len(symbols),
                )
                return SymbolSearchResult(success=True, symbols=symbols)

            finally:
                ib.disconnect()

        except Exception as e:
            logger.error("IBKR symbol search failed: %s", e)
            return SymbolSearchResult(success=False, message=str(e))
