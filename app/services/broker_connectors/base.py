"""
Base Broker Connector – abstract interface every broker must implement.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DiscoveredAccount:
    """An account discovered from the broker API."""
    account_id: str
    display_name: str | None = None
    account_type: str | None = None   # paper, live, margin …
    currency: str = "USD"
    extra: dict[str, Any] | None = None


@dataclass
class SymbolMatch:
    """A symbol returned from a broker symbol search."""
    symbol: str
    name: str | None = None
    asset_type: str | None = None     # STK, FUT, OPT, IND, …
    exchange: str | None = None
    currency: str = "USD"
    con_id: int | None = None
    extra: dict[str, Any] | None = None


@dataclass
class SymbolSearchResult:
    """Result of a symbol search operation."""
    success: bool
    message: str | None = None
    symbols: list[SymbolMatch] = field(default_factory=list)


@dataclass
class ConnectorResult:
    """Result of a connect / disconnect / health-check operation."""
    success: bool
    message: str | None = None
    accounts: list[DiscoveredAccount] = field(default_factory=list)


class BrokerConnector(ABC):
    """Abstract base class for broker connectors.

    Implementations must be **thread-safe** because they may be called from
    a ``ThreadPoolExecutor`` (blocking I/O libraries like ``ib_async`` require
    running outside the async event loop).
    """

    @property
    @abstractmethod
    def broker_type(self) -> str:
        """Return the broker identifier (e.g. ``'ibkr'``)."""

    @abstractmethod
    def connect(self, config: dict[str, Any]) -> ConnectorResult:
        """Connect to the broker and discover available accounts.

        Args:
            config: Broker-specific connection parameters
                    (host, port, client_id, api_key …).

        Returns:
            ConnectorResult with ``success=True`` and a list of
            ``DiscoveredAccount`` on success, or ``success=False``
            with an error ``message``.
        """

    @abstractmethod
    def disconnect(self, config: dict[str, Any]) -> ConnectorResult:
        """Disconnect from the broker.

        Args:
            config: Same configuration used for ``connect()``.

        Returns:
            ConnectorResult indicating outcome.
        """

    @abstractmethod
    def is_connected(self, config: dict[str, Any]) -> bool:
        """Quick liveness check – is the broker reachable right now?"""

    def search_symbols(self, config: dict[str, Any], query: str, asset_type: str | None = None) -> SymbolSearchResult:
        """Search for symbols/contracts on this broker.

        Default implementation returns an empty result.  Brokers that
        support symbol search should override this.

        Args:
            config: Broker-specific connection parameters.
            query:  Search string (ticker symbol or company name).
            asset_type: Optional filter (e.g. ``'stock'``, ``'futures'``).

        Returns:
            SymbolSearchResult with matched symbols.
        """
        return SymbolSearchResult(success=False, message="Symbol search not supported for this broker")
