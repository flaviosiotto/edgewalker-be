"""
Broker data types — shared dataclasses used by the connection manager
and gateway client.
"""
from __future__ import annotations

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
class ConnectorResult:
    """Result of a connect / disconnect / health-check operation."""
    success: bool
    message: str | None = None
    accounts: list[DiscoveredAccount] = field(default_factory=list)
