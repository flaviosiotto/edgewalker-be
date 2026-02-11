"""
Broker Connectors — shared data types for connection lifecycle.

IBKR operations are handled by per-Connection ``ibkr-gateway`` containers,
managed by ``ConnectionManager`` and accessed via ``IBKRGatewayClient``.
"""
from app.services.broker_connectors.base import ConnectorResult, DiscoveredAccount

__all__ = ["ConnectorResult", "DiscoveredAccount"]
