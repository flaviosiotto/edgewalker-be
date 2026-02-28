"""
Broker Connectors — shared data types for connection lifecycle.

Broker operations are handled by per-Connection gateway containers
(e.g. ``ibkr-gateway``, ``binance-gateway``), managed by
``ConnectionManager`` and accessed via ``GatewayClient``.
"""
from app.services.broker_connectors.base import ConnectorResult, DiscoveredAccount

__all__ = ["ConnectorResult", "DiscoveredAccount"]
