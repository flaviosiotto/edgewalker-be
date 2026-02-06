"""
Broker Connectors – abstraction layer for connecting to brokers/exchanges.

Each connector implements the BrokerConnector protocol:
  connect()       → establish connection, return discovered accounts
  disconnect()    → tear down connection
  is_connected()  → check liveness
"""
from app.services.broker_connectors.base import BrokerConnector, ConnectorResult

__all__ = ["BrokerConnector", "ConnectorResult"]
