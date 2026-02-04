"""
WebSocket Market Data API - Real-time market data streaming for UI clients.

Provides WebSocket endpoint for:
- Subscribing to real-time market data
- Receiving normalized tick, bar, quote events
- Real-time indicator calculation (using TA-Lib, same as backtesting)
- Best-effort delivery (via Redis Pub/Sub)
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import redis.asyncio as redis

from app.services.realtime_indicators import indicator_calculator

logger = logging.getLogger(__name__)

router = APIRouter(tags=["WebSocket Market Data"])


# Redis configuration (from environment or defaults)
import os
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))


@dataclass
class WebSocketClient:
    """Represents a connected WebSocket client."""
    websocket: WebSocket
    subscriptions: set[str] = field(default_factory=set)
    # symbol -> list of indicator configs
    indicators: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    
    async def send_json(self, data: dict) -> bool:
        """Send JSON message to client. Returns False if failed."""
        try:
            await self.websocket.send_json(data)
            return True
        except Exception as e:
            logger.warning(f"Failed to send to client: {e}")
            return False


class ConnectionManager:
    """
    Manages WebSocket connections and Redis Pub/Sub subscriptions.
    
    Routes market data from Redis Pub/Sub to subscribed WebSocket clients.
    """
    
    def __init__(self):
        self._clients: dict[str, WebSocketClient] = {}  # client_id -> client
        self._channel_clients: dict[str, set[str]] = {}  # channel -> set of client_ids
        
        self._redis: redis.Redis | None = None
        self._pubsub: redis.client.PubSub | None = None
        self._listener_task: asyncio.Task | None = None
        self._running = False
    
    async def start(self) -> None:
        """Start the connection manager and Redis listener."""
        if self._running:
            return
        
        try:
            self._redis = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True,
            )
            await self._redis.ping()
            
            self._pubsub = self._redis.pubsub()
            self._running = True
            
            # Start listener in background
            self._listener_task = asyncio.create_task(self._listen_redis())
            
            logger.info(f"ConnectionManager started, Redis: {REDIS_HOST}:{REDIS_PORT}")
            
        except Exception as e:
            logger.error(f"Failed to start ConnectionManager: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the connection manager."""
        self._running = False
        
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        
        if self._pubsub:
            await self._pubsub.close()
        
        if self._redis:
            await self._redis.close()
        
        logger.info("ConnectionManager stopped")
    
    async def connect(self, websocket: WebSocket) -> str:
        """
        Register a new WebSocket connection.
        
        Returns:
            Client ID
        """
        await websocket.accept()
        
        client_id = f"client-{id(websocket)}"
        self._clients[client_id] = WebSocketClient(websocket=websocket)
        
        logger.info(f"Client connected: {client_id}")
        return client_id
    
    async def disconnect(self, client_id: str) -> None:
        """Handle client disconnection."""
        client = self._clients.pop(client_id, None)
        if not client:
            return
        
        # Remove from all channel subscriptions
        for channel in client.subscriptions:
            if channel in self._channel_clients:
                self._channel_clients[channel].discard(client_id)
                
                # Unsubscribe from Redis if no more clients
                if not self._channel_clients[channel]:
                    del self._channel_clients[channel]
                    if self._pubsub:
                        await self._pubsub.unsubscribe(channel)
        
        logger.info(f"Client disconnected: {client_id}")
    
    async def subscribe(self, client_id: str, channels: list[str]) -> None:
        """
        Subscribe client to channels.
        
        Args:
            client_id: Client identifier
            channels: List of Redis Pub/Sub channels (e.g., "live:ticks:QQQ")
        """
        client = self._clients.get(client_id)
        if not client:
            return
        
        # Extract symbols from channels to request from collector
        symbols_to_request: set[str] = set()
        
        for channel in channels:
            # Add to client subscriptions
            client.subscriptions.add(channel)
            
            # Extract symbol from channel (e.g., "live:ticks:QQQ" -> "QQQ")
            parts = channel.split(":")
            if len(parts) >= 3:
                symbol = parts[2]
                symbols_to_request.add(symbol)
            
            # Add to channel -> clients mapping
            if channel not in self._channel_clients:
                self._channel_clients[channel] = set()
                # Subscribe to Redis channel
                if self._pubsub:
                    await self._pubsub.subscribe(channel)
            
            self._channel_clients[channel].add(client_id)
        
        # Request collector to start streaming these symbols
        for symbol in symbols_to_request:
            await self._request_collector_subscription(symbol)
        
        logger.info(f"Client {client_id} subscribed to {channels}")
    
    async def _request_collector_subscription(self, symbol: str) -> None:
        """
        Request the datasource-collector to start streaming a symbol.
        
        Sends a command via Redis Pub/Sub to control:subscribe channel.
        """
        if not self._redis:
            logger.warning(f"Cannot request subscription for {symbol}: Redis not connected")
            return
        
        try:
            command = {
                "symbol": symbol,
                "data_types": ["tick", "quote", "bar_5s"],
            }
            result = await self._redis.publish("control:subscribe", json.dumps(command))
            logger.info(f"Requested collector subscription for {symbol}, receivers: {result}")
        except Exception as e:
            logger.error(f"Failed to request collector subscription for {symbol}: {e}")
    
    async def unsubscribe(self, client_id: str, channels: list[str]) -> None:
        """Unsubscribe client from channels."""
        client = self._clients.get(client_id)
        if not client:
            return
        
        for channel in channels:
            client.subscriptions.discard(channel)
            
            if channel in self._channel_clients:
                self._channel_clients[channel].discard(client_id)
                
                # Unsubscribe from Redis if no more clients
                if not self._channel_clients[channel]:
                    del self._channel_clients[channel]
                    if self._pubsub:
                        await self._pubsub.unsubscribe(channel)
        
        logger.info(f"Client {client_id} unsubscribed from {channels}")
    
    async def configure_indicators(
        self, client_id: str, symbol: str, indicators: list[dict[str, Any]]
    ) -> None:
        """
        Configure indicators for a symbol for this client.
        
        The indicators will be calculated when bar data arrives and
        the values will be included in the market_data message.
        
        Args:
            client_id: Client identifier
            symbol: Trading symbol
            indicators: List of indicator configs, e.g.:
                [{"type": "SMA", "name": "sma_20", "params": {"period": 20}}]
        """
        logger.info(f"configure_indicators called: client={client_id}, symbol={symbol}, indicators={indicators}")
        
        client = self._clients.get(client_id)
        if not client:
            logger.warning(f"Client {client_id} not found for configure_indicators")
            return
        
        # Store indicators config for this client
        client.indicators[symbol] = indicators
        
        # Configure the global calculator
        # Merge all client indicators for this symbol
        all_indicators = []
        for c in self._clients.values():
            all_indicators.extend(c.indicators.get(symbol, []))
        
        # Deduplicate by name
        seen = set()
        unique_indicators = []
        for ind in all_indicators:
            name = ind.get("name", ind.get("type", "unknown"))
            if name not in seen:
                seen.add(name)
                unique_indicators.append(ind)
        
        indicator_calculator.configure_indicators(symbol, unique_indicators)
        logger.info(f"Client {client_id} configured {len(indicators)} indicators for {symbol}: {[i.get('name') for i in indicators]}")
    
    async def _listen_redis(self) -> None:
        """Listen for Redis Pub/Sub messages and route to clients."""
        logger.info("Redis listener started")
        while self._running:
            try:
                # Skip if no active subscriptions
                if not self._channel_clients:
                    await asyncio.sleep(0.5)
                    continue
                
                # Skip if pubsub not ready
                if not self._pubsub or not self._pubsub.subscribed:
                    logger.debug(f"Pubsub not ready: pubsub={self._pubsub is not None}, subscribed={getattr(self._pubsub, 'subscribed', False)}")
                    await asyncio.sleep(0.5)
                    continue
                
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0,
                )
                
                if message and message["type"] == "message":
                    channel = message["channel"]
                    data = message["data"]
                    logger.debug(f"Received message on {channel}")
                    
                    await self._broadcast_to_channel(channel, data)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in Redis listener: {e}")
                await asyncio.sleep(1)
    
    async def _broadcast_to_channel(self, channel: str, data: str) -> None:
        """Broadcast message to all clients subscribed to a channel."""
        client_ids = self._channel_clients.get(channel, set())
        
        if not client_ids:
            return
        
        # Parse JSON data
        try:
            parsed_data = json.loads(data)
        except json.JSONDecodeError:
            parsed_data = {"raw": data}
        
        # If this is a bar event, calculate indicators
        event_type = parsed_data.get("event_type", "")
        symbol = parsed_data.get("symbol", "")
        
        indicator_values = {}
        if event_type == "bar" and symbol:
            indicator_values = indicator_calculator.process_bar(symbol, parsed_data)
            logger.debug(f"Bar event for {symbol}: calculated {len(indicator_values)} indicator values: {list(indicator_values.keys())}")
            if indicator_values:
                parsed_data["indicators"] = indicator_values
        
        message = {
            "type": "market_data",
            "channel": channel,
            "data": parsed_data,
        }
        
        # Send to all subscribed clients
        disconnected = []
        for client_id in client_ids:
            client = self._clients.get(client_id)
            if client:
                success = await client.send_json(message)
                if not success:
                    disconnected.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected:
            await self.disconnect(client_id)
    
    def get_stats(self) -> dict[str, Any]:
        """Get connection statistics."""
        return {
            "clients_connected": len(self._clients),
            "channels_active": len(self._channel_clients),
            "subscriptions_by_channel": {
                ch: len(clients) for ch, clients in self._channel_clients.items()
            },
        }


# Global connection manager instance
connection_manager = ConnectionManager()


async def startup_websocket_manager():
    """Start the WebSocket connection manager."""
    try:
        await connection_manager.start()
        logger.info("WebSocket connection manager started")
    except Exception as e:
        logger.error(f"Failed to start WebSocket connection manager: {e}")


async def shutdown_websocket_manager():
    """Stop the WebSocket connection manager."""
    await connection_manager.stop()
    logger.info("WebSocket connection manager stopped")


@router.websocket("/ws/marketdata")
async def websocket_marketdata(websocket: WebSocket):
    """
    WebSocket endpoint for real-time market data.
    
    **Protocol:**
    
    Client → Server messages:
    ```json
    {"action": "subscribe", "channels": ["live:ticks:QQQ", "live:bars:QQQ:5s"]}
    {"action": "unsubscribe", "channels": ["live:ticks:QQQ"]}
    {"action": "ping"}
    ```
    
    Server → Client messages:
    ```json
    {"type": "connected", "client_id": "client-123"}
    {"type": "subscribed", "channels": ["live:ticks:QQQ"]}
    {"type": "market_data", "channel": "live:ticks:QQQ", "data": {...}}
    {"type": "pong"}
    {"type": "error", "message": "..."}
    ```
    
    **Available channels:**
    - `live:ticks:{symbol}` - Real-time trade ticks
    - `live:bars:{symbol}:{timeframe}` - Real-time bars (e.g., 5s, 1m)
    - `live:quotes:{symbol}` - Real-time bid/ask quotes
    """
    client_id = await connection_manager.connect(websocket)
    
    # Send connected message
    await websocket.send_json({
        "type": "connected",
        "client_id": client_id,
    })
    
    try:
        while True:
            # Wait for client messages
            data = await websocket.receive_json()
            
            action = data.get("action")
            
            if action == "subscribe":
                channels = data.get("channels", [])
                await connection_manager.subscribe(client_id, channels)
                await websocket.send_json({
                    "type": "subscribed",
                    "channels": channels,
                })
                
            elif action == "unsubscribe":
                channels = data.get("channels", [])
                await connection_manager.unsubscribe(client_id, channels)
                await websocket.send_json({
                    "type": "unsubscribed",
                    "channels": channels,
                })
                
            elif action == "configure_indicators":
                # Configure indicators for a symbol
                # Expected: {"action": "configure_indicators", "symbol": "QQQ", "indicators": [...]}
                symbol = data.get("symbol", "")
                indicators = data.get("indicators", [])
                if symbol:
                    await connection_manager.configure_indicators(client_id, symbol, indicators)
                    await websocket.send_json({
                        "type": "indicators_configured",
                        "symbol": symbol,
                        "count": len(indicators),
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Missing symbol for configure_indicators",
                    })
            
            elif action == "prefill_buffer":
                # Pre-fill indicator buffer with historical bars for warm-up
                # Expected: {"action": "prefill_buffer", "symbol": "QQQ", "bars": [...]}
                symbol = data.get("symbol", "")
                bars = data.get("bars", [])
                if symbol and bars:
                    logger.info(
                        f"prefill_buffer called: client={client_id}, symbol={symbol}, bars={len(bars)}"
                    )
                    indicator_calculator.prefill_buffer(symbol, bars)
                    await websocket.send_json({
                        "type": "buffer_prefilled",
                        "symbol": symbol,
                        "count": len(bars),
                        "buffer_size": indicator_calculator.get_buffer_size(symbol),
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Missing symbol or bars for prefill_buffer",
                    })
                
            elif action == "ping":
                await websocket.send_json({"type": "pong"})
                
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown action: {action}",
                })
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
    finally:
        await connection_manager.disconnect(client_id)


@router.get("/ws/stats")
async def websocket_stats():
    """Get WebSocket connection statistics."""
    return connection_manager.get_stats()
