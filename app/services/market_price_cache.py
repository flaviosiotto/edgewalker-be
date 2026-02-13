"""Redis market price cache — reads last known prices for PnL computation.

The ibkr-gateway publishes ticks to ``live:ticks:{symbol}`` via Redis
Pub/Sub.  This module subscribes to those channels and caches the last
known price per symbol, making it available for synchronous lookups
(e.g. from the positions API).

Also supports reading from Redis keys directly, in case a price was
SET by the gateway or another service.

Usage::

    from app.services.market_price_cache import price_cache

    # In FastAPI lifespan:
    await price_cache.start()

    # Anywhere:
    last = price_cache.get_last_price("MNQ")
    # → {"price": 21845.5, "timestamp": 1707784200.0, "symbol": "MNQ"}
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))


class MarketPriceCache:
    """In-memory cache of last known prices, fed by Redis Pub/Sub."""

    def __init__(self) -> None:
        self._prices: dict[str, dict[str, Any]] = {}  # symbol → {price, ts, bid, ask}
        self._redis: aioredis.Redis | None = None
        self._pubsub: aioredis.client.PubSub | None = None
        self._listener_task: asyncio.Task | None = None
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def start(self) -> None:
        """Connect to Redis and start background listener."""
        if self._running:
            return
        try:
            self._redis = aioredis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True,
            )
            await self._redis.ping()
            self._pubsub = self._redis.pubsub()

            # Subscribe to all tick and quote channels with pattern
            await self._pubsub.psubscribe("live:ticks:*", "live:quotes:*")

            self._running = True
            self._listener_task = asyncio.create_task(self._listen())
            logger.info("MarketPriceCache started (Redis %s:%s)", REDIS_HOST, REDIS_PORT)
        except Exception as e:
            logger.warning("MarketPriceCache failed to start: %s", e)

    async def stop(self) -> None:
        """Stop the listener and disconnect."""
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
        logger.info("MarketPriceCache stopped")

    # ── Price access ──────────────────────────────────────────────────

    def get_last_price(self, symbol: str) -> dict[str, Any] | None:
        """Return the last known price for a symbol, or None.

        Returns::

            {
                "price": 21845.5,
                "bid": 21845.25,
                "ask": 21845.75,
                "timestamp": 1707784200.123,
                "symbol": "MNQ",
                "age_seconds": 0.45,
            }
        """
        entry = self._prices.get(symbol)
        if entry is None:
            return None
        return {
            **entry,
            "age_seconds": round(time.time() - entry.get("timestamp", 0), 2),
        }

    def get_last_prices(self, symbols: list[str]) -> dict[str, dict[str, Any] | None]:
        """Batch lookup of last prices for multiple symbols."""
        return {s: self.get_last_price(s) for s in symbols}

    @property
    def cached_symbols(self) -> list[str]:
        """List symbols currently in the cache."""
        return list(self._prices.keys())

    # ── Background listener ───────────────────────────────────────────

    async def _listen(self) -> None:
        """Background task — reads Redis Pub/Sub messages and caches prices."""
        logger.info("MarketPriceCache listener started")
        try:
            while self._running:
                try:
                    message = await self._pubsub.get_message(
                        ignore_subscribe_messages=True,
                        timeout=1.0,
                    )
                    if message is None:
                        await asyncio.sleep(0.01)
                        continue

                    channel = message.get("channel", "")
                    data_raw = message.get("data", "")

                    if not isinstance(data_raw, str):
                        continue

                    try:
                        data = json.loads(data_raw)
                    except (json.JSONDecodeError, TypeError):
                        continue

                    # Extract symbol from channel pattern:
                    #   live:ticks:MNQ  → MNQ
                    #   live:quotes:MNQ → MNQ
                    parts = channel.split(":")
                    if len(parts) < 3:
                        continue
                    symbol = parts[2]

                    # Update price cache
                    ts = (
                        data.get("timestamp")
                        or data.get("ts")
                        or data.get("time")
                        or time.time()
                    )
                    # Convert ms timestamp to seconds if needed
                    if isinstance(ts, (int, float)) and ts > 1e12:
                        ts = ts / 1000.0

                    price = data.get("price") or data.get("last") or data.get("close")
                    if price is not None:
                        entry: dict[str, Any] = {
                            "symbol": symbol,
                            "price": float(price),
                            "timestamp": float(ts),
                        }
                        # Optionally capture bid/ask from quotes
                        if "bid" in data and data["bid"] is not None:
                            entry["bid"] = float(data["bid"])
                        if "ask" in data and data["ask"] is not None:
                            entry["ask"] = float(data["ask"])

                        self._prices[symbol] = entry

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.debug("MarketPriceCache listener error: %s", e)
                    await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass
        logger.info("MarketPriceCache listener stopped")


# ── Singleton ─────────────────────────────────────────────────────────

price_cache = MarketPriceCache()
