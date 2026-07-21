"""HTTP client for broker gateway microservices.

Used by the backend for non-order operations: health checks, account
discovery, symbol search, streaming control, and historical fetches.

Order placement is handled directly by the strategy runner — see
``strategy-runner/app/broker_client.py``.

All broker operations from the backend are routed through this client,
which talks to the per-Connection ``gw-{id}`` container over
the Docker network.

Usage::

    client = GatewayClient(connection_id=42, broker_type="ibkr")
    status = await client.get_status()
    accounts = await client.get_accounts()
    results = await client.search_symbols("NQ", asset_type="futures")
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Default gateway port (all gateways use the same port)
GATEWAY_PORT = 8080
DEFAULT_TIMEOUT = 30.0
FETCH_TIMEOUT = 300.0  # Historical fetches can be slow

# Prefix map: broker_type → container name prefix.
# Must stay in sync with GATEWAY_PREFIXES in shared/constants.py
# (runtime services) and GATEWAY_REGISTRY in connection_manager.py.
# All broker types now use the unified gateway image.
GATEWAY_PREFIXES: dict[str, str] = {
    "ibkr": "gw-",
    "binance": "gw-",
    "ctrader": "gw-",
}


def gateway_url_for(connection_id: int | str, broker_type: str) -> str:
    """Build the gateway base URL for a given connection.

    On the Docker network the container is named
    ``gw-{connection_id}`` and listens on ``GATEWAY_PORT``.

    Raises ``ValueError`` if *broker_type* is empty.
    """
    if not broker_type:
        raise ValueError("broker_type is required")
    prefix = GATEWAY_PREFIXES.get(broker_type, "gw-")
    return f"http://{prefix}{connection_id}:{GATEWAY_PORT}"


class GatewayClient:
    """Async HTTP client that talks to a broker gateway container."""

    def __init__(
        self,
        connection_id: int | str,
        *,
        broker_type: str,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._connection_id = connection_id
        self._broker_type = broker_type
        self._base_url = base_url or gateway_url_for(connection_id, broker_type)
        self._timeout = timeout

    @property
    def base_url(self) -> str:
        return self._base_url

    # ── Low-level helpers ─────────────────────────────────────────────────

    async def _get(self, path: str, params: dict | None = None, timeout: float | None = None) -> dict:
        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout or self._timeout,
        ) as client:
            resp = await client.get(path, params=params)
            resp.raise_for_status()
            return resp.json()

    async def _post(self, path: str, json: dict | None = None, timeout: float | None = None) -> dict:
        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout or self._timeout,
        ) as client:
            resp = await client.post(path, json=json)
            resp.raise_for_status()
            return resp.json()

    async def _delete(self, path: str, timeout: float | None = None) -> dict:
        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout or self._timeout,
        ) as client:
            resp = await client.delete(path)
            resp.raise_for_status()
            return resp.json()

    # ── Health / status ───────────────────────────────────────────────────

    async def health(self) -> dict:
        """Check gateway health."""
        return await self._get("/health")

    async def get_status(self, *, timeout: float | None = None) -> dict:
        """Get detailed connection status."""
        return await self._get("/status", timeout=timeout)

    async def is_connected(self) -> bool:
        """Quick liveness check — True if RT connection is up."""
        try:
            status = await self._get("/status", timeout=5.0)
            return status.get("connected", False)
        except Exception:
            return False

    # ── Connection lifecycle ──────────────────────────────────────────────

    async def connect(self) -> dict:
        """Trigger broker connection (if not already auto-connected)."""
        return await self._post("/connect")

    async def disconnect(self) -> dict:
        """Disconnect from the broker."""
        return await self._post("/disconnect")

    # ── Accounts ──────────────────────────────────────────────────────────

    async def get_accounts(self) -> list[dict]:
        """List managed accounts discovered by the gateway."""
        resp = await self._get("/accounts")
        return resp.get("accounts", [])

    # ── Symbol search ─────────────────────────────────────────────────────

    async def search_symbols(
        self,
        query: str,
        asset_type: str | None = None,
    ) -> list[dict]:
        """Search the broker for matching symbols."""
        params: dict[str, str] = {"query": query}
        if asset_type:
            params["asset_type"] = asset_type
        resp = await self._get("/search", params=params)
        return resp.get("symbols", [])

    # ── Streaming ─────────────────────────────────────────────────────────

    async def subscribe(
        self, symbol: str, data_types: list[str] | None = None
    ) -> dict:
        """Subscribe to real-time market data."""
        payload: dict[str, Any] = {"symbol": symbol}
        if data_types:
            payload["data_types"] = data_types
        return await self._post("/subscribe", json=payload)

    async def unsubscribe(self, symbol: str) -> dict:
        """Unsubscribe from real-time market data."""
        return await self._delete(f"/subscribe/{symbol}")

    async def get_subscriptions(self) -> list[str]:
        """List active streaming subscriptions."""
        resp = await self._get("/subscriptions")
        return resp.get("subscriptions", [])

    # ── Historical fetch ──────────────────────────────────────────────────

    async def fetch_historical(
        self,
        symbol: str,
        start: str,
        end: str,
        timeframe: str = "5m",
        source: str = "ibkr",
        asset_type: str = "stock",
        **kwargs: Any,
    ) -> dict:
        """Start an async historical fetch (returns task_id)."""
        payload = {
            "symbol": symbol,
            "source": source,
            "start": start,
            "end": end,
            "timeframe": timeframe,
            "asset_type": asset_type,
            **kwargs,
        }
        return await self._post("/fetch", json=payload)

    async def fetch_historical_sync(
        self,
        symbol: str,
        start: str,
        end: str,
        timeframe: str = "5m",
        source: str = "ibkr",
        asset_type: str = "stock",
        **kwargs: Any,
    ) -> dict:
        """Synchronous historical fetch — waits for completion."""
        payload = {
            "symbol": symbol,
            "source": source,
            "start": start,
            "end": end,
            "timeframe": timeframe,
            "asset_type": asset_type,
            **kwargs,
        }
        return await self._post("/fetch-sync", json=payload, timeout=FETCH_TIMEOUT)

    async def get_fetch_status(self, task_id: str) -> dict:
        """Check the status of an async fetch task."""
        return await self._get(f"/fetch-status/{task_id}")

    # ── Orders (read-only — placement is handled by the runner) ───────

    async def list_open_orders(self) -> list[dict]:
        """List open orders from the gateway (for reconciliation)."""
        resp = await self._get("/orders")
        return resp.get("orders", [])

    async def reread_orders(
        self,
        since: str | None = None,
        *,
        account: str | None = None,
        persist_checkpoint: bool = True,
    ) -> dict:
        """Trigger a gateway-side order reread, optionally scoped to one broker account."""
        payload: dict[str, Any] = {"persist_checkpoint": persist_checkpoint}
        if since:
            payload["since"] = since
        if account:
            payload["account"] = account
        return await self._post("/orders/reread", json=payload)

    async def reread_fills(
        self,
        since: str | None = None,
        *,
        account: str | None = None,
        persist_checkpoint: bool = True,
    ) -> dict:
        """Trigger a gateway-side fill reread, optionally scoped to one broker account."""
        payload: dict[str, Any] = {"persist_checkpoint": persist_checkpoint}
        if since:
            payload["since"] = since
        if account:
            payload["account"] = account
        return await self._post("/fills/reread", json=payload)

    async def list_positions(self, account: str | None = None) -> list[dict]:
        """List current positions from the gateway (for reconciliation)."""
        params = {"account": account} if account else None
        resp = await self._get("/positions", params=params)
        return resp.get("positions", [])

    async def close_position(
        self,
        position_id: str | int,
        *,
        quantity: float,
        account: str | None = None,
        symbol: str | None = None,
        extra: dict | None = None,
    ) -> dict:
        """Close a broker-native position by id.

        Only for when no runner owns the position (runner stopped): a live
        runner must stay the single commander so its ``_order_lock`` keeps
        closes serialized against the rule engine's own decisions.
        """
        payload: dict[str, Any] = {"quantity": quantity}
        if account:
            payload["account"] = account
        if symbol:
            payload["symbol"] = symbol
        if extra:
            payload["extra"] = extra
        return await self._post(f"/positions/{position_id}/close", json=payload)

    async def reread_positions(
        self,
        *,
        account: str | None = None,
        force_publish: bool = True,
    ) -> dict:
        """Trigger a gateway-side positions snapshot reread."""
        payload: dict[str, Any] = {"force_publish": force_publish}
        if account:
            payload["account"] = account
        return await self._post("/positions/reread", json=payload)
