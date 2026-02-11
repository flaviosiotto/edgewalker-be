"""HTTP client for the ibkr-gateway microservice.

All IBKR operations from the backend are routed through this client,
which talks to the per-Connection ``ibkr-gateway`` container over the
Docker network.

Usage::

    client = IBKRGatewayClient(connection_id=42)
    status = await client.get_status()
    accounts = await client.get_accounts()
    results = await client.search_symbols("NQ", asset_type="futures")
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Container naming convention — must match what ConnectionManager uses
# when spawning containers via Docker API.
CONTAINER_PREFIX = "ibkr-gw-"
GATEWAY_PORT = 8080
DEFAULT_TIMEOUT = 30.0
FETCH_TIMEOUT = 300.0  # Historical fetches can be slow


class IBKRGatewayClient:
    """Async HTTP client that talks to an ``ibkr-gateway`` container."""

    def __init__(
        self,
        connection_id: int | str,
        *,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._connection_id = connection_id
        self._base_url = base_url or self._default_url(connection_id)
        self._timeout = timeout

    @staticmethod
    def _default_url(connection_id: int | str) -> str:
        """Derive the gateway URL from the connection_id.

        On the Docker network the container is named
        ``ibkr-gw-{connection_id}`` and listens on port 8080.
        """
        return f"http://{CONTAINER_PREFIX}{connection_id}:{GATEWAY_PORT}"

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

    async def get_status(self) -> dict:
        """Get detailed connection status."""
        return await self._get("/status")

    async def is_connected(self) -> bool:
        """Quick liveness check — True if RT connection is up."""
        try:
            status = await self._get("/status", timeout=5.0)
            return status.get("connected", False)
        except Exception:
            return False

    # ── Connection lifecycle ──────────────────────────────────────────────

    async def connect(self) -> dict:
        """Trigger IBKR connection (if not already auto-connected)."""
        return await self._post("/connect")

    async def disconnect(self) -> dict:
        """Disconnect from IBKR."""
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
        """Search IBKR for matching symbols."""
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
