"""
Symbol Sync Handler - Utility functions for symbol search.

Provides cached symbol search (PostgreSQL) and live gateway symbol search
for API endpoints.
"""
from __future__ import annotations

import logging
from typing import Any

from sqlmodel import select

from app.db.database import get_session_context
from app.models.marketdata import SymbolCache

logger = logging.getLogger(__name__)


# ─── Utility functions for API use ───────────────────────────────────────────


def search_cached_symbols(
    query: str,
    broker_type: str | None = None,
    connection_id: int | None = None,
    asset_type: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Search symbols from cache, optionally scoped to a connection or broker_type."""
    with get_session_context() as session:
        stmt = select(SymbolCache)

        if connection_id is not None:
            stmt = stmt.where(SymbolCache.connection_id == connection_id)
        elif broker_type:
            stmt = stmt.where(SymbolCache.broker_type == broker_type)

        if asset_type:
            stmt = stmt.where(SymbolCache.asset_type == asset_type)

        query_lower = query.lower()
        stmt = stmt.where(
            (SymbolCache.symbol.ilike(f"%{query}%")) |
            (SymbolCache.search_text.ilike(f"%{query_lower}%"))
        )

        stmt = stmt.order_by(
            SymbolCache.symbol != query.upper(),
            ~SymbolCache.symbol.startswith(query.upper()),
            SymbolCache.symbol,
        )

        stmt = stmt.limit(limit)
        results = session.exec(stmt).all()

        return [
            {
                "id": r.id,
                "connection_id": r.connection_id,
                "broker_type": r.broker_type,
                "symbol": r.symbol,
                "name": r.name,
                "asset_type": r.asset_type,
                "exchange": r.exchange,
                "exchange_display": r.exchange_display,
                "currency": r.currency,
                "extra_data": r.extra_data,
                "created_at": r.created_at,
                "updated_at": r.updated_at,
            }
            for r in results
        ]


def search_gateway_symbols_by_id(
    query: str,
    connection_id: int,
    broker_type: str = "ibkr",
    asset_type: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Search symbols live via a connection's gateway (by numeric ID).

    For IBKR futures searches, checks Redis first for cached contract data
    (populated by ibkr-gateway).  Falls back to the gateway HTTP search.
    """
    import os, json, redis as sync_redis
    from app.services.gateway_client import gateway_url_for

    # ── Redis cache check (contracts populated by gateway) ──
    redis_host = os.getenv("REDIS_HOST", "redis")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    cache_key = f"{broker_type}:contracts:{connection_id}:{query.upper()}"
    try:
        r = sync_redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        cached = r.get(cache_key)
        r.close()
        if cached:
            contracts = json.loads(cached)
            if contracts:
                results = [
                    {
                        "symbol": s.get("symbol", ""),
                        "name": s.get("name"),
                        "asset_type": s.get("asset_type", ""),
                        "exchange": s.get("exchange"),
                        "currency": s.get("currency", "USD"),
                        "broker_type": broker_type,
                        "con_id": s.get("con_id"),
                        "extra_data": {
                            k: v for k, v in s.items()
                            if k not in ("symbol", "name", "asset_type", "exchange", "currency", "con_id")
                        },
                    }
                    for s in contracts
                ][:limit]
                return results
    except Exception:
        pass  # Redis unavailable — fall through to HTTP

    # ── Live search via gateway HTTP ──
    from app.services.connection_manager import get_connection_manager

    mgr = get_connection_manager()
    container = mgr._get_container(connection_id, broker_type)
    if container and container.status == "running":
        import httpx

        url = f"{gateway_url_for(connection_id, broker_type)}/search"
        params: dict[str, str] = {"query": query}
        if asset_type:
            params["asset_type"] = asset_type

        try:
            resp = httpx.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            symbols = data.get("symbols", [])[:limit]
            return [
                {
                    "symbol": s.get("symbol", ""),
                    "name": s.get("name"),
                    "asset_type": s.get("asset_type", ""),
                    "exchange": s.get("exchange"),
                    "currency": s.get("currency", "USD"),
                    "broker_type": broker_type,
                    "con_id": s.get("con_id"),
                    "extra_data": {
                        k: v for k, v in s.items()
                        if k not in ("symbol", "name", "asset_type", "exchange", "currency", "con_id")
                    },
                }
                for s in symbols
            ]
        except Exception as e:
            raise RuntimeError(
                f"Symbol search via gateway failed: {e}. "
                f"Is connection {connection_id} connected?"
            )

    raise RuntimeError(
        f"No gateway container running for connection {connection_id}. "
        f"Please connect first."
    )