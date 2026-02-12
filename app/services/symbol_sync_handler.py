"""
Symbol Sync Handler - Handles synchronization of symbols from connections.

This is the concrete implementation of BaseSyncHandler for symbol data.
It fetches symbols from various sources (based on connection broker_type)
and caches them in PostgreSQL.

Note: Blocking operations (yfinance, httpx) are run in thread pool
to avoid blocking the async event loop.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from typing import Any

from sqlmodel import Session, select, delete

from app.db.database import get_session_context
from app.models.connection import Connection
from app.models.marketdata import SymbolCache, SymbolSyncLog
from app.services.sync_manager import BaseSyncHandler, SyncType, FetchResult

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="symbol_sync_")


async def _run_blocking(func, *args, **kwargs):
    """Run a blocking function in thread pool."""
    loop = asyncio.get_event_loop()
    if kwargs:
        func = partial(func, **kwargs)
    return await loop.run_in_executor(_executor, func, *args)


class SymbolSyncHandler(BaseSyncHandler):
    """Handler for syncing symbols from connections."""

    @property
    def sync_type(self) -> SyncType:
        return SyncType.SYMBOLS

    async def fetch_current_data(self, conn: Connection) -> FetchResult:
        """Fetch symbols based on connection's broker_type."""
        if conn.broker_type == "yahoo":
            return await self._fetch_yahoo_symbols(conn)
        elif conn.broker_type == "ibkr":
            return await self._fetch_ibkr_symbols(conn)
        else:
            return FetchResult(
                available=True,
                data=[],
                content_hash=hashlib.sha256(b"[]").hexdigest()[:16],
            )

    def _compute_hash(self, symbols: list[dict]) -> str:
        symbols_sorted = sorted(symbols, key=lambda x: x.get("symbol", ""))
        hash_input = json.dumps(
            [s.get("symbol") for s in symbols_sorted],
            sort_keys=True
        )
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    async def get_cached_hash(self, conn: Connection) -> str | None:
        config = conn.config or {}
        return config.get("symbols_hash")

    async def apply_changes(
        self,
        conn: Connection,
        new_data: list[dict],
        is_full_refresh: bool,
    ) -> tuple[int, int, int]:
        """Apply symbol changes to the cache."""
        with get_session_context() as session:
            conn = session.get(Connection, conn.id)

            sync_log = SymbolSyncLog(
                connection_id=conn.id,
                connection_name=conn.name,
                started_at=datetime.utcnow(),
                status="running",
                symbols_fetched=len(new_data),
            )
            session.add(sync_log)
            conn.last_sync_status = "running"
            session.commit()
            session.refresh(sync_log)

            try:
                removed = 0

                if is_full_refresh:
                    count_stmt = select(SymbolCache).where(
                        SymbolCache.connection_id == conn.id
                    )
                    removed = len(session.exec(count_stmt).all())

                    stmt = delete(SymbolCache).where(
                        SymbolCache.connection_id == conn.id
                    )
                    session.exec(stmt)
                    session.commit()

                added, updated = self._upsert_symbols(session, conn, new_data)

                count_stmt = select(SymbolCache).where(
                    SymbolCache.connection_id == conn.id
                )
                conn.symbols_count = len(session.exec(count_stmt).all())
                conn.last_sync_at = datetime.now(timezone.utc)
                conn.last_sync_status = "success"
                conn.last_sync_error = None

                sync_log.status = "success"
                sync_log.completed_at = datetime.utcnow()
                sync_log.symbols_added = added
                sync_log.symbols_updated = updated
                sync_log.symbols_removed = removed
                sync_log.duration_seconds = (
                    sync_log.completed_at - sync_log.started_at
                ).total_seconds()

                session.commit()
                return added, updated, removed

            except Exception as e:
                sync_log.status = "error"
                sync_log.error_message = str(e)
                sync_log.completed_at = datetime.utcnow()
                conn.last_sync_status = "error"
                conn.last_sync_error = str(e)
                session.commit()
                raise

    async def update_connection_hash(self, conn: Connection, content_hash: str) -> None:
        with get_session_context() as session:
            conn = session.get(Connection, conn.id)
            config = dict(conn.config or {})
            config["symbols_hash"] = content_hash
            conn.config = config
            conn.updated_at = datetime.now(timezone.utc)
            session.commit()

    async def update_connection_status(
        self,
        conn: Connection,
        status: str,
        message: str | None = None
    ) -> None:
        with get_session_context() as session:
            conn = session.get(Connection, conn.id)
            conn.last_sync_status = status
            conn.last_sync_error = message
            conn.last_sync_at = datetime.now(timezone.utc)
            conn.updated_at = datetime.now(timezone.utc)

            sync_log = SymbolSyncLog(
                connection_id=conn.id,
                connection_name=conn.name,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                status=status,
                symbols_fetched=0,
                symbols_added=0,
                symbols_updated=0,
                symbols_removed=0,
                error_message=message,
                duration_seconds=0,
            )
            session.add(sync_log)
            session.commit()

    def _upsert_symbols(
        self,
        session: Session,
        conn: Connection,
        symbols: list[dict],
    ) -> tuple[int, int]:
        """Upsert symbols into cache."""
        added = 0
        updated = 0

        for sym_data in symbols:
            symbol = sym_data.get("symbol", "")
            if not symbol:
                continue

            stmt = select(SymbolCache).where(
                SymbolCache.connection_id == conn.id,
                SymbolCache.symbol == symbol,
            )
            existing = session.exec(stmt).first()

            search_text = f"{symbol} {sym_data.get('name', '')} {sym_data.get('exchange', '')}".lower()

            if existing:
                existing.name = sym_data.get("name", existing.name)
                existing.asset_type = sym_data.get("asset_type", existing.asset_type)
                existing.exchange = sym_data.get("exchange", existing.exchange)
                existing.exchange_display = sym_data.get("exchange_display", existing.exchange_display)
                existing.currency = sym_data.get("currency", existing.currency)
                existing.extra_data = sym_data.get("extra_data", existing.extra_data)
                existing.search_text = search_text
                existing.updated_at = datetime.utcnow()
                updated += 1
            else:
                new_symbol = SymbolCache(
                    connection_id=conn.id,
                    broker_type=conn.broker_type,
                    symbol=symbol,
                    name=sym_data.get("name", ""),
                    asset_type=sym_data.get("asset_type", "stock"),
                    exchange=sym_data.get("exchange", ""),
                    exchange_display=sym_data.get("exchange_display", ""),
                    currency=sym_data.get("currency", "USD"),
                    extra_data=sym_data.get("extra_data", {}),
                    search_text=search_text,
                )
                session.add(new_symbol)
                added += 1

        session.commit()
        return added, updated

    def _fetch_yahoo_symbols_blocking(self, symbol_lists: dict, symbol_categories: dict, all_symbols: list) -> tuple[list[dict], list[str]]:
        """Blocking operation to fetch Yahoo symbols - runs in thread pool."""
        import yfinance as yf

        results = []
        errors = []

        try:
            tickers = yf.Tickers(" ".join(all_symbols))

            for symbol in all_symbols:
                try:
                    ticker = tickers.tickers.get(symbol)
                    if not ticker:
                        continue

                    info = ticker.info
                    if not info or info.get("regularMarketPrice") is None:
                        continue

                    quote_type = info.get("quoteType", "EQUITY")
                    if quote_type == "ETF":
                        category = "etf"
                    elif quote_type == "INDEX":
                        category = "index"
                    else:
                        category = symbol_categories.get(symbol, "stock")

                    results.append({
                        "symbol": symbol,
                        "name": info.get("shortName") or info.get("longName", ""),
                        "asset_type": category,
                        "exchange": info.get("exchange", ""),
                        "exchange_display": info.get("exchangeTimezoneName", ""),
                        "currency": info.get("currency", "USD"),
                        "extra_data": {
                            "quote_type": quote_type,
                            "market_cap": info.get("marketCap"),
                            "avg_volume": info.get("averageDailyVolume10Day"),
                            "sector": info.get("sector"),
                            "industry": info.get("industry"),
                        },
                    })
                except Exception as e:
                    logger.debug(f"Failed to fetch {symbol}: {e}")
                    continue

        except Exception as e:
            errors.append(str(e))
            logger.warning(f"Failed to fetch from Yahoo Finance: {e}")

        return results, errors

    async def _fetch_yahoo_symbols(self, conn: Connection) -> FetchResult:
        """Fetch symbols from Yahoo Finance using yfinance library."""
        symbol_lists = {
            "indices": ["^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX", "^FTSE", "^N225", "^HSI"],
            "etfs": [
                "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VEA", "VWO",
                "EFA", "EEM", "AGG", "BND", "LQD", "HYG", "TLT", "IEF",
                "GLD", "SLV", "USO", "UNG", "XLF", "XLK", "XLE", "XLV",
                "XLI", "XLY", "XLP", "XLB", "XLU", "XLRE", "VNQ", "ARKK",
            ],
            "stocks": [
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK-B",
                "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "BAC", "ADBE",
                "CRM", "NFLX", "CMCSA", "PFE", "TMO", "ABT", "COST", "PEP", "AVGO",
                "CSCO", "ACN", "MRK", "NKE", "WMT", "LLY", "DHR", "MCD", "NEE",
                "AMD", "INTC", "IBM", "ORCL", "QCOM", "TXN", "AMAT", "LRCX", "MU",
            ],
        }

        all_symbols = []
        symbol_categories = {}
        for category, symbols in symbol_lists.items():
            for sym in symbols:
                all_symbols.append(sym)
                symbol_categories[sym] = category.rstrip("s")

        results, errors = await _run_blocking(
            self._fetch_yahoo_symbols_blocking,
            symbol_lists, symbol_categories, all_symbols
        )

        if not results and errors:
            return FetchResult(
                available=False,
                data=[],
                content_hash="",
                skip_reason=f"Yahoo Finance unavailable: {errors[0]}",
            )

        sorted_symbols = sorted(results, key=lambda x: x["symbol"])
        content_hash = hashlib.md5(
            json.dumps(sorted_symbols, sort_keys=True, default=str).encode()
        ).hexdigest()

        logger.info(f"Fetched {len(results)} symbols from Yahoo Finance")
        return FetchResult(available=True, data=results, content_hash=content_hash)

    def _check_ibkr_connection_blocking(self, connection_id: int, timeout: int) -> tuple[bool, str | None]:
        """Check IBKR connectivity via the connection's ibkr-gateway container."""
        try:
            import httpx

            resp = httpx.get(
                f"http://ibkr-gw-{connection_id}:8080/status",
                timeout=timeout,
            )
            data = resp.json()
            if data.get("connected"):
                return True, None
            return False, "Gateway running but not connected to IBKR"
        except Exception as e:
            return False, f"ibkr-gateway not reachable: {e}"

    async def _fetch_ibkr_symbols(self, conn: Connection) -> FetchResult:
        """Check IBKR gateway connection status.

        IBKR doesn't have a "list all symbols" API — symbols are
        discovered on-demand through contract search.
        """
        config = conn.config or {}
        timeout = config.get("timeout_s", 5)

        try:
            is_connected, error = await asyncio.wait_for(
                _run_blocking(
                    self._check_ibkr_connection_blocking,
                    conn.id, timeout
                ),
                timeout=timeout + 5
            )

            if is_connected:
                logger.info(f"IBKR gateway connected for connection {conn.name}")
                return FetchResult(
                    available=True,
                    data=[],
                    content_hash="connected",
                )
            else:
                return FetchResult(
                    available=False,
                    data=[],
                    content_hash="",
                    skip_reason=f"IBKR gateway not available: {error}",
                )

        except asyncio.TimeoutError:
            return FetchResult(
                available=False,
                data=[],
                content_hash="",
                skip_reason=f"IBKR gateway connection timeout for {conn.name}",
            )
        except Exception as e:
            return FetchResult(
                available=False,
                data=[],
                content_hash="",
                skip_reason=f"IBKR gateway not available: {str(e)}",
            )


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


def search_ibkr_symbols(
    query: str,
    connection_name: str,
    asset_type: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Search IBKR symbols live via a named connection's ibkr-gateway."""
    from app.services.connection_manager import get_connection_manager

    with get_session_context() as session:
        stmt = select(Connection).where(Connection.name == connection_name)
        connection = session.exec(stmt).first()

        if connection is None:
            raise ValueError(f"Connection '{connection_name}' not found")

        if connection.broker_type != "ibkr":
            raise ValueError(
                f"Connection '{connection_name}' is of type "
                f"'{connection.broker_type}', expected 'ibkr'"
            )

        connection_id = connection.id

    mgr = get_connection_manager()
    container = mgr._get_container(connection_id)
    if container and container.status == "running":
        import httpx

        url = f"http://ibkr-gw-{connection_id}:8080/search"
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
                    "broker_type": "ibkr",
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
                f"IBKR symbol search via gateway failed: {e}. "
                f"Is the connection '{connection_name}' connected?"
            )

    raise RuntimeError(
        f"No ibkr-gateway container running for connection '{connection_name}'. "
        f"Please connect first."
    )


def search_ibkr_symbols_by_id(
    query: str,
    connection_id: int,
    asset_type: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Search IBKR symbols live via a connection's ibkr-gateway (by numeric ID)."""
    from app.services.connection_manager import get_connection_manager

    mgr = get_connection_manager()
    container = mgr._get_container(connection_id)
    if container and container.status == "running":
        import httpx

        url = f"http://ibkr-gw-{connection_id}:8080/search"
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
                    "broker_type": "ibkr",
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
                f"IBKR symbol search via gateway failed: {e}. "
                f"Is connection {connection_id} connected?"
            )

    raise RuntimeError(
        f"No ibkr-gateway container running for connection {connection_id}. "
        f"Please connect first."
    )
