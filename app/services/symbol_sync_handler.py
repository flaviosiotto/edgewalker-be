"""
Symbol Sync Handler - Handles synchronization of symbols from data sources.

This is the concrete implementation of BaseSyncHandler for symbol data.
It fetches symbols from various sources and caches them in PostgreSQL.

Note: Blocking operations (yfinance, ib_async) are run in thread pool
to avoid blocking the async event loop.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from typing import Any

from sqlmodel import Session, select, delete

from app.db.database import get_session_context
from app.models.marketdata import DataSource, SymbolCache, SymbolSyncLog
from app.services.sync_manager import BaseSyncHandler, SyncType, FetchResult

logger = logging.getLogger(__name__)

# Thread pool for blocking I/O operations
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="symbol_sync_")


async def _run_blocking(func, *args, **kwargs):
    """Run a blocking function in thread pool."""
    loop = asyncio.get_event_loop()
    if kwargs:
        func = partial(func, **kwargs)
    return await loop.run_in_executor(_executor, func, *args)


class SymbolSyncHandler(BaseSyncHandler):
    """Handler for syncing symbols from data sources."""
    
    @property
    def sync_type(self) -> SyncType:
        return SyncType.SYMBOLS
    
    async def fetch_current_data(self, source: DataSource) -> FetchResult:
        """Fetch symbols from the source and compute content hash."""
        if source.source_type == "yahoo":
            return await self._fetch_yahoo_symbols(source)
        elif source.source_type == "ibkr":
            return await self._fetch_ibkr_symbols(source)
        else:
            # Unknown source type - return empty but available
            return FetchResult(
                available=True,
                data=[],
                content_hash=hashlib.sha256(b"[]").hexdigest()[:16],
            )
    
    def _compute_hash(self, symbols: list[dict]) -> str:
        """Compute content hash for change detection."""
        symbols_sorted = sorted(symbols, key=lambda x: x.get("symbol", ""))
        hash_input = json.dumps(
            [s.get("symbol") for s in symbols_sorted],
            sort_keys=True
        )
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    async def get_cached_hash(self, source: DataSource) -> str | None:
        """Get the hash stored in the source config."""
        config = source.config or {}
        return config.get("symbols_hash")
    
    async def apply_changes(
        self,
        source: DataSource,
        new_data: list[dict],
        is_full_refresh: bool,
    ) -> tuple[int, int, int]:
        """Apply symbol changes to the cache."""
        with get_session_context() as session:
            # Refresh source from this session
            source = session.get(DataSource, source.id)
            
            # Create sync log
            sync_log = SymbolSyncLog(
                source_id=source.id,
                source_name=source.name,
                started_at=datetime.utcnow(),
                status="running",
                symbols_fetched=len(new_data),
            )
            session.add(sync_log)
            source.last_sync_status = "running"
            session.commit()
            session.refresh(sync_log)
            
            try:
                removed = 0
                
                if is_full_refresh:
                    # Count existing before delete
                    count_stmt = select(SymbolCache).where(
                        SymbolCache.source_id == source.id
                    )
                    removed = len(session.exec(count_stmt).all())
                    
                    # Delete all existing
                    stmt = delete(SymbolCache).where(
                        SymbolCache.source_id == source.id
                    )
                    session.exec(stmt)
                    session.commit()
                
                # Upsert symbols
                added, updated = self._upsert_symbols(session, source, new_data)
                
                # Update source stats
                count_stmt = select(SymbolCache).where(
                    SymbolCache.source_id == source.id
                )
                source.symbols_count = len(session.exec(count_stmt).all())
                source.last_sync_at = datetime.utcnow()
                source.last_sync_status = "success"
                source.last_sync_error = None
                
                # Update sync log
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
                source.last_sync_status = "error"
                source.last_sync_error = str(e)
                session.commit()
                raise
    
    async def update_source_hash(self, source: DataSource, content_hash: str) -> None:
        """Store the content hash in source config."""
        with get_session_context() as session:
            source = session.get(DataSource, source.id)
            config = source.config or {}
            config["symbols_hash"] = content_hash
            source.config = config
            source.updated_at = datetime.utcnow()
            session.commit()
    
    async def update_source_status(
        self, 
        source: DataSource, 
        status: str, 
        message: str | None = None
    ) -> None:
        """Update source status for skipped/error states."""
        with get_session_context() as session:
            source = session.get(DataSource, source.id)
            source.last_sync_status = status
            source.last_sync_error = message
            source.last_sync_at = datetime.utcnow()
            source.updated_at = datetime.utcnow()
            
            # Create sync log for skipped state
            sync_log = SymbolSyncLog(
                source_id=source.id,
                source_name=source.name,
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
        source: DataSource,
        symbols: list[dict],
    ) -> tuple[int, int]:
        """Upsert symbols into cache."""
        added = 0
        updated = 0
        
        for sym_data in symbols:
            symbol = sym_data.get("symbol", "")
            if not symbol:
                continue
            
            # Check if exists
            stmt = select(SymbolCache).where(
                SymbolCache.source_id == source.id,
                SymbolCache.symbol == symbol,
            )
            existing = session.exec(stmt).first()
            
            # Build search text
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
                    source_id=source.id,
                    source_name=source.name,
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
            # Use yfinance to fetch all tickers at once
            tickers = yf.Tickers(" ".join(all_symbols))
            
            for symbol in all_symbols:
                try:
                    ticker = tickers.tickers.get(symbol)
                    if not ticker:
                        continue
                    
                    info = ticker.info
                    if not info or info.get("regularMarketPrice") is None:
                        continue
                    
                    # Determine category
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
    
    async def _fetch_yahoo_symbols(self, source: DataSource) -> FetchResult:
        """Fetch symbols from Yahoo Finance using yfinance library.
        
        Runs blocking yfinance operations in thread pool to avoid blocking event loop.
        """
        # Predefined lists of popular symbols
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
        
        # Flatten all symbols
        all_symbols = []
        symbol_categories = {}
        for category, symbols in symbol_lists.items():
            for sym in symbols:
                all_symbols.append(sym)
                symbol_categories[sym] = category.rstrip("s")
        
        # Run blocking yfinance call in thread pool
        results, errors = await _run_blocking(
            self._fetch_yahoo_symbols_blocking,
            symbol_lists, symbol_categories, all_symbols
        )
        
        # If we got no results and had errors, source is not available
        if not results and errors:
            return FetchResult(
                available=False,
                data=[],
                content_hash="",
                skip_reason=f"Yahoo Finance unavailable: {errors[0]}",
            )
        
        # Compute content hash
        sorted_symbols = sorted(results, key=lambda x: x["symbol"])
        content_hash = hashlib.md5(
            json.dumps(sorted_symbols, sort_keys=True, default=str).encode()
        ).hexdigest()
        
        logger.info(f"Fetched {len(results)} symbols from Yahoo Finance")
        
        return FetchResult(
            available=True,
            data=results,
            content_hash=content_hash,
        )
    
    def _check_ibkr_connection_blocking(self, host: str, port: int, client_id: int, timeout: int) -> tuple[bool, str | None]:
        """Blocking IBKR connection check - runs in thread pool."""
        try:
            from ib_async import IB
            
            ib = IB()
            ib.connect(host, port, clientId=client_id, timeout=timeout)
            
            is_connected = ib.isConnected()
            ib.disconnect()
            
            if is_connected:
                return True, None
            else:
                return False, f"IBKR gateway at {host}:{port} did not respond"
                
        except Exception as e:
            return False, str(e)
    
    async def _fetch_ibkr_symbols(self, source: DataSource) -> FetchResult:
        """Check IBKR gateway connection status.
        
        NOTE: IBKR doesn't have a "list all symbols" API like Yahoo.
        Symbols are discovered on-demand through contract search.
        
        This sync checks if the gateway is connected and returns
        appropriate status (connected/not_connected).
        
        Runs blocking ib_async operations in thread pool.
        """
        config = source.config or {}
        host = config.get("host", "127.0.0.1")
        port = config.get("port", 4001)
        client_id = config.get("client_id", 99)  # Use dedicated client_id for health checks
        timeout = config.get("timeout_s", 5)
        
        try:
            # Run blocking IBKR connection in thread pool
            is_connected, error = await asyncio.wait_for(
                _run_blocking(
                    self._check_ibkr_connection_blocking,
                    host, port, client_id, timeout
                ),
                timeout=timeout + 5
            )
            
            if is_connected:
                logger.info(f"IBKR gateway connected at {host}:{port}")
                return FetchResult(
                    available=True,
                    data=[],  # IBKR symbols are searched on-demand, not bulk loaded
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
                skip_reason=f"IBKR gateway connection timeout ({host}:{port})",
            )
        except Exception as e:
            return FetchResult(
                available=False,
                data=[],
                content_hash="",
                skip_reason=f"IBKR gateway not available: {str(e)}",
            )


# Utility functions for API use

def search_cached_symbols(
    query: str,
    source_name: str | None = None,
    asset_type: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Search symbols from cache."""
    with get_session_context() as session:
        stmt = select(SymbolCache)
        
        if source_name:
            stmt = stmt.where(SymbolCache.source_name == source_name)
        
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
                "source_id": r.source_id,
                "source_name": r.source_name,
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


def get_cache_stats() -> dict[str, Any]:
    """Get cache statistics."""
    with get_session_context() as session:
        stmt = select(DataSource).where(DataSource.is_active == True)
        sources = session.exec(stmt).all()
        
        stats = {
            "total_symbols": 0,
            "sources": [],
        }
        
        for source in sources:
            source_stats = {
                "name": source.name,
                "display_name": source.display_name,
                "symbols_count": source.symbols_count,
                "last_sync_at": source.last_sync_at,
                "last_sync_status": source.last_sync_status,
                "sync_enabled": source.sync_enabled,
            }
            stats["sources"].append(source_stats)
            stats["total_symbols"] += source.symbols_count
        
        return stats
