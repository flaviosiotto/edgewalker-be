"""
Sync Manager - Generic background synchronization orchestrator.

This module provides a lightweight, extensible framework for background
synchronization tasks. It runs per-datasource and can handle multiple
sync types (symbols, contracts, etc.) in the future.

Key features:
- Automatic startup sync (initial load or incremental)
- Configurable polling interval per datasource
- Incremental change detection via content hashing
- Non-blocking async execution
- Graceful shutdown
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from enum import Enum
from functools import partial
from typing import Any, Callable

from sqlmodel import Session, select

from app.core.config import settings
from app.db.database import get_session_context
from app.models.marketdata import DataSource

logger = logging.getLogger(__name__)


class SyncType(str, Enum):
    """Types of sync operations."""
    SYMBOLS = "symbols"
    # Future: CONTRACTS, INSTRUMENTS, etc.


class SyncStatus(str, Enum):
    """Status of a sync operation."""
    SUCCESS = "success"
    CONNECTED = "connected"      # Source connected (for connection-only checks like IBKR)
    NOT_CONNECTED = "not_connected"  # Source not available/connected
    SKIPPED = "skipped"          # Sync skipped for other reasons
    NO_CHANGES = "no_changes"    # Hash unchanged, no sync needed
    ERROR = "error"


class SyncResult:
    """Result of a sync operation."""
    
    def __init__(
        self,
        status: SyncStatus,
        sync_type: SyncType,
        source_name: str,
        items_fetched: int = 0,
        items_added: int = 0,
        items_updated: int = 0,
        items_removed: int = 0,
        is_incremental: bool = False,
        content_hash: str | None = None,
        message: str | None = None,
        error: str | None = None,
        duration_seconds: float = 0,
    ):
        self.status = status
        self.success = status in (SyncStatus.SUCCESS, SyncStatus.NO_CHANGES, SyncStatus.CONNECTED)
        self.sync_type = sync_type
        self.source_name = source_name
        self.items_fetched = items_fetched
        self.items_added = items_added
        self.items_updated = items_updated
        self.items_removed = items_removed
        self.is_incremental = is_incremental
        self.content_hash = content_hash
        self.message = message
        self.error = error
        self.duration_seconds = duration_seconds
    
    def __repr__(self) -> str:
        if self.status == SyncStatus.SUCCESS:
            return (
                f"SyncResult({self.sync_type.value}:{self.source_name} "
                f"+{self.items_added} ~{self.items_updated} -{self.items_removed} "
                f"in {self.duration_seconds:.1f}s)"
            )
        elif self.status == SyncStatus.CONNECTED:
            return f"SyncResult({self.sync_type.value}:{self.source_name} CONNECTED)"
        elif self.status == SyncStatus.NOT_CONNECTED:
            return f"SyncResult({self.sync_type.value}:{self.source_name} NOT_CONNECTED: {self.message})"
        elif self.status == SyncStatus.SKIPPED:
            return f"SyncResult({self.sync_type.value}:{self.source_name} SKIPPED: {self.message})"
        elif self.status == SyncStatus.NO_CHANGES:
            return f"SyncResult({self.sync_type.value}:{self.source_name} NO_CHANGES)"
        return f"SyncResult({self.sync_type.value}:{self.source_name} ERROR: {self.error})"


class FetchResult:
    """Result of fetching data from a source."""
    
    def __init__(
        self,
        available: bool,
        data: list[dict] | None = None,
        content_hash: str | None = None,
        skip_reason: str | None = None,
    ):
        self.available = available
        self.data = data or []
        self.content_hash = content_hash or ""
        self.skip_reason = skip_reason


class BaseSyncHandler(ABC):
    """Base class for sync handlers.
    
    Implement this for each sync type (symbols, contracts, etc.)
    """
    
    @property
    @abstractmethod
    def sync_type(self) -> SyncType:
        """The type of sync this handler performs."""
        pass
    
    @abstractmethod
    async def fetch_current_data(self, source: DataSource) -> FetchResult:
        """Fetch current data from the source.
        
        Returns:
            FetchResult with:
            - available: True if source is reachable
            - data: List of items fetched
            - content_hash: Hash for change detection
            - skip_reason: Reason if not available
        """
        pass
    
    @abstractmethod
    async def get_cached_hash(self, source: DataSource) -> str | None:
        """Get the hash of currently cached data."""
        pass
    
    @abstractmethod
    async def apply_changes(
        self,
        source: DataSource,
        new_data: list[dict],
        is_full_refresh: bool,
    ) -> tuple[int, int, int]:
        """Apply changes to the cache.
        
        Returns:
            Tuple of (added, updated, removed)
        """
        pass
    
    @abstractmethod
    async def update_source_hash(self, source: DataSource, content_hash: str) -> None:
        """Update the stored hash for the source."""
        pass
    
    @abstractmethod
    async def update_source_status(
        self, 
        source: DataSource, 
        status: str, 
        message: str | None = None
    ) -> None:
        """Update the source sync status (for skipped/error states)."""
        pass


async def run_in_executor(executor: ThreadPoolExecutor, func: Callable, *args, **kwargs) -> Any:
    """Run a blocking function in a thread pool executor.
    
    This allows blocking I/O operations (like yfinance) to run
    without blocking the async event loop.
    """
    loop = asyncio.get_event_loop()
    if kwargs:
        func = partial(func, **kwargs)
    return await loop.run_in_executor(executor, func, *args)


class SyncManager:
    """Manages background sync tasks for all data sources.
    
    Features:
    - Runs startup sync for all enabled sources
    - Polls sources based on their individual intervals
    - Detects changes incrementally via content hashing
    - Lightweight and non-blocking
    """
    
    def __init__(self):
        self._handlers: dict[SyncType, BaseSyncHandler] = {}
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._sync_in_progress: dict[tuple[int, SyncType], bool] = {}
        # Thread pool for blocking operations (yfinance, etc.)
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sync_")
    
    def register_handler(self, handler: BaseSyncHandler) -> None:
        """Register a sync handler for a sync type."""
        self._handlers[handler.sync_type] = handler
        logger.info(f"Registered sync handler for {handler.sync_type.value}")
    
    def is_sync_running(self, source_id: int, sync_type: SyncType) -> bool:
        """Check if sync is currently running for a source/type."""
        return self._sync_in_progress.get((source_id, sync_type), False)
    
    async def sync_source(
        self,
        source: DataSource,
        sync_type: SyncType,
        force_full: bool = False,
    ) -> SyncResult:
        """Sync a single source.
        
        Args:
            source: The data source to sync
            sync_type: Type of sync to perform
            force_full: Force full refresh instead of incremental
        
        Returns:
            SyncResult with details of the operation
        """
        handler = self._handlers.get(sync_type)
        if not handler:
            return SyncResult(
                status=SyncStatus.ERROR,
                sync_type=sync_type,
                source_name=source.name,
                error=f"No handler registered for {sync_type.value}",
            )
        
        key = (source.id, sync_type)
        if self._sync_in_progress.get(key, False):
            return SyncResult(
                status=SyncStatus.SKIPPED,
                sync_type=sync_type,
                source_name=source.name,
                message="Sync already in progress",
            )
        
        self._sync_in_progress[key] = True
        start_time = datetime.utcnow()
        
        try:
            # Fetch current data
            fetch_result = await handler.fetch_current_data(source)
            
            # Check if source is available
            if not fetch_result.available:
                logger.info(
                    f"Sync: {source.name}/{sync_type.value} not connected: "
                    f"{fetch_result.skip_reason}"
                )
                # Update source status to reflect not connected state
                await handler.update_source_status(
                    source, 
                    status="not_connected", 
                    message=fetch_result.skip_reason
                )
                return SyncResult(
                    status=SyncStatus.NOT_CONNECTED,
                    sync_type=sync_type,
                    source_name=source.name,
                    message=fetch_result.skip_reason,
                    duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
                )
            
            new_data = fetch_result.data
            new_hash = fetch_result.content_hash
            
            # If no data but available (connection check only, like IBKR)
            if not new_data and fetch_result.available:
                duration = (datetime.utcnow() - start_time).total_seconds()
                logger.info(f"Sync: {source.name}/{sync_type.value} connected (no bulk data)")
                await handler.update_source_status(source, status="connected", message=None)
                return SyncResult(
                    status=SyncStatus.CONNECTED,
                    sync_type=sync_type,
                    source_name=source.name,
                    duration_seconds=duration,
                )
            
            # Check if we need to sync (hash comparison)
            cached_hash = await handler.get_cached_hash(source)
            
            if not force_full and cached_hash and cached_hash == new_hash:
                # No changes detected
                logger.debug(f"No changes detected for {source.name}/{sync_type.value}")
                return SyncResult(
                    status=SyncStatus.NO_CHANGES,
                    sync_type=sync_type,
                    source_name=source.name,
                    items_fetched=len(new_data),
                    is_incremental=True,
                    content_hash=new_hash,
                    duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
                )
            
            is_incremental = cached_hash is not None and not force_full
            
            # Apply changes
            added, updated, removed = await handler.apply_changes(
                source,
                new_data,
                is_full_refresh=force_full or cached_hash is None,
            )
            
            # Update hash
            await handler.update_source_hash(source, new_hash)
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(
                f"Sync completed for {source.name}/{sync_type.value}: "
                f"+{added} ~{updated} -{removed} in {duration:.1f}s "
                f"({'incremental' if is_incremental else 'full'})"
            )
            
            return SyncResult(
                status=SyncStatus.SUCCESS,
                sync_type=sync_type,
                source_name=source.name,
                items_fetched=len(new_data),
                items_added=added,
                items_updated=updated,
                items_removed=removed,
                is_incremental=is_incremental,
                content_hash=new_hash,
                duration_seconds=duration,
            )
            
        except Exception as e:
            logger.exception(f"Sync failed for {source.name}/{sync_type.value}")
            # Update source status but don't raise
            try:
                await handler.update_source_status(source, status="error", message=str(e))
            except Exception:
                pass  # Don't let status update failure mask original error
            return SyncResult(
                status=SyncStatus.ERROR,
                sync_type=sync_type,
                source_name=source.name,
                error=str(e),
                duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
            )
        finally:
            self._sync_in_progress[key] = False
    
    async def _startup_sync(self) -> None:
        """Run initial sync for all enabled sources."""
        logger.info("Running startup sync for all enabled sources...")
        
        with get_session_context() as session:
            stmt = select(DataSource).where(
                DataSource.is_active == True,
                DataSource.sync_enabled == True,
            )
            sources = list(session.exec(stmt).all())
        
        if not sources:
            logger.info("No active sources with sync enabled")
            return
        
        # Run syncs concurrently but with a limit
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent syncs
        
        async def sync_with_limit(source: DataSource) -> None:
            async with semaphore:
                for sync_type in self._handlers.keys():
                    await self.sync_source(source, sync_type)
        
        tasks = [sync_with_limit(s) for s in sources]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"Startup sync completed for {len(sources)} sources")
    
    async def _polling_loop(self) -> None:
        """Main polling loop - checks sources and triggers syncs as needed."""
        poll_interval = getattr(settings, 'SYNC_POLL_INTERVAL_SECONDS', 60)
        
        while self._running:
            try:
                with get_session_context() as session:
                    stmt = select(DataSource).where(
                        DataSource.is_active == True,
                        DataSource.sync_enabled == True,
                    )
                    sources = list(session.exec(stmt).all())
                
                now = datetime.utcnow()
                
                for source in sources:
                    # Check if sync is needed based on interval
                    if source.last_sync_at is None:
                        needs_sync = True
                    else:
                        next_sync = source.last_sync_at + timedelta(
                            minutes=source.sync_interval_minutes
                        )
                        needs_sync = now >= next_sync
                    
                    if needs_sync:
                        for sync_type in self._handlers.keys():
                            if not self.is_sync_running(source.id, sync_type):
                                # Fire and forget - don't wait for completion
                                asyncio.create_task(
                                    self.sync_source(source, sync_type)
                                )
                
            except Exception as e:
                logger.exception("Error in sync polling loop")
            
            await asyncio.sleep(poll_interval)
    
    async def start(self) -> None:
        """Start the sync manager."""
        if self._running:
            logger.warning("Sync manager already running")
            return
        
        self._running = True
        logger.info("Starting sync manager...")
        
        # Start startup sync in background (don't wait - API should be immediately available)
        startup_task = asyncio.create_task(self._startup_sync())
        self._tasks.append(startup_task)
        
        # Start polling loop
        poll_task = asyncio.create_task(self._polling_loop())
        self._tasks.append(poll_task)
        
        logger.info("Sync manager started (startup sync running in background)")
    
    async def stop(self) -> None:
        """Stop the sync manager gracefully."""
        if not self._running:
            return
        
        logger.info("Stopping sync manager...")
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._tasks.clear()
        
        # Shutdown thread pool
        self._executor.shutdown(wait=False)
        
        logger.info("Sync manager stopped")


# Global instance
_sync_manager: SyncManager | None = None


def get_sync_manager() -> SyncManager:
    """Get the global sync manager instance."""
    global _sync_manager
    if _sync_manager is None:
        _sync_manager = SyncManager()
    return _sync_manager


async def start_sync_manager() -> None:
    """Initialize and start the sync manager with all handlers."""
    from app.services.symbol_sync_handler import SymbolSyncHandler
    
    manager = get_sync_manager()
    
    # Register handlers
    manager.register_handler(SymbolSyncHandler())
    
    # Start
    await manager.start()


async def stop_sync_manager() -> None:
    """Stop the sync manager."""
    manager = get_sync_manager()
    await manager.stop()
