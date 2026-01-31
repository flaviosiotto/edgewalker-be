"""
Market Data Models - Data sources and symbol cache.
"""
from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel, Field, Column, JSON


class DataSource(SQLModel, table=True):
    """Configuration for a market data source."""
    
    __tablename__ = "data_sources"
    
    id: int | None = Field(default=None, primary_key=True)
    
    # Source identification
    name: str = Field(index=True, unique=True, description="Unique source name (e.g., 'yahoo', 'ibkr')")
    display_name: str = Field(description="Human-readable name")
    source_type: str = Field(description="Type: 'yahoo', 'ibkr', 'custom'")
    
    # Connection settings (JSON for flexibility)
    config: dict = Field(default_factory=dict, sa_column=Column(JSON))
    
    # Supported features
    supports_stocks: bool = Field(default=True)
    supports_futures: bool = Field(default=False)
    supports_indices: bool = Field(default=False)
    supports_etfs: bool = Field(default=True)
    supports_realtime: bool = Field(default=False)
    
    # Status
    is_active: bool = Field(default=True)
    is_default: bool = Field(default=False, description="Default source for symbol search")
    
    # Sync settings
    sync_enabled: bool = Field(default=True, description="Enable automatic symbol sync")
    sync_interval_minutes: float = Field(default=1440, description="Sync interval in minutes (default: 24h). Use 0.5 for 30 sec")
    last_sync_at: datetime | None = Field(default=None)
    last_sync_status: str | None = Field(default=None, description="'success', 'error', 'running'")
    last_sync_error: str | None = Field(default=None)
    symbols_count: int = Field(default=0, description="Number of cached symbols")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class SymbolCache(SQLModel, table=True):
    """Cached symbol information from data sources."""
    
    __tablename__ = "symbol_cache"
    
    id: int | None = Field(default=None, primary_key=True)
    
    # Source reference
    source_id: int = Field(foreign_key="data_sources.id", index=True)
    source_name: str = Field(index=True, description="Denormalized for faster queries")
    
    # Symbol identification
    symbol: str = Field(index=True, description="Ticker symbol")
    name: str = Field(default="", description="Company/instrument name")
    
    # Classification
    asset_type: str = Field(index=True, description="stock, futures, index, etf, option, etc.")
    exchange: str = Field(default="", index=True)
    exchange_display: str = Field(default="")
    currency: str = Field(default="USD")
    
    # Additional data (JSON for flexibility)
    extra_data: dict = Field(default_factory=dict, sa_column=Column(JSON))
    
    # Search optimization
    search_text: str = Field(default="", description="Combined searchable text")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        # Composite unique constraint
        pass


class SymbolSyncLog(SQLModel, table=True):
    """Log of symbol sync operations."""
    
    __tablename__ = "symbol_sync_log"
    
    id: int | None = Field(default=None, primary_key=True)
    
    source_id: int = Field(foreign_key="data_sources.id", index=True)
    source_name: str = Field(index=True)
    
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = Field(default=None)
    
    status: str = Field(default="running", description="'running', 'success', 'error'")
    symbols_fetched: int = Field(default=0)
    symbols_added: int = Field(default=0)
    symbols_updated: int = Field(default=0)
    symbols_removed: int = Field(default=0)
    
    error_message: str | None = Field(default=None)
    
    # Duration in seconds
    duration_seconds: float | None = Field(default=None)
