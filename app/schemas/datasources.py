"""
Schemas for Data Sources API.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Data Source Schemas
# =============================================================================

class DataSourceConfig(BaseModel):
    """Configuration settings for a data source."""
    
    # Yahoo-specific
    timeout_s: int = Field(default=30, description="Request timeout in seconds")
    
    # IBKR-specific
    host: str | None = Field(default=None, description="IBKR gateway host")
    port: int | None = Field(default=None, description="IBKR gateway port")
    client_id: int | None = Field(default=None, description="IBKR client ID")
    
    # Generic
    api_key: str | None = Field(default=None, description="API key if required")
    rate_limit: int | None = Field(default=None, description="Max requests per minute")
    
    class Config:
        extra = "allow"  # Allow additional fields


class DataSourceCreate(BaseModel):
    """Schema for creating a new data source."""
    
    name: str = Field(..., min_length=1, max_length=50, description="Unique identifier")
    display_name: str = Field(..., min_length=1, max_length=100)
    source_type: str = Field(..., description="Type: 'yahoo', 'ibkr', 'custom'")
    
    config: dict[str, Any] = Field(default_factory=dict)
    
    supports_stocks: bool = True
    supports_futures: bool = False
    supports_indices: bool = False
    supports_etfs: bool = True
    supports_realtime: bool = False
    
    is_active: bool = True
    is_default: bool = False
    
    sync_enabled: bool = True
    sync_interval_minutes: float = Field(default=1440, ge=0.5, le=10080)  # 30 sec to 1 week


class DataSourceUpdate(BaseModel):
    """Schema for updating a data source."""
    
    display_name: str | None = None
    config: dict[str, Any] | None = None
    
    supports_stocks: bool | None = None
    supports_futures: bool | None = None
    supports_indices: bool | None = None
    supports_etfs: bool | None = None
    supports_realtime: bool | None = None
    
    is_active: bool | None = None
    is_default: bool | None = None
    
    sync_enabled: bool | None = None
    sync_interval_minutes: float | None = Field(default=None, ge=0.5, le=10080)


class DataSourceResponse(BaseModel):
    """Response schema for a data source."""
    
    id: int
    name: str
    display_name: str
    source_type: str
    
    config: dict[str, Any]
    
    supports_stocks: bool
    supports_futures: bool
    supports_indices: bool
    supports_etfs: bool
    supports_realtime: bool
    
    is_active: bool
    is_default: bool
    
    sync_enabled: bool
    sync_interval_minutes: float
    last_sync_at: datetime | None
    last_sync_status: str | None
    last_sync_error: str | None
    symbols_count: int
    
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class DataSourceListResponse(BaseModel):
    """Response for listing data sources."""
    
    sources: list[DataSourceResponse]
    count: int


# =============================================================================
# Symbol Cache Schemas
# =============================================================================

class CachedSymbolResponse(BaseModel):
    """Response schema for a cached symbol."""
    
    id: int
    source_id: int
    source_name: str
    
    symbol: str
    name: str
    asset_type: str
    exchange: str
    exchange_display: str
    currency: str
    
    extra_data: dict[str, Any]
    
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class SymbolSearchResponse(BaseModel):
    """Response for symbol search from cache."""
    
    symbols: list[CachedSymbolResponse]
    source: str | None = Field(None, description="Source filter used, or 'all'")
    asset_type: str | None = Field(None, description="Asset type filter used, or 'all'")
    query: str
    count: int
    from_cache: bool = True


# =============================================================================
# Sync Schemas
# =============================================================================

class SyncTriggerRequest(BaseModel):
    """Request to trigger a manual sync."""
    
    full_refresh: bool = Field(
        default=False,
        description="If true, clear cache and re-fetch all symbols"
    )


class SyncStatusResponse(BaseModel):
    """Response with sync status."""
    
    source_id: int
    source_name: str
    status: str
    last_sync_at: datetime | None
    symbols_count: int
    is_running: bool
    
    # Last sync details
    last_sync_duration_seconds: float | None = None
    last_sync_symbols_added: int | None = None
    last_sync_symbols_updated: int | None = None
    last_sync_error: str | None = None


class SyncLogResponse(BaseModel):
    """Response for a sync log entry."""
    
    id: int
    source_id: int
    source_name: str
    
    started_at: datetime
    completed_at: datetime | None
    
    status: str
    symbols_fetched: int
    symbols_added: int
    symbols_updated: int
    symbols_removed: int
    
    error_message: str | None
    duration_seconds: float | None
    
    class Config:
        from_attributes = True
