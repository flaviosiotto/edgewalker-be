"""
Market Data Models - Symbol cache (backed by Connection).
"""
from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel, Field, Column, JSON
from sqlalchemy import ForeignKey, Integer


class SymbolCache(SQLModel, table=True):
    """Cached symbol information, scoped per Connection."""
    
    __tablename__ = "symbol_cache"
    
    id: int | None = Field(default=None, primary_key=True)
    
    # Connection reference (cascade delete when connection is deleted)
    connection_id: int = Field(
        sa_column=Column(Integer, ForeignKey("connections.id", ondelete="CASCADE"), index=True)
    )
    broker_type: str = Field(index=True, description="Denormalized broker type for faster queries")
    
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


class SymbolSyncLog(SQLModel, table=True):
    """Log of symbol sync operations."""
    
    __tablename__ = "symbol_sync_log"
    
    id: int | None = Field(default=None, primary_key=True)
    
    # Connection reference (cascade delete when connection is deleted)
    connection_id: int = Field(
        sa_column=Column(Integer, ForeignKey("connections.id", ondelete="CASCADE"), index=True)
    )
    connection_name: str = Field(index=True)
    
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
