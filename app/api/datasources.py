"""
Data Sources API - Manage market data sources.

Provides CRUD operations for data sources.
Symbol cache sync is handled automatically by the background sync manager.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Depends, status
from sqlmodel import Session, select

from app.db.database import get_session
from app.models.marketdata import DataSource, SymbolCache, SymbolSyncLog
from app.schemas.datasources import (
    DataSourceCreate,
    DataSourceUpdate,
    DataSourceResponse,
    DataSourceListResponse,
    SyncLogResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasources", tags=["Data Sources"])


# =============================================================================
# Data Source CRUD
# =============================================================================

@router.get("", response_model=DataSourceListResponse)
def list_data_sources(
    active_only: bool = Query(False, description="Only return active sources"),
    session: Session = Depends(get_session),
):
    """
    List all configured data sources.
    
    Each source has automatic background sync that keeps symbols up-to-date.
    """
    stmt = select(DataSource)
    if active_only:
        stmt = stmt.where(DataSource.is_active == True)
    stmt = stmt.order_by(DataSource.is_default.desc(), DataSource.name)
    
    sources = session.exec(stmt).all()
    
    return DataSourceListResponse(
        sources=[DataSourceResponse.model_validate(s) for s in sources],
        count=len(sources),
    )


@router.get("/{source_id}", response_model=DataSourceResponse)
def get_data_source(
    source_id: int,
    session: Session = Depends(get_session),
):
    """
    Get a specific data source by ID.
    """
    source = session.get(DataSource, source_id)
    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data source {source_id} not found",
        )
    return DataSourceResponse.model_validate(source)


@router.post("", response_model=DataSourceResponse, status_code=status.HTTP_201_CREATED)
def create_data_source(
    data: DataSourceCreate,
    session: Session = Depends(get_session),
):
    """
    Create a new data source.
    
    The background sync manager will automatically start syncing symbols
    for this source based on the configured interval.
    
    **Source Types:**
    - `yahoo`: Yahoo Finance (free, limited history for intraday)
    - `ibkr`: Interactive Brokers (requires TWS/Gateway)
    - `custom`: Custom data source
    
    **Configuration:**
    The `config` field accepts source-specific settings:
    
    For Yahoo:
    ```json
    {"timeout_s": 30}
    ```
    
    For IBKR:
    ```json
    {"host": "127.0.0.1", "port": 4001, "client_id": 101}
    ```
    """
    # Check for duplicate name
    stmt = select(DataSource).where(DataSource.name == data.name)
    if session.exec(stmt).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Data source with name '{data.name}' already exists",
        )
    
    # If this is default, unset other defaults
    if data.is_default:
        stmt = select(DataSource).where(DataSource.is_default == True)
        for other in session.exec(stmt).all():
            other.is_default = False
    
    source = DataSource(**data.model_dump())
    session.add(source)
    session.commit()
    session.refresh(source)
    
    logger.info(f"Created data source: {source.name}")
    
    return DataSourceResponse.model_validate(source)


@router.patch("/{source_id}", response_model=DataSourceResponse)
def update_data_source(
    source_id: int,
    data: DataSourceUpdate,
    session: Session = Depends(get_session),
):
    """
    Update a data source.
    
    Changes to `sync_enabled` or `sync_interval_minutes` will be picked up
    automatically by the background sync manager.
    """
    source = session.get(DataSource, source_id)
    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data source {source_id} not found",
        )
    
    update_data = data.model_dump(exclude_unset=True)
    
    # Handle is_default
    if update_data.get("is_default"):
        stmt = select(DataSource).where(
            DataSource.is_default == True,
            DataSource.id != source_id,
        )
        for other in session.exec(stmt).all():
            other.is_default = False
    
    for key, value in update_data.items():
        setattr(source, key, value)
    
    source.updated_at = datetime.utcnow()
    session.commit()
    session.refresh(source)
    
    logger.info(f"Updated data source: {source.name}")
    
    return DataSourceResponse.model_validate(source)


@router.delete("/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_data_source(
    source_id: int,
    session: Session = Depends(get_session),
):
    """
    Delete a data source and its cached symbols.
    """
    source = session.get(DataSource, source_id)
    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data source {source_id} not found",
        )
    
    source_name = source.name
    
    # Delete cached symbols
    stmt = select(SymbolCache).where(SymbolCache.source_id == source_id)
    for symbol in session.exec(stmt).all():
        session.delete(symbol)
    
    # Delete sync logs
    stmt = select(SymbolSyncLog).where(SymbolSyncLog.source_id == source_id)
    for log in session.exec(stmt).all():
        session.delete(log)
    
    session.delete(source)
    session.commit()
    
    logger.info(f"Deleted data source: {source_name}")


# =============================================================================
# Sync Status (read-only)
# =============================================================================

@router.get("/{source_id}/sync/logs", response_model=list[SyncLogResponse])
def get_sync_logs(
    source_id: int,
    limit: int = Query(20, ge=1, le=100),
    session: Session = Depends(get_session),
):
    """
    Get sync log history for a data source.
    
    Sync runs automatically in the background - this endpoint shows the history.
    """
    source = session.get(DataSource, source_id)
    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data source {source_id} not found",
        )
    
    stmt = (
        select(SymbolSyncLog)
        .where(SymbolSyncLog.source_id == source_id)
        .order_by(SymbolSyncLog.started_at.desc())
        .limit(limit)
    )
    logs = session.exec(stmt).all()
    
    return [SyncLogResponse.model_validate(log) for log in logs]


# =============================================================================
# Seed Default Sources
# =============================================================================

@router.post("/seed-defaults", response_model=DataSourceListResponse)
def seed_default_sources(
    session: Session = Depends(get_session),
):
    """
    Seed the database with default data sources (Yahoo, IBKR).
    
    This is idempotent - existing sources won't be modified.
    The background sync manager will automatically start syncing
    symbols for newly created sources.
    """
    defaults = [
        {
            "name": "yahoo",
            "display_name": "Yahoo Finance",
            "source_type": "yahoo",
            "config": {"timeout_s": 30},
            "supports_stocks": True,
            "supports_futures": False,
            "supports_indices": True,
            "supports_etfs": True,
            "supports_realtime": False,
            "is_default": True,
            "sync_enabled": True,
            "sync_interval_minutes": 1440,  # Daily
        },
        {
            "name": "ibkr",
            "display_name": "Interactive Brokers",
            "source_type": "ibkr",
            "config": {"host": "127.0.0.1", "port": 4001, "client_id": 101},
            "supports_stocks": True,
            "supports_futures": True,
            "supports_indices": True,
            "supports_etfs": True,
            "supports_realtime": True,
            "is_default": False,
            "sync_enabled": False,  # Requires active connection
            "sync_interval_minutes": 10080,  # Weekly
        },
    ]
    
    created = []
    for default in defaults:
        stmt = select(DataSource).where(DataSource.name == default["name"])
        if not session.exec(stmt).first():
            source = DataSource(**default)
            session.add(source)
            created.append(source)
    
    session.commit()
    
    stmt = select(DataSource).order_by(DataSource.is_default.desc(), DataSource.name)
    sources = session.exec(stmt).all()
    
    logger.info(f"Seeded {len(created)} default data sources")
    
    return DataSourceListResponse(
        sources=[DataSourceResponse.model_validate(s) for s in sources],
        count=len(sources),
    )
