"""Watchlist items of a user (migr. 079)."""
from __future__ import annotations

from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from app.models.connection import Connection
from app.models.watchlist import WatchlistItem
from app.schemas.watchlist import WatchlistItemCreate

MAX_ITEMS_PER_USER = 500


def owned_connection(session: Session, connection_id: int, user_id: int) -> Connection:
    """The connection, or 404 when it is not the caller's."""
    connection = session.exec(
        select(Connection).where(Connection.id == connection_id, Connection.user_id == user_id)
    ).first()
    if connection is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found")
    return connection


def list_items(session: Session, user_id: int, connection_id: int | None = None) -> list[WatchlistItem]:
    stmt = select(WatchlistItem).where(WatchlistItem.user_id == user_id)
    if connection_id is not None:
        stmt = stmt.where(WatchlistItem.connection_id == connection_id)
    return list(session.exec(stmt.order_by(WatchlistItem.created_at, WatchlistItem.id)).all())


def _find(session: Session, user_id: int, connection_id: int, symbol: str) -> WatchlistItem | None:
    return session.exec(
        select(WatchlistItem).where(
            WatchlistItem.user_id == user_id,
            WatchlistItem.connection_id == connection_id,
            WatchlistItem.symbol == symbol,
        )
    ).first()


def add_item(session: Session, payload: WatchlistItemCreate, user_id: int) -> tuple[WatchlistItem, bool]:
    """Add a symbol; adding one already there returns it. ``(item, created)``."""
    owned_connection(session, payload.connection_id, user_id)
    existing = _find(session, user_id, payload.connection_id, payload.symbol)
    if existing is not None:
        return existing, False
    count = session.exec(
        select(func.count()).select_from(WatchlistItem).where(WatchlistItem.user_id == user_id)
    ).one()
    if count >= MAX_ITEMS_PER_USER:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"The watchlist holds at most {MAX_ITEMS_PER_USER} symbols",
        )
    item = WatchlistItem(
        user_id=user_id,
        connection_id=payload.connection_id,
        symbol=payload.symbol,
        name=payload.name,
        asset_type=(payload.asset_type or None),
        exchange=payload.exchange,
        currency=payload.currency,
        info=payload.info or {},
    )
    session.add(item)
    try:
        session.commit()
    except IntegrityError:
        # Two adds of the same symbol at once: the other one won.
        session.rollback()
        existing = _find(session, user_id, payload.connection_id, payload.symbol)
        if existing is None:
            raise
        return existing, False
    session.refresh(item)
    return item, True


def remove_item(session: Session, item_id: int, user_id: int) -> None:
    item = session.exec(
        select(WatchlistItem).where(WatchlistItem.id == item_id, WatchlistItem.user_id == user_id)
    ).first()
    if item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Watchlist item not found")
    session.delete(item)
    session.commit()
