"""Watchlist API (migr. 079): the symbols a user keeps an eye on, per connection."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlmodel import Session

from app.db.database import get_session
from app.models.user import User
from app.schemas.watchlist import SymbolQuotesResponse, WatchlistItemCreate, WatchlistItemRead
from app.services import symbol_catalog_service as catalog
from app.services import watchlist_service
from app.utils.auth_utils import get_current_active_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/watchlist", tags=["Watchlist"])


def _read(item) -> WatchlistItemRead:
    return WatchlistItemRead.model_validate(item, from_attributes=True)


@router.get("/", response_model=list[WatchlistItemRead])
def list_watchlist_endpoint(
    connection_id: int | None = None,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    return [_read(item) for item in watchlist_service.list_items(session, current_user.id, connection_id)]


@router.post("/", response_model=WatchlistItemRead)
def add_watchlist_item_endpoint(
    payload: WatchlistItemCreate,
    response: Response,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Add a symbol of a connection. Adding one already there returns it (200);
    a new one answers 201."""
    item, created = watchlist_service.add_item(session, payload, current_user.id)
    response.status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
    return _read(item)


@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_watchlist_item_endpoint(
    item_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    watchlist_service.remove_item(session, item_id, current_user.id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/quotes", response_model=SymbolQuotesResponse)
def watchlist_quotes_endpoint(
    connection_id: int = Query(..., description="Connection whose watchlist symbols are quoted"),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Last price and 24 h change of the caller's watchlist symbols on one
    connection. 424 when the gateway does not answer and nothing is cached."""
    connection = watchlist_service.owned_connection(session, connection_id, current_user.id)
    broker_type = connection.broker_type
    symbols = [item.symbol for item in watchlist_service.list_items(session, current_user.id, connection_id)]
    # The gateway call must not hold the request's DB connection.
    session.close()
    try:
        return catalog.get_quotes(connection_id, broker_type, symbols)
    except catalog.CatalogUnavailableError as exc:
        logger.info("Watchlist quotes unavailable on connection %s: %s", connection_id, exc)
        raise HTTPException(status_code=status.HTTP_424_FAILED_DEPENDENCY, detail=str(exc)) from exc
