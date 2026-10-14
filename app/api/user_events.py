"""Outcomes of the user's long-running tasks (migr. 070): the Activity page.

Read-only. Rows are written by the webhook event sources; the push that
tells the FE to refetch rides ``GET /connections/events``.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session

from app.db.database import get_session
from app.models.user import User
from app.models.user_event import UserEvent
from app.schemas.user_event import UserEventDetail, UserEventRead, UserEventTypeSpec
from app.services import user_events
from app.utils.auth_utils import get_current_active_user

router = APIRouter(prefix="/events", tags=["Activity"])


def _read(row: UserEvent) -> UserEventRead:
    return UserEventRead(
        id=row.id,
        type=row.type,
        severity=row.severity,
        entity_type=row.entity_type,
        entity_id=row.entity_id,
        entity_name=row.entity_name,
        message=row.message,
        has_details=bool(row.details),
        data=row.data if isinstance(row.data, dict) else {},
        links=row.links if isinstance(row.links, dict) else {},
        created_at=row.created_at,
    )


@router.get("/types", response_model=list[UserEventTypeSpec])
def list_types(current_user: User = Depends(get_current_active_user)):
    return [UserEventTypeSpec(**e) for e in user_events.catalog()]


@router.get("/", response_model=list[UserEventRead])
def list_events(
    since_id: Optional[int] = Query(default=None, ge=0, description="only events newer than this id"),
    before_id: Optional[int] = Query(default=None, ge=1, description="pagination: events older than this id"),
    severity: Optional[list[str]] = Query(default=None),
    entity_type: Optional[list[str]] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    rows = user_events.list_events(
        session,
        current_user.id,
        since_id=since_id,
        before_id=before_id,
        severity=severity,
        entity_type=entity_type,
        limit=limit,
    )
    return [_read(r) for r in rows]


@router.get("/{event_id}", response_model=UserEventDetail)
def get_event(
    event_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    row = user_events.get_event(session, current_user.id, event_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")
    return UserEventDetail(**_read(row).model_dump(), details=row.details)
