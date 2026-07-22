"""Administrator-only user access management.

Covers the family&friends controls: the approval queue for accounts that
registered without an invite, and the allowlist of pre-authorised emails.
"""

from typing import Annotated, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, status
from sqlmodel import Session

from app.db.database import get_session
from app.schemas.auth import (
    AllowlistCreateRequest,
    AllowlistEntryRead,
    MessageResponse,
    PendingUserRead,
)
from app.models.user import User
from app.services.registration_service import (
    add_allowlist_entry,
    approve_user,
    delete_allowlist_entry,
    list_allowlist,
    list_users_with_status,
    reject_user,
)
from app.utils.auth_utils import get_current_admin_user

router = APIRouter(
    prefix="/admin",
    tags=["Administration"],
    dependencies=[Depends(get_current_admin_user)],
)


@router.get("/users", response_model=list[PendingUserRead])
def list_users_endpoint(
    status_filter: Optional[str] = None,
    session: Session = Depends(get_session),
):
    """List accounts, optionally narrowed to one lifecycle status."""
    return list_users_with_status(session, status_filter)


@router.post("/users/{user_id}/approve", response_model=PendingUserRead)
def approve_user_endpoint(
    user_id: int,
    admin: Annotated[User, Depends(get_current_admin_user)],
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
):
    return approve_user(session, user_id, admin, background_tasks=background_tasks)


@router.post("/users/{user_id}/reject", response_model=PendingUserRead)
def reject_user_endpoint(
    user_id: int,
    admin: Annotated[User, Depends(get_current_admin_user)],
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
    notify: bool = True,
):
    return reject_user(session, user_id, admin, background_tasks=background_tasks, notify=notify)


@router.get("/allowlist", response_model=list[AllowlistEntryRead])
def list_allowlist_endpoint(session: Session = Depends(get_session)):
    return list_allowlist(session)


@router.post(
    "/allowlist",
    response_model=AllowlistEntryRead,
    status_code=status.HTTP_201_CREATED,
)
def add_allowlist_endpoint(
    payload: AllowlistCreateRequest,
    admin: Annotated[User, Depends(get_current_admin_user)],
    session: Session = Depends(get_session),
):
    return add_allowlist_entry(session, payload.email, payload.note, admin)


@router.delete("/allowlist/{entry_id}", response_model=MessageResponse)
def delete_allowlist_endpoint(
    entry_id: int,
    session: Session = Depends(get_session),
):
    delete_allowlist_entry(session, entry_id)
    return {"message": "Voce rimossa dall'allowlist"}
