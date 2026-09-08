from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlmodel import Session

from app.db.database import get_session
from app.models.user import User
from app.services.onboarding_service import activate_onboarding, prepare_onboarding, update_onboarding
from app.utils.auth_utils import get_current_active_user

router = APIRouter(prefix="/onboarding", tags=["Onboarding"])


class GuideProgress(BaseModel):
    dismissed: bool = False
    step: int = Field(default=0, ge=0, le=4)
    track: Literal["forex", "bitcoin", "welcome"] = "forex"


class ActivateOnboarding(BaseModel):
    account_id: int = Field(gt=0)
    symbol: str = Field(min_length=1, max_length=100)


@router.post("/prepare")
def prepare(session: Session = Depends(get_session), user: User = Depends(get_current_active_user)):
    return prepare_onboarding(session, user.id)


@router.patch("")
def progress(payload: GuideProgress, session: Session = Depends(get_session), user: User = Depends(get_current_active_user)):
    return update_onboarding(session, user.id, dismissed=payload.dismissed, step=payload.step, track=payload.track)


@router.post("/activate")
def activate(payload: ActivateOnboarding, session: Session = Depends(get_session), user: User = Depends(get_current_active_user)):
    return activate_onboarding(session, user.id, payload.account_id, payload.symbol)