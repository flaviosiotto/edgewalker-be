from fastapi import APIRouter, Depends
from sqlmodel import Session

from app.db.database import get_session
from app.schemas.user import UserCreate, UserRead
from app.services.user_service import create_user, list_users
from app.utils.auth_utils import get_current_active_user
from app.models.user import User

router = APIRouter(prefix="/users", tags=["Users"])


@router.post("/", response_model=UserRead)
def register_user(payload: UserCreate, session: Session = Depends(get_session)):
    user = create_user(session, payload)
    return user


@router.get("/", response_model=list[UserRead])
def read_users(session: Session = Depends(get_session)):
    return list_users(session)


@router.get("/me", response_model=UserRead)
def read_current_user(current_user: User = Depends(get_current_active_user)):
    return current_user
