from sqlmodel import Session, select
from fastapi import HTTPException, status

from app.models.user import User
from app.schemas.user import UserCreate
from app.utils.auth_utils import get_password_hash


def create_user(session: Session, payload: UserCreate) -> User:
    existing = session.exec(
        select(User).where((User.email == payload.email) | (User.username == payload.username))
    ).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email o username già in uso"
        )

    user = User(
        email=payload.email,
        username=payload.username,
        hashed_password=get_password_hash(payload.password),
        is_active=True
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user
