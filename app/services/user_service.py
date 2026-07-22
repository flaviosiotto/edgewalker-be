import logging

from sqlmodel import Session, select
from fastapi import HTTPException, status

from app.core.config import settings
from app.models.user import User
from app.schemas.user import UserCreate
from app.utils.auth_utils import get_password_hash

logger = logging.getLogger(__name__)

ALLOWED_ROLES = {"user", "admin"}


def create_user(session: Session, payload: UserCreate) -> User:
    existing = session.exec(
        select(User).where((User.email == payload.email) | (User.username == payload.username))
    ).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email o username già in uso"
        )

    role = payload.role or "user"
    if role not in ALLOWED_ROLES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Ruolo non valido: {role}",
        )

    user = User(
        email=payload.email,
        username=payload.username,
        hashed_password=get_password_hash(payload.password),
        is_active=True,
        role=role,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def ensure_bootstrap_admin(session: Session) -> None:
    """Seed the first administrator on an empty installation.

    This replaces the previous rule where whoever reached the unauthenticated
    ``POST /users/`` first was granted the admin role — a privilege escalation
    for anyone who could reach the API before the owner did. Seeding is driven
    by configuration and only ever runs while the user table is empty.
    """
    if session.exec(select(User.id)).first() is not None:
        return

    if not settings.BOOTSTRAP_ADMIN_EMAIL or not settings.BOOTSTRAP_ADMIN_PASSWORD:
        logger.warning(
            "No users exist and BOOTSTRAP_ADMIN_EMAIL/BOOTSTRAP_ADMIN_PASSWORD are unset: "
            "nobody can log in. Set both and restart to seed the first administrator."
        )
        return

    admin = User(
        email=settings.BOOTSTRAP_ADMIN_EMAIL,
        username=settings.BOOTSTRAP_ADMIN_USERNAME,
        hashed_password=get_password_hash(settings.BOOTSTRAP_ADMIN_PASSWORD),
        is_active=True,
        role="admin",
    )
    session.add(admin)
    session.commit()
    logger.info("Seeded bootstrap administrator %s", settings.BOOTSTRAP_ADMIN_EMAIL)


def list_users(session: Session) -> list[User]:
    statement = select(User)
    return list(session.exec(statement).all())
