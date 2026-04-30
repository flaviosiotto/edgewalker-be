import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, status
from sqlmodel import Session, select

from app.core.config import settings
from app.models.password_reset_token import PasswordResetToken
from app.models.user import User
from app.utils.auth_utils import (
    get_password_hash,
    get_user_by_username_or_email,
    verify_password,
)

GENERIC_PASSWORD_RESET_MESSAGE = (
    "Se l'account esiste, e' stato generato un token di reset password."
)


def validate_password_strength(password: str) -> None:
    if len(password) < 8:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="La password deve avere almeno 8 caratteri")
    if not any(char.isupper() for char in password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="La password deve contenere almeno una lettera maiuscola")
    if not any(char.islower() for char in password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="La password deve contenere almeno una lettera minuscola")
    if not any(char.isdigit() for char in password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="La password deve contenere almeno un numero")


def _hash_reset_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _invalidate_active_tokens(session: Session, user_id: int, used_at: datetime) -> None:
    statement = select(PasswordResetToken).where(
        PasswordResetToken.user_id == user_id,
        PasswordResetToken.used_at.is_(None),
    )
    for reset_token in session.exec(statement).all():
        reset_token.used_at = used_at
        session.add(reset_token)


def change_password(session: Session, user: User, current_password: str, new_password: str) -> None:
    if not verify_password(current_password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="La password attuale non e' corretta")

    validate_password_strength(new_password)
    if verify_password(new_password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="La nuova password deve essere diversa da quella attuale")

    now = datetime.now(timezone.utc)
    user.hashed_password = get_password_hash(new_password)
    _invalidate_active_tokens(session, user.id, now)
    session.add(user)
    session.commit()


def request_password_reset(session: Session, identifier: str) -> dict[str, object]:
    normalized_identifier = identifier.strip()
    if not normalized_identifier:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email o username obbligatori")

    user = get_user_by_username_or_email(normalized_identifier, session)
    if user is None or not user.is_active:
        return {"message": GENERIC_PASSWORD_RESET_MESSAGE, "reset_token": None, "expires_at": None}

    now = datetime.now(timezone.utc)
    _invalidate_active_tokens(session, user.id, now)

    raw_token = secrets.token_urlsafe(32)
    expires_at = now + timedelta(minutes=settings.PASSWORD_RESET_TOKEN_EXPIRE_MINUTES)
    reset_token = PasswordResetToken(
        user_id=user.id,
        token_hash=_hash_reset_token(raw_token),
        expires_at=expires_at,
    )
    session.add(reset_token)
    session.commit()

    return {
        "message": GENERIC_PASSWORD_RESET_MESSAGE,
        "reset_token": raw_token if settings.PASSWORD_RESET_DEBUG_RETURN_TOKEN else None,
        "expires_at": expires_at if settings.PASSWORD_RESET_DEBUG_RETURN_TOKEN else None,
    }


def confirm_password_reset(session: Session, token: str, new_password: str) -> None:
    normalized_token = token.strip()
    if not normalized_token:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Token di reset obbligatorio")

    validate_password_strength(new_password)

    now = datetime.now(timezone.utc)
    token_hash = _hash_reset_token(normalized_token)
    statement = select(PasswordResetToken).where(
        PasswordResetToken.token_hash == token_hash,
        PasswordResetToken.used_at.is_(None),
        PasswordResetToken.expires_at > now,
    )
    reset_token = session.exec(statement).first()
    if reset_token is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Il token di reset non e' valido o e' scaduto",
        )

    user = session.get(User, reset_token.user_id)
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Utente non disponibile per il reset password")
    if verify_password(new_password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="La nuova password deve essere diversa da quella attuale")

    user.hashed_password = get_password_hash(new_password)
    _invalidate_active_tokens(session, user.id, now)
    reset_token.used_at = now
    session.add(user)
    session.add(reset_token)
    session.commit()