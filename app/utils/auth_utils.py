from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlmodel import Session, select

from app.core.config import settings
from app.models.user import User
from app.db.database import get_session

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_ROOT_PATH}/auth/token")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def _encode_token(payload: dict[str, Any]) -> str:
    return jwt.encode(payload, settings.jwt_signing_key, algorithm=settings.ALGORITHM)


def _build_token_payload(
    data: dict[str, Any],
    *,
    token_type: str,
    audience: str,
    expires_delta: timedelta,
    purpose: Optional[str] = None,
) -> dict[str, Any]:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode.update(
        {
            "aud": audience,
            "exp": expire,
            "iss": settings.JWT_ISSUER,
            "type": token_type,
        }
    )
    if purpose:
        to_encode["purpose"] = purpose
    return to_encode


def create_access_token(
    data: dict[str, Any],
    expires_delta: Optional[timedelta] = None,
    audience: Optional[str] = None,
    purpose: str = "ui_auth",
) -> str:
    payload = _build_token_payload(
        data,
        token_type="access",
        audience=audience or settings.ACCESS_TOKEN_AUDIENCE,
        expires_delta=expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
        purpose=purpose,
    )
    return _encode_token(payload)


def create_refresh_token(
    data: dict[str, Any],
    expires_delta: Optional[timedelta] = None,
    audience: Optional[str] = None,
) -> str:
    payload = _build_token_payload(
        data,
        token_type="refresh",
        audience=audience or settings.REFRESH_TOKEN_AUDIENCE,
        expires_delta=expires_delta or timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
        purpose="refresh_auth",
    )
    return _encode_token(payload)


def create_delegated_token(
    data: dict[str, Any],
    *,
    audience: str,
    purpose: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    payload = _build_token_payload(
        data,
        token_type="delegated",
        audience=audience,
        expires_delta=expires_delta or timedelta(minutes=settings.DELEGATED_TOKEN_EXPIRE_MINUTES),
        purpose=purpose,
    )
    return _encode_token(payload)


def decode_token(token: str, audience: Optional[str] = None) -> Optional[dict[str, Any]]:
    try:
        decode_kwargs: dict[str, Any] = {
            "algorithms": [settings.ALGORITHM],
            "issuer": settings.JWT_ISSUER,
            "options": {"verify_aud": audience is not None},
        }
        if audience is not None:
            decode_kwargs["audience"] = audience
        return jwt.decode(token, settings.jwt_verifying_key, **decode_kwargs)
    except JWTError:
        return None


def decode_token_for_audiences(token: str, audiences: list[str]) -> Optional[dict[str, Any]]:
    for audience in audiences:
        payload = decode_token(token, audience=audience)
        if payload is not None:
            return payload
    return None


def get_user_by_email(email: str, session: Session) -> Optional[User]:
    statement = select(User).where(User.email == email)
    return session.exec(statement).first()


def get_user_by_username_or_email(username_or_email: str, session: Session) -> Optional[User]:
    statement = select(User).where(
        (User.username == username_or_email) | (User.email == username_or_email)
    )
    return session.exec(statement).first()


def authenticate_user(username_or_email: str, password: str, session: Session) -> Optional[User]:
    user = get_user_by_username_or_email(username_or_email, session)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    if not user.is_active:
        return None
    return user


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_session)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = decode_token_for_audiences(
        token,
        [settings.ACCESS_TOKEN_AUDIENCE, settings.RUNNER_TOKEN_AUDIENCE],
    )
    if payload is None:
        raise credentials_exception

    email: str | None = payload.get("sub")
    token_type: str | None = payload.get("type")
    if email is None or token_type != "access":
        raise credentials_exception

    user = get_user_by_email(email, session)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    if not current_user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")
    return current_user


async def get_current_admin_user(
    current_user: User = Depends(get_current_active_user),
) -> User:
    if current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return current_user


def create_user_delegated_token(
    session: Session,
    *,
    user_id: int,
    audience: str,
    purpose: str,
    extra_claims: Optional[dict[str, Any]] = None,
    expires_delta: Optional[timedelta] = None,
) -> str:
    user = session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"User {user_id} not found")

    claims: dict[str, Any] = {
        "sub": user.email,
        "uid": user.id,
        "username": user.username,
        "role": user.role,
    }
    if extra_claims:
        claims.update(extra_claims)

    return create_delegated_token(
        claims,
        audience=audience,
        purpose=purpose,
        expires_delta=expires_delta,
    )
