from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import HTTPException, status
from jose import jwt
from sqlmodel import Session

from app.core.config import settings
from app.models.user import User
from app.utils.auth_utils import create_access_token, create_user_delegated_token


def _get_user(session: Session, user_id: int) -> User:
    user = session.get(User, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )
    return user


def issue_n8n_webhook_auth_token(
    session: Session,
    *,
    user_id: int | None,
    purpose: str,
    extra_claims: dict[str, Any] | None = None,
) -> str | None:
    if user_id is None:
        return None

    if settings.N8N_WEBHOOK_JWT_SHARED_SECRET:
        user = _get_user(session, user_id)

        issued_at = datetime.now(timezone.utc)
        payload: dict[str, Any] = {
            "sub": user.email,
            "uid": user.id,
            "username": user.username,
            "role": user.role,
            "aud": settings.n8n_webhook_jwt_audience,
            "exp": issued_at + timedelta(minutes=settings.N8N_WEBHOOK_JWT_EXPIRE_MINUTES),
            "iat": issued_at,
            "iss": settings.n8n_webhook_jwt_issuer,
            "type": "webhook",
            "purpose": purpose,
        }
        if extra_claims:
            payload.update(extra_claims)

        return jwt.encode(
            payload,
            settings.N8N_WEBHOOK_JWT_SHARED_SECRET,
            algorithm="HS256",
        )

    return create_user_delegated_token(
        session,
        user_id=user_id,
        audience=settings.N8N_TOKEN_AUDIENCE,
        purpose=purpose,
        extra_claims=extra_claims,
    )


def build_n8n_webhook_auth_headers(token: str | None) -> dict[str, str]:
    if not token:
        return {}

    return {"Authorization": f"Bearer {token}"}


def issue_n8n_api_access_token(
    session: Session,
    *,
    user_id: int | None,
    purpose: str,
    extra_claims: dict[str, Any] | None = None,
) -> tuple[str | None, datetime | None]:
    if user_id is None:
        return None, None

    user = _get_user(session, user_id)
    expires_delta = timedelta(minutes=settings.AGENT_CALLBACK_TOKEN_EXPIRE_MINUTES)
    expires_at = datetime.now(timezone.utc) + expires_delta
    claims: dict[str, Any] = {
        "sub": user.email,
        "uid": user.id,
        "username": user.username,
        "role": user.role,
    }
    if extra_claims:
        claims.update(extra_claims)

    token = create_access_token(
        claims,
        expires_delta=expires_delta,
        audience=settings.ACCESS_TOKEN_AUDIENCE,
        purpose=purpose,
    )
    return token, expires_at


def build_n8n_api_auth_metadata(
    *,
    user_id: int | None,
    purpose: str,
    token: str | None = None,
    expires_at: datetime | None = None,
) -> dict[str, Any]:
    metadata = {
        "mode": "authorization_header",
        "scheme": "Bearer",
        "auth_mode": "ui_access_token",
        "token_type": "access",
        "audience": settings.ACCESS_TOKEN_AUDIENCE,
        "issuer": settings.JWT_ISSUER,
        "purpose": purpose,
        "user_id": user_id,
    }
    if token:
        metadata["token"] = token
    if expires_at is not None:
        metadata["expires_at"] = expires_at.isoformat()
    return metadata