from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import HTTPException, status
from jose import jwt
from sqlmodel import Session

from app.core.config import settings
from app.models.user import User
from app.utils.auth_utils import create_user_delegated_token


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
        user = session.get(User, user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User {user_id} not found",
            )

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


def build_n8n_api_auth_metadata(
    *,
    user_id: int | None,
    purpose: str,
    token: str | None = None,
) -> dict[str, Any]:
    uses_shared_secret = bool(settings.N8N_WEBHOOK_JWT_SHARED_SECRET)
    metadata = {
        "mode": "authorization_header",
        "scheme": "Bearer",
        "auth_mode": "shared_secret_jwt" if uses_shared_secret else "backend_delegated_jwt",
        "token_type": "jwt" if uses_shared_secret else "delegated",
        "audience": (
            settings.n8n_webhook_jwt_audience if uses_shared_secret else settings.N8N_TOKEN_AUDIENCE
        ),
        "issuer": settings.n8n_webhook_jwt_issuer if uses_shared_secret else settings.JWT_ISSUER,
        "purpose": purpose,
        "user_id": user_id,
    }
    if token:
        metadata["token"] = token
    return metadata