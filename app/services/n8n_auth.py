from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import HTTPException, status
from jose import jwt
from sqlmodel import Session, select

from app.core.config import settings
from app.models.strategy import LiveStatus, StrategyLive
from app.models.user import User
from app.utils.auth_utils import (
    create_access_token,
    create_user_delegated_token,
)


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
    backend_api: dict[str, Any] | None = None,
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
    if backend_api is not None:
        metadata["backend_api"] = backend_api
    return metadata


_DEFAULT_N8N_CONSULTATIVE_SCOPES = ["accounts:read", "account_orders:read"]


def _resolve_active_strategy_live(
    session: Session,
    *,
    chat_id: int | None,
    strategy_id: int | None,
) -> StrategyLive | None:
    """Find an active StrategyLive bound to the given chat or strategy."""
    if chat_id is not None:
        sl = session.exec(
            select(StrategyLive)
            .where(StrategyLive.chat_id == chat_id)
            .where(StrategyLive.status != LiveStatus.STOPPED.value)
            .order_by(StrategyLive.id.desc())
        ).first()
        if sl is not None:
            return sl
    if strategy_id is not None:
        sl = session.exec(
            select(StrategyLive)
            .where(StrategyLive.strategy_id == strategy_id)
            .where(StrategyLive.status != LiveStatus.STOPPED.value)
            .order_by(StrategyLive.id.desc())
        ).first()
        if sl is not None:
            return sl
    return None


def build_n8n_backend_api_metadata(
    session: Session,
    *,
    user_id: int | None,
    chat_id: int | None = None,
    strategy_id: int | None = None,
) -> dict[str, Any] | None:
    """Issue a consultative `agent_backend_consult` token bound to a live session.

    The token has `no_expiry=True` and is gated at decode time by
    `StrategyLive.status` (see `get_current_consultative_principal`).
    Returns None if no active live session can be resolved from
    `chat_id` or `strategy_id` — in that case the agent has no live
    context to read account data against.
    """
    if user_id is None:
        return None

    sl = _resolve_active_strategy_live(
        session, chat_id=chat_id, strategy_id=strategy_id
    )
    if sl is None:
        return None

    token = create_user_delegated_token(
        session,
        user_id=user_id,
        audience=settings.AGENT_TOKEN_AUDIENCE,
        purpose="agent_backend_consult",
        no_expiry=True,
        extra_claims={
            "strategy_id": sl.strategy_id,
            "live_id": sl.id,
            "connection_id": sl.connection_id,
            "account_id": sl.account_id,
            "scopes": list(_DEFAULT_N8N_CONSULTATIVE_SCOPES),
        },
    )
    return {
        "token": token,
        "token_type": "Bearer",
        "audience": settings.AGENT_TOKEN_AUDIENCE,
        "purpose": "agent_backend_consult",
        "scopes": list(_DEFAULT_N8N_CONSULTATIVE_SCOPES),
        "account_id": sl.account_id,
        "connection_id": sl.connection_id,
        "live_id": sl.id,
        "strategy_id": sl.strategy_id,
        "paths": {
            "accounts": "/accounts",
            "account": f"/accounts/{sl.account_id}" if sl.account_id else None,
            "account_orders": f"/accounts/{sl.account_id}/orders" if sl.account_id else None,
        },
    }