"""Personal access tokens (PAT) for machine clients — MCP server, scripts.

A PAT is an opaque bearer token ``ewp_<random>`` presented in the same
``Authorization`` header as the JWTs. Storage is hash-only (SHA-256), so a
leaked database dump cannot be replayed. Authorization is coarse-grained via
scopes checked centrally against the request method + path (see
``required_scope_for``), NOT per-endpoint annotations: the auth dependencies in
``auth_utils`` call ``enforce_pat_scope`` before handing the user to the
endpoint. The per-user data isolation itself is unchanged — every service
already filters on ``current_user.id``.
"""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import HTTPException, status
from sqlmodel import Session, select

from app.core.config import settings
from app.models.personal_access_token import PersonalAccessToken
from app.models.user import User

PAT_TOKEN_PREFIX = "ewp_"

SCOPE_READ = "read"
SCOPE_WRITE = "write"
SCOPE_TRADE = "trade"
VALID_SCOPES = {SCOPE_READ, SCOPE_WRITE, SCOPE_TRADE}

#: How stale last_used_at may get before we spend a write refreshing it.
_LAST_USED_REFRESH_SECONDS = 60

_READ_METHODS = {"GET", "HEAD", "OPTIONS"}

#: Identity and administration surfaces are never reachable with a PAT: a
#: leaked token must not be able to rotate credentials, mint further tokens or
#: touch other accounts.
_FORBIDDEN_PREFIXES = ("/auth", "/admin", "/pats", "/lab")

#: Mutations on these surfaces move real money (orders, positions, live
#: sessions, runner callbacks) and require the explicit ``trade`` scope.
_TRADE_PREFIXES = ("/accounts", "/live", "/runners")


def hash_pat_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def mint_personal_access_token(
    session: Session,
    *,
    user: User,
    name: str,
    scopes: list[str],
    expires_in_days: Optional[int] = None,
) -> tuple[str, PersonalAccessToken]:
    """Create a PAT and return ``(raw_token, record)``.

    The raw token exists only in this return value: the caller must surface it
    to the user immediately, it cannot be recovered later.
    """
    cleaned_name = name.strip()
    if not cleaned_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Token name is required")
    if len(cleaned_name) > 120:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Token name is too long (max 120 chars)")

    normalized_scopes = sorted({scope.strip().lower() for scope in scopes if scope.strip()})
    invalid = set(normalized_scopes) - VALID_SCOPES
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid scopes: {', '.join(sorted(invalid))}. Valid scopes: {', '.join(sorted(VALID_SCOPES))}",
        )
    if not normalized_scopes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one scope is required")

    expires_at: Optional[datetime] = None
    if expires_in_days is not None:
        if expires_in_days <= 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="expires_in_days must be positive")
        expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)

    raw_token = PAT_TOKEN_PREFIX + secrets.token_urlsafe(32)
    pat = PersonalAccessToken(
        user_id=user.id,
        name=cleaned_name,
        token_hash=hash_pat_token(raw_token),
        token_prefix=raw_token[:12],
        scopes=normalized_scopes,
        expires_at=expires_at,
    )
    session.add(pat)
    session.commit()
    session.refresh(pat)
    return raw_token, pat


def list_personal_access_tokens(session: Session, user_id: int) -> list[PersonalAccessToken]:
    statement = (
        select(PersonalAccessToken)
        .where(PersonalAccessToken.user_id == user_id)
        .order_by(PersonalAccessToken.created_at.desc())
    )
    return list(session.exec(statement).all())


def revoke_personal_access_token(session: Session, user_id: int, pat_id: int) -> PersonalAccessToken:
    pat = session.get(PersonalAccessToken, pat_id)
    if pat is None or pat.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Personal access token not found")
    if pat.revoked_at is None:
        pat.revoked_at = datetime.now(timezone.utc)
        session.add(pat)
        session.commit()
        session.refresh(pat)
    return pat


def resolve_personal_access_token(
    session: Session, raw_token: str
) -> Optional[tuple[User, PersonalAccessToken]]:
    """Return ``(user, pat)`` for a valid raw token, ``None`` otherwise.

    Refreshes ``last_used_at`` at most once per minute so the lookup stays
    read-only on the hot path.
    """
    statement = select(PersonalAccessToken).where(
        PersonalAccessToken.token_hash == hash_pat_token(raw_token)
    )
    pat = session.exec(statement).first()
    if pat is None or pat.revoked_at is not None:
        return None

    now = datetime.now(timezone.utc)
    if pat.expires_at is not None:
        expires_at = pat.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at <= now:
            return None

    user = session.get(User, pat.user_id)
    if user is None:
        return None

    last_used = pat.last_used_at
    if last_used is not None and last_used.tzinfo is None:
        last_used = last_used.replace(tzinfo=timezone.utc)
    if last_used is None or (now - last_used).total_seconds() > _LAST_USED_REFRESH_SECONDS:
        pat.last_used_at = now
        session.add(pat)
        session.commit()

    return user, pat


def _normalize_api_path(path: str) -> str:
    root = settings.API_ROOT_PATH
    if root and path.startswith(root):
        path = path[len(root):] or "/"
    return path


def required_scope_for(method: str, path: str) -> Optional[str]:
    """Map a request to the scope it needs, or ``None`` when PATs are banned.

    The policy is deliberately coarse and centralized: read methods need
    ``read``, mutations need ``write``, and mutations on the trading surfaces
    (accounts/live/runners) need ``trade``. Identity management (/auth, /admin,
    /pats, /users except GET /users/me) is unreachable with a PAT.
    """
    path = _normalize_api_path(path)
    method = method.upper()

    if path.startswith(_FORBIDDEN_PREFIXES):
        return None
    if path.startswith("/users"):
        return SCOPE_READ if path == "/users/me" and method in _READ_METHODS else None

    if method in _READ_METHODS:
        return SCOPE_READ
    if path.startswith(_TRADE_PREFIXES):
        return SCOPE_TRADE
    return SCOPE_WRITE


def enforce_pat_scope(method: str, path: str, scopes: list[str]) -> None:
    required = required_scope_for(method, path)
    if required is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This endpoint is not accessible with a personal access token",
        )
    if required not in scopes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Personal access token is missing the required '{required}' scope",
        )
