"""Google Sign-In: Authorization Code flow with PKCE, mediated by the backend.

The browser never sees the client secret and never receives tokens in a URL.
The callback hands the SPA a single-use exchange code, which the SPA trades for
the real token pair over a normal POST.

Transient state (PKCE verifier, exchange codes) lives in Redis with a TTL, which
gives single-use semantics and expiry for free.
"""

import base64
import hashlib
import json
import logging
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlencode

import httpx
from fastapi import BackgroundTasks, HTTPException, status
from jose import JWTError, jwt
from sqlmodel import Session, select

from app.core.config import settings
from app.models.access_control import AuthProvider, UserIdentity
from app.models.user import User, UserStatus, apply_status
from app.services.email_service import queue_email
from app.services.email_templates import (
    account_approved_email,
    account_pending_approval_email,
)
from app.services.registration_service import (
    consume_allowlist_entry,
    normalize_email,
    resolve_signup_outcome,
)
from app.utils.redis_client import get_redis

logger = logging.getLogger(__name__)

GOOGLE_AUTHORIZATION_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
GOOGLE_JWKS_URI = "https://www.googleapis.com/oauth2/v3/certs"
GOOGLE_ISSUERS = ("https://accounts.google.com", "accounts.google.com")

_STATE_PREFIX = "oauth:google:state:"
_EXCHANGE_PREFIX = "oauth:exchange:"

# Google rotates its signing keys slowly; caching avoids a fetch per login while
# still picking up rotations well within Google's publication window.
_JWKS_CACHE_TTL_SECONDS = 3600
_jwks_cache: dict[str, Any] = {"fetched_at": 0.0, "keys": None}


class OAuthError(Exception):
    """A failure that must be reported to the SPA as a redirect, not a 500."""


def _require_enabled() -> None:
    if not settings.google_oauth_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="L'accesso con Google non e' configurato su questo ambiente.",
        )


def _require_redis():
    client = get_redis()
    if client is None:
        # Unlike rate limiting, this cannot fail open: without the stored PKCE
        # verifier the flow is not verifiable, so refuse it outright.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Servizio di autenticazione temporaneamente non disponibile.",
        )
    return client


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def build_authorization_url(redirect_to: Optional[str] = None) -> str:
    """Start the flow: mint PKCE + state, stash them, return Google's URL."""
    _require_enabled()
    client = _require_redis()

    code_verifier = _b64url(secrets.token_bytes(64))
    code_challenge = _b64url(hashlib.sha256(code_verifier.encode("ascii")).digest())
    state = _b64url(secrets.token_bytes(32))

    client.setex(
        f"{_STATE_PREFIX}{state}",
        settings.OAUTH_STATE_TTL_SECONDS,
        json.dumps({"code_verifier": code_verifier, "redirect_to": redirect_to or ""}),
    )

    query = urlencode(
        {
            "client_id": settings.GOOGLE_OAUTH_CLIENT_ID,
            "redirect_uri": settings.GOOGLE_OAUTH_REDIRECT_URI,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            # Ask for a fresh consent screen only when we need it; "select_account"
            # still lets a user pick a different Google account.
            "prompt": "select_account",
        }
    )
    return f"{GOOGLE_AUTHORIZATION_ENDPOINT}?{query}"


def _consume_state(state: str) -> dict[str, Any]:
    client = _require_redis()
    key = f"{_STATE_PREFIX}{state}"
    # GETDEL makes the state single-use: a replayed callback finds nothing.
    raw = client.getdel(key)
    if not raw:
        raise OAuthError("Sessione di accesso scaduta o non valida. Riprova.")
    return json.loads(raw)


def _fetch_jwks() -> dict[str, Any]:
    now = time.time()
    if (
        _jwks_cache["keys"] is not None
        and now - _jwks_cache["fetched_at"] < _JWKS_CACHE_TTL_SECONDS
    ):
        return _jwks_cache["keys"]

    with httpx.Client(timeout=settings.OAUTH_HTTP_TIMEOUT_SECONDS) as client:
        response = client.get(GOOGLE_JWKS_URI)
        response.raise_for_status()
        keys = response.json()

    _jwks_cache["keys"] = keys
    _jwks_cache["fetched_at"] = now
    return keys


def _exchange_code_for_tokens(code: str, code_verifier: str) -> dict[str, Any]:
    with httpx.Client(timeout=settings.OAUTH_HTTP_TIMEOUT_SECONDS) as client:
        response = client.post(
            GOOGLE_TOKEN_ENDPOINT,
            data={
                "code": code,
                "client_id": settings.GOOGLE_OAUTH_CLIENT_ID,
                "client_secret": settings.GOOGLE_OAUTH_CLIENT_SECRET,
                "redirect_uri": settings.GOOGLE_OAUTH_REDIRECT_URI,
                "grant_type": "authorization_code",
                "code_verifier": code_verifier,
            },
        )

    if response.status_code != 200:
        logger.warning(
            "Google token exchange failed: %s %s", response.status_code, response.text[:400]
        )
        raise OAuthError("Google ha rifiutato l'accesso. Riprova.")

    return response.json()


def verify_google_id_token(id_token: str) -> dict[str, Any]:
    """Validate signature, audience, issuer and expiry of Google's ID token."""
    try:
        claims = jwt.decode(
            id_token,
            _fetch_jwks(),
            algorithms=["RS256"],
            audience=settings.GOOGLE_OAUTH_CLIENT_ID,
            options={"verify_at_hash": False},
        )
    except JWTError as exc:
        logger.warning("Google ID token failed validation: %s", exc)
        raise OAuthError("Token Google non valido.") from exc

    if claims.get("iss") not in GOOGLE_ISSUERS:
        raise OAuthError("Token Google non valido.")

    return claims


def _unique_username(session: Session, email: str) -> str:
    """Derive a free username from the address' local part."""
    base = "".join(ch for ch in email.split("@")[0].lower() if ch.isalnum() or ch in "._-")
    base = base.strip("._-") or "user"
    base = base[:48]

    candidate = base
    suffix = 1
    while session.exec(select(User).where(User.username == candidate)).first() is not None:
        suffix += 1
        candidate = f"{base}{suffix}"
        if suffix > 1000:  # pathological; fall back to something certainly free
            candidate = f"{base}-{secrets.token_hex(4)}"
            break
    return candidate


def _issue_exchange_code(user_id: int) -> str:
    client = _require_redis()
    code = _b64url(secrets.token_bytes(32))
    client.setex(f"{_EXCHANGE_PREFIX}{code}", settings.OAUTH_EXCHANGE_TTL_SECONDS, str(user_id))
    return code


def consume_exchange_code(code: str) -> int:
    """Trade a one-time code for the user id it was minted for."""
    client = _require_redis()
    raw = client.getdel(f"{_EXCHANGE_PREFIX}{code}")
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Codice di accesso scaduto o gia' utilizzato.",
        )
    return int(raw)


def complete_google_login(
    session: Session,
    *,
    code: str,
    state: str,
    background_tasks: Optional[BackgroundTasks] = None,
) -> tuple[Optional[str], str, Optional[str]]:
    """Finish the flow.

    Returns ``(exchange_code, status_value, redirect_to)``. ``exchange_code`` is
    None whenever the account exists but must not receive tokens yet, e.g. it is
    waiting in the approval queue.
    """
    _require_enabled()

    stored = _consume_state(state)
    tokens = _exchange_code_for_tokens(code, stored["code_verifier"])

    id_token = tokens.get("id_token")
    if not id_token:
        raise OAuthError("Google non ha restituito un token di identita'.")

    claims = verify_google_id_token(id_token)

    subject = claims.get("sub")
    email = claims.get("email")
    if not subject or not email:
        raise OAuthError("Google non ha restituito un indirizzo email.")

    # An unverified Google address must never be trusted: accepting one would
    # let anyone claim an existing local account by registering that address at
    # Google without proving control of it.
    if not claims.get("email_verified"):
        raise OAuthError(
            "L'indirizzo email del tuo account Google non risulta verificato."
        )

    email = normalize_email(email)
    now = datetime.now(timezone.utc)
    redirect_to = stored.get("redirect_to") or None

    identity = session.exec(
        select(UserIdentity).where(
            UserIdentity.provider == AuthProvider.GOOGLE.value,
            UserIdentity.provider_subject == subject,
        )
    ).first()

    is_new_account = False

    if identity is not None:
        user = session.get(User, identity.user_id)
        if user is None:
            raise OAuthError("Account non disponibile.")
        identity.last_login_at = now
        identity.email = email
        session.add(identity)
    else:
        user = session.exec(select(User).where(User.email == email)).first()

        if user is not None:
            # Link the provider to the account that already owns this verified
            # address.
            session.add(
                UserIdentity(
                    user_id=user.id,
                    provider=AuthProvider.GOOGLE.value,
                    provider_subject=subject,
                    email=email,
                    raw_profile=_safe_profile(claims),
                    last_login_at=now,
                )
            )
            if user.email_verified_at is None:
                user.email_verified_at = now
        else:
            user = _create_user_from_google(session, claims, email, subject, now)
            is_new_account = True

    if user.is_active:
        user.last_login_at = now

    session.add(user)
    session.commit()
    session.refresh(user)

    account_status = user.status
    recipient_email, display_name = user.email, user.display_name

    # Only on first contact: an existing user signing in again must not be
    # emailed every time.
    if is_new_account:
        notify_google_signup(
            account_status,
            email=recipient_email,
            display_name=display_name,
            background_tasks=background_tasks,
        )

    if user.is_active:
        return _issue_exchange_code(user.id), account_status, redirect_to
    return None, account_status, redirect_to


def _safe_profile(claims: dict[str, Any]) -> dict[str, Any]:
    """Keep only the profile fields worth storing; drop token machinery."""
    return {
        key: claims.get(key)
        for key in ("email", "email_verified", "name", "given_name", "family_name", "picture", "hd")
        if claims.get(key) is not None
    }


def _create_user_from_google(
    session: Session,
    claims: dict[str, Any],
    email: str,
    subject: str,
    now: datetime,
) -> User:
    """Register a brand new account coming from Google.

    Routed through the same gate as local signup: Google must not be a way
    around the family&friends phase.
    """
    outcome = resolve_signup_outcome(session, email)

    user = User(
        email=email,
        username=_unique_username(session, email),
        first_name=claims.get("given_name"),
        last_name=claims.get("family_name"),
        # No local password: this account signs in through Google only.
        hashed_password=None,
        role="user",
        email_verified_at=now,
    )
    apply_status(user, outcome)
    if outcome == UserStatus.ACTIVE:
        user.approved_at = now

    session.add(user)
    session.commit()
    session.refresh(user)

    session.add(
        UserIdentity(
            user_id=user.id,
            provider=AuthProvider.GOOGLE.value,
            provider_subject=subject,
            email=email,
            raw_profile=_safe_profile(claims),
            last_login_at=now,
        )
    )
    if outcome == UserStatus.ACTIVE:
        consume_allowlist_entry(session, user)
    session.commit()
    session.refresh(user)
    return user


def notify_google_signup(
    user_status: str,
    *,
    email: str,
    display_name: str,
    background_tasks: Optional[BackgroundTasks] = None,
) -> None:
    if user_status == UserStatus.ACTIVE.value:
        subject, text_body, html_body = account_approved_email(display_name=display_name)
    elif user_status == UserStatus.PENDING_APPROVAL.value:
        subject, text_body, html_body = account_pending_approval_email(display_name=display_name)
    else:
        return

    queue_email(
        background_tasks,
        to_address=email,
        subject=subject,
        text_body=text_body,
        html_body=html_body,
    )
