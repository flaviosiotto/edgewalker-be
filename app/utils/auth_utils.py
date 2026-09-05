from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Any, Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlmodel import Session, select

from app.core.config import settings
from app.models.strategy import BacktestResult, BacktestStatus, LiveStatus, Strategy, StrategyLive
from app.models.user import User
from app.db.database import get_session
from app.services.pat_service import (
    PAT_TOKEN_PREFIX,
    enforce_pat_scope,
    resolve_personal_access_token,
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_ROOT_PATH}/auth/token")


@dataclass
class AuthPrincipal:
    user: User
    claims: dict[str, Any]


def verify_password(plain_password: str, hashed_password: Optional[str]) -> bool:
    # Accounts that only authenticate through an external provider have no local
    # hash. Passing None straight to passlib raises, so no password can ever
    # match for them.
    if not hashed_password:
        return False
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
    expires_delta: Optional[timedelta],
    purpose: Optional[str] = None,
) -> dict[str, Any]:
    to_encode = data.copy()
    to_encode.update(
        {
            "aud": audience,
            "iss": settings.JWT_ISSUER,
            "type": token_type,
        }
    )
    if expires_delta is not None:
        to_encode["exp"] = datetime.now(timezone.utc) + expires_delta
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
    no_expiry: bool = False,
) -> str:
    effective_expires_delta = None if no_expiry else (
        expires_delta or timedelta(minutes=settings.DELEGATED_TOKEN_EXPIRE_MINUTES)
    )
    payload = _build_token_payload(
        data,
        token_type="delegated",
        audience=audience,
        expires_delta=effective_expires_delta,
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


def _credentials_exception(detail: str = "Could not validate credentials") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _load_principal_from_payload(
    payload: dict[str, Any],
    session: Session,
    *,
    allowed_token_types: set[str],
    allowed_purposes: set[str] | None = None,
    credentials_exception: HTTPException,
) -> AuthPrincipal:
    email: str | None = payload.get("sub")
    token_type: str | None = payload.get("type")
    purpose: str | None = payload.get("purpose")
    if email is None or token_type not in allowed_token_types:
        raise credentials_exception
    if allowed_purposes is not None and purpose not in allowed_purposes:
        raise credentials_exception

    user = get_user_by_email(email, session)
    if user is None:
        raise credentials_exception

    return AuthPrincipal(user=user, claims=payload)


def _try_pat_principal(
    request: Optional[Request],
    token: str,
    session: Session,
) -> Optional[AuthPrincipal]:
    """Resolve an opaque ``ewp_`` personal access token, or return ``None``.

    ``None`` means "not a PAT, try the JWT path". A malformed/revoked/expired
    PAT raises immediately: an ``ewp_`` string can never be a valid JWT, so
    falling through would only produce a less accurate error. Scope enforcement
    happens here, centrally, against the request method + path — endpoints stay
    unannotated.
    """
    if not token.startswith(PAT_TOKEN_PREFIX):
        return None

    resolved = resolve_personal_access_token(session, token)
    if resolved is None:
        raise _credentials_exception("Invalid, expired or revoked personal access token")

    user, pat = resolved
    if request is not None:
        enforce_pat_scope(request.method, request.url.path, pat.scopes)

    claims = {
        "sub": user.email,
        "uid": user.id,
        "username": user.username,
        "role": user.role,
        "type": "pat",
        "purpose": "pat_access",
        "scopes": pat.scopes,
        "pat_id": pat.id,
        "pat_name": pat.name,
    }
    return AuthPrincipal(user=user, claims=claims)


def get_user_by_email(email: str, session: Session) -> Optional[User]:
    statement = select(User).where(User.email == email)
    return session.exec(statement).first()


def get_user_by_username_or_email(username_or_email: str, session: Session) -> Optional[User]:
    statement = select(User).where(
        (User.username == username_or_email) | (User.email == username_or_email)
    )
    return session.exec(statement).first()


#: Hash of a value nobody can supply, used to spend the same CPU time on a
#: missing account as on a real one.
_DUMMY_PASSWORD_HASH = pwd_context.hash("edgewalker-nonexistent-account-placeholder")


def authenticate_user(username_or_email: str, password: str, session: Session) -> Optional[User]:
    """Return the user when the password matches, regardless of account status.

    Status is deliberately NOT enforced here: the login endpoint needs to tell a
    pending or rejected account why it cannot sign in, and it can only do that
    safely once the caller has proved knowledge of the password. Every caller
    must therefore gate on the returned user's status.
    """
    user = get_user_by_username_or_email(username_or_email, session)
    if not user or not user.hashed_password:
        # Verify against a throwaway hash anyway. Returning immediately would
        # answer far faster than a real bcrypt check, turning response time into
        # an oracle for which addresses are registered.
        verify_password(password, _DUMMY_PASSWORD_HASH)
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def _release_auth_connection(session: Session) -> None:
    """End the auth lookup's read-only transaction.

    Auth runs first in every request and only reads; without this, the
    transaction it opens keeps the pooled connection checked out for the
    entire request — including endpoints that then block on HTTP/docker for
    minutes (the 19/08 pool-saturation incident). Sessions run with
    ``expire_on_commit=False``, so the resolved principal stays readable and
    the endpoint's own queries lazily re-acquire a connection.
    """
    if session.in_transaction():
        session.commit()


async def get_current_user(
    request: Request,
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_session)
) -> User:
    credentials_exception = _credentials_exception()

    pat_principal = _try_pat_principal(request, token, session)
    if pat_principal is not None:
        _release_auth_connection(session)
        return pat_principal.user

    payload = decode_token_for_audiences(
        token,
        [settings.ACCESS_TOKEN_AUDIENCE],
    )
    if payload is None:
        raise credentials_exception

    user = _load_principal_from_payload(
        payload,
        session,
        allowed_token_types={"access"},
        credentials_exception=credentials_exception,
    ).user
    _release_auth_connection(session)
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    if not current_user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")
    return current_user


async def get_current_active_principal(
    request: Request,
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_session),
) -> AuthPrincipal:
    """Same auth chain as ``get_current_active_user`` but keeps the token claims.

    Used by endpoints that need to know WHO is calling beyond the user — e.g.
    the ``purpose`` claim distinguishes the FE (``ui_auth``) from n8n-issued
    agent tokens (``n8n_chat_api_access``, ...) and personal access tokens
    (``pat_access``).
    """
    credentials_exception = _credentials_exception()

    pat_principal = _try_pat_principal(request, token, session)
    if pat_principal is not None:
        if not pat_principal.user.is_active:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")
        _release_auth_connection(session)
        return pat_principal

    payload = decode_token_for_audiences(
        token,
        [settings.ACCESS_TOKEN_AUDIENCE],
    )
    if payload is None:
        raise credentials_exception

    principal = _load_principal_from_payload(
        payload,
        session,
        allowed_token_types={"access"},
        credentials_exception=credentials_exception,
    )
    if not principal.user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")
    _release_auth_connection(session)
    return principal


async def get_current_admin_user(
    current_user: User = Depends(get_current_active_user),
) -> User:
    if current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return current_user


async def get_current_runner_principal(
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_session),
) -> AuthPrincipal:
    credentials_exception = _credentials_exception("Could not validate runner credentials")
    payload = decode_token_for_audiences(token, [settings.RUNNER_TOKEN_AUDIENCE])
    if payload is None:
        raise credentials_exception

    principal = _load_principal_from_payload(
        payload,
        session,
        allowed_token_types={"delegated"},
        allowed_purposes={"runner_backend"},
        credentials_exception=credentials_exception,
    )

    runner_live_id = principal.claims.get("live_id")
    runner_backtest_id = principal.claims.get("backtest_id")
    runner_strategy_id = principal.claims.get("strategy_id")

    if runner_live_id is not None:
        strategy_live = session.get(StrategyLive, runner_live_id)
        if strategy_live is None:
            raise _credentials_exception("Runner live session not found")

        if strategy_live.status == LiveStatus.STOPPED.value:
            raise _credentials_exception("Runner live session is no longer active")

        if runner_strategy_id is not None and str(runner_strategy_id) != str(strategy_live.strategy_id):
            raise _credentials_exception("Runner token does not match this strategy")

        _release_auth_connection(session)
        return principal

    if runner_backtest_id is None:
        raise _credentials_exception("Runner token is missing live or backtest binding")

    backtest = session.get(BacktestResult, runner_backtest_id)
    if backtest is None:
        raise _credentials_exception("Runner backtest not found")

    if backtest.status not in {BacktestStatus.PENDING.value, BacktestStatus.RUNNING.value}:
        raise _credentials_exception("Runner backtest is no longer active")

    strategy = session.get(Strategy, backtest.strategy_id)
    if strategy is None or strategy.user_id != principal.user.id:
        raise _credentials_exception("Runner token does not match this backtest owner")

    if runner_strategy_id is not None and str(runner_strategy_id) != str(backtest.strategy_id):
        raise _credentials_exception("Runner token does not match this strategy")

    _release_auth_connection(session)
    return principal


async def get_current_consultative_principal(
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_session),
) -> AuthPrincipal:
    credentials_exception = _credentials_exception("Could not validate consultative API credentials")
    payload = decode_token_for_audiences(
        token,
        [settings.AGENT_TOKEN_AUDIENCE, settings.N8N_TOKEN_AUDIENCE],
    )
    if payload is None:
        raise credentials_exception

    principal = _load_principal_from_payload(
        payload,
        session,
        allowed_token_types={"delegated"},
        allowed_purposes={"agent_backend_consult", "n8n_backend_consult"},
        credentials_exception=credentials_exception,
    )

    # Tokens issued with no_expiry are gated by the session lifecycle:
    # stopping the live session (or finishing the backtest) implicitly revokes
    # them.
    live_id = principal.claims.get("live_id")
    if live_id is not None:
        strategy_live = session.get(StrategyLive, live_id)
        if strategy_live is None:
            raise _credentials_exception("Consultative token live session not found")
        if strategy_live.status == LiveStatus.STOPPED.value:
            raise _credentials_exception("Consultative token live session is no longer active")

    backtest_id = principal.claims.get("backtest_id")
    if backtest_id is not None:
        backtest = session.get(BacktestResult, backtest_id)
        if backtest is None:
            raise _credentials_exception("Consultative token backtest not found")

    _release_auth_connection(session)
    return principal


async def get_current_active_or_runner_user(
    request: Request,
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_session),
) -> User:
    pat_principal = _try_pat_principal(request, token, session)
    if pat_principal is not None:
        if not pat_principal.user.is_active:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")
        _release_auth_connection(session)
        return pat_principal.user

    access_payload = decode_token(token, audience=settings.ACCESS_TOKEN_AUDIENCE)
    if access_payload is not None:
        user = _load_principal_from_payload(
            access_payload,
            session,
            allowed_token_types={"access"},
            credentials_exception=_credentials_exception(),
        ).user
        if not user.is_active:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")
        _release_auth_connection(session)
        return user

    runner_principal = await get_current_runner_principal(token=token, session=session)
    if not runner_principal.user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")
    return runner_principal.user


async def get_current_active_or_consultative_principal(
    request: Request,
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_session),
) -> AuthPrincipal:
    """User JWT / PAT / delegated n8n token / agent consultative token.

    Same chain as ``get_current_active_or_consultative_user`` but keeps the
    claims, for endpoints that must tell WHO is calling — e.g. PATCH
    /strategies/{id} derives ``origin=user|agent`` from ``purpose``. The
    consultative token carries ``purpose=agent_backend_consult``, so it lands
    on ``agent`` like the other n8n-issued tokens.
    """
    pat_principal = _try_pat_principal(request, token, session)
    if pat_principal is not None:
        if not pat_principal.user.is_active:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")
        _release_auth_connection(session)
        return pat_principal

    access_payload = decode_token(token, audience=settings.ACCESS_TOKEN_AUDIENCE)
    if access_payload is not None:
        principal = _load_principal_from_payload(
            access_payload,
            session,
            allowed_token_types={"access"},
            credentials_exception=_credentials_exception(),
        )
        if not principal.user.is_active:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")
        _release_auth_connection(session)
        return principal

    consultative_principal = await get_current_consultative_principal(token=token, session=session)
    if not consultative_principal.user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")
    return consultative_principal


async def get_current_active_or_consultative_user(
    request: Request,
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_session),
) -> User:
    principal = await get_current_active_or_consultative_principal(
        request=request, token=token, session=session
    )
    return principal.user


def create_user_delegated_token(
    session: Session,
    *,
    user_id: int,
    audience: str,
    purpose: str,
    extra_claims: Optional[dict[str, Any]] = None,
    expires_delta: Optional[timedelta] = None,
    no_expiry: bool = False,
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
        no_expiry=no_expiry,
    )


async def get_current_ai_usage_principal(
    request: Request,
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_session),
) -> AuthPrincipal:
    """Who may report AI usage for a turn (POST /ai-usage/report).

    The n8n workflow ends a turn with whichever token it was handed in
    ``metadata.api_auth`` — a UI access token (``n8n_chat_api_access``), a
    consultative token (``agent_backend_consult`` / ``n8n_backend_consult``)
    or a runner-callback token — and the strategy-runner reports its own
    estimate with its delegated runner token. All of them resolve to the
    user the usage is charged to.
    """
    credentials_exception = _credentials_exception("Could not validate usage reporter credentials")

    pat_principal = _try_pat_principal(request, token, session)
    if pat_principal is not None:
        if not pat_principal.user.is_active:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")
        _release_auth_connection(session)
        return pat_principal

    access_payload = decode_token(token, audience=settings.ACCESS_TOKEN_AUDIENCE)
    if access_payload is not None:
        principal = _load_principal_from_payload(
            access_payload,
            session,
            allowed_token_types={"access"},
            credentials_exception=credentials_exception,
        )
        _release_auth_connection(session)
        return principal

    delegated_payload = decode_token_for_audiences(
        token,
        [settings.RUNNER_TOKEN_AUDIENCE, settings.AGENT_TOKEN_AUDIENCE, settings.N8N_TOKEN_AUDIENCE],
    )
    if delegated_payload is None:
        raise credentials_exception
    principal = _load_principal_from_payload(
        delegated_payload,
        session,
        allowed_token_types={"delegated"},
        allowed_purposes={
            "runner_backend",
            "agent_backend_consult",
            "n8n_backend_consult",
            "agent_runner_callback",
        },
        credentials_exception=credentials_exception,
    )
    _release_auth_connection(session)
    return principal
