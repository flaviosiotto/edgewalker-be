from datetime import timedelta
from typing import Annotated, Optional, Dict, Any, cast

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session

from app.db.database import get_session
from app.schemas.auth import (
    MessageResponse,
    PasswordChangeRequest,
    PasswordResetConfirmRequest,
    PasswordResetRequest,
    PasswordResetRequestResponse,
    RefreshTokenRequest,
    Token,
    TokenWithRefresh,
)
from app.services.password_reset_service import (
    change_password,
    confirm_password_reset,
    request_password_reset,
)
from app.utils.auth_utils import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_user_by_email,
    get_current_active_user,
)
from app.utils.rate_limit import check_rate_limit
from app.core.config import settings
from app.models.user import User

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
    responses={404: {"description": "Not found"}},
)


def _client_ip(request: Request) -> str:
    """Best-effort client address, honouring the reverse proxy in front of us."""
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _enforce_password_reset_quota(request: Request, identifier: str) -> None:
    """Cap reset requests per identifier and per caller.

    Both counters matter: the per-identifier one stops a single mailbox from
    being flooded, the per-IP one stops a single caller from walking a list of
    addresses.
    """
    quotas = (
        (
            f"password-reset:id:{identifier.strip().lower()}",
            settings.PASSWORD_RESET_MAX_PER_IDENTIFIER_PER_HOUR,
        ),
        (
            f"password-reset:ip:{_client_ip(request)}",
            settings.PASSWORD_RESET_MAX_PER_IP_PER_HOUR,
        ),
    )

    for key, limit in quotas:
        decision = check_rate_limit(key, limit=limit, window_seconds=3600)
        if not decision.allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Troppe richieste di reset password. Riprova piu' tardi.",
                headers={"Retry-After": str(decision.retry_after_seconds)},
            )


@router.post("/token", response_model=TokenWithRefresh)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    session: Session = Depends(get_session)
):
    user = authenticate_user(form_data.username, form_data.password, session)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenziali non valide",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    token_claims = {"sub": user.email, "uid": user.id, "role": user.role}
    access_token = create_access_token(data=token_claims, expires_delta=access_token_expires)
    refresh_token = create_refresh_token(data=token_claims)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


@router.post("/refresh", response_model=Token)
async def refresh_access_token(refresh_data: RefreshTokenRequest, session: Session = Depends(get_session)):
    payload = cast(
        Optional[Dict[str, Any]],
        decode_token(refresh_data.refresh_token, audience=settings.REFRESH_TOKEN_AUDIENCE),
    )

    if payload is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token non valido")

    email = cast(Optional[str], payload.get("sub") if payload else None)
    token_type = cast(Optional[str], payload.get("type") if payload else None)

    if email is None or token_type != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token non valido")

    user = get_user_by_email(email, session)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Utente non trovato")

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    new_access_token = create_access_token(
        data={"sub": email, "uid": user.id, "role": user.role},
        expires_delta=access_token_expires,
    )

    return {"access_token": new_access_token, "token_type": "bearer"}


@router.post("/logout")
async def logout(current_user: Annotated[User, Depends(get_current_active_user)]):
    return {"message": "Logout effettuato con successo. Cancella i token dal client."}


@router.post("/password/change", response_model=MessageResponse)
async def change_password_endpoint(
    payload: PasswordChangeRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
):
    change_password(
        session,
        current_user,
        payload.current_password,
        payload.new_password,
        background_tasks=background_tasks,
    )
    return {"message": "Password aggiornata con successo"}


@router.post("/password-reset/request", response_model=PasswordResetRequestResponse)
async def request_password_reset_endpoint(
    payload: PasswordResetRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
):
    _enforce_password_reset_quota(request, payload.identifier)
    return request_password_reset(session, payload.identifier, background_tasks=background_tasks)


@router.post("/password-reset/confirm", response_model=MessageResponse)
async def confirm_password_reset_endpoint(
    payload: PasswordResetConfirmRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
):
    confirm_password_reset(
        session,
        payload.token,
        payload.new_password,
        background_tasks=background_tasks,
    )
    return {"message": "Password reimpostata con successo"}
