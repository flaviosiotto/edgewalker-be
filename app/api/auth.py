from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional, Dict, Any, cast

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session

from app.db.database import get_session
from app.schemas.auth import (
    EmailVerificationRequest,
    EmailVerificationResponse,
    MessageResponse,
    PasswordChangeRequest,
    PasswordResetConfirmRequest,
    PasswordResetRequest,
    PasswordResetRequestResponse,
    RefreshTokenRequest,
    RegistrationRequest,
    RegistrationResponse,
    ResendVerificationRequest,
    Token,
    TokenWithRefresh,
)
from app.services.registration_service import (
    confirm_email_verification,
    register_user,
    resend_verification_email,
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
from app.models.user import User, UserStatus

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


#: Why a given account cannot sign in. Only ever shown to a caller who already
#: proved knowledge of the password, so it leaks nothing to an anonymous prober.
_LOGIN_BLOCKED_REASONS = {
    UserStatus.PENDING_EMAIL.value: (
        "Devi confermare il tuo indirizzo email prima di accedere. "
        "Controlla la posta o richiedi un nuovo link."
    ),
    UserStatus.PENDING_APPROVAL.value: (
        "Il tuo account e' in attesa di approvazione da parte di un amministratore."
    ),
    UserStatus.REJECTED.value: "La richiesta di accesso e' stata rifiutata.",
    UserStatus.SUSPENDED.value: "Questo account e' sospeso.",
}


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

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=_LOGIN_BLOCKED_REASONS.get(user.status, "Questo account non e' utilizzabile."),
        )

    user.last_login_at = datetime.now(timezone.utc)
    session.add(user)
    session.commit()
    session.refresh(user)

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


def _enforce_ip_quota(request: Request, bucket: str, limit: int) -> None:
    decision = check_rate_limit(
        f"{bucket}:ip:{_client_ip(request)}", limit=limit, window_seconds=3600
    )
    if not decision.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Troppe richieste. Riprova piu' tardi.",
            headers={"Retry-After": str(decision.retry_after_seconds)},
        )


@router.post(
    "/register",
    response_model=RegistrationResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def register_endpoint(
    payload: RegistrationRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
):
    _enforce_ip_quota(request, "register", settings.REGISTRATION_MAX_PER_IP_PER_HOUR)
    return register_user(session, payload, background_tasks=background_tasks)


@router.post("/email/verify", response_model=EmailVerificationResponse)
async def verify_email_endpoint(
    payload: EmailVerificationRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
):
    return confirm_email_verification(session, payload.token, background_tasks=background_tasks)


@router.post("/email/resend", response_model=RegistrationResponse)
async def resend_verification_endpoint(
    payload: ResendVerificationRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
):
    _enforce_ip_quota(
        request, "verify-resend", settings.EMAIL_VERIFICATION_MAX_RESENDS_PER_HOUR
    )
    return resend_verification_email(session, payload.email, background_tasks=background_tasks)


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
