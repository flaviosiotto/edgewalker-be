from datetime import timedelta
from typing import Annotated, Optional, Dict, Any, cast

from fastapi import APIRouter, Depends, HTTPException, status
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
from app.core.config import settings
from app.models.user import User

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
    responses={404: {"description": "Not found"}},
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
    session: Session = Depends(get_session),
):
    change_password(session, current_user, payload.current_password, payload.new_password)
    return {"message": "Password aggiornata con successo"}


@router.post("/password-reset/request", response_model=PasswordResetRequestResponse)
async def request_password_reset_endpoint(
    payload: PasswordResetRequest,
    session: Session = Depends(get_session),
):
    return request_password_reset(session, payload.identifier)


@router.post("/password-reset/confirm", response_model=MessageResponse)
async def confirm_password_reset_endpoint(
    payload: PasswordResetConfirmRequest,
    session: Session = Depends(get_session),
):
    confirm_password_reset(session, payload.token, payload.new_password)
    return {"message": "Password reimpostata con successo"}
