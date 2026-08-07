from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from app.db.database import get_session
from app.schemas.pat import PatCreate, PatCreated, PatRead
from app.services.pat_service import (
    list_personal_access_tokens,
    mint_personal_access_token,
    revoke_personal_access_token,
)
from app.utils.auth_utils import AuthPrincipal, get_current_active_principal

router = APIRouter(prefix="/pats", tags=["Personal Access Tokens"])


def _require_interactive_session(principal: AuthPrincipal) -> None:
    """Token management is reserved to a logged-in UI session.

    A PAT (or an n8n-issued token) must never be able to mint or revoke other
    PATs: that would turn any leaked token into a self-renewing credential.
    """
    if principal.claims.get("purpose") != "ui_auth":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Personal access tokens can only be managed from an interactive login session",
        )


@router.post("/", response_model=PatCreated, status_code=status.HTTP_201_CREATED)
def create_pat(
    payload: PatCreate,
    session: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_active_principal),
):
    """Create a personal access token. The raw token is returned once, here."""
    _require_interactive_session(principal)
    raw_token, pat = mint_personal_access_token(
        session,
        user=principal.user,
        name=payload.name,
        scopes=payload.scopes,
        expires_in_days=payload.expires_in_days,
    )
    return PatCreated(
        id=pat.id,
        name=pat.name,
        token_prefix=pat.token_prefix,
        scopes=pat.scopes,
        expires_at=pat.expires_at,
        last_used_at=pat.last_used_at,
        revoked_at=pat.revoked_at,
        created_at=pat.created_at,
        token=raw_token,
    )


@router.get("/", response_model=list[PatRead])
def list_pats(
    session: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_active_principal),
):
    _require_interactive_session(principal)
    return list_personal_access_tokens(session, principal.user.id)


@router.delete("/{pat_id}", response_model=PatRead)
def revoke_pat(
    pat_id: int,
    session: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_active_principal),
):
    """Revoke a token (soft delete: it stays listed with its revoked_at)."""
    _require_interactive_session(principal)
    return revoke_personal_access_token(session, principal.user.id, pat_id)
