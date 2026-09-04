"""Studio access token: PAT (or UI session) -> credential for studio-svc.

studio-svc only verifies JWTs (audiences ``edgewalker-ui`` /
``edgewalker-studio``); personal access tokens are opaque strings the
backend alone can resolve. Machine clients that hold a PAT — the MCP server
above all — exchange it here for a short-lived delegated token that
studio-svc accepts, carrying the PAT scopes so studio-svc can still tell a
read-only token from a write one. Nothing is stored: the token dies on its
own after ``STUDIO_ACCESS_TOKEN_EXPIRE_MINUTES``.

Same mechanism as the ``studio_token`` minted by ``/lab/launch`` for the
Lab container (audience ``edgewalker-studio``), with a different purpose so
studio-svc can keep the secret-value endpoint for the Lab only.
"""

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.config import settings
from app.schemas.lab import StudioAccessToken
from app.utils.auth_utils import (
    AuthPrincipal,
    create_delegated_token,
    get_current_active_principal,
)

router = APIRouter(prefix="/users/me", tags=["Studios"])

_ALL_SCOPES = ["read", "write", "trade"]


@router.get("/studio-token", response_model=StudioAccessToken)
async def mint_studio_access_token(
    principal: AuthPrincipal = Depends(get_current_active_principal),
):
    """Mint a short-lived studio-svc credential for the caller (scope: read).

    Accepted callers: personal access tokens (the MCP server) and interactive
    UI sessions. Other delegated tokens (n8n, runners) are refused: they
    never need to touch the Studi.
    """
    purpose = principal.claims.get("purpose")
    if purpose == "pat_access":
        scopes = list(principal.claims.get("scopes") or [])
    elif purpose == "ui_auth":
        scopes = _ALL_SCOPES
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Studio access tokens are only issued to personal access tokens or UI sessions",
        )
    expires = timedelta(minutes=settings.STUDIO_ACCESS_TOKEN_EXPIRE_MINUTES)
    token = create_delegated_token(
        data={"sub": principal.user.email, "uid": principal.user.id, "scopes": scopes},
        audience=settings.STUDIO_TOKEN_AUDIENCE,
        purpose="studio_api",
        expires_delta=expires,
    )
    return StudioAccessToken(
        access_token=token,
        expires_in=int(expires.total_seconds()),
        scopes=scopes,
    )
