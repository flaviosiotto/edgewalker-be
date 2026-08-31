"""Studio Lab (JupyterHub) launch endpoint.

Hands the FE a one-shot login URL for the embedded JupyterLab: a delegated
JWT with a dedicated audience/purpose and a TTL of a couple of minutes. The
token only has to survive the iframe's first redirect — after login the hub's
own session cookie takes over, so FE token rotation never matters here.
"""

from datetime import timedelta
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.config import settings
from app.schemas.lab import LabLaunch
from app.utils.auth_utils import (
    AuthPrincipal,
    create_delegated_token,
    get_current_active_principal,
)

router = APIRouter(prefix="/lab", tags=["Studio Lab"])


@router.post("/launch", response_model=LabLaunch)
def launch_lab(
    principal: AuthPrincipal = Depends(get_current_active_principal),
):
    """Mint a short-lived launch URL for the current user's Lab session.

    Interactive UI sessions only: a PAT (or any delegated token) must never
    open a Lab, where arbitrary user code runs with the user's identity.
    """
    if principal.claims.get("purpose") != "ui_auth":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The Studio Lab can only be launched from an interactive login session",
        )
    token = create_delegated_token(
        data={
            "sub": principal.user.email,
            "uid": principal.user.id,
            "role": principal.user.role,
        },
        audience=settings.LAB_TOKEN_AUDIENCE,
        purpose="lab_launch",
        expires_delta=timedelta(minutes=settings.LAB_LAUNCH_TOKEN_EXPIRE_MINUTES),
    )
    base = settings.LAB_PUBLIC_URL.rstrip("/")
    return LabLaunch(lab_url=f"{base}/hub/login?{urlencode({'token': token})}")
