"""Studio Lab (JupyterHub) launch endpoint.

Hands the FE a one-shot login URL for the embedded JupyterLab: a delegated
JWT with a dedicated audience/purpose and a TTL of a couple of minutes. The
token only has to survive the iframe's first redirect — after login the hub's
own session cookie takes over, so FE token rotation never matters here.
"""

import asyncio
import logging
from datetime import timedelta
from urllib.parse import urlencode, urlparse

import httpx
from fastapi import APIRouter, Depends, HTTPException, status

from app.core.config import settings
from app.schemas.lab import LabLaunch, LabLaunchRequest
from app.utils.auth_utils import (
    AuthPrincipal,
    create_delegated_token,
    get_current_active_principal,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/lab", tags=["Studio Lab"])


async def _stop_user_server(uid: int) -> None:
    """Best-effort stop of the user's single-user server via the hub API.

    The respawn that follows re-runs the workspace sync hook, so freshly
    created studios appear in ~/work/studios. Requires LAB_HUB_API_TOKEN
    (service `edgewalker-backend` in jupyterhub_config); silently skipped
    when unset.
    """
    if not settings.LAB_HUB_API_TOKEN:
        return
    base = settings.LAB_HUB_API_URL.rstrip("/") + "/hub/api"
    headers = {"Authorization": f"token {settings.LAB_HUB_API_TOKEN}"}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.delete(f"{base}/users/{uid}/server", headers=headers)
            if response.status_code not in (202, 204, 400, 404):
                response.raise_for_status()
            # 202 = stop avviato: aspetta (poco) che il server sparisca, così
            # il login successivo spawna subito invece di fallire su "pending".
            if response.status_code == 202:
                for _ in range(8):
                    await asyncio.sleep(1)
                    user = await client.get(f"{base}/users/{uid}", headers=headers)
                    if user.status_code != 200 or not user.json().get("server"):
                        break
    except httpx.HTTPError as exc:
        logger.warning("lab fresh restart: hub API unreachable (%s)", exc)


@router.post("/launch", response_model=LabLaunch)
async def launch_lab(
    payload: LabLaunchRequest | None = None,
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
    # Credenziale di sessione per il container Lab: audience dedicata che
    # solo studio-svc accetta, così il codice dei notebook può salvare
    # versioni/lanciare run dei PROPRI Studi ma non toccare il resto
    # dell'API (niente trading). L'hub la inietta nell'env del container
    # via auth_state al momento dello spawn.
    studio_token = create_delegated_token(
        data={"sub": principal.user.email, "uid": principal.user.id},
        audience=settings.STUDIO_TOKEN_AUDIENCE,
        purpose="lab_api",
        expires_delta=timedelta(minutes=settings.LAB_STUDIO_TOKEN_EXPIRE_MINUTES),
    )
    token = create_delegated_token(
        data={
            "sub": principal.user.email,
            "uid": principal.user.id,
            "role": principal.user.role,
            "studio_token": studio_token,
        },
        audience=settings.LAB_TOKEN_AUDIENCE,
        purpose="lab_launch",
        expires_delta=timedelta(minutes=settings.LAB_LAUNCH_TOKEN_EXPIRE_MINUTES),
    )
    if payload and payload.fresh:
        await _stop_user_server(principal.user.id)

    base = settings.LAB_PUBLIC_URL.rstrip("/")
    query: dict[str, str] = {"token": token}
    if payload and payload.next_path:
        # Atterraggio diretto su un file del workspace (UI Notebook):
        # next deve essere un path assoluto sotto il prefisso dell'hub.
        prefix = urlparse(base).path or ""
        query["next"] = f"{prefix}/user/{principal.user.id}/{payload.next_path}"
    return LabLaunch(lab_url=f"{base}/hub/login?{urlencode(query)}")
