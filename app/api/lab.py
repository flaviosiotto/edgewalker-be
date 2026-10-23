"""Studio Lab (JupyterHub) launch endpoint.

Hands the FE a one-shot login URL for the embedded JupyterLab: a delegated
JWT with a dedicated audience/purpose and a TTL of a couple of minutes. The
token only has to survive the iframe's first redirect — after login the hub's
own session cookie takes over, so FE token rotation never matters here.
"""

import json
import logging
from datetime import timedelta
from urllib.parse import urlencode, urlparse

import httpx
from fastapi import APIRouter, Depends, HTTPException, status

from app.core.config import settings
from app.schemas.lab import (
    LabLaunch,
    LabLaunchRequest,
    LabSessionRequest,
    LabThemeRequest,
    LabWorkspaceFile,
    LabWorkspaceFileWrite,
)
from app.services import lab_hub
from app.services.lab_hub import LabSession
from app.utils.auth_utils import (
    AuthPrincipal,
    create_delegated_token,
    get_current_active_principal,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/lab", tags=["Studio Lab"])


def _hub_api_client() -> httpx.AsyncClient:
    """Client sull'API REST dell'hub (porta interna 8081) col token del
    service edgewalker-backend."""
    return httpx.AsyncClient(
        base_url=settings.LAB_HUB_API_URL.rstrip("/") + "/hub/api",
        headers={"Authorization": f"token {settings.LAB_HUB_API_TOKEN}"},
        timeout=10.0,
    )


async def _stop_user_server(uid: int) -> None:
    """Best-effort stop of the user's single-user server via the hub API.

    The respawn that follows re-runs the workspace sync hook, so freshly
    created studios appear in ~/work/studios. Requires LAB_HUB_API_TOKEN
    (service `edgewalker-backend` in jupyterhub_config); silently skipped
    when unset.
    """
    if not settings.LAB_HUB_API_TOKEN:
        return
    try:
        async with _hub_api_client() as client:
            await lab_hub.stop_server(client, uid)
    except httpx.HTTPError as exc:
        logger.warning("lab stop: hub API unreachable (%s)", exc)


def _studio_token(principal: AuthPrincipal) -> str:
    # Credenziale di sessione per il container Lab: audience dedicata che
    # solo studio-svc accetta, così il codice dei notebook può salvare
    # versioni/lanciare run dei PROPRI Studi ma non toccare il resto
    # dell'API (niente trading). L'hub la inietta nell'env del container
    # via auth_state al momento dello spawn.
    return create_delegated_token(
        data={"sub": principal.user.email, "uid": principal.user.id},
        audience=settings.STUDIO_TOKEN_AUDIENCE,
        purpose="lab_api",
        expires_delta=timedelta(minutes=settings.LAB_STUDIO_TOKEN_EXPIRE_MINUTES),
    )


@router.get("/session", response_model=LabSession)
async def get_lab_session(principal: AuthPrincipal = Depends(get_current_active_principal)):
    """State of the user's Lab session (ready / pending / stopping / stopped).

    The page polls it while the server starts and, in edit mode, to notice a
    session culled for inactivity (overlay "resume" instead of a hub page).
    """
    _ui_only(principal)
    _require_hub_token()
    try:
        async with _hub_api_client() as client:
            return await lab_hub.fetch_session(client, principal.user.id)
    except httpx.HTTPError as exc:
        logger.warning("lab session state: hub API unreachable (%s)", exc)
        return LabSession(state="unavailable", message=f"hub unreachable: {exc}")


@router.post("/session", response_model=LabSession)
async def create_lab_session(
    payload: LabSessionRequest,
    principal: AuthPrincipal = Depends(get_current_active_principal),
):
    """Spawn the user's Lab server through the hub API (idempotent: a
    running server is left alone).

    The hub user is created on first use (same username = uid as the JWT
    login), the auth_state carries what the spawn hooks need (studio token,
    theme), then the server is started. The FE polls GET /lab/session and
    mounts the iframe (POST /lab/launch) only when the state is ready, so the
    hub's own "not running" / "spawn pending" pages never show up.
    """
    _ui_only(principal)
    _require_hub_token()
    uid = principal.user.id
    auth_state = {
        "uid": uid,
        "email": principal.user.email,
        "studio_token": _studio_token(principal),
        "lab_theme": payload.theme,
    }
    try:
        async with _hub_api_client() as client:
            await lab_hub.ensure_user(client, uid)
            current = await lab_hub.fetch_session(client, uid)
            if current.state in ("ready", "pending"):
                return current
            if current.state == "stopping":
                # Lo stop appena chiesto (Riavvia sessione) deve finire prima.
                await lab_hub.stop_server(client, uid)
            await lab_hub.set_auth_state(client, uid, auth_state)
            return await lab_hub.start_server(client, uid)
    except httpx.HTTPError as exc:
        logger.warning("lab session start: hub API error (%s)", exc)
        return LabSession(state="unavailable", message=f"hub unreachable: {exc}")


@router.delete("/session", status_code=204)
async def stop_lab_session(principal: AuthPrincipal = Depends(get_current_active_principal)):
    """Stop the user's Lab server ("restart session": the respawn re-runs
    the workspace sync hook)."""
    _ui_only(principal)
    _require_hub_token()
    await _stop_user_server(principal.user.id)


def _require_hub_token() -> None:
    if not settings.LAB_HUB_API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="LAB_HUB_API_TOKEN not configured",
        )


def _user_server_base(uid: int) -> str:
    # Attraverso il proxy dell'hub (porta 8000): il token di service con
    # scope access:servers viene validato dal server utente contro l'hub.
    return f"{settings.LAB_HUB_PROXY_URL.rstrip('/')}/user/{uid}"


def _hub_auth_headers() -> dict[str, str]:
    return {"Authorization": f"token {settings.LAB_HUB_API_TOKEN}"}


def _ui_only(principal: AuthPrincipal) -> None:
    if principal.claims.get("purpose") != "ui_auth":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="UI sessions only")


def _check_path(path: str) -> None:
    if not path or path.startswith("/") or ".." in path or "?" in path or "#" in path:
        raise HTTPException(status_code=422, detail="path must be a plain relative path")


def _contents_url(uid: int, path: str) -> str:
    return f"{_user_server_base(uid)}/api/contents/{path}"


def _lab_unreachable(exc: httpx.HTTPError) -> HTTPException:
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Lab session unreachable: {exc}")


_NOT_RUNNING = HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Lab session not running")


@router.get("/workspace-file", response_model=LabWorkspaceFile)
async def read_workspace_file(
    path: str,
    principal: AuthPrincipal = Depends(get_current_active_principal),
):
    """Read a file from the user's running Lab workspace (raw text).

    The page snapshots the notebook being edited into a studio version
    (Save version, Run now, leaving edit mode). 409 when the Lab session is
    not running.
    """
    _ui_only(principal)
    _check_path(path)
    _require_hub_token()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                _contents_url(principal.user.id, path),
                params={"content": "1", "type": "file", "format": "text"},
                headers=_hub_auth_headers(),
            )
    except httpx.HTTPError as exc:
        raise _lab_unreachable(exc) from None
    if response.status_code == 404:
        raise HTTPException(status_code=404, detail="File not found in the Lab workspace")
    if response.status_code != 200:
        # Server spento: l'hub risponde con la pagina 424/503, non col file.
        raise _NOT_RUNNING
    data = response.json()
    return LabWorkspaceFile(
        path=path,
        content=data.get("content") or "",
        last_modified=data.get("last_modified"),
    )


@router.put("/workspace-file", response_model=LabWorkspaceFile)
async def write_workspace_file(
    payload: LabWorkspaceFileWrite,
    principal: AuthPrincipal = Depends(get_current_active_principal),
):
    """Write a text file into the user's running Lab workspace.

    The studio versions (studio-svc) are the truth and the workspace follows:
    entering edit mode the page writes the current version when it is newer
    than the file (a version saved by an upload, by MCP or by an agent while
    the Lab was off). 409 when the Lab session is not running.
    """
    _ui_only(principal)
    _check_path(payload.path)
    _require_hub_token()
    body = {"type": "file", "format": "text", "content": payload.content}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.put(
                _contents_url(principal.user.id, payload.path), json=body, headers=_hub_auth_headers(),
            )
    except httpx.HTTPError as exc:
        raise _lab_unreachable(exc) from None
    if response.status_code not in (200, 201):
        raise _NOT_RUNNING
    data = response.json()
    return LabWorkspaceFile(path=payload.path, content=payload.content, last_modified=data.get("last_modified"))


@router.delete("/workspace-file", status_code=204)
async def delete_workspace_file(
    path: str,
    principal: AuthPrincipal = Depends(get_current_active_principal),
):
    """Remove a file from the user's running Lab workspace (deleted Studio).

    Best-effort by design: 204 also when the file is already gone; 409 when
    the session is not running (the spawn hook removes orphans anyway).
    """
    _ui_only(principal)
    _check_path(path)
    _require_hub_token()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.delete(_contents_url(principal.user.id, path), headers=_hub_auth_headers())
    except httpx.HTTPError as exc:
        raise _lab_unreachable(exc) from None
    if response.status_code not in (204, 404):
        raise _NOT_RUNNING


@router.post("/theme", status_code=204)
async def set_lab_theme(
    payload: LabThemeRequest,
    principal: AuthPrincipal = Depends(get_current_active_principal),
):
    """Align the running Jupyter session's theme with the app theme.

    Best-effort by design: a stopped session simply returns 204 (the spawn
    hook applies the theme claim at the next start).
    """
    _ui_only(principal)
    _require_hub_token()
    jupyter_theme = "JupyterLab Light" if payload.theme == "light" else "JupyterLab Dark"
    url = (f"{_user_server_base(principal.user.id)}"
           "/lab/api/settings/@jupyterlab/apputils-extension:themes")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.put(
                url,
                json={"raw": json.dumps({"theme": jupyter_theme, "adaptive-theme": False})},
                headers=_hub_auth_headers(),
            )
    except httpx.HTTPError as exc:
        logger.warning("lab theme update skipped (%s)", exc)


@router.post("/launch", response_model=LabLaunch)
async def launch_lab(
    payload: LabLaunchRequest,
    principal: AuthPrincipal = Depends(get_current_active_principal),
):
    """Mint a short-lived launch URL for the current user's Lab session, landing
    on one notebook (``next_path`` is required: the file browser is not a
    surface the UI exposes).

    Interactive UI sessions only: a PAT (or any delegated token) must never
    open a Lab, where arbitrary user code runs with the user's identity.
    """
    if principal.claims.get("purpose") != "ui_auth":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The Studio Lab can only be launched from an interactive login session",
        )
    # Lo stesso studio_token finisce nell'auth_state anche da POST /lab/session:
    # il login lo riscrive (identico scopo), così lo spawn ha sempre una
    # credenziale fresca qualunque sia la strada.
    studio_token = _studio_token(principal)
    token = create_delegated_token(
        data={
            "sub": principal.user.email,
            "uid": principal.user.id,
            "role": principal.user.role,
            "studio_token": studio_token,
            # Tema dell'app al lancio: l'hook di spawn imposta il tema
            # Jupyter corrispondente.
            "lab_theme": payload.theme,
        },
        audience=settings.LAB_TOKEN_AUDIENCE,
        purpose="lab_launch",
        expires_delta=timedelta(minutes=settings.LAB_LAUNCH_TOKEN_EXPIRE_MINUTES),
    )
    if payload.fresh:
        await _stop_user_server(principal.user.id)

    base = settings.LAB_PUBLIC_URL.rstrip("/")
    # Atterraggio diretto sul file del workspace (UI Notebook): next deve
    # essere un path assoluto sotto il prefisso dell'hub.
    prefix = urlparse(base).path or ""
    query = {"token": token, "next": f"{prefix}/user/{principal.user.id}/{payload.next_path}"}
    return LabLaunch(lab_url=f"{base}/hub/login?{urlencode(query)}")
