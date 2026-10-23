"""Client minimo dell'API REST di JupyterHub per la sessione Lab dell'utente.

Il backend pilota lo spawn al posto delle pagine dell'hub ("Server not
running" / "spawn pending"): il FE monta l'iframe solo quando il server e
pronto, cosi l'utente non vede mai il chrome di JupyterHub. Tutte le chiamate
usano il token del service ``edgewalker-backend`` (jupyterhub_config: scope
``admin:users`` per creare l'utente hub e scrivere l'auth_state, ``servers``
per avviare, ``delete:servers`` per fermare, ``read:users`` per lo stato).

Funzioni pure rispetto a FastAPI: ricevono un ``httpx.AsyncClient`` gia
puntato su ``<hub>/hub/api`` (testabili con ``httpx.MockTransport``).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal, Optional

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)

SessionState = Literal["ready", "pending", "stopping", "stopped", "failed", "unavailable"]


class LabSession(BaseModel):
    """Stato della sessione Lab dell'utente, come lo legge il FE."""

    state: SessionState
    # Messaggio per l'utente quando state = failed | unavailable.
    message: Optional[str] = None


def _server_entry(user: dict[str, Any]) -> Optional[dict[str, Any]]:
    servers = user.get("servers")
    if isinstance(servers, dict) and "" in servers:
        return servers[""]
    return None


def session_from_user(user: dict[str, Any]) -> LabSession:
    """Traduce il modello utente dell'hub (GET /users/{name}) nello stato.

    ``servers`` contiene solo i server attivi: assente = spento. Un server con
    ``pending`` = "spawn" sta partendo, "stop" si sta fermando.
    """
    server = _server_entry(user)
    if server is None:
        return LabSession(state="stopped")
    pending = server.get("pending")
    if pending == "spawn":
        return LabSession(state="pending")
    if pending == "stop":
        return LabSession(state="stopping")
    if server.get("ready"):
        return LabSession(state="ready")
    return LabSession(state="pending")


async def fetch_session(client: httpx.AsyncClient, uid: int) -> LabSession:
    """Stato corrente; utente hub mai visto = spento."""
    response = await client.get(f"/users/{uid}")
    if response.status_code == 404:
        return LabSession(state="stopped")
    response.raise_for_status()
    return session_from_user(response.json())


async def ensure_user(client: httpx.AsyncClient, uid: int) -> None:
    """Crea l'utente hub se manca (username = uid, come l'authenticator)."""
    response = await client.get(f"/users/{uid}")
    if response.status_code == 200:
        return
    if response.status_code != 404:
        response.raise_for_status()
    created = await client.post(f"/users/{uid}", json={})
    if created.status_code not in (201, 409):
        created.raise_for_status()


async def set_auth_state(client: httpx.AsyncClient, uid: int, auth_state: dict[str, Any]) -> None:
    """Scrive l'auth_state che ``pre_spawn_start`` inietta nel container
    (studio_token, tema): lo stesso che il login con JWT salverebbe."""
    response = await client.patch(f"/users/{uid}", json={"auth_state": auth_state})
    response.raise_for_status()


async def start_server(client: httpx.AsyncClient, uid: int) -> LabSession:
    """Avvia il server se non gira: 201 pronto, 202 in avvio, 400 gia
    attivo/pending (si rilegge lo stato), 5xx = spawn fallito subito."""
    response = await client.post(f"/users/{uid}/server", json={})
    if response.status_code == 201:
        return LabSession(state="ready")
    if response.status_code == 202:
        return LabSession(state="pending")
    if response.status_code == 400:
        return await fetch_session(client, uid)
    detail = _error_detail(response)
    logger.warning("lab spawn failed for uid %s: %s %s", uid, response.status_code, detail)
    return LabSession(state="failed", message=detail or f"hub {response.status_code}")


async def stop_server(client: httpx.AsyncClient, uid: int, *, wait_seconds: int = 8) -> None:
    """Ferma il server (best-effort) e aspetta poco che sparisca, cosi un
    avvio immediatamente dopo non fallisce su "pending stop"."""
    response = await client.delete(f"/users/{uid}/server")
    if response.status_code not in (202, 204, 400, 404):
        response.raise_for_status()
    if response.status_code != 202:
        return
    for _ in range(max(wait_seconds, 0)):
        await asyncio.sleep(1)
        if (await fetch_session(client, uid)).state == "stopped":
            return


def _error_detail(response: httpx.Response) -> Optional[str]:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200] or None
    if isinstance(body, dict):
        return str(body.get("message") or body.get("detail") or "")[:200] or None
    return None
