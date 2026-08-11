"""Launch endpoints for the interactive IB Gateway (TWS) login popup.

The public URL paths keep the historical ``/client-portal/*`` names for
Traefik/Dokploy routing continuity (see tws_launch_service docstring).
"""
from __future__ import annotations

from http import HTTPStatus
import logging
import secrets

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse, Response

from app.services.tws_launch_service import (
    get_tws_launch_cookie_name,
    get_tws_launch_cookie_ttl_seconds,
    get_tws_launch_session,
    normalize_launch_path_prefix,
    tws_path_prefix,
    tws_routing_base_url,
    validate_tws_launch_access,
)


logger = logging.getLogger(__name__)

router = APIRouter(include_in_schema=False)


@router.get("/client-portal/launch/{launch_token}")
async def start_tws_launch(launch_token: str, request: Request):
    launch_session = await get_tws_launch_session(launch_token)
    if launch_session is None:
        raise HTTPException(status_code=404, detail="Launch session not found or expired")

    try:
        connection_id = int(launch_session.get("connection_id"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=404, detail="Launch session not found or expired")

    prefix = normalize_launch_path_prefix(launch_session.get("path_prefix")) or tws_path_prefix(connection_id)
    # Cache-busting nonce so the browser can never serve a stale SPA shell
    # cached for /ib-access/* and instead hits the forwardAuth gate + twsgw.
    cache_bust = secrets.token_urlsafe(8)
    # Use an ABSOLUTE redirect URL (scheme + routing host): uvicorn runs with
    # --root-path /api, and a relative Location could inherit the /api prefix
    # and miss the Traefik twsgw router (which matches PathPrefix(/ib-access/<id>)).
    routing_base_url = tws_routing_base_url()
    path_with_cache_bust = f"{prefix}/?_cb={cache_bust}"
    redirect_url = (
        f"{routing_base_url}{path_with_cache_bust}"
        if routing_base_url
        else path_with_cache_bust
    )

    response = RedirectResponse(url=redirect_url, status_code=HTTPStatus.TEMPORARY_REDIRECT)
    # The launch URL is stable across auth-status polls; the browser must never
    # cache this 307 or it would replay a stale Location.
    response.headers["Cache-Control"] = "no-store"
    response.set_cookie(
        key=get_tws_launch_cookie_name(),
        value=launch_token,
        max_age=get_tws_launch_cookie_ttl_seconds(),
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
        path=prefix,
    )
    return response


@router.api_route(
    "/client-portal/access-check/{connection_id}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
)
async def tws_access_check(connection_id: int, request: Request):
    """forwardAuth gate invoked by Traefik before forwarding the browser to the
    per-connection IB Gateway (noVNC) container. Returns 200 only when the
    launch cookie maps to a live launch session owning *connection_id*.
    """
    launch_token = request.cookies.get(get_tws_launch_cookie_name(), "")
    if not await validate_tws_launch_access(launch_token, connection_id):
        raise HTTPException(status_code=HTTPStatus.UNAUTHORIZED, detail="Unauthorized")
    return Response(status_code=HTTPStatus.OK)
