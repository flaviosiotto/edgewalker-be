from __future__ import annotations

from http import HTTPStatus
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse, Response

from app.services.client_portal_launch_service import (
    get_client_portal_launch_cookie_name,
    get_client_portal_launch_cookie_ttl_seconds,
    get_client_portal_launch_session,
    is_client_portal_access_request,
    proxy_http_request,
    resolve_client_portal_proxy_target,
)


logger = logging.getLogger(__name__)

router = APIRouter(include_in_schema=False)

_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def _ensure_access_request(request: Request) -> None:
    if not is_client_portal_access_request(request):
        raise HTTPException(status_code=404, detail="Not found")


def _copy_response_headers(response: Response, upstream_headers, *, request: Request, upstream_base_url: str) -> None:
    for key, value in upstream_headers.multi_items():
        lower = key.lower()
        if lower in _HOP_BY_HOP_HEADERS or lower == "content-length":
            continue
        if lower == "location":
            value = _rewrite_location_header(value, request=request, upstream_base_url=upstream_base_url)
        response.headers.append(key, value)


def _rewrite_location_header(location: str, *, request: Request, upstream_base_url: str) -> str:
    if not location:
        return location
    if location.startswith("/"):
        return location

    if location.startswith(upstream_base_url):
        suffix = location[len(upstream_base_url):]
        if not suffix.startswith("/"):
            suffix = f"/{suffix}"
        return suffix

    return location


@router.get("/client-portal/launch/{launch_token}")
async def start_client_portal_launch(launch_token: str, request: Request):
    _ensure_access_request(request)

    launch_session = await get_client_portal_launch_session(launch_token)
    if launch_session is None:
        raise HTTPException(status_code=404, detail="Launch session not found or expired")

    response = RedirectResponse(url="/", status_code=HTTPStatus.TEMPORARY_REDIRECT)
    response.set_cookie(
        key=get_client_portal_launch_cookie_name(),
        value=launch_token,
        max_age=get_client_portal_launch_cookie_ttl_seconds(),
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
        path="/",
    )
    return response


@router.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
)
async def client_portal_access_proxy(path: str, request: Request):
    _ensure_access_request(request)

    upstream_base_url, verify_ssl = await resolve_client_portal_proxy_target(request)
    upstream_response = await proxy_http_request(
        request,
        upstream_base_url=upstream_base_url,
        verify_ssl=verify_ssl,
    )

    status_code = upstream_response.status_code
    if request.method.upper() == "HEAD" or 100 <= status_code < 200 or status_code in {204, 304}:
        response = Response(status_code=status_code)
    else:
        response = Response(content=upstream_response.content, status_code=status_code)

    _copy_response_headers(
        response,
        upstream_response.headers,
        request=request,
        upstream_base_url=upstream_base_url,
    )
    return response