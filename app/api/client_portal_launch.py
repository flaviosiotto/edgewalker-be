from __future__ import annotations

from http import HTTPStatus
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse, Response

from app.services.connection_manager import get_connection_manager
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

_CLIENT_PORTAL_LOGIN_PATH = "/sso/Login?forwardTo=22&RL=1&ip2loc=US"
_DISPATCHER_MESSAGE_MARKER = "edgewalker:client-portal-dispatcher-submit"
_DISPATCHER_BRIDGE_SCRIPT = (
    "<script>(function(){"
    "try{if(window.opener&&!window.opener.closed){window.opener.postMessage({type:'edgewalker:client-portal-dispatcher-submit'},'*');}}catch(error){}"
    "})();</script>"
)

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


def _copy_response_headers(
    response: Response,
    upstream_headers,
    *,
    request: Request,
    upstream_base_url: str,
    excluded_headers: set[str] | None = None,
) -> None:
    excluded = {header.lower() for header in (excluded_headers or set())}
    for key, value in upstream_headers.multi_items():
        lower = key.lower()
        if lower in _HOP_BY_HOP_HEADERS or lower == "content-length" or lower in excluded:
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


def _is_dispatcher_success_response(
    request: Request,
    status_code: int,
    content_type: str | None,
    body: bytes,
) -> bool:
    path = (str(request.scope.get("path") or "/").rstrip("/") or "/")
    if path != "/sso/Dispatcher":
        return False
    if request.method.upper() == "HEAD" or status_code >= 400:
        return False

    normalized_content_type = (content_type or "").lower()
    if normalized_content_type and "text/html" not in normalized_content_type and "text/plain" not in normalized_content_type:
        return False

    normalized_body = body.lower()
    if b"client login succeeds" in normalized_body:
        return True

    # The unauthenticated login shell can still render on /sso/Dispatcher with a
    # 200 response; do not treat that page as a completed authorization.
    if (
        b"jquery-3.7.0/jquery.min.js" in normalized_body
        or b"forge-1.3.1.all.min.js" in normalized_body
        or b"js.cookie.min.js" in normalized_body
    ):
        return False

    return request.method.upper() == "POST"


async def _mark_dispatcher_received_from_launch(launch_token: str) -> None:
    launch_session = await get_client_portal_launch_session(launch_token)
    if launch_session is None:
        return

    try:
        connection_id = int(launch_session.get("connection_id"))
    except (TypeError, ValueError):
        return

    try:
        await get_connection_manager().mark_client_portal_dispatcher_received(connection_id)
    except Exception as exc:
        logger.warning("Failed to persist Client Portal Dispatcher for connection %s: %s", connection_id, exc)


def _dispatcher_bridge_response_content(body: bytes, content_type: str | None) -> str:
    normalized_content_type = (content_type or "").lower()

    if _DISPATCHER_MESSAGE_MARKER.encode("utf-8") in body:
        try:
            return body.decode("utf-8")
        except UnicodeDecodeError:
            return (
                "<!DOCTYPE html>"
                "<html><head><meta charset='utf-8'><title>Interactive Brokers</title>"
                "<meta name='viewport' content='width=device-width, initial-scale=1'></head>"
                f"<body>{_DISPATCHER_BRIDGE_SCRIPT}Authorization received. Edgewalker is waiting for the brokerage session.</body></html>"
            )

    if "text/html" not in normalized_content_type:
        return (
            "<!DOCTYPE html>"
            "<html><head><meta charset='utf-8'><title>Interactive Brokers</title>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'></head>"
            f"<body>{_DISPATCHER_BRIDGE_SCRIPT}Authorization received. Edgewalker is waiting for the brokerage session.</body></html>"
        )

    try:
        html = body.decode("utf-8")
    except UnicodeDecodeError:
        return (
            "<!DOCTYPE html>"
            "<html><head><meta charset='utf-8'><title>Interactive Brokers</title>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'></head>"
            f"<body>{_DISPATCHER_BRIDGE_SCRIPT}Authorization received. Edgewalker is waiting for the brokerage session.</body></html>"
        )

    lowered_html = html.lower()
    body_close_index = lowered_html.rfind("</body>")
    html_close_index = lowered_html.rfind("</html>")

    if body_close_index != -1:
        return f"{html[:body_close_index]}{_DISPATCHER_BRIDGE_SCRIPT}{html[body_close_index:]}"

    if html_close_index != -1:
        return f"{html[:html_close_index]}{_DISPATCHER_BRIDGE_SCRIPT}{html[html_close_index:]}"

    return f"{html}{_DISPATCHER_BRIDGE_SCRIPT}"


@router.get("/client-portal/launch/{launch_token}")
async def start_client_portal_launch(launch_token: str, request: Request):
    _ensure_access_request(request)

    launch_session = await get_client_portal_launch_session(launch_token)
    if launch_session is None:
        raise HTTPException(status_code=404, detail="Launch session not found or expired")

    response = RedirectResponse(url=_CLIENT_PORTAL_LOGIN_PATH, status_code=HTTPStatus.TEMPORARY_REDIRECT)
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


async def _proxy_client_portal_access(request: Request):
    launch_token, upstream_base_url, verify_ssl = await resolve_client_portal_proxy_target(request)
    upstream_response = await proxy_http_request(
        request,
        launch_token=launch_token,
        upstream_base_url=upstream_base_url,
        verify_ssl=verify_ssl,
    )

    if _is_dispatcher_success_response(
        request,
        upstream_response.status_code,
        upstream_response.headers.get("content-type"),
        upstream_response.content,
    ):
        await _mark_dispatcher_received_from_launch(launch_token)
        response = Response(
            content=_dispatcher_bridge_response_content(
                upstream_response.content,
                upstream_response.headers.get("content-type"),
            ),
            status_code=HTTPStatus.OK,
            media_type="text/html",
            headers={
                "Cache-Control": "no-store",
                "Pragma": "no-cache",
            },
        )
        _copy_response_headers(
            response,
            upstream_response.headers,
            request=request,
            upstream_base_url=upstream_base_url,
            excluded_headers={"content-type", "content-disposition", "cache-control", "pragma"},
        )
        return response

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


@router.api_route(
    "/",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
)
async def client_portal_access_proxy_root(request: Request):
    _ensure_access_request(request)
    return await _proxy_client_portal_access(request)


@router.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
)
async def client_portal_access_proxy(path: str, request: Request):
    _ensure_access_request(request)
    return await _proxy_client_portal_access(request)