from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel
from sqlmodel import Session

from app.core.config import settings
from app.db.database import get_session
from app.models.strategy import BacktestResult, BacktestStatus, Strategy, StrategyLive
from app.utils.auth_utils import AuthPrincipal, create_user_delegated_token, get_current_runner_principal

router = APIRouter(prefix="/runners", tags=["Runners"])

LIVE_RUNNER_CONTAINER_PREFIX = os.getenv("LIVE_RUNNER_CONTAINER_PREFIX", "edgewalker-live-")
BACKTEST_RUNNER_CONTAINER_PREFIX = os.getenv("BACKTEST_RUNNER_CONTAINER_PREFIX", "edgewalker-backtest-runner-")
RUNNER_PROXY_TIMEOUT_SECONDS = float(os.getenv("RUNNER_PROXY_TIMEOUT_SECONDS", "30"))
_STREAM_ID_RE = re.compile(r"^(strategy|backtest)-(\d+)$")
_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-encoding",
    "content-length",
}

_DEFAULT_AGENT_RUNNER_SCOPES = [
    "runner:read",
    "runner:orders:write",
    "runner:alerts:write",
    "runner:decision:write",
]
_DEFAULT_AGENT_CONSULTATIVE_SCOPES = [
    "accounts:read",
    "account_orders:read",
]


class AgentApiTokenResponse(BaseModel):
    token: str
    token_type: str = "Bearer"
    expires_at: datetime
    audience: str
    purpose: str
    scopes: list[str]
    backend_api: dict[str, Any] | None = None


def _runner_base_url_for_stream(stream_id: str) -> str:
    match = _STREAM_ID_RE.fullmatch(stream_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Runner not found")

    kind, raw_id = match.groups()
    if kind == "strategy":
        return f"http://{LIVE_RUNNER_CONTAINER_PREFIX}{raw_id}:8080"
    return f"http://{BACKTEST_RUNNER_CONTAINER_PREFIX}{raw_id}:8080"


def _proxy_headers(request: Request) -> dict[str, str]:
    return {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _HOP_BY_HOP_HEADERS
    }


def _response_headers(response: httpx.Response) -> dict[str, str]:
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() not in _HOP_BY_HOP_HEADERS
    }


@router.post("/agent-token", response_model=AgentApiTokenResponse)
def issue_runner_agent_token_endpoint(
    session: Session = Depends(get_session),
    runner_principal: AuthPrincipal = Depends(get_current_runner_principal),
):
    """Issue a scoped agent callback token for the active runner."""
    live_id = runner_principal.claims.get("live_id")
    backtest_id = runner_principal.claims.get("backtest_id")

    expires_at = datetime.now(timezone.utc) + timedelta(days=365)
    purpose = "agent_runner_callback"
    token_claims: dict[str, Any]
    backend_api: dict[str, Any] | None = None

    if live_id is not None:
        sl = session.get(StrategyLive, live_id)
        if sl is None:
            raise HTTPException(status_code=404, detail="Live session not found")
        strategy = session.get(Strategy, sl.strategy_id)
        if strategy is None or strategy.user_id != runner_principal.user.id:
            raise HTTPException(status_code=404, detail="Live session not found")

        token_claims = {
            "strategy_id": sl.strategy_id,
            "live_id": sl.id,
            "stream_id": f"strategy-{sl.id}",
            "connection_id": sl.connection_id,
            "account_id": sl.account_id,
            "scopes": _DEFAULT_AGENT_RUNNER_SCOPES,
        }

        consultative_purpose = "agent_backend_consult"
        consultative_token = create_user_delegated_token(
            session,
            user_id=runner_principal.user.id,
            audience=settings.AGENT_TOKEN_AUDIENCE,
            purpose=consultative_purpose,
            no_expiry=True,
            extra_claims={
                "strategy_id": sl.strategy_id,
                "live_id": sl.id,
                "connection_id": sl.connection_id,
                "account_id": sl.account_id,
                "scopes": _DEFAULT_AGENT_CONSULTATIVE_SCOPES,
            },
        )
        backend_api = {
            "token": consultative_token,
            "token_type": "Bearer",
            "expires_at": expires_at,
            "audience": settings.AGENT_TOKEN_AUDIENCE,
            "purpose": consultative_purpose,
            "scopes": list(_DEFAULT_AGENT_CONSULTATIVE_SCOPES),
            "account_id": sl.account_id,
            "connection_id": sl.connection_id,
        }
    elif backtest_id is not None:
        backtest = session.get(BacktestResult, backtest_id)
        if backtest is None:
            raise HTTPException(status_code=404, detail="Backtest not found")
        if backtest.status not in {BacktestStatus.PENDING.value, BacktestStatus.RUNNING.value}:
            raise HTTPException(status_code=403, detail="Backtest is no longer active")
        strategy = session.get(Strategy, backtest.strategy_id)
        if strategy is None or strategy.user_id != runner_principal.user.id:
            raise HTTPException(status_code=404, detail="Backtest not found")

        token_claims = {
            "strategy_id": backtest.strategy_id,
            "backtest_id": backtest.id,
            "stream_id": f"backtest-{backtest.id}",
            "connection_id": strategy.connection_id,
            "scopes": _DEFAULT_AGENT_RUNNER_SCOPES,
        }
    else:
        raise HTTPException(status_code=403, detail="Runner token is missing live or backtest binding")

    token = create_user_delegated_token(
        session,
        user_id=runner_principal.user.id,
        audience=settings.AGENT_TOKEN_AUDIENCE,
        purpose=purpose,
        no_expiry=True,
        extra_claims=token_claims,
    )
    return AgentApiTokenResponse(
        token=token,
        expires_at=expires_at,
        audience=settings.AGENT_TOKEN_AUDIENCE,
        purpose=purpose,
        scopes=list(_DEFAULT_AGENT_RUNNER_SCOPES),
        backend_api=backend_api,
    )


@router.api_route(
    "/{stream_id}/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
@router.api_route(
    "/{stream_id}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
async def proxy_runner_request(stream_id: str, request: Request, path: str = ""):
    """Proxy public runner API calls to the internal live/backtest runner."""
    base_url = _runner_base_url_for_stream(stream_id)
    target_path = path.strip("/")
    target_url = f"{base_url}/{target_path}" if target_path else base_url
    if request.url.query:
        target_url = f"{target_url}?{request.url.query}"

    try:
        async with httpx.AsyncClient(timeout=RUNNER_PROXY_TIMEOUT_SECONDS) as client:
            proxied = await client.request(
                request.method,
                target_url,
                content=await request.body(),
                headers=_proxy_headers(request),
            )
    except httpx.ConnectError as exc:
        raise HTTPException(status_code=404, detail="Runner not reachable") from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Runner request timed out") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Runner proxy failed: {exc}") from exc

    return Response(
        content=proxied.content,
        status_code=proxied.status_code,
        headers=_response_headers(proxied),
        media_type=proxied.headers.get("content-type"),
    )
