from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session

from app.core.config import settings
from app.db.database import get_session
from app.models.strategy import BacktestResult, BacktestStatus, Strategy, StrategyLive
from app.utils.auth_utils import AuthPrincipal, create_user_delegated_token, get_current_runner_principal

router = APIRouter(prefix="/runners", tags=["Runners"])

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