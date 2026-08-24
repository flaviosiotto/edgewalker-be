import asyncio
import json
import logging
from collections.abc import AsyncIterator
from fastapi import APIRouter, Depends, Header, Request, status
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from app.db.database import get_session
from app.models.user import User
from app.schemas.strategy import (
    StrategyCreate,
    StrategyRead,
    StrategyUpdate,
    RuleTriggerRequest,
    RuleTriggerResponse,
    LayoutConfigUpdate,
)
from app.schemas.chat import ChatCreate, ChatRead
from app.services.live_summary_service import build_live_summary
from app.services.strategy_service import (
    create_strategy,
    delete_strategy,
    get_strategy,
    list_strategies,
    update_strategy,
    list_strategy_chats,
    create_strategy_chat,
    get_strategy_chat,
    update_strategy_chat,
    delete_strategy_chat,
    trigger_rule_agent,
    update_strategy_layout,
)
from app.utils.auth_utils import (
    AuthPrincipal,
    get_current_active_principal,
    get_current_active_user,
)
from app.utils.redis_async import close_pubsub, get_async_redis
from app.utils.redis_client import get_redis

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/strategies", tags=["Strategies"])

# Redis channel used to notify FE clients that a strategy definition changed
# (e.g. the design agent's update_strategy tool). Relayed via the SSE endpoint
# GET /strategies/{id}/events. Best-effort: Redis down = no live notification.
STRATEGY_EVENTS_CHANNEL_PREFIX = "strategy-events:"
STRATEGY_EVENT_HEARTBEAT_SECONDS = 15.0


def _publish_strategy_updated(strategy, *, origin: str, request_id: str | None) -> None:
    client = get_redis()
    if client is None:
        return
    payload = {
        "v": 1,
        "type": "strategy_updated",
        "strategy_id": strategy.id,
        "updated_at": strategy.updated_at.isoformat() if strategy.updated_at else None,
        "origin": origin,
        "request_id": request_id,
    }
    try:
        client.publish(
            f"{STRATEGY_EVENTS_CHANNEL_PREFIX}{strategy.id}",
            json.dumps(payload, default=str),
        )
    except Exception as exc:  # noqa: BLE001 - never fail the PATCH on Redis errors
        logger.debug("strategy-events publish failed (non-critical): %s", exc)


def _serialize_strategy_with_live(session: Session, strategy) -> StrategyRead:
    """Build a StrategyRead and attach `live_summary` if a live session exists."""
    payload = StrategyRead.model_validate(strategy)
    sl = strategy.live
    if sl is not None:
        payload.live_summary = build_live_summary(session, sl)
    return payload


@router.post("/", response_model=StrategyRead)
def create_strategy_endpoint(
    payload: StrategyCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    strategy = create_strategy(session, payload, current_user.id)
    return _serialize_strategy_with_live(session, strategy)


@router.get("/", response_model=list[StrategyRead])
def list_strategies_endpoint(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    strategies = list_strategies(session, current_user.id)
    return [_serialize_strategy_with_live(session, s) for s in strategies]


@router.get("/{strategy_id}", response_model=StrategyRead)
def get_strategy_endpoint(
    strategy_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    strategy = get_strategy(session, strategy_id, current_user.id)
    return _serialize_strategy_with_live(session, strategy)


@router.patch("/{strategy_id}", response_model=StrategyRead)
def update_strategy_endpoint(
    strategy_id: int,
    payload: StrategyUpdate,
    session: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_active_principal),
    x_client_request_id: str | None = Header(default=None, alias="X-Client-Request-Id"),
):
    strategy = update_strategy(session, strategy_id, payload, principal.user.id)
    # FE tokens carry purpose=ui_auth; everything else (n8n_chat_api_access &
    # co.) is an agent-side writer. The FE uses `origin` to ignore its own
    # autosave echoes and reload only on agent edits.
    origin = "user" if principal.claims.get("purpose") == "ui_auth" else "agent"
    _publish_strategy_updated(strategy, origin=origin, request_id=x_client_request_id)
    return _serialize_strategy_with_live(session, strategy)


@router.get("/{strategy_id}/events")
async def strategy_events_endpoint(
    strategy_id: int,
    request: Request,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """SSE stream of ``strategy_updated`` events for a strategy.

    Relays the Redis channel ``strategy-events:{id}``; degrades to
    heartbeat-only when Redis is unavailable.
    """
    # Ownership check (404/403 like the other strategy endpoints).
    get_strategy(session, strategy_id, current_user.id)

    async def event_stream() -> AsyncIterator[str]:
        pubsub = None
        try:
            redis_client = get_async_redis()
            if redis_client is not None:
                try:
                    pubsub = redis_client.pubsub()
                    await pubsub.subscribe(f"{STRATEGY_EVENTS_CHANNEL_PREFIX}{strategy_id}")
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "strategy %s: events relay unavailable: %s", strategy_id, exc
                    )
                    if pubsub is not None:
                        await close_pubsub(pubsub)
                        pubsub = None

            yield f"event: ready\ndata: {json.dumps({'strategy_id': strategy_id})}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                if pubsub is None:
                    await asyncio.sleep(STRATEGY_EVENT_HEARTBEAT_SECONDS)
                    yield ": ping\n\n"
                    continue
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=STRATEGY_EVENT_HEARTBEAT_SECONDS,
                )
                if message is None:
                    yield ": ping\n\n"
                    continue
                if message.get("type") != "message":
                    continue
                try:
                    payload = json.loads(message.get("data") or "")
                except (TypeError, ValueError):
                    continue
                if payload.get("type") != "strategy_updated":
                    continue
                yield f"event: strategy_updated\ndata: {json.dumps(payload)}\n\n"
        finally:
            if pubsub is not None:
                await close_pubsub(pubsub)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.delete("/{strategy_id}")
def delete_strategy_endpoint(
    strategy_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    delete_strategy(session, strategy_id, current_user.id)
    return {"status": "ok"}


@router.patch("/{strategy_id}/layout", response_model=StrategyRead)
def update_strategy_layout_endpoint(
    strategy_id: int,
    payload: LayoutConfigUpdate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Update only the UI layout configuration for a strategy."""
    strategy = update_strategy_layout(session, strategy_id, payload, current_user.id)
    return _serialize_strategy_with_live(session, strategy)


# ─── CHAT ENDPOINTS ───


@router.get("/{strategy_id}/chats", response_model=list[ChatRead])
def list_strategy_chats_endpoint(
    strategy_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """List all chats for a strategy."""
    return list_strategy_chats(session, strategy_id, current_user.id)


@router.post("/{strategy_id}/chats", response_model=ChatRead)
def create_strategy_chat_endpoint(
    strategy_id: int,
    payload: ChatCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Create a new chat for a strategy."""
    return create_strategy_chat(session, strategy_id, payload, current_user.id)


@router.get("/{strategy_id}/chats/{chat_id}", response_model=ChatRead)
def get_strategy_chat_endpoint(
    strategy_id: int,
    chat_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Get a specific chat for a strategy."""
    return get_strategy_chat(session, strategy_id, chat_id, current_user.id)


@router.patch("/{strategy_id}/chats/{chat_id}", response_model=ChatRead)
def update_strategy_chat_endpoint(
    strategy_id: int,
    chat_id: int,
    payload: ChatCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Update a chat for a strategy."""
    return update_strategy_chat(session, strategy_id, chat_id, payload, current_user.id)


@router.delete("/{strategy_id}/chats/{chat_id}")
def delete_strategy_chat_endpoint(
    strategy_id: int,
    chat_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Delete a chat from a strategy."""
    delete_strategy_chat(session, strategy_id, chat_id, current_user.id)
    return {"status": "ok"}


# ─── RULE TRIGGER ENDPOINTS ───


@router.post("/trigger-agent", response_model=RuleTriggerResponse)
def trigger_agent_endpoint(
    payload: RuleTriggerRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Trigger an agent webhook when an ask_agent rule is activated.
    
    This endpoint is called during backtest execution when a rule with
    action='ask_agent' has its conditions satisfied.
    
    The rule_context should contain:
    - rule_name: Name of the triggered rule
    - timestamp: When the rule was triggered
    - bar_data: Current bar OHLCV data
    - indicators: Current indicator values
    - position: Current position info
    - conditions_matched: List of matched conditions
    """
    result = trigger_rule_agent(
        session=session,
        agent_id=payload.agent_id,
        chat_id=payload.chat_id,
        rule_context=payload.rule_context,
        webhook_url=payload.webhook_url,
        user_id=current_user.id,
    )
    return RuleTriggerResponse(status="ok", agent_response=result)
