import asyncio
import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from app.db.database import get_session
from app.models.user import User
from app.schemas.chat import ChatHistoryPage, ChatSendMessageRequest, ChatSendMessageResponse
from app.services.chat_realtime import subscribe, unsubscribe
from app.services.chat_service import (
    get_chat_session_id,
    list_chat_history,
    send_chat_message,
    stream_chat_message,
)
from app.utils.auth_utils import get_current_active_user
from app.utils.redis_async import close_pubsub, get_async_redis

logger = logging.getLogger(__name__)

CHAT_EVENT_HEARTBEAT_SECONDS = 15.0

# Redis channel where runners publish streamed agent-turn chunks
# (see strategy-runner ChatStreamPublisher). Payload types map 1:1 onto the
# SSE events emitted to the frontend.
CHAT_STREAM_CHANNEL_PREFIX = "chat-stream:"
_AGENT_TURN_EVENT_MAP = {
    "turn_start": "agent_turn_start",
    "delta": "agent_delta",
    "turn_end": "agent_turn_end",
    "turn_error": "agent_turn_error",
}

router = APIRouter(prefix="/chats", tags=["chats"])


@router.get("/{chat_id}/messages", response_model=ChatHistoryPage)
def list_chat_messages_endpoint(
    chat_id: int,
    limit: int = Query(default=20, ge=1, le=100),
    before_id: int | None = Query(default=None, ge=1),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    return list_chat_history(
        session,
        chat_id=chat_id,
        user_id=current_user.id,
        limit=limit,
        before_id=before_id,
    )


@router.post("/{chat_id}/messages", response_model=ChatSendMessageResponse)
def send_chat_message_endpoint(
    chat_id: int,
    payload: ChatSendMessageRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    return send_chat_message(
        session,
        chat_id=chat_id,
        user_id=current_user.id,
        text=payload.text,
        metadata=payload.metadata,
    )


@router.post("/{chat_id}/messages/stream")
async def stream_chat_message_endpoint(
    chat_id: int,
    payload: ChatSendMessageRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    request_id, event_stream = await stream_chat_message(
        session,
        chat_id=chat_id,
        user_id=current_user.id,
        text=payload.text,
        metadata=payload.metadata,
    )
    return StreamingResponse(
        event_stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Chat-Request-Id": request_id,
        },
    )


@router.get("/{chat_id}/events")
async def chat_events_endpoint(
    chat_id: int,
    request: Request,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """SSE stream with chat realtime events.

    Emits ``new_message`` whenever a row is inserted in ``n8n_chat_histories``
    for this chat's session (frontends then refresh via
    ``GET /chats/{id}/messages``), plus ``agent_turn_start`` / ``agent_delta``
    / ``agent_turn_end`` / ``agent_turn_error`` relayed from the Redis channel
    ``chat-stream:{session_id}`` where runners publish the n8n agent stream.
    Redis being unavailable degrades to new_message-only behaviour.
    """
    session_id = get_chat_session_id(session, chat_id=chat_id, user_id=current_user.id)
    # Release the pooled DB connection before streaming: the get_session
    # dependency is torn down only when the response completes, and an SSE
    # response lives for minutes — a held session sits idle in transaction
    # until the Postgres guardrail kills it after 60s.
    session.close()
    queue = subscribe(session_id)

    async def event_stream() -> AsyncIterator[str]:
        events: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()
        pump_tasks: list[asyncio.Task] = []
        pubsub = None

        async def pump_pg_notify() -> None:
            while True:
                row_id = await queue.get()
                await events.put(("new_message", {"id": row_id}))

        pump_tasks.append(asyncio.create_task(pump_pg_notify()))

        redis_client = get_async_redis()
        if redis_client is not None:
            try:
                pubsub = redis_client.pubsub()
                await pubsub.subscribe(f"{CHAT_STREAM_CHANNEL_PREFIX}{session_id}")

                async def pump_agent_stream() -> None:
                    while True:
                        message = await pubsub.get_message(
                            ignore_subscribe_messages=True, timeout=5.0
                        )
                        if not message or message.get("type") != "message":
                            continue
                        try:
                            payload = json.loads(message.get("data") or "")
                        except (TypeError, ValueError):
                            continue
                        event_name = _AGENT_TURN_EVENT_MAP.get(payload.get("type"))
                        if event_name:
                            await events.put((event_name, payload))

                pump_tasks.append(asyncio.create_task(pump_agent_stream()))
            except Exception as exc:  # noqa: BLE001 - relay is best-effort
                logger.warning("chat %s: agent stream relay unavailable: %s", chat_id, exc)
                if pubsub is not None:
                    await close_pubsub(pubsub)
                    pubsub = None

        try:
            # Initial connected marker so clients know the subscription is live.
            yield f"event: ready\ndata: {json.dumps({'session_id': session_id})}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event_name, payload = await asyncio.wait_for(
                        events.get(), timeout=CHAT_EVENT_HEARTBEAT_SECONDS
                    )
                except asyncio.TimeoutError:
                    # SSE comment as keep-alive (ignored by EventSource clients)
                    yield ": ping\n\n"
                    continue
                yield f"event: {event_name}\ndata: {json.dumps(payload)}\n\n"
        finally:
            for task in pump_tasks:
                task.cancel()
            unsubscribe(session_id, queue)
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