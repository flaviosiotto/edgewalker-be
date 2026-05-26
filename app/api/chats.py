import asyncio
import json
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

CHAT_EVENT_HEARTBEAT_SECONDS = 15.0

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
    """SSE stream that pushes a ``new_message`` event whenever a row is
    inserted in ``n8n_chat_histories`` for this chat's session.

    Frontends use this signal to call ``GET /chats/{id}/messages`` and refresh
    the conversation without polling.
    """
    session_id = get_chat_session_id(session, chat_id=chat_id, user_id=current_user.id)
    queue = subscribe(session_id)

    async def event_stream() -> AsyncIterator[str]:
        try:
            # Initial connected marker so clients know the subscription is live.
            yield f"event: ready\ndata: {json.dumps({'session_id': session_id})}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    row_id = await asyncio.wait_for(
                        queue.get(), timeout=CHAT_EVENT_HEARTBEAT_SECONDS
                    )
                except asyncio.TimeoutError:
                    # SSE comment as keep-alive (ignored by EventSource clients)
                    yield ": ping\n\n"
                    continue
                payload = json.dumps({"id": row_id})
                yield f"event: new_message\ndata: {payload}\n\n"
        finally:
            unsubscribe(session_id, queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )