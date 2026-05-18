from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from app.db.database import get_session
from app.models.user import User
from app.schemas.chat import ChatHistoryPage, ChatSendMessageRequest, ChatSendMessageResponse
from app.services.chat_service import list_chat_history, send_chat_message, stream_chat_message
from app.utils.auth_utils import get_current_active_user

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