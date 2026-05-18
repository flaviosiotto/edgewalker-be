from __future__ import annotations

import codecs
import json
import uuid
from collections.abc import AsyncIterator
from datetime import date, datetime
from typing import Any

import httpx
from fastapi import HTTPException, status
from sqlalchemy.orm import selectinload
from sqlmodel import Session, select

from app.models.agent import Agent, Chat
from app.models.n8n_chat_history import N8nChatHistory
from app.schemas.chat import ChatHistoryMessageRead, ChatHistoryPage, ChatSendMessageResponse
from app.services.live_runner_service import _rewrite_webhook_for_docker
from app.services.n8n_auth import (
    build_n8n_api_auth_metadata,
    build_n8n_webhook_auth_headers,
    issue_n8n_api_access_token,
    issue_n8n_webhook_auth_token,
)

MAX_CHAT_PAGE_SIZE = 100
DEFAULT_STREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=60.0, pool=60.0)
DEFAULT_SEND_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _sse_event(event: str, payload: dict[str, Any]) -> str:
    return (
        f"event: {event}\n"
        f"data: {json.dumps(payload, ensure_ascii=False, default=_json_default)}\n\n"
    )


def _get_owned_chat(session: Session, chat_id: int, user_id: int) -> Chat:
    chat = session.exec(
        select(Chat)
        .options(selectinload(Chat.agent))
        .where(Chat.id == chat_id)
        .where(Chat.user_id == user_id)
    ).first()
    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
    return chat


def _resolve_chat_agent(session: Session, chat: Chat) -> Agent:
    if chat.agent and chat.agent.n8n_webhook:
        return chat.agent

    if chat.id_agent is not None:
        agent = session.get(Agent, chat.id_agent)
        if agent and agent.user_id == chat.user_id and agent.n8n_webhook:
            return agent

    if chat.strategy_id is not None:
        from app.services.strategy_service import resolve_strategy_manager_agent_id

        fallback_agent_id = resolve_strategy_manager_agent_id(
            session,
            chat.strategy_id,
            user_id=chat.user_id,
        )
        if fallback_agent_id is not None:
            agent = session.get(Agent, fallback_agent_id)
            if agent and agent.user_id == chat.user_id and agent.n8n_webhook:
                return agent

    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="No n8n webhook configured for this chat",
    )


def _chat_session_id(chat: Chat) -> str:
    if chat.n8n_session_id:
        return chat.n8n_session_id
    if chat.id is not None:
        return str(chat.id)
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Chat session id is not available",
    )


def _build_webhook_payload(
    chat: Chat,
    *,
    text: str,
    metadata: dict[str, Any] | None,
    request_id: str,
    api_auth_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    session_id = _chat_session_id(chat)
    base_metadata = dict(metadata or {})
    existing_api_auth_metadata = base_metadata.get("api_auth")
    legacy_auth_metadata = base_metadata.get("auth")
    merged_api_auth_metadata: dict[str, Any] | None = None
    if isinstance(existing_api_auth_metadata, dict):
        merged_api_auth_metadata = dict(existing_api_auth_metadata)
    elif isinstance(legacy_auth_metadata, dict):
        merged_api_auth_metadata = dict(legacy_auth_metadata)
    elif api_auth_metadata:
        merged_api_auth_metadata = {}

    if merged_api_auth_metadata is not None and api_auth_metadata:
        merged_api_auth_metadata.update(api_auth_metadata)

    base_metadata.update(
        {
            "chat_id": session_id,
            "edgewalker_chat_id": chat.id,
            "request_id": request_id,
        }
    )
    base_metadata.pop("auth", None)
    if merged_api_auth_metadata is not None:
        base_metadata["api_auth"] = merged_api_auth_metadata
    return {
        "action": "sendMessage",
        "sessionId": session_id,
        "chatInput": text,
        "metadata": base_metadata,
    }


def _coerce_message_dict(raw_message: Any) -> dict[str, Any]:
    if isinstance(raw_message, dict):
        message = dict(raw_message)
        for key in ("text", "output", "message", "content"):
            value = message.get(key)
            if isinstance(value, str):
                normalized = _extract_n8n_stream_text(value)
                if normalized is not None:
                    message[key] = normalized
        return message
    if isinstance(raw_message, str):
        normalized = _extract_n8n_stream_text(raw_message)
        return {"type": "ai", "text": normalized if normalized is not None else raw_message}
    return {"type": "ai", "text": json.dumps(raw_message, ensure_ascii=False, default=_json_default)}


def _extract_n8n_stream_text(raw_value: str) -> str | None:
    stripped = raw_value.strip()
    if not stripped or not stripped.startswith("{"):
        return None

    lines = [line.strip() for line in raw_value.splitlines() if line.strip()]
    if not lines:
        return None

    parts: list[str] = []
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict):
            return None

        event_type = payload.get("type")
        if event_type not in {"begin", "item", "end"}:
            return None

        if event_type == "item" and isinstance(payload.get("content"), str):
            parts.append(payload["content"])

    return "".join(parts)


def _parse_n8n_stream_event(raw_value: str) -> tuple[bool, str | None]:
    stripped = raw_value.strip()
    if not stripped or not stripped.startswith("{") or '"type"' not in stripped:
        return False, None

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return False, None

    if not isinstance(payload, dict):
        return False, None

    event_type = payload.get("type")
    if event_type not in {"begin", "item", "end"}:
        return False, None

    if event_type == "item" and isinstance(payload.get("content"), str):
        return True, payload["content"]

    extracted = _extract_text(payload)
    if extracted and extracted != json.dumps(payload, ensure_ascii=False, default=_json_default):
        return True, extracted
    return True, None


def _extract_text(payload: Any) -> str:
    if isinstance(payload, str):
        normalized = _extract_n8n_stream_text(payload)
        return normalized if normalized is not None else payload
    if isinstance(payload, dict):
        for key in ("text", "output", "message", "content"):
            value = payload.get(key)
            if isinstance(value, str):
                normalized = _extract_n8n_stream_text(value)
                return normalized if normalized is not None else value
        return json.dumps(payload, ensure_ascii=False, default=_json_default)
    return str(payload)


def _default_sender_kind(message_type: str | None) -> str | None:
    if message_type == "human":
        return "user"
    if message_type == "ai":
        return "agent"
    return message_type


def _default_sender_label(sender_kind: str | None) -> str | None:
    if sender_kind == "user":
        return "You"
    if sender_kind == "agent":
        return "Agent"
    if sender_kind == "system":
        return "System"
    return None


def _serialize_history_message(entry: N8nChatHistory) -> ChatHistoryMessageRead:
    message = _coerce_message_dict(entry.message)
    metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
    message_type = message.get("type") if isinstance(message.get("type"), str) else None

    if isinstance(message.get("sender_kind"), str):
        sender_kind = message["sender_kind"]
    elif isinstance(metadata.get("sender_kind"), str):
        sender_kind = metadata["sender_kind"]
    elif isinstance(message.get("sender"), str):
        sender_kind = message["sender"]
    else:
        sender_kind = _default_sender_kind(message_type)

    if isinstance(message.get("sender_label"), str):
        sender_label = message["sender_label"]
    elif isinstance(metadata.get("sender_label"), str):
        sender_label = metadata["sender_label"]
    elif isinstance(metadata.get("agent_name"), str):
        sender_label = metadata["agent_name"]
    else:
        sender_label = _default_sender_label(sender_kind)

    fmt = message.get("format") if isinstance(message.get("format"), str) else None

    return ChatHistoryMessageRead(
        id=entry.id or 0,
        session_id=entry.session_id,
        message=message,
        text=_extract_text(message),
        message_type=message_type,
        sender_kind=sender_kind,
        sender_label=sender_label,
        timestamp=entry.created_at,
        metadata=metadata,
        format=fmt,
    )


def list_chat_history(
    session: Session,
    *,
    chat_id: int,
    user_id: int,
    limit: int = 20,
    before_id: int | None = None,
) -> ChatHistoryPage:
    chat = _get_owned_chat(session, chat_id, user_id)
    safe_limit = max(1, min(limit, MAX_CHAT_PAGE_SIZE))
    session_id = _chat_session_id(chat)

    stmt = select(N8nChatHistory).where(N8nChatHistory.session_id == session_id)
    if before_id is not None:
        stmt = stmt.where(N8nChatHistory.id < before_id)

    rows = list(session.exec(stmt.order_by(N8nChatHistory.id.desc()).limit(safe_limit + 1)).all())
    has_more = len(rows) > safe_limit
    if has_more:
        rows = rows[:safe_limit]

    rows.reverse()
    items = [_serialize_history_message(row) for row in rows]
    next_before = items[0].id if has_more and items else None
    return ChatHistoryPage(
        chat_id=chat.id or chat_id,
        session_id=session_id,
        items=items,
        limit=safe_limit,
        next_before=next_before,
        has_more=has_more,
    )


def send_chat_message(
    session: Session,
    *,
    chat_id: int,
    user_id: int,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> ChatSendMessageResponse:
    normalized_text = text.strip()
    if not normalized_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Message text is required",
        )

    chat = _get_owned_chat(session, chat_id, user_id)
    agent = _resolve_chat_agent(session, chat)
    request_id = str(uuid.uuid4())
    session_id = _chat_session_id(chat)
    webhook_auth_token = issue_n8n_webhook_auth_token(
        session,
        user_id=chat.user_id,
        purpose="n8n_chat_message",
        extra_claims={
            "agent_id": agent.id_agent,
            "chat_id": chat.id,
            "request_id": request_id,
            "session_id": session_id,
        },
    )
    api_auth_token, api_auth_expires_at = issue_n8n_api_access_token(
        session,
        user_id=chat.user_id,
        purpose="n8n_chat_api_access",
        extra_claims={
            "agent_id": agent.id_agent,
            "chat_id": chat.id,
            "session_id": session_id,
        },
    )
    api_auth_metadata = build_n8n_api_auth_metadata(
        user_id=chat.user_id,
        purpose="n8n_chat_api_access",
        token=api_auth_token,
        expires_at=api_auth_expires_at,
    )
    webhook_payload = _build_webhook_payload(
        chat,
        text=normalized_text,
        metadata=metadata,
        request_id=request_id,
        api_auth_metadata=api_auth_metadata,
    )

    headers = build_n8n_webhook_auth_headers(webhook_auth_token)

    try:
        with httpx.Client(timeout=DEFAULT_SEND_TIMEOUT) as client:
            response = client.post(
                _rewrite_webhook_for_docker(agent.n8n_webhook),
                json=webhook_payload,
                headers=headers,
            )
            response.raise_for_status()
    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Cannot connect to agent webhook: {exc}",
        )
    except httpx.HTTPStatusError as exc:
        body = exc.response.text[:500] if exc.response.text else ""
        detail = f"Agent webhook returned {exc.response.status_code}"
        if body:
            detail = f"{detail}: {body}"
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail)
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to call agent webhook: {exc}",
        )

    return ChatSendMessageResponse(
        status="ok",
        chat_id=chat.id or chat_id,
        session_id=session_id,
        request_id=request_id,
    )


def _extract_ndjson_chunk(line: str) -> str | None:
    handled, emitted = _parse_n8n_stream_event(line)
    if handled:
        return emitted

    stripped = line.strip()
    if not stripped:
        return None

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    extracted = _extract_text(payload)
    if extracted and extracted != json.dumps(payload, ensure_ascii=False, default=_json_default):
        return extracted
    return None


def _drain_stream_buffer(buffer: str, full_text: str) -> tuple[str, str, list[str]]:
    emitted_parts: list[str] = []

    while "\n" in buffer:
        line, buffer = buffer.split("\n", 1)
        handled, emitted = _parse_n8n_stream_event(line)
        if handled:
            if emitted:
                full_text += emitted
                emitted_parts.append(emitted)
            continue

        raw_line = f"{line}\n"
        full_text += raw_line
        emitted_parts.append(raw_line)

    return buffer, full_text, emitted_parts


def _emit_terminal_buffer(buffer: str, full_text: str) -> tuple[str, str | None]:
    stripped = buffer.strip()
    if not stripped:
        return full_text, None

    handled, emitted = _parse_n8n_stream_event(buffer)
    if handled:
        if emitted:
            full_text += emitted
            return full_text, emitted
        return full_text, None

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        full_text += buffer
        return full_text, buffer

    extracted = _extract_text(payload)
    if extracted:
        full_text += extracted
        return full_text, extracted
    return full_text, None


async def stream_chat_message(
    session: Session,
    *,
    chat_id: int,
    user_id: int,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[str, AsyncIterator[str]]:
    normalized_text = text.strip()
    if not normalized_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Message text is required",
        )

    chat = _get_owned_chat(session, chat_id, user_id)
    agent = _resolve_chat_agent(session, chat)
    request_id = str(uuid.uuid4())
    session_id = _chat_session_id(chat)
    webhook_auth_token = issue_n8n_webhook_auth_token(
        session,
        user_id=chat.user_id,
        purpose="n8n_chat_stream",
        extra_claims={
            "agent_id": agent.id_agent,
            "chat_id": chat.id,
            "request_id": request_id,
            "session_id": session_id,
        },
    )
    api_auth_token, api_auth_expires_at = issue_n8n_api_access_token(
        session,
        user_id=chat.user_id,
        purpose="n8n_chat_api_access",
        extra_claims={
            "agent_id": agent.id_agent,
            "chat_id": chat.id,
            "session_id": session_id,
        },
    )
    api_auth_metadata = build_n8n_api_auth_metadata(
        user_id=chat.user_id,
        purpose="n8n_chat_api_access",
        token=api_auth_token,
        expires_at=api_auth_expires_at,
    )
    webhook_payload = _build_webhook_payload(
        chat,
        text=normalized_text,
        metadata=metadata,
        request_id=request_id,
        api_auth_metadata=api_auth_metadata,
    )
    headers = build_n8n_webhook_auth_headers(webhook_auth_token)
    headers["Accept"] = "text/plain"

    async def event_stream() -> AsyncIterator[str]:
        decoder = codecs.getincrementaldecoder("utf-8")()
        buffer = ""
        full_text = ""

        yield _sse_event(
            "message_start",
            {
                "request_id": request_id,
                "chat_id": chat.id,
                "session_id": session_id,
            },
        )

        try:
            async with httpx.AsyncClient(timeout=DEFAULT_STREAM_TIMEOUT) as client:
                async with client.stream(
                    "POST",
                    _rewrite_webhook_for_docker(agent.n8n_webhook),
                    json=webhook_payload,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    content_type = (response.headers.get("content-type") or "").lower()

                    if "application/json" in content_type:
                        payload = await response.aread()
                        decoded = payload.decode("utf-8", errors="replace").strip()
                        if decoded:
                            full_text, emitted = _emit_terminal_buffer(decoded, full_text)
                            if emitted:
                                yield _sse_event(
                                    "message_chunk",
                                    {
                                        "request_id": request_id,
                                        "delta": emitted,
                                    },
                                )
                    else:
                        async for chunk in response.aiter_bytes():
                            if not chunk:
                                continue

                            text_chunk = decoder.decode(chunk)
                            if not text_chunk:
                                continue

                            buffer += text_chunk

                            buffer, full_text, emitted_parts = _drain_stream_buffer(buffer, full_text)
                            for emitted in emitted_parts:
                                if emitted:
                                    yield _sse_event(
                                        "message_chunk",
                                        {
                                            "request_id": request_id,
                                            "delta": emitted,
                                        },
                                    )

                            probe = buffer.lstrip()
                            if buffer and not probe.startswith("{"):
                                full_text += buffer
                                yield _sse_event(
                                    "message_chunk",
                                    {
                                        "request_id": request_id,
                                        "delta": buffer,
                                    },
                                )
                                buffer = ""

                        tail = decoder.decode(b"", final=True)
                        if tail:
                            buffer += tail

                        buffer, full_text, emitted_parts = _drain_stream_buffer(buffer, full_text)
                        for emitted in emitted_parts:
                            if emitted:
                                yield _sse_event(
                                    "message_chunk",
                                    {
                                        "request_id": request_id,
                                        "delta": emitted,
                                    },
                                )

                        if buffer:
                            full_text, emitted = _emit_terminal_buffer(buffer, full_text)
                            if emitted:
                                yield _sse_event(
                                    "message_chunk",
                                    {
                                        "request_id": request_id,
                                        "delta": emitted,
                                    },
                                )

            yield _sse_event(
                "message_end",
                {
                    "request_id": request_id,
                    "chat_id": chat.id,
                    "session_id": session_id,
                    "full_text": full_text,
                },
            )
        except httpx.ConnectError as exc:
            yield _sse_event(
                "message_error",
                {
                    "request_id": request_id,
                    "detail": f"Cannot connect to agent webhook: {exc}",
                },
            )
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500] if exc.response.text else ""
            detail = f"Agent webhook returned {exc.response.status_code}"
            if body:
                detail = f"{detail}: {body}"
            yield _sse_event(
                "message_error",
                {
                    "request_id": request_id,
                    "detail": detail,
                },
            )
        except httpx.HTTPError as exc:
            yield _sse_event(
                "message_error",
                {
                    "request_id": request_id,
                    "detail": f"Failed to call agent webhook: {exc}",
                },
            )

    return request_id, event_stream()