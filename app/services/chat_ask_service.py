"""Backend side of the mid-turn questions (``edgewalker_platform.user_ask``).

The agent service owns the question and waits for the answer; here we only
(1) tell a client what is pending on a chat session and (2) hand the owner's
answer to the waiting turn. Redis down = no pending question to show and a
503 on the answer: the turn then ends by its own timeout.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import HTTPException, status

from edgewalker_platform import user_ask

from app.utils.redis_async import get_async_redis

logger = logging.getLogger(__name__)


async def read_pending_ask(session_id: str) -> dict[str, Any] | None:
    """The question in flight on a chat session, if any and not expired."""
    client = get_async_redis()
    if client is None:
        return None
    try:
        ask = user_ask.loads(await client.get(user_ask.pending_key(session_id)))
    except Exception as exc:  # noqa: BLE001 - display only
        logger.debug("pending ask read failed (non-critical): %s", exc)
        return None
    if ask is None or user_ask.is_expired(ask):
        return None
    return ask


async def answer_ask(
    session_id: str, ask_id: str, *, option: int | None = None, text: str | None = None, skip: bool = False, user_id: int | None = None
) -> dict[str, Any]:
    """Deliver the owner's answer to the turn that is waiting for it.

    404 no question pending on the session (or another one), 410 expired,
    409 already answered, 422 answer not valid for that question.
    """
    client = get_async_redis()
    if client is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Answers are unavailable right now")
    try:
        ask = user_ask.loads(await client.get(user_ask.pending_key(session_id)))
    except Exception as exc:  # noqa: BLE001
        logger.warning("pending ask read failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Answers are unavailable right now") from exc
    if ask is None or str(ask.get("id")) != str(ask_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No such question is pending on this chat")
    if user_ask.is_expired(ask):
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="The question has expired")
    try:
        answer = user_ask.build_answer(ask, option=option, text=text, skip=skip)
    except user_ask.AskError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if user_id is not None:
        answer["user_id"] = user_id
    key = user_ask.answer_key(ask["id"])
    try:
        # First answer wins: a second tab or a double click must not queue a
        # second one for a question the turn has already moved past.
        pushed = await client.rpush(key, json.dumps(answer, ensure_ascii=False))
        await client.expire(key, user_ask.ANSWER_TTL_SECONDS)
        if pushed and int(pushed) > 1:
            await client.rpop(key)
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="The question has already been answered")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("ask answer delivery failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Answers are unavailable right now") from exc
    return {"ask_id": ask["id"], "status": answer["status"], "answer": answer["answer"]}
