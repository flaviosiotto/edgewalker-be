"""Mid-turn questions: the I/O (``user_ask`` holds the protocol).

Two callers. The agent service opens a question, waits for the answer and
closes it (:func:`open_ask`, :func:`wait_answer`, :func:`close_ask`); the
chat's owner sees it and answers (:func:`read_pending_ask`,
:func:`answer_ask`). Redis down = nothing to ask or answer (503): the agent
then goes on without the question.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from fastapi import HTTPException, status

from app.services import user_ask
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


def _unavailable(exc: Exception | None = None) -> HTTPException:
    if exc is not None:
        logger.warning("ask storage unavailable: %s", exc)
    return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Questions are unavailable right now")


async def open_ask(session_id: str, **fields: Any) -> dict[str, Any]:
    """Store a new question for the chat session and announce it to its
    clients. 422 malformed question, 409 another one is already pending."""
    client = get_async_redis()
    if client is None:
        raise _unavailable()
    try:
        ask = user_ask.build_ask(session_id=session_id, **fields)
    except user_ask.AskError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        stored = await client.set(
            user_ask.pending_key(session_id), json.dumps(ask, ensure_ascii=False), ex=user_ask.pending_ttl_seconds(ask), nx=True
        )
        if stored:
            await client.publish(user_ask.channel(session_id), user_ask.ask_event(ask))
    except Exception as exc:  # noqa: BLE001
        raise _unavailable(exc) from exc
    if not stored:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Another question is already pending on this chat")
    return ask


async def _finish(client: Any, session_id: str, ask: dict[str, Any], *, status_: str, answer: str = "") -> None:
    """The question is over: nothing stays pending and the clients are told."""
    await client.delete(user_ask.pending_key(session_id), user_ask.answer_key(ask["id"]))
    await client.publish(user_ask.channel(session_id), user_ask.resolved_event(ask, status=status_, answer=answer))


async def _pending(client: Any, session_id: str, ask_id: str) -> dict[str, Any] | None:
    ask = user_ask.loads(await client.get(user_ask.pending_key(session_id)))
    return ask if ask is not None and str(ask.get("id")) == str(ask_id) else None


async def wait_answer(session_id: str, ask_id: str, *, seconds: float) -> dict[str, Any]:
    """Long poll of the asker: block up to ``seconds`` for the answer.

    ``{"status": "pending"}`` nothing yet (ask again); ``answered`` /
    ``skipped`` with the answer — the question is then closed; ``timeout``
    its deadline passed (closed too); ``gone`` no such question is pending.
    """
    client = get_async_redis()
    if client is None:
        raise _unavailable()
    try:
        ask = await _pending(client, session_id, ask_id)
        if ask is None:
            return {"status": "gone"}
        left = float(ask["expires_at"]) / 1000.0 - time.time()
        if left <= 0:
            await _finish(client, session_id, ask, status_=user_ask.STATUS_TIMEOUT)
            return {"status": user_ask.STATUS_TIMEOUT}
        wait = max(1, int(min(max(1.0, seconds), user_ask.WAIT_MAX_SECONDS, left) + 0.999))
        got = await client.blpop(user_ask.answer_key(ask["id"]), timeout=wait)
        answer = user_ask.loads(got[1]) if got else None
        if answer is None:
            return {"status": "pending"}
        outcome = user_ask.STATUS_SKIPPED if answer.get("status") == user_ask.STATUS_SKIPPED else user_ask.STATUS_ANSWERED
        text = str(answer.get("answer") or "") if outcome == user_ask.STATUS_ANSWERED else ""
        await _finish(client, session_id, ask, status_=outcome, answer=text)
        return {"status": outcome, "answer": text, "option": answer.get("option")}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _unavailable(exc) from exc


async def close_ask(session_id: str, ask_id: str, *, reason: str = user_ask.STATUS_CANCELLED) -> dict[str, Any]:
    """The asker gives up on its question (turn over or cancelled). Idempotent."""
    client = get_async_redis()
    if client is None:
        raise _unavailable()
    outcome = reason if reason in (user_ask.STATUS_TIMEOUT, user_ask.STATUS_CANCELLED) else user_ask.STATUS_CANCELLED
    try:
        ask = await _pending(client, session_id, ask_id)
        if ask is not None:
            await _finish(client, session_id, ask, status_=outcome)
    except Exception as exc:  # noqa: BLE001
        raise _unavailable(exc) from exc
    return {"status": outcome if ask is not None else "gone"}
