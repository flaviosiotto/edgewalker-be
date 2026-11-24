"""Questions an agent asks the trader in the middle of a turn: the protocol.

One mechanism for every "I need the user to pick / confirm before I go on":
the turn stays open, the question reaches the chat, the answer comes back to
the same turn. The backend owns it end to end; nobody else touches Redis:

* the **agent service** opens a question, waits for its answer and closes it
  through the backend API (``POST /chats/{id}/asks``, ``GET .../wait``,
  ``DELETE ...``), like every other thing it does on the user's behalf;
* the **frontend** learns about it from the chat event stream (SSE
  ``agent_ask`` / ``agent_ask_resolved``; the ``ready`` event carries it for
  a page that reloads) and answers with ``POST .../answer``.

State (``<sid>`` = chat session id):

* ``chat-ask:<sid>``            JSON of the pending ask (:func:`build_ask`), TTL
  a little past its deadline. One pending ask per session.
* ``chat-ask-answer:<ask_id>``  list holding the one answer (:func:`build_answer`).
* ``chat-stream:<sid>``         pub/sub: ``{"v": 1, "type": "ask", "ask": {...}}``
  when asked, ``{"v": 1, "type": "ask_resolved", "ask_id", "status", "answer"}``
  when it ends (``answered`` | ``skipped`` | ``timeout`` | ``cancelled``).

This module is the pure part (shapes, limits, validation);
``chat_ask_service`` does the I/O.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

ASK_KEY_PREFIX = "chat-ask:"
ANSWER_KEY_PREFIX = "chat-ask-answer:"
CHAT_STREAM_CHANNEL_PREFIX = "chat-stream:"

KIND_CHOICE = "choice"
KIND_CONFIRM = "confirm"
KINDS = (KIND_CHOICE, KIND_CONFIRM)

STATUS_ANSWERED = "answered"
STATUS_SKIPPED = "skipped"
STATUS_TIMEOUT = "timeout"
STATUS_CANCELLED = "cancelled"

DEFAULT_TIMEOUT_SECONDS = 180
MAX_TIMEOUT_SECONDS = 900
#: the pending key outlives the deadline a little: the asker deletes it itself
PENDING_TTL_MARGIN_SECONDS = 30
#: an answer nobody collects (asker gone) does not stay in Redis
ANSWER_TTL_SECONDS = 300
#: longest single long poll of the asker (it polls again, with a keep-alive in between)
WAIT_MAX_SECONDS = 20

QUESTION_MAX_CHARS = 300
OPTION_MAX_CHARS = 80
OPTIONS_MIN, OPTIONS_MAX = 2, 6
ANSWER_TEXT_MAX_CHARS = 500


class AskError(ValueError):
    """The question or the answer does not respect the protocol."""


def pending_key(session_id: Any) -> str:
    return f"{ASK_KEY_PREFIX}{session_id}"


def answer_key(ask_id: Any) -> str:
    return f"{ANSWER_KEY_PREFIX}{ask_id}"


def channel(session_id: Any) -> str:
    return f"{CHAT_STREAM_CHANNEL_PREFIX}{session_id}"


def _line(value: Any, limit: int) -> str:
    return " ".join(str(value if value is not None else "").split())[:limit]


def clean_options(options: Any) -> list[str]:
    """Distinct, short button labels; raises :class:`AskError` otherwise."""
    if isinstance(options, str) or not isinstance(options, (list, tuple)):
        raise AskError("options e un array JSON di stringhe (es. [\"V1\", \"V2\"]), non una stringa.")
    labels: list[str] = []
    for raw in options:
        label = " ".join(str((raw.get("label") if isinstance(raw, dict) else raw) or "").split())
        if not label:
            continue
        if len(label) > OPTION_MAX_CHARS:
            raise AskError(f"opzione troppo lunga (max {OPTION_MAX_CHARS} caratteri): un etichetta da pulsante, i dettagli stanno nel testo.")
        if label.lower() not in {x.lower() for x in labels}:
            labels.append(label)
    if not OPTIONS_MIN <= len(labels) <= OPTIONS_MAX:
        raise AskError(f"servono da {OPTIONS_MIN} a {OPTIONS_MAX} opzioni distinte.")
    return labels


def build_ask(
    *,
    session_id: Any,
    question: Any,
    options: Any,
    kind: str = KIND_CHOICE,
    allow_text: bool = True,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    turn_id: str = "",
    request_id: str = "",
    agent_name: str = "",
    now: float | None = None,
) -> dict[str, Any]:
    """The pending ask as stored and shown. Times are epoch milliseconds."""
    text = _line(question, QUESTION_MAX_CHARS)
    if not text:
        raise AskError("question vuota: scrivi la domanda in una riga.")
    if kind not in KINDS:
        raise AskError(f"kind deve essere uno di {KINDS}.")
    labels = clean_options(options)
    seconds = max(10.0, min(float(timeout_seconds or DEFAULT_TIMEOUT_SECONDS), float(MAX_TIMEOUT_SECONDS)))
    created = time.time() if now is None else now
    return {
        "id": uuid.uuid4().hex,
        "session_id": str(session_id),
        "kind": kind,
        "question": text,
        "options": [{"label": label} for label in labels],
        "allow_text": bool(allow_text),
        "turn_id": turn_id,
        "request_id": request_id,
        "agent_name": agent_name,
        "created_at": int(created * 1000),
        "expires_at": int((created + seconds) * 1000),
    }


def pending_ttl_seconds(ask: dict[str, Any], *, now: float | None = None) -> int:
    current = time.time() if now is None else now
    left = float(ask.get("expires_at") or 0) / 1000.0 - current
    return max(1, int(left) + PENDING_TTL_MARGIN_SECONDS)


def is_expired(ask: dict[str, Any], *, now: float | None = None) -> bool:
    current = time.time() if now is None else now
    return current * 1000 >= float(ask.get("expires_at") or 0)


def build_answer(ask: dict[str, Any], *, option: Any = None, text: Any = None, skip: bool = False) -> dict[str, Any]:
    """Validate what the client sent against the pending ask and return the
    answer the asker receives: ``{"status", "answer", "option"}``. Exactly one
    of ``option`` (index of a button), ``text`` (free answer, when the ask
    allows it) or ``skip``."""
    if skip:
        return {"status": STATUS_SKIPPED, "answer": "", "option": None}
    options = ask.get("options") or []
    if option is not None:
        try:
            index = int(option)
        except (TypeError, ValueError) as exc:
            raise AskError("option deve essere l indice di una delle opzioni.") from exc
        if isinstance(option, bool) or not 0 <= index < len(options):
            raise AskError("option fuori dall elenco delle opzioni.")
        return {"status": STATUS_ANSWERED, "answer": str(options[index].get("label") or ""), "option": index}
    free = " ".join(str(text or "").split())[:ANSWER_TEXT_MAX_CHARS]
    if not free:
        raise AskError("risposta vuota: scegli un opzione, scrivi una risposta oppure salta.")
    if not ask.get("allow_text", True):
        raise AskError("questa domanda accetta solo una delle opzioni.")
    match = next((i for i, o in enumerate(options) if str(o.get("label") or "").lower() == free.lower()), None)
    return {"status": STATUS_ANSWERED, "answer": str(options[match]["label"]) if match is not None else free, "option": match}


def ask_event(ask: dict[str, Any]) -> str:
    return json.dumps({"v": 1, "type": "ask", "request_id": ask.get("request_id") or "", "ask": ask}, ensure_ascii=False)


def resolved_event(ask: dict[str, Any], *, status: str, answer: str = "") -> str:
    return json.dumps(
        {"v": 1, "type": "ask_resolved", "request_id": ask.get("request_id") or "", "ask_id": ask.get("id"), "status": status, "answer": answer},
        ensure_ascii=False,
    )


def loads(raw: Any) -> dict[str, Any] | None:
    """A stored ask / answer back from Redis (bytes or str); None when unusable."""
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "replace")
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None
