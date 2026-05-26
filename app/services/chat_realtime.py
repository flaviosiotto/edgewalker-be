"""In-process pub/sub bridge between PostgreSQL LISTEN/NOTIFY and SSE consumers.

A single dedicated psycopg2 connection runs LISTEN on `chat_history_changes`.
Every notification is dispatched to all asyncio queues registered for the
corresponding ``session_id``. SSE endpoints register/unregister their queue
on connect/disconnect.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import select
import threading
from collections import defaultdict
from typing import Optional

import psycopg2
import psycopg2.extensions

from app.core.config import settings

logger = logging.getLogger(__name__)

CHAT_NOTIFY_CHANNEL = "chat_history_changes"

# session_id -> set of asyncio.Queue[int]   (queues receive new row ids)
_subscribers: dict[str, set[asyncio.Queue]] = defaultdict(set)
_subscribers_lock = threading.Lock()

_listener_thread: Optional[threading.Thread] = None
_listener_stop = threading.Event()
_listener_loop: Optional[asyncio.AbstractEventLoop] = None


def _dispatch(session_id: str, row_id: int) -> None:
    """Push ``row_id`` to all queues registered for ``session_id``.

    Runs on the listener thread; uses ``call_soon_threadsafe`` to hand the
    item to each queue's owning event loop.
    """
    with _subscribers_lock:
        queues = list(_subscribers.get(session_id, ()))
    if not queues or _listener_loop is None:
        return

    def _put(queue: asyncio.Queue, value: int) -> None:
        try:
            queue.put_nowait(value)
        except asyncio.QueueFull:
            # Drop oldest if a slow consumer is backed up
            with contextlib.suppress(asyncio.QueueEmpty):
                queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(value)

    for queue in queues:
        _listener_loop.call_soon_threadsafe(_put, queue, row_id)


def _listener_main() -> None:
    """Listener thread loop. Keeps a dedicated connection in autocommit and
    waits for NOTIFY payloads using ``select``.
    """
    backoff = 1.0
    while not _listener_stop.is_set():
        conn: Optional[psycopg2.extensions.connection] = None
        try:
            conn = psycopg2.connect(settings.DATABASE_URL)
            conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            with conn.cursor() as cur:
                cur.execute(f"LISTEN {CHAT_NOTIFY_CHANNEL};")
            logger.info("Chat realtime listener connected (LISTEN %s)", CHAT_NOTIFY_CHANNEL)
            backoff = 1.0

            while not _listener_stop.is_set():
                # Wait up to 5s for either data or stop signal
                readable, _, _ = select.select([conn], [], [], 5.0)
                if not readable:
                    continue
                conn.poll()
                while conn.notifies:
                    notify = conn.notifies.pop(0)
                    try:
                        payload = json.loads(notify.payload)
                        session_id = payload.get("session_id")
                        row_id = payload.get("id")
                    except (ValueError, TypeError):
                        continue
                    if not session_id or not isinstance(row_id, int):
                        continue
                    _dispatch(str(session_id), row_id)
        except Exception as exc:  # noqa: BLE001
            if _listener_stop.is_set():
                break
            logger.warning(
                "Chat realtime listener error: %s; reconnecting in %.1fs",
                exc, backoff,
            )
            _listener_stop.wait(backoff)
            backoff = min(backoff * 2, 30.0)
        finally:
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.close()


def start_chat_realtime() -> None:
    global _listener_thread, _listener_loop
    if _listener_thread is not None and _listener_thread.is_alive():
        return
    try:
        _listener_loop = asyncio.get_running_loop()
    except RuntimeError:
        _listener_loop = asyncio.get_event_loop()
    _listener_stop.clear()
    _listener_thread = threading.Thread(
        target=_listener_main,
        name="chat-history-listener",
        daemon=True,
    )
    _listener_thread.start()
    logger.info("Chat realtime listener thread started")


def stop_chat_realtime() -> None:
    global _listener_thread
    _listener_stop.set()
    thread = _listener_thread
    if thread is not None:
        thread.join(timeout=6.0)
    _listener_thread = None


def subscribe(session_id: str) -> asyncio.Queue:
    queue: asyncio.Queue = asyncio.Queue(maxsize=64)
    with _subscribers_lock:
        _subscribers[session_id].add(queue)
    return queue


def unsubscribe(session_id: str, queue: asyncio.Queue) -> None:
    with _subscribers_lock:
        bucket = _subscribers.get(session_id)
        if not bucket:
            return
        bucket.discard(queue)
        if not bucket:
            _subscribers.pop(session_id, None)
