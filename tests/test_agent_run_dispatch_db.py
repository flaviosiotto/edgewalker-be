"""The run executor and the real ``send_chat_message`` against a fake agent-svc:
no pooled DB connection may stay checked out while the turn runs (the turn
lasts minutes: 13/10 incident, pool starvation pattern), and the run's own
deadline, not the chat's 120 s, bounds the call."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import event
from sqlmodel import Session, select

import app.models.agent_turn  # noqa: E402,F401
import app.models.connection  # noqa: E402,F401
import app.models.live_trading  # noqa: E402,F401
import app.models.strategy  # noqa: E402,F401
import app.models.strategy_template  # noqa: E402,F401
import app.models.agent_model  # noqa: E402,F401

pytest.importorskip("pgserver")


@pytest.fixture
def session(app_engine):
    with Session(app_engine, expire_on_commit=False) as s:
        yield s


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "SECRETS_ENCRYPTION_KEY", None)
    monkeypatch.setattr(settings, "SECRET_KEY", "unit-test-secret-key-for-runs")


@pytest.fixture
def tenant(session):
    from app.models.connection import Account, Connection
    from app.models.user import User

    now = datetime.now(timezone.utc)
    stale = session.exec(select(User).where(User.email == "dispatch@example.com")).first()
    if stale is not None:
        session.delete(stale)
        session.commit()
    user = User(email="dispatch@example.com", username="user_dispatch", hashed_password="x")
    session.add(user)
    session.flush()
    conn = Connection(user_id=user.id, name="conn-dispatch", broker_type="binance", status="connected", created_at=now, updated_at=now)
    session.add(conn)
    session.flush()
    acc = Account(connection_id=conn.id, account_id="acc-dispatch", currency="USDT", created_at=now, updated_at=now)
    session.add(acc)
    session.commit()
    yield user, conn, acc
    session.delete(user)
    session.commit()


def _find(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find(v, key)
            if found is not None:
                return found
    if isinstance(obj, list):
        for v in obj:
            found = _find(v, key)
            if found is not None:
                return found
    return None


def _write_answer(session_id, request_id, text):
    import app.db.database as database
    from app.models.n8n_chat_history import N8nChatHistory

    with database.get_session_context() as own:
        own.add(N8nChatHistory(session_id=str(session_id), message={
            "type": "ai", "content": text, "tool_calls": [], "additional_kwargs": {},
            "response_metadata": {"edgewalker": {"turn_usage": {"correlation_id": request_id, "credits": 0.5}}},
            "metadata": {},
        }))


class _FakeStream:
    """agent-svc streams the turn: chunks come with pauses, the answer row is
    written to the chat by agent-svc itself (here: by ``on_chunk``)."""

    def __init__(self, chunks, *, pause=0.0, on_chunk=None):
        self.chunks, self.pause, self.on_chunk = chunks, pause, on_chunk

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    def iter_bytes(self):
        import time

        for i, chunk in enumerate(self.chunks):
            if self.pause:
                time.sleep(self.pause)
            if self.on_chunk:
                self.on_chunk(i)
            yield chunk


class _FakeRedis:
    def __init__(self):
        self.published = []

    def publish(self, channel, message):
        import json

        self.published.append((channel, json.loads(message)))


def _fake_agent_svc(monkeypatch, *, pause=0.0, answer_at=0, answer_text="Fatto.", n_chunks=1, chunks=None):
    """Fake ``httpx.Client`` for chat_service: records what the backend holds
    and configures, and plays agent-svc writing the answer at chunk ``answer_at``
    (None = never)."""
    import app.db.database as database
    from app.services import chat_service

    held: set[int] = set()
    seen = {}

    def on_checkout(dbapi_conn, rec, proxy):
        held.add(id(rec))

    def on_checkin(dbapi_conn, rec):
        held.discard(id(rec))

    event.listen(database.engine, "checkout", on_checkout)
    event.listen(database.engine, "checkin", on_checkin)

    class FakeClient:
        def __init__(self, **kw):
            seen["timeout"] = kw.get("timeout")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None, headers=None):  # chat path (no deadline): not used by runs
            raise AssertionError("runs must stream the turn under a deadline")

        def stream(self, method, url, json=None, headers=None):
            seen["outstanding_during_turn"] = len(held)
            seen["url"] = url
            request_id = _find(json, "request_id")
            session_id = _find(json, "session_id") or _find(json, "sessionId")
            seen["request_id"] = request_id

            def on_chunk(i):
                if answer_at is not None and i == answer_at:
                    _write_answer(session_id, request_id, answer_text)

            return _FakeStream(chunks if chunks is not None else [b"{}\n"] * n_chunks, pause=pause, on_chunk=on_chunk)

    monkeypatch.setattr(chat_service.httpx, "Client", FakeClient)
    seen["redis"] = _FakeRedis()
    import app.utils.redis_client as redis_client

    monkeypatch.setattr(redis_client, "get_redis", lambda: seen["redis"])

    def cleanup():
        event.remove(database.engine, "checkout", on_checkout)
        event.remove(database.engine, "checkin", on_checkin)

    return seen, cleanup


def _queued_run(session, tenant, task="Briefing"):
    from app.schemas.agent import AgentCreate
    from app.schemas.agent_run import RunCreate
    from app.services import agent_run_service as runs
    from app.services.agent_service import create_agent

    user, _, _ = tenant
    agent, _ = create_agent(session, AgentCreate(agent_name="Runner"), user.id)
    return runs.create_run(session, user_id=user.id, agent_id=agent.id_agent, payload=RunCreate(task=task, scope="agent"), created_by=None)


def test_execute_run_holds_no_db_connection_and_uses_the_run_deadline(session, tenant, monkeypatch):
    from app.core.config import settings
    from app.services import agent_run_service as runs

    run = _queued_run(session, tenant)
    seen, cleanup = _fake_agent_svc(monkeypatch)
    try:
        done = runs.execute_run(session, run.id)
    finally:
        cleanup()
    assert done.status == "succeeded", done.error
    assert done.result == "Fatto."
    assert seen["outstanding_during_turn"] == 0, seen
    assert seen["timeout"].read == float(settings.AGENT_RUN_TURN_TIMEOUT_SECONDS)
    assert seen["url"] == settings.AGENT_SVC_WEBHOOK_URL


def test_execute_run_relays_the_live_turn_events_of_the_chat(session, tenant, monkeypatch):
    """agent-svc NDJSON → Redis chat-stream events (what GET /chats/{id}/events
    serves): text deltas, tool calls, start/end; split across chunks."""
    from app.services import agent_run_service as runs
    from app.services.chat_service import CHAT_STREAM_CHANNEL_PREFIX

    run = _queued_run(session, tenant)
    ndjson = (
        b'{"type":"begin","metadata":{"turn_id":"t1"}}\n'
        b'{"type":"item","content":"","metadata":{"tool_call":"live_dashboard"}}\n'
        b'{"type":"item","content":"Briefing: ","metadata":{}}\n{"type":"item","content":"tutto ok.","metadata":{}}\n'
        b'{"type":"end","metadata":{"turn_id":"t1"}}\n'
    )
    half = len(ndjson) // 2
    seen, cleanup = _fake_agent_svc(monkeypatch, chunks=[ndjson[:half], ndjson[half:]], answer_at=1)
    try:
        done = runs.execute_run(session, run.id)
    finally:
        cleanup()
    assert done.status == "succeeded", done.error
    published = seen["redis"].published
    assert {c for c, _ in published} == {f"{CHAT_STREAM_CHANNEL_PREFIX}{run.chat_id}"}
    events = [p for _, p in published]
    assert [e["type"] for e in events] == ["turn_start", "tool_call", "delta", "delta", "turn_end"]
    assert all(e["request_id"] == seen["request_id"] and e["v"] == 1 for e in events)
    assert events[1]["name"] == "live_dashboard"
    assert events[2]["delta"] + events[3]["delta"] == "Briefing: tutto ok."
    assert events[4]["full_len"] == len("Briefing: tutto ok.")


def test_execute_run_past_the_deadline_fails_with_a_pointer_to_the_chat(session, tenant, monkeypatch):
    from app.core.config import settings
    from app.services import agent_run_service as runs

    monkeypatch.setattr(settings, "AGENT_RUN_TURN_TIMEOUT_SECONDS", 1)
    run = _queued_run(session, tenant)
    seen, cleanup = _fake_agent_svc(monkeypatch, pause=0.6, n_chunks=4, answer_at=None)
    try:
        done = runs.execute_run(session, run.id)
    finally:
        cleanup()
    assert done.status == "failed"
    assert "did not finish within 1s" in done.error and "answer will appear in the chat" in done.error
    assert done.request_id is None


def test_execute_run_past_the_deadline_still_takes_a_late_answer(session, tenant, monkeypatch):
    from app.core.config import settings
    from app.services import agent_run_service as runs

    monkeypatch.setattr(settings, "AGENT_RUN_TURN_TIMEOUT_SECONDS", 1)
    run = _queued_run(session, tenant)
    # the answer lands at chunk 1 (t≈1.2 s) and the deadline trips at the same chunk
    seen, cleanup = _fake_agent_svc(monkeypatch, pause=0.6, n_chunks=4, answer_at=1, answer_text="Fatto tardi.")
    try:
        done = runs.execute_run(session, run.id)
    finally:
        cleanup()
    assert done.status == "succeeded", done.error
    assert done.result == "Fatto tardi."


def test_submit_run_executes_on_the_run_pool_in_its_own_session(session, tenant, monkeypatch):
    from app.models.agent_model import AgentRun
    from app.services import agent_run_service as runs

    run = _queued_run(session, tenant)
    seen, cleanup = _fake_agent_svc(monkeypatch, answer_text="Dal pool.")
    try:
        future = runs.submit_run(run.id)
        done = future.result(timeout=30)
    finally:
        cleanup()
    assert done is not None and done.status == "succeeded", getattr(done, "error", None)
    session.expire_all()
    assert session.get(AgentRun, run.id).result == "Dal pool."
