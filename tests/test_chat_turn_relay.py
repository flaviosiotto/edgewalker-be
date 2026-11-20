"""Live state of an agent turn: what the chat shows while the agent works.

Covers the relay (events published on the chat's channel + the state a client
reads when it connects mid-turn) and the user's streamed reply (activity
events, keep-alive during a silent reasoning step, relay that survives the
client).
"""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from app.services import chat_service
from app.services.chat_service import CHAT_STREAM_CHANNEL_PREFIX, CHAT_TURN_STATE_PREFIX, TurnRelay


class FakeRedis:
    def __init__(self):
        self.published: list[dict] = []
        self.keys: dict[str, str] = {}

    def publish(self, channel, data):
        assert channel == f"{CHAT_STREAM_CHANNEL_PREFIX}s1"
        self.published.append(json.loads(data))

    def set(self, key, value, ex=None):
        assert ex
        self.keys[key] = value

    def delete(self, key):
        self.keys.pop(key, None)

    def state(self):
        raw = self.keys.get(f"{CHAT_TURN_STATE_PREFIX}s1")
        return json.loads(raw) if raw else None


@pytest.fixture
def redis(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(TurnRelay, "_client", lambda self: fake)
    return fake


def _item(content="", **metadata):
    return json.dumps({"type": "item", "content": content, "metadata": metadata}) + "\n"


BEGIN = json.dumps({"type": "begin", "metadata": {}}) + "\n"
END = json.dumps({"type": "end", "metadata": {}}) + "\n"


def test_relay_tracks_what_the_turn_is_doing(redis):
    relay = TurnRelay("s1", "r1", asker={"chat_input": "riprova", "sender_kind": "user"})

    assert relay.feed(BEGIN + _item(activity="thinking")) == [relay.state]
    assert redis.state()["phase"] == "thinking"
    assert redis.published[0]["type"] == "turn_start"
    assert redis.published[0]["chat_input"] == "riprova"

    changes = relay.feed(_item(tool_call="get_strategy") + _item(tool_call="update_strategy"))
    assert [(c["phase"], c["name"], c["tools"]) for c in changes] == [
        ("tool", "get_strategy", 1),
        ("tool", "update_strategy", 2),
    ]
    assert redis.state() == {**relay.state, "phase": "tool", "name": "update_strategy", "tools": 2}

    # Back to the model after the tools, then the reply starts streaming.
    assert relay.feed(_item(activity="thinking"))[0]["phase"] == "thinking"
    changes = relay.feed(_item("Fatto. ") + _item("Ho salvato."))
    assert [c["phase"] for c in changes] == ["writing"]
    assert redis.state()["request_id"] == "r1"

    assert relay.feed(END) == []
    assert redis.state() is None
    assert [p["type"] for p in redis.published] == [
        "turn_start", "tool_call", "tool_call", "activity", "delta", "delta", "turn_end",
    ]


def test_relay_drops_the_state_when_the_turn_breaks(redis):
    relay = TurnRelay("s1", "r1")
    relay.feed(BEGIN)
    assert redis.state() is not None
    relay.error("agent-svc gone")
    assert redis.state() is None
    assert redis.published[-1]["type"] == "turn_error"
    relay.touch()  # a late heartbeat must not resurrect it
    assert redis.state() is None


def _patch_chat_plumbing(monkeypatch, handler):
    chat = SimpleNamespace(id=7, user_id=1)
    agent = SimpleNamespace(id_agent=3)
    monkeypatch.setattr(chat_service, "_get_owned_chat", lambda *a, **k: chat)
    monkeypatch.setattr(chat_service, "_ensure_chat_accepts_messages", lambda *a, **k: None)
    monkeypatch.setattr(chat_service, "check_ai_budget", lambda *a, **k: None)
    monkeypatch.setattr(chat_service, "_resolve_chat_agent", lambda *a, **k: agent)
    monkeypatch.setattr(chat_service, "_chat_session_id", lambda c: "s1")
    monkeypatch.setattr(chat_service, "issue_n8n_webhook_auth_token", lambda *a, **k: "tok")
    monkeypatch.setattr(chat_service, "_build_chat_api_auth_metadata", lambda *a, **k: {})
    monkeypatch.setattr(chat_service, "_build_webhook_payload", lambda *a, **k: {})
    monkeypatch.setattr(chat_service, "build_n8n_webhook_auth_headers", lambda t: {})
    monkeypatch.setattr(chat_service, "agent_webhook_url", lambda: "http://agent/webhook")
    monkeypatch.setattr(chat_service, "persist_asker_message", lambda *a, **k: 1)
    estimates: list[dict] = []
    monkeypatch.setattr(chat_service, "_record_turn_estimate", lambda **k: estimates.append(k))
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        chat_service.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )
    return estimates


class _Body(httpx.AsyncByteStream):
    def __init__(self, steps):
        self.steps = steps

    async def __aiter__(self):
        for delay, chunk in self.steps:
            await asyncio.sleep(delay)
            yield chunk.encode()


def _parse(blocks):
    events = []
    for block in blocks:
        if block.startswith(":"):
            events.append(("ping", None))
            continue
        name, data = block.strip().split("\n", 1)
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


def test_user_stream_reports_activity_and_keeps_the_connection_alive(redis, monkeypatch):
    monkeypatch.setattr(chat_service, "CHAT_STREAM_HEARTBEAT_SECONDS", 0.05)
    steps = [
        (0, BEGIN + _item(activity="thinking")),
        (0.18, _item(tool_call="update_strategy")),  # silent reasoning step
        (0, _item(activity="thinking") + _item("Fatto.")),
        (0, END),
    ]
    estimates = _patch_chat_plumbing(monkeypatch, lambda request: httpx.Response(200, stream=_Body(steps)))

    async def run():
        _, stream = await chat_service.stream_chat_message(
            SimpleNamespace(close=lambda: None), chat_id=7, user_id=1, text="riprova"
        )
        return [block async for block in stream]

    events = _parse(asyncio.run(run()))
    names = [name for name, _ in events]
    assert names[0] == "message_start" and names[-1] == "message_end"
    assert names.count("ping") >= 2
    activity = [(d["phase"], d.get("name"), d["tools"]) for n, d in events if n == "message_activity"]
    assert activity == [
        ("thinking", None, 0),
        ("tool", "update_strategy", 1),
        ("thinking", None, 1),
        ("writing", None, 1),
    ]
    assert events[-1][1]["full_text"] == "Fatto."
    assert estimates and estimates[0]["response_chars"] == len("Fatto.")
    assert redis.state() is None


def test_user_stream_relay_outlives_the_client(redis, monkeypatch):
    steps = [(0, BEGIN), (0.1, _item(tool_call="get_strategy")), (0.1, _item("Ecco.") + END)]
    _patch_chat_plumbing(monkeypatch, lambda request: httpx.Response(200, stream=_Body(steps)))

    async def run():
        _, stream = await chat_service.stream_chat_message(
            SimpleNamespace(close=lambda: None), chat_id=7, user_id=1, text="riprova"
        )
        await stream.__anext__()  # message_start
        await stream.__anext__()  # first activity, then the page goes away
        await stream.aclose()
        mid = redis.state()
        await asyncio.gather(*list(chat_service._STREAM_PUMPS))
        return mid

    mid = asyncio.run(run())
    assert mid and mid["request_id"]
    assert [p["type"] for p in redis.published] == ["turn_start", "tool_call", "delta", "turn_end"]
    assert redis.state() is None


def test_user_stream_reports_an_agent_error(redis, monkeypatch):
    _patch_chat_plumbing(monkeypatch, lambda request: httpx.Response(503, text="busy"))

    async def run():
        _, stream = await chat_service.stream_chat_message(
            SimpleNamespace(close=lambda: None), chat_id=7, user_id=1, text="riprova"
        )
        return [block async for block in stream]

    events = _parse(asyncio.run(run()))
    assert events[-1][0] == "message_error"
    assert "503" in events[-1][1]["detail"] and "busy" in events[-1][1]["detail"]
