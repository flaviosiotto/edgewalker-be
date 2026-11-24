"""Mid-turn questions: what the backend shows and how it delivers the answer."""
import asyncio
import json
import time

import pytest
from fastapi import HTTPException

from app.schemas.agent_model import SkillRead
from app.services import chat_ask_service, user_ask
from app.services.chat_service import TurnRelay


class FakeRedis:
    def __init__(self):
        self.values: dict[str, str] = {}
        self.lists: dict[str, list[str]] = {}
        self.expires: dict[str, int] = {}
        self.published: list[tuple[str, dict]] = []

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, ex=None, nx=False):
        if nx and key in self.values:
            return None
        self.values[key] = value
        return True

    async def rpush(self, key, value):
        self.lists.setdefault(key, []).append(value)
        return len(self.lists[key])

    async def rpop(self, key):
        return self.lists[key].pop()

    async def blpop(self, key, timeout=0):
        self.waits = getattr(self, "waits", []) + [timeout]
        items = self.lists.get(key) or []
        return (key, items.pop(0)) if items else None

    async def expire(self, key, seconds):
        self.expires[key] = seconds

    async def delete(self, *keys):
        for key in keys:
            self.values.pop(key, None)
            self.lists.pop(key, None)

    async def publish(self, channel, payload):
        self.published.append((channel, json.loads(payload)))


@pytest.fixture
def redis(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(chat_ask_service, "get_async_redis", lambda: fake)
    return fake


def _pending(redis, **kw):
    ask = user_ask.build_ask(session_id="621", question="Quale variante?", options=["V1", "V2"], **kw)
    redis.values[user_ask.pending_key("621")] = json.dumps(ask)
    return ask


def _status(coro) -> int:
    with pytest.raises(HTTPException) as err:
        asyncio.run(coro)
    return err.value.status_code


def test_pending_ask_is_read_until_it_expires(redis):
    assert asyncio.run(chat_ask_service.read_pending_ask("621")) is None
    ask = _pending(redis)
    assert asyncio.run(chat_ask_service.read_pending_ask("621"))["id"] == ask["id"]
    _pending(redis, now=time.time() - 1000)
    assert asyncio.run(chat_ask_service.read_pending_ask("621")) is None


def test_answer_reaches_the_waiting_turn_once(redis):
    ask = _pending(redis)
    out = asyncio.run(chat_ask_service.answer_ask("621", ask["id"], option=1, user_id=7))
    assert out == {"ask_id": ask["id"], "status": "answered", "answer": "V2"}
    key = user_ask.answer_key(ask["id"])
    assert json.loads(redis.lists[key][0]) == {"status": "answered", "answer": "V2", "option": 1, "user_id": 7}
    assert redis.expires[key] == user_ask.ANSWER_TTL_SECONDS
    assert _status(chat_ask_service.answer_ask("621", ask["id"], option=0)) == 409
    assert len(redis.lists[key]) == 1


def test_answer_refusals(redis):
    assert _status(chat_ask_service.answer_ask("621", "nope", option=0)) == 404
    ask = _pending(redis)
    assert _status(chat_ask_service.answer_ask("621", "other-id", option=0)) == 404
    assert _status(chat_ask_service.answer_ask("621", ask["id"], option=5)) == 422
    assert _status(chat_ask_service.answer_ask("621", ask["id"])) == 422
    old = _pending(redis, now=time.time() - 1000)
    assert _status(chat_ask_service.answer_ask("621", old["id"], option=0)) == 410


def test_answer_without_redis_is_503(monkeypatch):
    monkeypatch.setattr(chat_ask_service, "get_async_redis", lambda: None)
    assert _status(chat_ask_service.answer_ask("621", "x", skip=True)) == 503


def test_relay_reports_the_waiting_phase_once(monkeypatch):
    published: list[dict] = []
    relay = TurnRelay("621", "req-1")
    monkeypatch.setattr(relay, "_client", lambda: None)
    monkeypatch.setattr(relay, "_publish", published.append)
    line = json.dumps({"type": "item", "content": "", "metadata": {"activity": "waiting_user"}}) + "\n"
    relay.feed(json.dumps({"type": "begin", "metadata": {}}) + "\n")
    changes = relay.feed(line + line)
    assert [c["phase"] for c in changes] == ["waiting"]
    assert [p["phase"] for p in published if p["type"] == "activity"] == ["waiting"]
    assert relay.feed(json.dumps({"type": "item", "content": "ok", "metadata": {}}) + "\n")[0]["phase"] == "writing"


def test_skill_read_exposes_declared_inputs():
    skill = SkillRead(
        id=0, name="x", description="", version=1, origin="user", updated_at="2026-11-24T00:00:00", created_at="2026-11-24T00:00:00",
        body="# X\n\n## Input\n- modo (scelta: a | b; default a): come\n",
    )
    assert skill.model_dump()["inputs"] == [
        {"key": "modo", "type": "choice", "label": "come", "options": ["a", "b"], "required": False, "default": "a"}
    ]


def _open(redis, **kw):
    fields = {"question": "Quale variante?", "options": ["V1", "V2"], "turn_id": "t1", "request_id": "r1", **kw}
    return asyncio.run(chat_ask_service.open_ask("621", **fields))


def test_open_stores_announces_and_refuses_a_second_question(redis):
    ask = _open(redis)
    assert json.loads(redis.values["chat-ask:621"])["id"] == ask["id"]
    assert redis.published == [("chat-stream:621", {"v": 1, "type": "ask", "request_id": "r1", "ask": ask})]
    assert _status(chat_ask_service.open_ask("621", question="Altra?", options=["a", "b"])) == 409
    assert _status(chat_ask_service.open_ask("622", question="", options=["a", "b"])) == 422
    assert _status(chat_ask_service.open_ask("622", question="q", options=["solo una"])) == 422
    assert len(redis.published) == 1


def test_wait_returns_pending_then_the_answer_and_closes_the_question(redis):
    ask = _open(redis)
    assert asyncio.run(chat_ask_service.wait_answer("621", ask["id"], seconds=5)) == {"status": "pending"}
    assert redis.waits == [5]
    asyncio.run(chat_ask_service.answer_ask("621", ask["id"], text="v2"))
    assert asyncio.run(chat_ask_service.wait_answer("621", ask["id"], seconds=999)) == {"status": "answered", "answer": "V2", "option": 1}
    assert redis.waits[-1] == user_ask.WAIT_MAX_SECONDS
    assert redis.values == {} and redis.lists == {}
    assert redis.published[-1][1] == {"v": 1, "type": "ask_resolved", "request_id": "r1", "ask_id": ask["id"], "status": "answered", "answer": "V2"}
    assert asyncio.run(chat_ask_service.wait_answer("621", ask["id"], seconds=5)) == {"status": "gone"}
    assert asyncio.run(chat_ask_service.read_pending_ask("621")) is None


def test_wait_skip_and_deadline(redis):
    ask = _open(redis)
    asyncio.run(chat_ask_service.answer_ask("621", ask["id"], skip=True))
    assert asyncio.run(chat_ask_service.wait_answer("621", ask["id"], seconds=5)) == {"status": "skipped", "answer": "", "option": None}
    assert redis.published[-1][1]["status"] == "skipped"
    old = _pending(redis, now=time.time() - 1000)
    assert asyncio.run(chat_ask_service.wait_answer("621", old["id"], seconds=5)) == {"status": "timeout"}
    assert redis.values == {} and redis.published[-1][1]["status"] == "timeout"


def test_close_is_idempotent_and_tells_the_clients(redis):
    ask = _open(redis)
    assert asyncio.run(chat_ask_service.close_ask("621", ask["id"], reason="timeout")) == {"status": "timeout"}
    assert redis.values == {} and redis.published[-1][1]["status"] == "timeout"
    assert asyncio.run(chat_ask_service.close_ask("621", ask["id"])) == {"status": "gone"}
    assert asyncio.run(chat_ask_service.close_ask("621", "other", reason="boh")) == {"status": "gone"}
    assert len(redis.published) == 2
