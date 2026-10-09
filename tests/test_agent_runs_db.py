"""Runs (migr. 069, bridge F4) against an embedded Postgres: creation and
chat resolution, budget, execution as a chat turn (dispatch monkeypatched),
Paperclip heartbeat + callback, A2A task mapping.
Run: ``venv/bin/python -m pytest tests/test_agent_runs_db.py``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from fastapi import HTTPException
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
    stale = session.exec(select(User).where(User.email == "run@example.com")).first()
    if stale is not None:
        session.delete(stale)
        session.commit()
    user = User(email="run@example.com", username="user_run", hashed_password="x")
    session.add(user)
    session.flush()
    conn = Connection(user_id=user.id, name="conn-run", broker_type="binance", status="connected", created_at=now, updated_at=now)
    session.add(conn)
    session.flush()
    acc = Account(connection_id=conn.id, account_id="acc-run", currency="USDT", created_at=now, updated_at=now)
    session.add(acc)
    session.commit()
    yield user, conn, acc
    session.delete(user)
    session.commit()


def _agent(session, user, name="Runner", **kw):
    from app.schemas.agent import AgentCreate
    from app.services.agent_service import create_agent

    return create_agent(session, AgentCreate(agent_name=name, **kw), user.id)


def _fake_dispatch(monkeypatch, *, answer="Fatto: le live vanno bene.", credits=0.7, raise_exc=None):
    """Stand-in for chat_service.send_chat_message: writes the agent's answer
    row the way agent-svc does (turn_usage with the request id)."""
    from app.schemas.chat import ChatSendMessageResponse
    from app.services import chat_service

    calls = []

    def fake(session, *, chat_id, user_id, text, metadata=None, sender_kind="user", sender_label=None):
        calls.append({"chat_id": chat_id, "text": text, "metadata": metadata, "sender_kind": sender_kind, "sender_label": sender_label})
        if raise_exc:
            raise raise_exc
        request_id = f"req-{len(calls)}"
        chat_service.persist_asker_message(session, session_id=str(chat_id), text=text, sender_kind=sender_kind, sender_label=sender_label, metadata={"request_id": request_id})
        from app.models.n8n_chat_history import N8nChatHistory

        session.add(N8nChatHistory(session_id=str(chat_id), message={
            "type": "ai", "content": answer, "tool_calls": [], "additional_kwargs": {},
            "response_metadata": {"edgewalker": {"turn_usage": {"correlation_id": request_id, "credits": credits, "input_tokens": 120, "output_tokens": 40, "tokens_cached": 10, "wallet_cents": 3}}},
            "metadata": {"agent_name": "Runner"},
        }))
        session.commit()
        return ChatSendMessageResponse(status="ok", chat_id=chat_id, session_id=str(chat_id), request_id=request_id)

    monkeypatch.setattr(chat_service, "send_chat_message", fake)
    return calls


def test_create_run_resolves_chat_and_enforces_rules(session, tenant):
    from app.models.agent import Chat
    from app.models.agent_model import AgentRun
    from app.schemas.agent_run import RunCreate
    from app.schemas.strategy import StrategyCreate
    from app.services import agent_run_service as runs
    from app.services.strategy_service import create_strategy

    user, conn, acc = tenant
    agent, default_chat = _agent(session, user)
    run = runs.create_run(session, user_id=user.id, agent_id=agent.id_agent, payload=RunCreate(task="Briefing", context={"from": "test"}))
    assert run.status == "queued" and run.chat_id == default_chat.id and run.scope == "agent"

    with pytest.raises(HTTPException) as exc:
        runs.create_run(session, user_id=user.id, agent_id=agent.id_agent, payload=RunCreate(task="x", scope="strategy"))
    assert exc.value.status_code == 400
    definition = {"strategy": {
        "name": "r", "symbol": "BTCUSDT", "asset": "crypto", "timeframe": "5m", "timezone": "UTC", "params": {},
        "indicators": [], "charts": [{"id": "main", "symbol": "BTCUSDT", "asset": "crypto", "timeframe": "5m", "indicators": []}], "rules": [],
    }}
    strategy = create_strategy(session, StrategyCreate(name="R", definition=definition, account_id=acc.id), user.id)
    run2 = runs.create_run(session, user_id=user.id, agent_id=agent.id_agent, payload=RunCreate(task="Aggiungi un filtro", scope="strategy", strategy_id=strategy.id))
    chat = session.get(Chat, run2.chat_id)
    assert chat.strategy_id == strategy.id and chat.id_agent == agent.id_agent
    run3 = runs.create_run(session, user_id=user.id, agent_id=agent.id_agent, payload=RunCreate(task="Ancora", scope="strategy", strategy_id=strategy.id))
    assert run3.chat_id == run2.chat_id  # reused

    ext, _ = _agent(session, user, "Ext", kind="external")
    with pytest.raises(HTTPException) as exc:
        runs.create_run(session, user_id=user.id, agent_id=ext.id_agent, payload=RunCreate(task="x"))
    assert exc.value.status_code == 422

    # budget: 1 credit/month, 1.5 already spent → 402
    agent.budget_credits_month = 1
    session.add(agent)
    run.cost_credits = Decimal("1.5")
    session.add(run)
    session.commit()
    assert runs.credits_spent_this_month(session, agent.id_agent) == 1.5
    with pytest.raises(HTTPException) as exc:
        runs.create_run(session, user_id=user.id, agent_id=agent.id_agent, payload=RunCreate(task="x"))
    assert exc.value.status_code == 402
    assert [r.id for r in runs.list_runs(session, user_id=user.id, agent_id=agent.id_agent)] == [run3.id, run2.id, run.id]
    assert isinstance(session.get(AgentRun, run.id), AgentRun)


def test_execute_run_copies_answer_usage_and_emits_event(session, tenant, monkeypatch):
    from app.models.webhook import WebhookDelivery
    from app.schemas.agent_run import RunCreate
    from app.services import agent_run_service as runs
    from app.services import webhook_service as ws

    user, _, _ = tenant
    agent, _ = _agent(session, user)
    sub, _ = ws.create_subscription(session, user_id=user.id, name="hook", url="https://example.com/h", events=["agent.run.completed"])
    calls = _fake_dispatch(monkeypatch)
    run = runs.create_run(session, user_id=user.id, agent_id=agent.id_agent, payload=RunCreate(task="Come stanno le live?", context={"issue": "PC-12"}, external_run_id="ext-1"))
    done = runs.execute_run(session, run.id)
    assert done.status == "succeeded" and done.result == "Fatto: le live vanno bene."
    assert float(done.cost_credits) == 0.7 and done.usage["input_tokens"] == 120 and done.request_id == "req-1"
    assert calls[0]["sender_kind"] == "external_agent" and calls[0]["sender_label"] == "API · run #%d" % run.id
    assert calls[0]["text"].startswith("Come stanno le live?") and "- issue: PC-12" in calls[0]["text"]
    assert calls[0]["metadata"]["run_id"] == run.id
    deliveries = session.exec(select(WebhookDelivery).where(WebhookDelivery.subscription_id == sub.id)).all()
    assert len(deliveries) == 1 and deliveries[0].payload["data"]["status"] == "succeeded" and deliveries[0].payload["data"]["external_run_id"] == "ext-1"

    # dispatch failure → failed, with the detail
    _fake_dispatch(monkeypatch, raise_exc=HTTPException(status_code=402, detail="AI credits exhausted"))
    run2 = runs.create_run(session, user_id=user.id, agent_id=agent.id_agent, payload=RunCreate(task="x"))
    failed = runs.execute_run(session, run2.id)
    assert failed.status == "failed" and failed.error == "AI credits exhausted"
    assert runs.execute_run(session, run2.id).status == "failed"  # idempotent: not queued any more


def test_paperclip_heartbeat_and_callback(session, tenant, monkeypatch):
    import httpx

    from app.schemas.agent_run import PaperclipHeartbeat
    from app.services import agent_run_service as runs

    user, _, _ = tenant
    agent, _ = _agent(session, user)
    _fake_dispatch(monkeypatch, answer="Task PC-12 completato.")
    posted = []

    class FakeClient:
        def __init__(self, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None, headers=None):
            posted.append((url, json, headers))
            return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(runs.httpx, "Client", FakeClient)
    beat = PaperclipHeartbeat.model_validate({
        "runId": "run_abc", "agentId": "ag_1", "companyId": "co_1", "taskId": "task_9", "issueId": "task_9", "wakeReason": "task_assigned",
        "issueIds": ["task_9"], "context": {"taskId": "task_9", "wakeReason": "task_assigned", "paperclipWorkspace": {"cwd": "/w"}},
        "paperclipApiUrl": "https://paperclip.example/", "paperclipApiKey": "pk_secret", "task": "Fai un backtest della strategia 3 sull ultimo mese.",
        "priority": "high",
    })
    run = runs.run_from_paperclip(session, user_id=user.id, agent_id=agent.id_agent, beat=beat)
    assert run.source == "paperclip" and run.external_run_id == "run_abc"
    assert run.callback_url == "https://paperclip.example/api/heartbeat-runs/run_abc/callback"
    assert run.callback_auth and run.callback_auth != "pk_secret" and runs._decrypt(run.callback_auth) == "pk_secret"
    assert run.task.startswith("Richiesta da Paperclip (task_assigned: task task_9)") and "Fai un backtest" in run.task
    assert run.context["paperclip"]["taskId"] == "task_9" and "paperclipWorkspace" not in run.context["paperclip"] and run.context["priority"] == "high"

    done = runs.execute_run(session, run.id)
    assert done.status == "succeeded" and done.callback_status == "delivered"
    url, body, headers = posted[0]
    assert url == run.callback_url and headers["Authorization"] == "Bearer pk_secret"
    assert body["status"] == "succeeded" and body["result"] == "Task PC-12 completato." and body["provider"] == "edgewalker"
    assert body["usage"] == {"inputTokens": 120, "outputTokens": 40, "cachedInputTokens": 10} and body["costUsd"] == 0.03
    assert "errorMessage" not in body

    # no api url → no callback, heartbeat without an explicit task gets a default text
    beat2 = PaperclipHeartbeat(runId="run_2", wakeReason="heartbeat")
    run2 = runs.run_from_paperclip(session, user_id=user.id, agent_id=agent.id_agent, beat=beat2)
    assert run2.callback_url is None and "briefing" in run2.task
    done2 = runs.execute_run(session, run2.id)
    assert done2.callback_status is None and len(posted) == 1

    # generic callback body for API runs
    _fake_dispatch(monkeypatch, raise_exc=RuntimeError("boom"))
    from app.schemas.agent_run import RunCreate

    run3 = runs.create_run(session, user_id=user.id, agent_id=agent.id_agent, payload=RunCreate(task="x", callback_url="https://me.example/cb", callback_auth="Bearer tok"))
    done3 = runs.execute_run(session, run3.id)
    assert done3.status == "failed" and posted[-1][2]["Authorization"] == "Bearer tok"
    assert posted[-1][1]["status"] == "failed" and posted[-1][1]["error"] == "boom" and posted[-1][1]["run_id"] == run3.id


def test_a2a_task_and_agent_card(session, tenant, monkeypatch):
    from app.schemas.agent_run import RunCreate
    from app.services import agent_run_service as runs

    user, _, _ = tenant
    agent, _ = _agent(session, user, "Card Agent")
    card = runs.agent_card(agent, base_url="https://api.edgewalker.tech")
    assert card["url"] == f"https://api.edgewalker.tech/a2a/agents/{agent.id_agent}" and card["name"] == "Card Agent"
    assert {s["id"] for s in card["skills"]} == {"briefing", "design", "review"} and card["metadata"]["edgewalker"]["slug"] == "card-agent"
    assert runs.a2a_message_text({"parts": [{"kind": "text", "text": "Ciao"}, {"kind": "file"}, {"text": " mondo"}]}) == "Ciao\n mondo"

    _fake_dispatch(monkeypatch, answer="Risposta A2A")
    run = runs.create_run(session, user_id=user.id, agent_id=agent.id_agent, payload=RunCreate(task="Ciao", source="a2a", external_run_id="msg-1"))
    task = runs.a2a_task(run)
    assert task["status"]["state"] == "submitted" and task["id"] == str(run.id) and "artifacts" not in task
    done = runs.execute_run(session, run.id)
    task = runs.a2a_task(done)
    assert task["status"]["state"] == "completed" and task["artifacts"][0]["parts"][0]["text"] == "Risposta A2A"
    assert json.dumps(task)  # JSON-safe
