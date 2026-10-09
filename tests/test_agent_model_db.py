"""Agent model F3 (migr. 068) against an embedded Postgres: policy + slug on
the agent, skills (CRUD + from playbook), memory, action requests
(create → chat row + webhook, reject, approve → executor, expiry), history
search. Run: ``venv/bin/python -m pytest tests/test_agent_model_db.py``.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

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
    monkeypatch.setattr(settings, "SECRET_KEY", "unit-test-secret-key-for-agent-model")


@pytest.fixture
def tenant(session):
    from app.models.connection import Account, Connection
    from app.models.user import User

    now = datetime.now(timezone.utc)
    stale = session.exec(select(User).where(User.email == "am@example.com")).first()
    if stale is not None:
        session.delete(stale)
        session.commit()
    user = User(email="am@example.com", username="user_am", hashed_password="x")
    session.add(user)
    session.flush()
    conn = Connection(user_id=user.id, name="conn-am", broker_type="binance", status="connected", created_at=now, updated_at=now)
    session.add(conn)
    session.flush()
    acc = Account(connection_id=conn.id, account_id="acc-am", currency="USDT", created_at=now, updated_at=now)
    session.add(acc)
    session.commit()
    yield user, conn, acc
    session.delete(user)
    session.commit()


def _agent(session, user, name="Aurelio", **kw):
    from app.schemas.agent import AgentCreate
    from app.services.agent_service import create_agent

    agent, chat = create_agent(session, AgentCreate(agent_name=name, **kw), user.id)
    return agent, chat


def test_agent_policy_slug_and_external_rules(session, tenant):
    from app.schemas.agent import AgentRead, AgentUpdate
    from app.services.agent_service import update_agent

    user, _, _ = tenant
    agent, chat = _agent(session, user, "Marco Polo!", tool_policy={"trading": "ask", "strategies": "off"})
    assert agent.slug == "marco-polo" and chat is not None
    twin, _ = _agent(session, user, "marco polo")
    assert twin.slug == "marco-polo-2"
    read = AgentRead.model_validate(agent, from_attributes=True)
    assert read.tool_policy == {"trading": "ask", "strategies": "off"}
    assert read.tool_policy_effective["trading"] == "ask" and read.tool_policy_effective["strategies"] == "off"
    assert read.tool_policy_effective["backtests"] == "allow"

    # propose autonomy → trading asks by default
    updated = update_agent(session, agent.id_agent, AgentUpdate(tool_policy={}, settings={"autonomy": "propose"}), user.id)
    assert AgentRead.model_validate(updated, from_attributes=True).tool_policy_effective["trading"] == "ask"
    # a rename keeps the slug
    updated = update_agent(session, agent.id_agent, AgentUpdate(agent_name="Marco"), user.id)
    assert updated.slug == "marco-polo"

    with pytest.raises(ValueError):
        AgentUpdate(tool_policy={"strategies": "ask"})

    ext, no_chat = _agent(session, user, "Claude", kind="external")
    assert no_chat is None
    assert AgentRead.model_validate(ext, from_attributes=True).tool_policy_effective["trading"] == "off"
    with pytest.raises(HTTPException) as exc:
        update_agent(session, ext.id_agent, AgentUpdate(tool_policy={"trading": "allow"}), user.id)
    assert exc.value.status_code == 400
    update_agent(session, ext.id_agent, AgentUpdate(tool_policy={"trading": "off", "strategies": "allow"}), user.id)


def test_skills_crud_and_from_playbook(session, tenant):
    from app.models.agent_lesson import AgentLesson
    from app.schemas.agent_model import SkillCreate, SkillFromPlaybook, SkillUpdate
    from app.schemas.strategy import StrategyCreate
    from app.services import agent_skill_service as skills
    from app.services.strategy_service import create_strategy

    user, conn, acc = tenant
    row = skills.create_skill(session, user.id, SkillCreate(name="morning-routine", description="How I start", body="# Routine\n1. dashboard"))
    assert row.version == 1 and row.origin == "user"
    with pytest.raises(HTTPException) as exc:
        skills.create_skill(session, user.id, SkillCreate(name="morning-routine", body="x"))
    assert exc.value.status_code == 409
    row = skills.update_skill(session, user.id, "morning-routine", SkillUpdate(body="# Routine v2"))
    assert row.version == 2
    assert skills.render_skill_markdown(row).startswith("---\nname: morning-routine\ndescription: How I start\n---\n")
    assert [s.name for s in skills.list_skills(session, user.id, names=["morning-routine", "nope"])] == ["morning-routine"]

    definition = {"strategy": {
        "name": "pb", "symbol": "BTCUSDT", "asset": "crypto", "timeframe": "5m", "timezone": "UTC", "params": {},
        "indicators": [], "charts": [{"id": "main", "symbol": "BTCUSDT", "asset": "crypto", "timeframe": "5m", "indicators": []}], "rules": [],
    }}
    strategy = create_strategy(session, StrategyCreate(name="PB", definition=definition, account_id=acc.id), user.id)
    # agent_lessons is created by the lessons migration, not by create_all
    AgentLesson.__table__.create(session.get_bind(), checkfirst=True)
    with pytest.raises(HTTPException) as exc:
        skills.skill_from_playbook(session, user.id, SkillFromPlaybook(name="pb-skill", strategy_id=strategy.id))
    assert exc.value.status_code == 404
    session.add(AgentLesson(strategy_id=strategy.id, user_id=user.id, lesson="Non inseguire il breakout.", context="apertura", confidence=0.7, source="backtest", scope="strategy"))
    session.add(AgentLesson(strategy_id=strategy.id, user_id=user.id, lesson="Ipotesi: ATR basso = niente.", confidence=0.4, source="backtest", scope="strategy"))
    session.commit()
    skill = skills.skill_from_playbook(session, user.id, SkillFromPlaybook(name="pb-skill", strategy_id=strategy.id))
    assert skill.origin == "playbook" and skill.source_strategy_id == strategy.id
    assert "[validated, confidence 0.70] Non inseguire il breakout. — when: apertura" in skill.body
    assert "[hypothesis, confidence 0.40]" in skill.body

    skills.delete_skill(session, user.id, "pb-skill")
    with pytest.raises(HTTPException):
        skills.get_skill(session, user.id, "pb-skill")


def test_memory_upsert_and_limits(session, tenant):
    from app.services import agent_memory_service as mem

    user, _, _ = tenant
    agent, _ = _agent(session, user, "Mem")
    assert [m.content for m in mem.list_memory(session, agent)] == ["", "", ""]
    out = mem.set_memory(session, agent, "user_profile", "  Preferisce MNQ.  ", updated_by="agent")
    assert out.content == "Preferisce MNQ." and out.updated_by == "agent"
    out = mem.set_memory(session, agent, "user_profile", "Preferisce NQ.", updated_by="user")
    assert out.updated_by == "user"
    rows = mem.list_memory(session, agent)
    assert rows[0].kind == "user_profile" and rows[0].content == "Preferisce NQ."
    with pytest.raises(HTTPException) as exc:
        mem.set_memory(session, agent, "user_profile", "x" * 5000, updated_by="user")
    assert exc.value.status_code == 422
    with pytest.raises(HTTPException):
        mem.set_memory(session, agent, "dreams", "x", updated_by="user")
    ext, _ = _agent(session, user, "Ext", kind="external")
    with pytest.raises(HTTPException):
        mem.set_memory(session, ext, "user_profile", "x", updated_by="user")


def _history_rows(session, chat):
    from sqlalchemy import text

    return session.execute(
        text("SELECT message FROM n8n_chat_histories WHERE session_id = :sid ORDER BY id"), {"sid": str(chat.id)}
    ).scalars().all()


def test_action_requests_lifecycle(session, tenant, monkeypatch):
    from app.models.webhook import WebhookDelivery
    from app.schemas.agent_model import ActionRequestCreate
    from app.services import agent_action_service as actions
    from app.services import webhook_service as ws

    user, conn, acc = tenant
    agent, chat = _agent(session, user, "Asker", settings={"autonomy": "propose"})
    sub, _ = ws.create_subscription(session, user_id=user.id, name="hook", url="https://example.com/h", events=["agent.action.requested", "agent.action.decided"])

    payload = ActionRequestCreate(
        tool_name="place_order",
        args={"symbol": "BTCUSDT", "side": "buy", "quantity": 0.1, "order_type": "limit", "limit_price": 59000, "stop_loss_price": 58000},
        rationale="Pullback sulla EMA 20", chat_id=chat.id, account_id=acc.id,
    )
    with pytest.raises(HTTPException) as exc:
        actions.create_request(session, user_id=user.id, agent_id=agent.id_agent, payload=ActionRequestCreate(tool_name="update_strategy", account_id=acc.id))
    assert exc.value.status_code == 400
    row = actions.create_request(session, user_id=user.id, agent_id=agent.id_agent, payload=payload)
    assert row.status == "pending" and row.account_id == acc.id and row.chat_row_id is not None
    assert actions.summarize(row.tool_name, row.args) == "Ordine: BUY 0.1 BTCUSDT limit @ 59000 (SL 58000)"
    rows = _history_rows(session, chat)
    assert rows[-1]["type"] == "system" and rows[-1]["metadata"]["action_request"]["id"] == row.id
    assert rows[-1]["metadata"]["action_request"]["status"] == "pending"
    deliveries = session.exec(select(WebhookDelivery).where(WebhookDelivery.subscription_id == sub.id)).all()
    assert [d.event_type for d in deliveries] == ["agent.action.requested"]
    assert deliveries[0].payload["data"]["summary"].startswith("Ordine: BUY")

    # reject
    decided = asyncio.run(actions.decide(session, user_id=user.id, request_id=row.id, approve=False, decided_by={"via": "ui", "user_id": user.id}, note="troppo presto"))
    assert decided.status == "rejected" and decided.decided_by["note"] == "troppo presto"
    rows = _history_rows(session, chat)
    assert rows[-1]["content"].startswith("Richiesta RIFIUTATA") and rows[0]["metadata"]["action_request"]["status"] == "rejected"
    with pytest.raises(HTTPException) as exc:
        asyncio.run(actions.decide(session, user_id=user.id, request_id=row.id, approve=True, decided_by={}))
    assert exc.value.status_code == 409

    # approve → executor (monkeypatched broker command)
    calls = []

    async def fake_place(session_, account, **kw):
        calls.append((account.id, kw))
        return {"order_id": "o-1", "status": "submitted"}

    import app.services.order_command_service as occ

    monkeypatch.setattr(occ, "place_account_order", fake_place)
    row2 = actions.create_request(session, user_id=user.id, agent_id=agent.id_agent, payload=payload)
    approved = asyncio.run(actions.decide(session, user_id=user.id, request_id=row2.id, approve=True, decided_by={"via": "ui", "user_id": user.id}))
    assert approved.status == "approved" and approved.result == {"order_id": "o-1", "status": "submitted"}
    assert calls[0][0] == acc.id and calls[0][1]["side"] == "buy" and calls[0][1]["limit_price"] == 59000.0
    assert calls[0][1]["extra"]["actor"]["agent_id"] == agent.id_agent and calls[0][1]["extra"]["approval"]["request_id"] == row2.id
    assert _history_rows(session, chat)[-1]["content"].startswith("Richiesta APPROVATA")

    # executor failure → failed, error recorded
    async def boom(session_, account, **kw):
        raise RuntimeError("gateway down")

    monkeypatch.setattr(occ, "place_account_order", boom)
    row3 = actions.create_request(session, user_id=user.id, agent_id=agent.id_agent, payload=payload)
    failed = asyncio.run(actions.decide(session, user_id=user.id, request_id=row3.id, approve=True, decided_by={}))
    assert failed.status == "failed" and "gateway down" in failed.error

    # expiry sweep
    row4 = actions.create_request(session, user_id=user.id, agent_id=agent.id_agent, payload=payload)
    row4.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    session.add(row4)
    session.commit()
    assert actions.expire_pending(session) == 1
    session.refresh(row4)
    assert row4.status == "expired"
    assert [r.status for r in actions.list_requests(session, user_id=user.id, status_filter="pending")] == []
    kinds = [d.event_type for d in session.exec(select(WebhookDelivery).where(WebhookDelivery.subscription_id == sub.id)).all()]
    assert kinds.count("agent.action.requested") == 4 and kinds.count("agent.action.decided") == 4

    # external agents cannot queue
    ext, _ = _agent(session, user, "Ext2", kind="external")
    with pytest.raises(HTTPException) as exc:
        actions.create_request(session, user_id=user.id, agent_id=ext.id_agent, payload=payload)
    assert exc.value.status_code == 422


def test_history_search(session, tenant):
    from app.services.agent_history_service import search_history
    from app.services.chat_service import persist_asker_message

    user, _, _ = tenant
    agent, chat = _agent(session, user, "Hist")
    persist_asker_message(session, session_id=str(chat.id), text="Preferisco operare su MNQ al mattino", sender_kind="user")
    persist_asker_message(session, session_id=str(chat.id), text="Ricevuto: MNQ al mattino.", sender_kind="agent", message_type="ai")
    persist_asker_message(session, session_id=str(chat.id), text="(tool noise)", sender_kind="tool_result", message_type="tool")
    hits = search_history(session, user_id=user.id, agent_id=agent.id_agent, query="mnq")
    assert [h.type for h in hits] == ["ai", "human"] and all("MNQ" in h.text for h in hits)
    assert hits[0].chat_id == chat.id
    assert search_history(session, user_id=user.id, agent_id=agent.id_agent, query="x") == []
    assert search_history(session, user_id=user.id, agent_id=agent.id_agent + 999, query="mnq") == []
