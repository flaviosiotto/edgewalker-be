"""Agent bridge F1 (migration 066) — service rules against an embedded Postgres.

* a PAT may be bound to one of the user's agents; bound to an *external*
  agent it never gets ``trade``;
* an external agent is never the manager of a strategy nor the answerer of
  an ``ask_agent`` rule (422), while a hosted one is;
* an external agent is never the default agent and gets no default chat;
* the actor of the request is recorded on the strategy.
Run: ``venv/bin/python -m pytest tests/test_agent_bridge_db.py``.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi import HTTPException
from sqlmodel import Session, select

pytest.importorskip("pgserver")


@pytest.fixture
def session(app_engine):
    with Session(app_engine, expire_on_commit=False) as s:
        yield s


@pytest.fixture(autouse=True)
def _no_actor_leak():
    """The auth helpers set the actor contextvar and, outside a request, nobody
    resets it: start every test with no actor, as a fresh request would."""
    from app.core.actor import reset_current_actor, set_current_actor

    token = set_current_actor(None)
    yield
    reset_current_actor(token)


@pytest.fixture
def tenant(session):
    from app.models.connection import Account, Connection
    from app.models.user import User

    now = datetime.now(timezone.utc)
    stale = session.exec(select(User).where(User.email == "bridge@example.com")).first()
    if stale is not None:
        session.delete(stale)
        session.commit()
    user = User(email="bridge@example.com", username="user_bridge", hashed_password="x")
    session.add(user)
    session.flush()
    conn = Connection(user_id=user.id, name="conn-bridge", broker_type="binance", status="connected",
                      created_at=now, updated_at=now)
    session.add(conn)
    session.flush()
    acc = Account(connection_id=conn.id, account_id="acc-bridge", currency="USDT", created_at=now, updated_at=now)
    session.add(acc)
    session.commit()
    yield user, conn, acc
    session.delete(user)
    session.commit()


def _agent(session, user, name, kind="hosted", is_default=False):
    from app.schemas.agent import AgentCreate
    from app.services.agent_service import create_agent

    agent, chat = create_agent(session, AgentCreate(agent_name=name, kind=kind, is_default=is_default), user.id)
    return agent, chat


def _definition(rule_agent_id=None):
    rules = []
    if rule_agent_id is not None:
        rules.append({
            "name": "ask", "action": "ask_agent", "agent_id": rule_agent_id, "logic": "and",
            "conditions": [{"field": "price.close", "op": "greater_than", "value": 0}], "size": 1, "enabled": True,
        })
    return {"strategy": {
        "name": "bridge", "symbol": "BTCUSDT", "asset": "crypto", "timeframe": "5m", "timezone": "UTC", "params": {},
        "indicators": [], "charts": [{"id": "main", "symbol": "BTCUSDT", "asset": "crypto", "timeframe": "5m", "indicators": []}],
        "rules": rules,
    }}


def test_external_agent_has_no_chat_and_is_never_default(session, tenant):
    user, _, _ = tenant
    hosted, hosted_chat = _agent(session, user, "Aurelio", is_default=True)
    external, external_chat = _agent(session, user, "Claude Code", kind="external", is_default=True)
    assert hosted.kind == "hosted" and hosted_chat is not None and hosted.is_default
    assert external.kind == "external" and external_chat is None and not external.is_default


def test_pat_bound_to_external_agent_refuses_trade(session, tenant):
    from app.services.pat_service import mint_personal_access_token

    user, _, _ = tenant
    external, _ = _agent(session, user, "OpenClaw", kind="external")
    hosted, _ = _agent(session, user, "Livia")

    with pytest.raises(HTTPException) as exc:
        mint_personal_access_token(session, user=user, name="t", scopes=["read", "trade"], agent_id=external.id_agent)
    assert exc.value.status_code == 400

    raw, pat = mint_personal_access_token(session, user=user, name="t", scopes=["read", "write"], agent_id=external.id_agent)
    assert raw.startswith("ewp_") and pat.agent_id == external.id_agent

    # a hosted agent's token may trade (it is the user's own trading agent)
    _, pat2 = mint_personal_access_token(session, user=user, name="t2", scopes=["read", "trade"], agent_id=hosted.id_agent)
    assert pat2.agent_id == hosted.id_agent


def test_pat_bound_to_someone_elses_agent_is_refused(session, tenant):
    from app.models.user import User
    from app.services.pat_service import mint_personal_access_token

    user, _, _ = tenant
    other = session.exec(select(User).where(User.email == "bridge-other@example.com")).first()
    if other is None:
        other = User(email="bridge-other@example.com", username="user_bridge_other", hashed_password="x")
        session.add(other)
        session.commit()
    foreign, _ = _agent(session, other, "Foreign")
    try:
        with pytest.raises(HTTPException) as exc:
            mint_personal_access_token(session, user=user, name="t", scopes=["read"], agent_id=foreign.id_agent)
        assert exc.value.status_code == 404
    finally:
        session.delete(other)
        session.commit()


def test_pat_principal_carries_the_bound_agent(session, tenant):
    from app.core.actor import current_actor
    from app.services.pat_service import mint_personal_access_token
    from app.utils.auth_utils import _try_pat_principal

    user, _, _ = tenant
    external, _ = _agent(session, user, "Hermes", kind="external")
    raw, _ = mint_personal_access_token(session, user=user, name="t", scopes=["read"], agent_id=external.id_agent)

    principal = _try_pat_principal(None, raw, session)
    assert principal is not None
    assert principal.claims["agent_id"] == external.id_agent
    assert principal.claims["agent_kind"] == "external"
    actor = current_actor()
    assert actor is not None and actor.via == "pat" and actor.agent_id == external.id_agent


def test_external_agent_cannot_manage_a_strategy_nor_answer_rules(session, tenant):
    from app.schemas.strategy import StrategyCreate, StrategyUpdate
    from app.services.strategy_service import create_strategy, update_strategy

    user, _, acc = tenant
    external, _ = _agent(session, user, "Paperclip CEO", kind="external")
    hosted, _ = _agent(session, user, "Viktor")

    with pytest.raises(HTTPException) as exc:
        create_strategy(session, StrategyCreate(name="S1", definition=_definition(), account_id=acc.id,
                                                manager_agent_id=external.id_agent), user.id)
    assert exc.value.status_code == 422

    with pytest.raises(HTTPException) as exc:
        create_strategy(session, StrategyCreate(name="S2", definition=_definition(rule_agent_id=external.id_agent),
                                                account_id=acc.id), user.id)
    assert exc.value.status_code == 422

    strategy = create_strategy(session, StrategyCreate(name="S3", definition=_definition(rule_agent_id=hosted.id_agent),
                                                       account_id=acc.id, manager_agent_id=hosted.id_agent), user.id)
    assert strategy.manager_agent_id == hosted.id_agent

    with pytest.raises(HTTPException) as exc:
        update_strategy(session, strategy.id, StrategyUpdate(manager_agent_id=external.id_agent), user.id)
    assert exc.value.status_code == 422
    with pytest.raises(HTTPException) as exc:
        update_strategy(session, strategy.id, StrategyUpdate(definition=_definition(rule_agent_id=external.id_agent)), user.id)
    assert exc.value.status_code == 422


def test_strategy_records_the_actor(session, tenant):
    from app.core.actor import Actor, reset_current_actor, set_current_actor
    from app.schemas.strategy import StrategyCreate, StrategyUpdate
    from app.services.strategy_service import create_strategy, update_strategy

    user, _, acc = tenant
    # outside a request there is no actor: nothing recorded, nothing broken
    strategy = create_strategy(session, StrategyCreate(name="S4", definition=_definition(), account_id=acc.id), user.id)
    assert strategy.updated_by is None

    token = set_current_actor(Actor(via="pat", user_id=user.id, agent_id=9, agent_name="Analyst", pat_id=1, pat_name="cc"))
    try:
        updated = update_strategy(session, strategy.id, StrategyUpdate(description="by the agent"), user.id)
    finally:
        reset_current_actor(token)
    assert updated.updated_by == {"via": "pat", "user_id": user.id, "agent_id": 9, "agent_name": "Analyst", "pat_id": 1, "pat_name": "cc"}
