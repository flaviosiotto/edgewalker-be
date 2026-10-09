"""Agent model F3 routes: skills, memory, history search, action requests.

Three routers so each surface gets the right auth:

* ``skills_router`` (``/skills``): reads for the UI, PATs and the hosted
  agent's turn token; writes for the UI and PATs only.
* ``agents_ext_router`` (``/agents/{id}/...``): memory (read any, write by
  the user or by the agent's turn), history search, the agent's requests
  and the creation of a request (agent-svc, from a live turn).
* ``action_router`` (``/action-requests``): list and decide. A decision is
  the user's manual act: UI session, or a PAT holding the ``trade`` scope
  (which an external agent's PAT can never hold, decision D3). Turn tokens
  of the hosted agent cannot decide their own requests.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlmodel import Session

from app.core.actor import current_actor, current_actor_dict
from app.db.database import get_session
from app.models.user import User
from app.schemas.agent_model import (
    ActionRequestCreate,
    ActionRequestDecision,
    ActionRequestRead,
    HistoryHit,
    MemoryRead,
    MemoryUpdate,
    SkillCreate,
    SkillFromPlaybook,
    SkillRead,
    SkillSummary,
    SkillUpdate,
)
from app.services import agent_action_service as actions
from app.services import agent_history_service, agent_memory_service, agent_skill_service
from app.services.agent_service import get_agent
from app.utils.auth_utils import (
    AuthPrincipal,
    get_current_active_or_consultative_principal,
    get_current_active_or_consultative_user,
    get_current_active_principal,
    get_current_active_user,
)

skills_router = APIRouter(prefix="/skills", tags=["Skills"])
agents_ext_router = APIRouter(prefix="/agents", tags=["Agents"])
action_router = APIRouter(prefix="/action-requests", tags=["Action requests"])


# ── skills ──────────────────────────────────────────────────────────────────


@skills_router.get("", response_model=list[SkillSummary])
def list_skills_endpoint(
    names: Optional[str] = Query(default=None, description="comma-separated allowlist"),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    wanted = [n.strip() for n in names.split(",") if n.strip()] if names else None
    return agent_skill_service.list_skills(session, current_user.id, names=wanted)


@skills_router.post("", response_model=SkillRead, status_code=status.HTTP_201_CREATED)
def create_skill_endpoint(
    payload: SkillCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    return agent_skill_service.create_skill(session, current_user.id, payload)


@skills_router.post("/from-playbook", response_model=SkillRead, status_code=status.HTTP_201_CREATED)
def skill_from_playbook_endpoint(
    payload: SkillFromPlaybook,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    return agent_skill_service.skill_from_playbook(session, current_user.id, payload)


@skills_router.get("/{name}", response_model=SkillRead)
def get_skill_endpoint(
    name: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    return agent_skill_service.get_skill(session, current_user.id, name)


@skills_router.get("/{name}/SKILL.md", response_class=Response)
def get_skill_markdown_endpoint(
    name: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    row = agent_skill_service.get_skill(session, current_user.id, name)
    return Response(agent_skill_service.render_skill_markdown(row), media_type="text/markdown; charset=utf-8")


@skills_router.patch("/{name}", response_model=SkillRead)
def update_skill_endpoint(
    name: str,
    payload: SkillUpdate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    return agent_skill_service.update_skill(session, current_user.id, name, payload)


@skills_router.delete("/{name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_skill_endpoint(
    name: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    agent_skill_service.delete_skill(session, current_user.id, name)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ── memory / history ────────────────────────────────────────────────────────


@agents_ext_router.get("/{agent_id}/memory", response_model=list[MemoryRead])
def get_memory_endpoint(
    agent_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    agent = get_agent(session, agent_id, current_user.id)
    return agent_memory_service.list_memory(session, agent)


@agents_ext_router.put("/{agent_id}/memory/{kind}", response_model=MemoryRead)
def set_memory_endpoint(
    agent_id: int,
    kind: str,
    payload: MemoryUpdate,
    session: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_active_or_consultative_principal),
):
    agent = get_agent(session, agent_id, principal.user.id)
    actor = current_actor()
    # A turn token of ANOTHER agent must not write this agent's memory.
    if actor is not None and actor.via == "agent" and actor.agent_id not in (None, agent.id_agent):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="An agent can only update its own memory")
    updated_by = "agent" if actor is not None and actor.via == "agent" else "user"
    return agent_memory_service.set_memory(session, agent, kind, payload.content, updated_by=updated_by)


@agents_ext_router.get("/{agent_id}/history/search", response_model=list[HistoryHit])
def search_history_endpoint(
    agent_id: int,
    q: str = Query(min_length=2, max_length=200),
    limit: int = Query(default=20, ge=1, le=50),
    chat_id: Optional[int] = Query(default=None),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    agent = get_agent(session, agent_id, current_user.id)
    return agent_history_service.search_history(
        session, user_id=current_user.id, agent_id=agent.id_agent, query=q, limit=limit, chat_id=chat_id
    )


# ── action requests ─────────────────────────────────────────────────────────


def _read(session: Session, row) -> ActionRequestRead:
    from app.models.agent import Agent

    agent = session.get(Agent, row.agent_id)
    return actions.to_read(row, agent_name=agent.agent_name if agent else None)


@agents_ext_router.post("/{agent_id}/action-requests", response_model=ActionRequestRead, status_code=status.HTTP_201_CREATED)
def create_action_request_endpoint(
    agent_id: int,
    payload: ActionRequestCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Queue a trading action for approval (called by agent-svc from a turn
    whose tool policy says ``ask``; the UI may also use it to test)."""
    actor = current_actor()
    if actor is not None and actor.via == "agent" and actor.agent_id not in (None, agent_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="An agent can only queue its own requests")
    row = actions.create_request(session, user_id=current_user.id, agent_id=agent_id, payload=payload)
    return _read(session, row)


@agents_ext_router.get("/{agent_id}/action-requests", response_model=list[ActionRequestRead])
def list_agent_action_requests_endpoint(
    agent_id: int,
    status_filter: Optional[str] = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=200),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    get_agent(session, agent_id, current_user.id)
    rows = actions.list_requests(session, user_id=current_user.id, agent_id=agent_id, status_filter=status_filter, limit=limit)
    return [_read(session, r) for r in rows]


@action_router.get("", response_model=list[ActionRequestRead])
def list_action_requests_endpoint(
    status_filter: Optional[str] = Query(default=None, alias="status"),
    live_id: Optional[int] = Query(default=None),
    chat_id: Optional[int] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    rows = actions.list_requests(
        session, user_id=current_user.id, status_filter=status_filter, live_id=live_id, chat_id=chat_id, limit=limit
    )
    return [_read(session, r) for r in rows]


@action_router.get("/{request_id}", response_model=ActionRequestRead)
def get_action_request_endpoint(
    request_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    return _read(session, actions.get_request(session, user_id=current_user.id, request_id=request_id))


def _require_decider(principal: AuthPrincipal) -> dict:
    """UI session, or a PAT with the trade scope. Never a turn token."""
    claims = principal.claims or {}
    if claims.get("type") == "pat":
        scopes = claims.get("scopes") or []
        if "trade" not in scopes:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Deciding an action request needs the 'trade' scope")
    elif str(claims.get("purpose") or "") != "ui_auth":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only the user (UI or a trade-scoped token) can decide")
    return current_actor_dict() or {"via": "ui", "user_id": principal.user.id}


@action_router.post("/{request_id}/approve", response_model=ActionRequestRead)
async def approve_action_request_endpoint(
    request_id: int,
    payload: ActionRequestDecision | None = None,
    session: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_active_principal),
):
    decided_by = _require_decider(principal)
    row = await actions.decide(
        session, user_id=principal.user.id, request_id=request_id, approve=True, decided_by=decided_by,
        note=payload.note if payload else None,
    )
    return _read(session, row)


@action_router.post("/{request_id}/reject", response_model=ActionRequestRead)
async def reject_action_request_endpoint(
    request_id: int,
    payload: ActionRequestDecision | None = None,
    session: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_active_principal),
):
    decided_by = _require_decider(principal)
    row = await actions.decide(
        session, user_id=principal.user.id, request_id=request_id, approve=False, decided_by=decided_by,
        note=payload.note if payload else None,
    )
    return _read(session, row)
