"""The chat of a Studio (docs/valutazione-studi-pubblicazione-e-agente.md, D11).

One chat per Studio, created on first use with the Studio's agent
(``studio.agent_id``, chosen when the Studio is created among the agents
whose tool policy enables the ``studios`` tools). The ``studio`` table is
studio-svc's (same database): it is read raw, like webhook_sources does.
agent-svc routes the chat to the ``studio`` scope from ``chat.studio_id``.
"""
from __future__ import annotations

from datetime import datetime

from fastapi import HTTPException, status
from sqlalchemy import text
from sqlalchemy.orm import selectinload
from sqlmodel import Session, select

from edgewalker_platform import agent_tools

from app.models.agent import Agent, Chat


def _studio_row(session: Session, studio_id: int, user_id: int) -> dict:
    row = session.execute(
        text("SELECT id, user_id, name, agent_id FROM studio WHERE id = :id AND deleted_at IS NULL"),
        {"id": studio_id},
    ).mappings().first()
    if row is None or int(row["user_id"]) != int(user_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Studio not found")
    return dict(row)


def studio_agent_allowed(agent: Agent) -> bool:
    """A hosted agent whose policy does not switch the Studio tools off."""
    if agent.kind != "hosted":
        return False
    policy = agent_tools.group_policy(agent.tool_policy or {}, kind=agent.kind, autonomy=(agent.settings or {}).get("autonomy") or "execute")
    return policy.get("studios") != "off"


def get_or_create_studio_chat(session: Session, studio_id: int, user_id: int) -> Chat:
    """The Studio's chat, created on first use. 409 when the Studio has no
    agent or its agent cannot work on Studi (policy off, external)."""
    studio = _studio_row(session, studio_id, user_id)
    agent_id = studio.get("agent_id")
    if agent_id is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This Studio has no agent: pick one in its settings")
    agent = session.get(Agent, int(agent_id))
    if agent is None or agent.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="The Studio's agent no longer exists: pick another one")
    if not studio_agent_allowed(agent):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The Studio's agent cannot work on Studi (tool policy): pick another agent or enable the Studio tools",
        )
    chat = session.exec(
        select(Chat).options(selectinload(Chat.agent)).where(Chat.studio_id == studio_id).order_by(Chat.id.desc())
    ).first()
    if chat is None:
        chat = Chat(
            user_id=user_id,
            id_agent=agent.id_agent,
            studio_id=studio_id,
            nome=str(studio.get("name") or f"Studio {studio_id}"),
            descrizione="Chat dello Studio",
            chat_type=Chat.ChatType.STUDIO,
            created_at=datetime.now(),
        )
        session.add(chat)
        session.commit()
    elif chat.id_agent != agent.id_agent:
        # The Studio's agent changed: the conversation continues with the new one.
        chat.id_agent = agent.id_agent
        session.add(chat)
        session.commit()
    return session.exec(select(Chat).options(selectinload(Chat.agent)).where(Chat.id == chat.id)).first()
