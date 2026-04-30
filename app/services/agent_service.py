from datetime import datetime
from sqlmodel import Session, select
from fastapi import HTTPException, status
from sqlalchemy.orm import selectinload

from app.models.agent import Agent, Chat
from app.schemas.agent import AgentCreate, AgentUpdate
from app.schemas.chat import ChatCreate


DEFAULT_CHAT_NAME = "Default"
DEFAULT_CHAT_DESCRIPTION = "Chat predefinita"


def create_agent(session: Session, payload: AgentCreate, user_id: int) -> tuple[Agent, Chat]:
    existing = session.exec(
        select(Agent)
        .where(Agent.user_id == user_id)
        .where(Agent.agent_name == payload.agent_name)
    ).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Nome agente già in uso",
        )

    agent = Agent(
        user_id=user_id,
        agent_name=payload.agent_name,
        n8n_webhook=payload.n8n_webhook,
        is_default=payload.is_default,
    )
    session.add(agent)
    session.commit()
    session.refresh(agent)

    chat = Chat(
        user_id=user_id,
        id_agent=agent.id_agent,
        nome=f"{payload.agent_name} {DEFAULT_CHAT_NAME}",
        descrizione=DEFAULT_CHAT_DESCRIPTION,
        chat_type=Chat.ChatType.USER,
        created_at=datetime.now(),
    )
    session.add(chat)
    session.commit()
    session.refresh(chat)

    return agent, chat


def list_agents(session: Session, user_id: int) -> list[Agent]:
    return list(session.exec(select(Agent).where(Agent.user_id == user_id)).all())


def get_agent(session: Session, agent_id: int, user_id: int | None = None) -> Agent:
    statement = select(Agent).where(Agent.id_agent == agent_id)
    if user_id is not None:
        statement = statement.where(Agent.user_id == user_id)
    agent = session.exec(statement).first()
    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agente non trovato")
    return agent


def update_agent(session: Session, agent_id: int, payload: AgentUpdate, user_id: int | None = None) -> Agent:
    agent = get_agent(session, agent_id, user_id)

    if payload.agent_name and payload.agent_name != agent.agent_name:
        existing = session.exec(
            select(Agent)
            .where(Agent.user_id == agent.user_id)
            .where(Agent.agent_name == payload.agent_name)
        ).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Nome agente già in uso",
            )

    if payload.agent_name is not None:
        agent.agent_name = payload.agent_name
    if payload.n8n_webhook is not None:
        agent.n8n_webhook = payload.n8n_webhook
    if payload.is_default is not None:
        agent.is_default = payload.is_default

    session.add(agent)
    session.commit()
    session.refresh(agent)
    return agent


def delete_agent(session: Session, agent_id: int, user_id: int | None = None) -> None:
    agent = get_agent(session, agent_id, user_id)
    session.delete(agent)
    session.commit()


def create_chat_for_agent(session: Session, agent_id: int, payload: ChatCreate, user_id: int) -> Chat:
    agent = get_agent(session, agent_id, user_id)

    chat = Chat(
        user_id=user_id,
        id_agent=agent.id_agent,
        nome=payload.nome,
        descrizione=payload.descrizione,
        chat_type=payload.chat_type or Chat.ChatType.USER,
        created_at=datetime.now(),
    )
    session.add(chat)
    session.commit()
    session.refresh(chat)
    return session.exec(
        select(Chat)
        .options(selectinload(Chat.agent))
        .where(Chat.id == chat.id)
    ).first()


def list_chats_for_agent(session: Session, agent_id: int, user_id: int) -> list[Chat]:
    get_agent(session, agent_id, user_id)
    statement = (
        select(Chat)
        .options(selectinload(Chat.agent))
        .where(Chat.id_agent == agent_id)
        .where(Chat.user_id == user_id)
    )
    return list(session.exec(statement).all())
