from datetime import datetime
from sqlmodel import Session, select
from fastapi import HTTPException, status

from app.models.agent import Agent, Chat
from app.schemas.agent import AgentCreate, AgentUpdate
from app.schemas.chat import ChatCreate


DEFAULT_CHAT_NAME = "Default"
DEFAULT_CHAT_DESCRIPTION = "Chat predefinita"


def create_agent(session: Session, payload: AgentCreate) -> tuple[Agent, Chat]:
    existing = session.exec(select(Agent).where(Agent.agent_name == payload.agent_name)).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Nome agente già in uso",
        )

    agent = Agent(
        agent_name=payload.agent_name,
        n8n_webhook=payload.n8n_webhook,
        is_default=payload.is_default,
    )
    session.add(agent)
    session.commit()
    session.refresh(agent)

    chat = Chat(
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


def list_agents(session: Session) -> list[Agent]:
    return list(session.exec(select(Agent)).all())


def get_agent(session: Session, agent_id: int) -> Agent:
    agent = session.get(Agent, agent_id)
    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agente non trovato")
    return agent


def update_agent(session: Session, agent_id: int, payload: AgentUpdate) -> Agent:
    agent = get_agent(session, agent_id)

    if payload.agent_name and payload.agent_name != agent.agent_name:
        existing = session.exec(select(Agent).where(Agent.agent_name == payload.agent_name)).first()
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


def delete_agent(session: Session, agent_id: int) -> None:
    agent = get_agent(session, agent_id)
    session.delete(agent)
    session.commit()


def create_chat_for_agent(session: Session, agent_id: int, payload: ChatCreate) -> Chat:
    agent = get_agent(session, agent_id)

    chat = Chat(
        id_agent=agent.id_agent,
        nome=payload.nome,
        descrizione=payload.descrizione,
        chat_type=payload.chat_type or Chat.ChatType.USER,
        user_id=payload.user_id,
        created_at=datetime.now(),
    )
    session.add(chat)
    session.commit()
    session.refresh(chat)
    return chat


def list_chats_for_agent(session: Session, agent_id: int) -> list[Chat]:
    get_agent(session, agent_id)
    statement = select(Chat).where(Chat.id_agent == agent_id)
    return list(session.exec(statement).all())
