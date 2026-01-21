from fastapi import APIRouter, Depends
from sqlmodel import Session

from app.db.database import get_session
from app.schemas.agent import AgentCreate, AgentRead, AgentUpdate
from app.schemas.chat import ChatCreate, ChatRead
from app.services.agent_service import (
    create_agent,
    create_chat_for_agent,
    delete_agent,
    get_agent,
    list_agents,
    list_chats_for_agent,
    update_agent,
)

router = APIRouter(prefix="/agents", tags=["Agents"])


@router.post("/", response_model=AgentRead)
def create_agent_endpoint(payload: AgentCreate, session: Session = Depends(get_session)):
    agent, _chat = create_agent(session, payload)
    return agent


@router.get("/", response_model=list[AgentRead])
def list_agents_endpoint(session: Session = Depends(get_session)):
    return list_agents(session)


@router.get("/{agent_id}", response_model=AgentRead)
def get_agent_endpoint(agent_id: int, session: Session = Depends(get_session)):
    return get_agent(session, agent_id)


@router.patch("/{agent_id}", response_model=AgentRead)
def update_agent_endpoint(agent_id: int, payload: AgentUpdate, session: Session = Depends(get_session)):
    return update_agent(session, agent_id, payload)


@router.delete("/{agent_id}")
def delete_agent_endpoint(agent_id: int, session: Session = Depends(get_session)):
    delete_agent(session, agent_id)
    return {"status": "ok"}


@router.post("/{agent_id}/chats", response_model=ChatRead)
def create_chat_for_agent_endpoint(
    agent_id: int,
    payload: ChatCreate,
    session: Session = Depends(get_session),
):
    return create_chat_for_agent(session, agent_id, payload)


@router.get("/{agent_id}/chats", response_model=list[ChatRead])
def list_chats_for_agent_endpoint(agent_id: int, session: Session = Depends(get_session)):
    return list_chats_for_agent(session, agent_id)
