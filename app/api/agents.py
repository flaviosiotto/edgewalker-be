from fastapi import APIRouter, Depends
from sqlmodel import Session

from app.db.database import get_session
from app.models.user import User
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
from app.utils.auth_utils import get_current_active_user

router = APIRouter(prefix="/agents", tags=["Agents"])


@router.post("/", response_model=AgentRead)
def create_agent_endpoint(
    payload: AgentCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    agent, _chat = create_agent(session, payload, current_user.id)
    return agent


@router.get("/", response_model=list[AgentRead])
def list_agents_endpoint(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    return list_agents(session, current_user.id)


@router.get("/{agent_id}", response_model=AgentRead)
def get_agent_endpoint(
    agent_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    return get_agent(session, agent_id, current_user.id)


@router.patch("/{agent_id}", response_model=AgentRead)
def update_agent_endpoint(
    agent_id: int,
    payload: AgentUpdate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    return update_agent(session, agent_id, payload, current_user.id)


@router.delete("/{agent_id}")
def delete_agent_endpoint(
    agent_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    delete_agent(session, agent_id, current_user.id)
    return {"status": "ok"}


@router.post("/{agent_id}/chats", response_model=ChatRead)
def create_chat_for_agent_endpoint(
    agent_id: int,
    payload: ChatCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    return create_chat_for_agent(session, agent_id, payload, current_user.id)


@router.get("/{agent_id}/chats", response_model=list[ChatRead])
def list_chats_for_agent_endpoint(
    agent_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    return list_chats_for_agent(session, agent_id, current_user.id)
