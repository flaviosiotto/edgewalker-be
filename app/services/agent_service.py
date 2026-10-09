import re
from datetime import datetime
from sqlmodel import Session, select
from fastapi import HTTPException, status
from sqlalchemy.orm import selectinload

from edgewalker_platform import agent_tools

from app.core.config import settings
from app.models.agent import Agent, Chat
from app.schemas.agent import AgentCreate, AgentUpdate, AgentSettings
from app.schemas.chat import ChatCreate


DEFAULT_CHAT_NAME = "Default"
DEFAULT_CHAT_DESCRIPTION = "Chat predefinita"


def is_hosted_agent(agent: Agent) -> bool:
    return (getattr(agent, "kind", None) or "hosted") == "hosted"


def require_hosted_agent(agent: Agent, *, role: str = "manager") -> Agent:
    """v1 rule of the agent bridge (decision D1): only hosted agents run
    strategies. An external agent (the identity of an agent living in the
    user's orchestrator) cannot be the manager of a strategy, a live or a
    backtest, nor answer an ``ask_agent`` rule. Enforced here, never by a
    schema constraint, so the scenario can be opened later."""
    if not is_hosted_agent(agent):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Agent '{agent.agent_name}' is external and cannot be the {role}: "
                "in EdgeWalker trading is done by hosted agents through strategies"
            ),
        )
    return agent


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    base = _SLUG_RE.sub("-", str(name or "").lower()).strip("-")
    return base[:56] or "agent"


def unique_slug(session: Session, user_id: int, name: str, *, exclude_id: int | None = None) -> str:
    """A slug unique among the user's agents: the slugified name, with a
    numeric suffix when taken. Stable once assigned (renames keep it)."""
    base = slugify(name)
    taken = {
        row.slug
        for row in session.exec(select(Agent).where(Agent.user_id == user_id)).all()
        if row.slug and row.id_agent != exclude_id
    }
    if base not in taken:
        return base
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"


def policy_for_kind(policy: dict[str, str] | None, kind: str) -> dict[str, str]:
    """An external agent never trades (D1/D3): a policy that tries to allow a
    trade-scoped tool or group for it is refused, not silently ignored."""
    clean = agent_tools.normalize_policy(policy or {})
    if kind == "hosted":
        return clean
    offending = []
    for key, value in clean.items():
        if value == "off":
            continue
        spec = agent_tools.get_tool(key)
        if (spec is not None and spec.scope == "trade") or key in ("trading", "live"):
            offending.append(key)
    if offending:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"An external agent never trades: {', '.join(offending)} can only be 'off'",
        )
    return clean


def create_agent(session: Session, payload: AgentCreate, user_id: int) -> tuple[Agent, Chat | None]:
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
        kind=payload.kind,
        # Legacy column: always the agent-svc endpoint, payload value ignored.
        n8n_webhook=settings.AGENT_SVC_WEBHOOK_URL,
        # An external agent is never preselected: it cannot run anything.
        is_default=payload.is_default and payload.kind == "hosted",
        avatar=payload.avatar,
        accent_color=payload.accent_color,
        avatar_url=payload.avatar_url,
        description=payload.description,
        risk_profile=payload.risk_profile,
        persona=payload.persona or {},
        settings=payload.settings.model_dump(),
        tool_policy=policy_for_kind(payload.tool_policy, payload.kind),
        skills=list(payload.skills or []),
        budget_credits_month=payload.budget_credits_month,
        slug=unique_slug(session, user_id, payload.agent_name),
    )
    session.add(agent)
    session.commit()
    session.refresh(agent)

    if payload.kind != "hosted":
        # No chat for an identity that never answers a turn.
        return agent, None

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
    """Every agent of the user — there is no kind to filter on any more.

    The design/running split was retired with migr. 049: all agents can both
    design and trade, so the FE shows one list everywhere.
    """
    statement = select(Agent).where(Agent.user_id == user_id)
    return list(session.exec(statement).all())


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
    # payload.n8n_webhook is accepted for API compatibility but ignored: the
    # execution endpoint is always AGENT_SVC_WEBHOOK_URL.
    if payload.is_default is not None:
        agent.is_default = payload.is_default and is_hosted_agent(agent)
    if payload.avatar is not None:
        agent.avatar = payload.avatar
    if payload.accent_color is not None:
        agent.accent_color = payload.accent_color
    # avatar_url and description are nullable: an explicit null clears them,
    # so they are keyed off the fields actually present in the PATCH body.
    fields_set = payload.model_fields_set
    if "avatar_url" in fields_set:
        agent.avatar_url = payload.avatar_url
    if "description" in fields_set:
        agent.description = payload.description
    if payload.risk_profile is not None:
        agent.risk_profile = payload.risk_profile
    if payload.persona is not None:
        agent.persona = payload.persona
    if payload.settings is not None:
        # PATCH semantics: keys left out keep their current value.
        current = AgentSettings.model_validate(agent.settings or {}).model_dump()
        agent.settings = {**current, **payload.settings.model_dump(exclude_unset=True)}
    if payload.tool_policy is not None:
        # The whole document is replaced: a group left out goes back to the default.
        agent.tool_policy = policy_for_kind(payload.tool_policy, agent.kind or "hosted")
    if payload.skills is not None:
        agent.skills = list(payload.skills)
    if "budget_credits_month" in payload.model_fields_set:
        agent.budget_credits_month = payload.budget_credits_month
    if not agent.slug:
        agent.slug = unique_slug(session, agent.user_id, agent.agent_name, exclude_id=agent.id_agent)

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
