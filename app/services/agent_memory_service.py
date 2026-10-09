"""Curated memory of a hosted agent (migr. 068, decision D8): three bounded
text slots, written by the agent with ``memory_update`` or edited by the
user on the agent page. No vector store: traceability over recall.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import HTTPException, status
from sqlmodel import Session, select

from app.models.agent import Agent
from app.models.agent_model import MEMORY_KINDS, AgentMemory
from app.schemas.agent_model import MEMORY_MAX_CHARS, MemoryRead


def list_memory(session: Session, agent: Agent) -> list[MemoryRead]:
    rows = {
        r.kind: r
        for r in session.exec(select(AgentMemory).where(AgentMemory.agent_id == agent.id_agent)).all()
    }
    out: list[MemoryRead] = []
    for kind in MEMORY_KINDS:
        row = rows.get(kind)
        out.append(
            MemoryRead(
                kind=kind,  # type: ignore[arg-type]
                content=row.content if row else "",
                updated_at=row.updated_at if row else None,
                updated_by=row.updated_by if row else None,
            )
        )
    return out


def set_memory(session: Session, agent: Agent, kind: str, content: str, *, updated_by: str) -> MemoryRead:
    if kind not in MEMORY_KINDS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown memory kind '{kind}'")
    if (agent.kind or "hosted") != "hosted":
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Only hosted agents have a memory")
    text = str(content or "").strip()
    if len(text) > MEMORY_MAX_CHARS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Memory '{kind}' is limited to {MEMORY_MAX_CHARS} characters",
        )
    row = session.exec(
        select(AgentMemory).where(AgentMemory.agent_id == agent.id_agent, AgentMemory.kind == kind)
    ).first()
    if row is None:
        row = AgentMemory(agent_id=agent.id_agent, kind=kind)
    row.content = text
    row.updated_at = datetime.now(timezone.utc)
    row.updated_by = "agent" if updated_by == "agent" else "user"
    session.add(row)
    session.commit()
    session.refresh(row)
    return MemoryRead(kind=kind, content=row.content, updated_at=row.updated_at, updated_by=row.updated_by)  # type: ignore[arg-type]
