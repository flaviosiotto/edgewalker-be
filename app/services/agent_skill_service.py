"""User skills (migr. 068): SKILL.md documents the user writes, or promotes
from a playbook, that a hosted agent loads on demand and an external agent
reads through MCP. Text only: a skill never declares tools (decision D9).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import HTTPException, status
from sqlmodel import Session, select

from app.models.agent_lesson import AgentLesson
from app.models.agent_model import AgentSkill
from app.models.strategy import Strategy
from app.schemas.agent_model import SkillCreate, SkillFromPlaybook, SkillUpdate


def _now() -> datetime:
    return datetime.now(timezone.utc)


def list_skills(session: Session, user_id: int, *, names: Optional[list[str]] = None) -> list[AgentSkill]:
    stmt = select(AgentSkill).where(AgentSkill.user_id == user_id)
    if names:
        stmt = stmt.where(AgentSkill.name.in_([n.lower() for n in names]))  # type: ignore[attr-defined]
    return list(session.exec(stmt.order_by(AgentSkill.name)).all())


RUN_SKILL_ORIGIN = "paperclip"


def run_skills(session: Session, user_id: int, run_id: Optional[int]) -> list[dict[str, Any]]:
    """The ephemeral skills of a run (``run.context["skills"]``, shipped by the
    Paperclip adapter), shaped like ``SkillRead`` rows: they are visible to the
    turn of that run only and never written to ``agent_skill``. Unknown or
    foreign runs → empty (the turn simply sees the user's skills)."""
    if run_id is None:
        return []
    from app.services.agent_run_service import get_run  # local: avoids an import cycle

    try:
        run = get_run(session, user_id=user_id, run_id=run_id)
    except HTTPException:
        return []
    raw = (run.context or {}).get("skills") if isinstance(run.context, dict) else None
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip().lower()
        body = item.get("body")
        if not name or name in seen or not isinstance(body, str) or not body.strip():
            continue
        seen.add(name)
        out.append({
            "id": 0,
            "name": name,
            "description": str(item.get("description") or ""),
            "version": 1,
            "origin": RUN_SKILL_ORIGIN,
            "source_strategy_id": None,
            "source_backtest_id": None,
            "created_at": run.created_at,
            "updated_at": run.created_at,
            "body": body,
        })
    return out


def list_skills_for_turn(session: Session, user_id: int, *, names: Optional[list[str]] = None, run_id: Optional[int] = None) -> list[Any]:
    """``GET /skills`` as the turn sees it: the run's ephemeral skills first
    (always, regardless of the agent allowlist), then the user's skills."""
    extra = run_skills(session, user_id, run_id)
    taken = {row["name"] for row in extra}
    rows = [row for row in list_skills(session, user_id, names=names) if row.name not in taken]
    return [*extra, *rows]


def get_skill_for_turn(session: Session, user_id: int, name: str, *, run_id: Optional[int] = None) -> Any:
    wanted = str(name).strip().lower()
    for row in run_skills(session, user_id, run_id):
        if row["name"] == wanted:
            return row
    return get_skill(session, user_id, name)


def get_skill(session: Session, user_id: int, name: str) -> AgentSkill:
    row = session.exec(
        select(AgentSkill).where(AgentSkill.user_id == user_id, AgentSkill.name == str(name).strip().lower())
    ).first()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Skill '{name}' not found")
    return row


def create_skill(session: Session, user_id: int, payload: SkillCreate, *, origin: str = "user", **source) -> AgentSkill:
    existing = session.exec(
        select(AgentSkill).where(AgentSkill.user_id == user_id, AgentSkill.name == payload.name)
    ).first()
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Skill '{payload.name}' already exists")
    row = AgentSkill(
        user_id=user_id,
        name=payload.name,
        description=payload.description or "",
        body=payload.body,
        origin=origin,
        source_strategy_id=source.get("source_strategy_id"),
        source_backtest_id=source.get("source_backtest_id"),
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def update_skill(session: Session, user_id: int, name: str, payload: SkillUpdate) -> AgentSkill:
    row = get_skill(session, user_id, name)
    changed = False
    if payload.description is not None and payload.description != row.description:
        row.description = payload.description
        changed = True
    if payload.body is not None and payload.body != row.body:
        row.body = payload.body
        row.version = int(row.version or 1) + 1
        changed = True
    if changed:
        row.updated_at = _now()
        session.add(row)
        session.commit()
        session.refresh(row)
    return row


def delete_skill(session: Session, user_id: int, name: str) -> None:
    row = get_skill(session, user_id, name)
    session.delete(row)
    session.commit()


def render_skill_markdown(row: AgentSkill) -> str:
    """The SKILL.md text (agentskills.io frontmatter + body)."""
    description = (row.description or "").replace("\n", " ").strip()
    return f"---\nname: {row.name}\ndescription: {description}\n---\n\n{row.body.strip()}\n"


def skill_from_playbook(session: Session, user_id: int, payload: SkillFromPlaybook) -> AgentSkill:
    """Promote a playbook (the strategy's current one, or a backtest's output)
    to a skill. The lessons become a checklist the agent reads as procedural
    knowledge; the playbook itself is untouched (decision D10)."""
    from app.services import playbook_service

    strategy = session.get(Strategy, payload.strategy_id)
    if strategy is None or strategy.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Strategy not found")
    if payload.backtest_id is not None:
        rows = playbook_service.run_rows(session, payload.backtest_id, status_filter="active")
        if not rows:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="That backtest has no active lessons")
        if any(r.strategy_id != strategy.id for r in rows):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Backtest does not belong to this strategy")
        label = f"backtest #{payload.backtest_id}"
    else:
        rows = playbook_service.current_rows(session, strategy, status_filter="active")
        if not rows:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="The strategy has no active lessons")
        label = "current playbook"
    body = _lessons_markdown(strategy, rows, label)
    description = payload.description or f"Lessons learned on strategy '{strategy.name}' ({label})"
    return create_skill(
        session,
        user_id,
        SkillCreate(name=payload.name, description=description[:1024], body=body),
        origin="playbook",
        source_strategy_id=strategy.id,
        source_backtest_id=payload.backtest_id,
    )


def _lessons_markdown(strategy: Strategy, rows: list[AgentLesson], label: str) -> str:
    lines = [
        f"# Lessons from '{strategy.name}' ({label})",
        "",
        "Apply these as execution notes inside the plan of the strategy you are working on:",
        "they never replace the strategy's own rules. Confidence >= 0.6 means validated across runs;",
        "below it the lesson is a hypothesis under observation.",
        "",
    ]
    for row in sorted(rows, key=lambda r: (-(r.confidence or 0.0), r.id or 0)):
        tag = "validated" if (row.confidence or 0.0) >= 0.6 else "hypothesis"
        line = f"- [{tag}, confidence {row.confidence:.2f}] {row.lesson.strip()}"
        if row.context:
            line += f" — when: {row.context.strip()}"
        lines.append(line)
    return "\n".join(lines) + "\n"
