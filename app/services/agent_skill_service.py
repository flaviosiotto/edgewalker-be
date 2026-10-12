"""Skills (migr. 068): SKILL.md documents a hosted agent loads on demand and
an external agent reads through MCP. Two kinds share one catalogue: the
**system skills** shipped with the platform kit (read-only, same for every
user) and the **user skills** written by the trader. Text only: a skill
never declares tools (decision D9). A run may add its own ephemeral skills
(Paperclip) on top, see ``run_skills``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import HTTPException, status
from sqlmodel import Session, select

from edgewalker_platform.skills import system_skill_names, system_skills

from app.models.agent_model import AgentSkill
from app.schemas.agent_model import SkillCreate, SkillUpdate

SYSTEM_ORIGIN = "system"
#: fixed timestamp of the system skills (they change with the platform release)
_SYSTEM_SKILLS_AT = datetime(2026, 10, 12, tzinfo=timezone.utc)


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
            "read_only": True,
            "source_strategy_id": None,
            "source_backtest_id": None,
            "created_at": run.created_at,
            "updated_at": run.created_at,
            "body": body,
        })
    return out


def list_skills_for_turn(session: Session, user_id: int, *, names: Optional[list[str]] = None, run_id: Optional[int] = None) -> list[Any]:
    """``GET /skills``: the run's ephemeral skills first (always, regardless
    of the agent allowlist), then the system skills, then the user's skills.
    ``names`` (the agent allowlist) filters system and user skills alike."""
    extra = run_skills(session, user_id, run_id)
    taken = {row["name"] for row in extra}
    wanted = {n.strip().lower() for n in names} if names else None
    system = [row for row in system_skill_rows() if row["name"] not in taken and (wanted is None or row["name"] in wanted)]
    taken |= {row["name"] for row in system}
    rows = [row for row in list_skills(session, user_id, names=names) if row.name not in taken]
    return [*extra, *system, *rows]


def get_skill_for_turn(session: Session, user_id: int, name: str, *, run_id: Optional[int] = None) -> Any:
    wanted = str(name).strip().lower()
    for row in [*run_skills(session, user_id, run_id), *system_skill_rows()]:
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


def system_skill_rows() -> list[dict[str, Any]]:
    """The platform skills as ``SkillRead`` rows (id 0, read-only)."""
    return [
        {
            "id": 0, "name": skill.name, "description": skill.description, "version": 1,
            "origin": SYSTEM_ORIGIN, "read_only": True, "source_strategy_id": None, "source_backtest_id": None,
            "created_at": _SYSTEM_SKILLS_AT, "updated_at": _SYSTEM_SKILLS_AT, "body": skill.body,
        }
        for skill in system_skills()
    ]


def _assert_not_system(name: str, action: str) -> None:
    if name in system_skill_names():
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Skill '{name}' is a system skill: it cannot be {action}")


def create_skill(session: Session, user_id: int, payload: SkillCreate, *, origin: str = "user", **source) -> AgentSkill:
    if payload.name in system_skill_names():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Skill '{payload.name}' is a system skill: choose another name")
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
    _assert_not_system(str(name).strip().lower(), "edited")
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
    _assert_not_system(str(name).strip().lower(), "deleted")
    row = get_skill(session, user_id, name)
    session.delete(row)
    session.commit()


def render_skill_markdown(row: AgentSkill) -> str:
    """The SKILL.md text (agentskills.io frontmatter + body)."""
    description = (row.description or "").replace("\n", " ").strip()
    return f"---\nname: {row.name}\ndescription: {description}\n---\n\n{row.body.strip()}\n"
