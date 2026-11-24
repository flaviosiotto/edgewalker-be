"""System skills as the backend serves them.

Most are still shipped with the platform kit (``edgewalker_platform.skills``);
the ones in ``app/system_skills/<name>/SKILL.md`` belong to the backend and
win over a kit skill of the same name. New and revised system skills go here:
they are backend content, and editing the kit rebuilds every service.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from edgewalker_platform.skills import SYSTEM_SKILLS_DIR, SystemSkill, _NAME_RE, parse_skill_markdown

BACKEND_SKILLS_DIR = Path(__file__).resolve().parents[1] / "system_skills"


def _read(root: Path) -> dict[str, SystemSkill]:
    """Every valid ``<root>/<name>/SKILL.md`` (same rules as the kit loader:
    folder name = frontmatter name, valid skill name, non-empty body)."""
    out: dict[str, SystemSkill] = {}
    if not root.is_dir():
        return out
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        skill_md = folder / "SKILL.md"
        if not skill_md.is_file():
            continue
        fields, body = parse_skill_markdown(skill_md.read_text(encoding="utf-8"))
        name = (fields.get("name") or folder.name).strip().lower()
        if name != folder.name or not _NAME_RE.match(name) or not body:
            continue
        out[name] = SystemSkill(name=name, description=(fields.get("description") or "")[:1024], body=body[:20_000])
    return out


@lru_cache(maxsize=1)
def system_skills() -> tuple[SystemSkill, ...]:
    merged = {**_read(SYSTEM_SKILLS_DIR), **_read(BACKEND_SKILLS_DIR)}
    return tuple(merged[name] for name in sorted(merged))


def system_skill_names() -> frozenset[str]:
    return frozenset(skill.name for skill in system_skills())
