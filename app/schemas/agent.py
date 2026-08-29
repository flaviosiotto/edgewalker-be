import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# The agent has no `kind` any more (migr. 049): every agent can both design a
# strategy and trade it. What it has instead is an identity — an avatar preset,
# an accent colour, a risk profile and a free-form persona — which is rendered
# by the FE and shipped to n8n as `metadata.agent`.
RiskProfile = Literal["conservative", "balanced", "aggressive"]

# Keys of the FE's inline SVG avatar set (AgentAvatar.vue). Not validated as an
# enum on purpose: the FE falls back to initials on an unknown key, and a new
# avatar must not require a backend deploy.
DEFAULT_AVATAR = "robot"
DEFAULT_ACCENT_COLOR = "#6f42c1"

_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def _validate_accent_color(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip()
    if not _HEX_COLOR_RE.match(normalized):
        raise ValueError("accent_color must be a hex colour in the form #RRGGBB")
    return normalized.lower()


class AgentPersonaFields(BaseModel):
    """Persona attributes shared by create/read/update.

    Declared once so the three schemas cannot drift; `AgentUpdate` re-declares
    them as optional because it is a PATCH body.
    """

    avatar: str = Field(default=DEFAULT_AVATAR, max_length=64)
    accent_color: str = Field(default=DEFAULT_ACCENT_COLOR, max_length=16)
    avatar_url: Optional[str] = Field(default=None, max_length=1024)
    description: Optional[str] = None
    risk_profile: RiskProfile = "balanced"
    persona: dict[str, Any] = Field(default_factory=dict)

    @field_validator("accent_color")
    @classmethod
    def _check_accent_color(cls, value: str) -> str:
        return _validate_accent_color(value)


class AgentCreate(AgentPersonaFields):
    agent_name: str
    n8n_webhook: str
    is_default: bool = False


class AgentRead(AgentPersonaFields):
    id_agent: int
    agent_name: str
    n8n_webhook: str
    is_default: bool


class AgentUpdate(BaseModel):
    agent_name: Optional[str] = None
    n8n_webhook: Optional[str] = None
    is_default: Optional[bool] = None
    avatar: Optional[str] = Field(default=None, max_length=64)
    accent_color: Optional[str] = Field(default=None, max_length=16)
    avatar_url: Optional[str] = Field(default=None, max_length=1024)
    description: Optional[str] = None
    risk_profile: Optional[RiskProfile] = None
    persona: Optional[dict[str, Any]] = None

    @field_validator("accent_color")
    @classmethod
    def _check_accent_color(cls, value: Optional[str]) -> Optional[str]:
        return _validate_accent_color(value)


class AgentReadWithMeta(AgentRead):
    created_default_chat_id: Optional[int] = None
    created_default_chat_name: Optional[str] = None


def build_agent_persona_block(agent: Any) -> dict[str, Any]:
    """The `metadata.agent` block sent to n8n with every webhook call.

    Single definition so the chat, streaming and rule-trigger payloads cannot
    diverge — the n8n prompt renders it verbatim as `== CHI SEI ==`.
    """
    persona = getattr(agent, "persona", None)
    return {
        "id": agent.id_agent,
        "name": agent.agent_name,
        "avatar": getattr(agent, "avatar", None) or DEFAULT_AVATAR,
        "accent_color": getattr(agent, "accent_color", None) or DEFAULT_ACCENT_COLOR,
        "avatar_url": getattr(agent, "avatar_url", None),
        "description": getattr(agent, "description", None),
        "risk_profile": getattr(agent, "risk_profile", None) or "balanced",
        "persona": persona if isinstance(persona, dict) else {},
    }
