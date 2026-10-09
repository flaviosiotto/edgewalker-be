import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, computed_field, field_validator

from edgewalker_platform import agent_tools

# The agent has no `kind` any more (migr. 049): every agent can both design a
# strategy and trade it. What it has instead is an identity — an avatar preset,
# an accent colour, a risk profile and a free-form persona — which is rendered
# by the FE and shipped to n8n as `metadata.agent`.
RiskProfile = Literal["conservative", "balanced", "aggressive"]

# migr. 066: hosted = runs in agent-svc; external = identity of an agent that
# runs in the user's orchestrator and uses EdgeWalker through MCP / the API
# with a PAT bound to it (never trades, never a manager — v1).
AgentKind = Literal["hosted", "external"]

# Keys of the FE's inline SVG avatar set (AgentAvatar.vue). Not validated as an
# enum on purpose: the FE falls back to initials on an unknown key, and a new
# avatar must not require a backend deploy.
DEFAULT_AVATAR = "analyst"
DEFAULT_ACCENT_COLOR = "#6c757d"

_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def _validate_accent_color(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip()
    if not _HEX_COLOR_RE.match(normalized):
        raise ValueError("accent_color must be a hex colour in the form #RRGGBB")
    return normalized.lower()


def _validate_tool_policy(value: Optional[dict[str, str]]) -> dict[str, str]:
    if value is None:
        return {}
    try:
        return agent_tools.normalize_policy(value)
    except ValueError as exc:
        raise ValueError(f"tool_policy: {exc}") from exc


_SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")


def _validate_skill_names(value: Optional[list[str]]) -> list[str]:
    names: list[str] = []
    for raw in value or []:
        name = str(raw or "").strip().lower()
        if not _SKILL_NAME_RE.match(name):
            raise ValueError(f"skills: '{raw}' is not a valid skill name (lowercase letters, digits and dashes)")
        if name not in names:
            names.append(name)
    return names


# User-facing behaviour knobs (migr. 058). The reasoning level is the only
# thing the trader chooses about the model; the platform maps it to a real
# model per plan in ai_model_policy (admin). No model name ever lives here.
ReasoningLevel = Literal["quick", "balanced", "deep"]
Autonomy = Literal["propose", "execute"]


class AgentSettings(BaseModel):
    reasoning: ReasoningLevel = "balanced"
    # propose = the agent only suggests; execute = it may place/close orders
    # and manage alerts in operational turns (live/backtest).
    autonomy: Autonomy = "execute"
    include_chart: bool = True
    lessons_enabled: bool = True

    model_config = {"extra": "ignore"}


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
    settings: AgentSettings = Field(default_factory=AgentSettings)
    # migr. 068: {"<tool|group>": "allow"|"ask"|"off"} (edgewalker_platform.agent_tools);
    # empty = defaults derived from settings.autonomy. Allowlist of the user's
    # skill names the agent loads (empty = all).
    tool_policy: dict[str, str] = Field(default_factory=dict)
    skills: list[str] = Field(default_factory=list)
    # migr. 069: monthly credit cap for the runs handed to this agent from outside.
    budget_credits_month: Optional[int] = Field(default=None, ge=0, le=1_000_000)

    @field_validator("accent_color")
    @classmethod
    def _check_accent_color(cls, value: str) -> str:
        return _validate_accent_color(value)

    @field_validator("tool_policy")
    @classmethod
    def _check_tool_policy(cls, value: dict[str, str]) -> dict[str, str]:
        return _validate_tool_policy(value)

    @field_validator("skills")
    @classmethod
    def _check_skills(cls, value: list[str]) -> list[str]:
        return _validate_skill_names(value)


class AgentCreate(AgentPersonaFields):
    agent_name: str
    # Immutable after creation: a hosted agent has chats, lessons and runs
    # that an external one cannot have, and vice versa.
    kind: AgentKind = "hosted"
    # Execution engine address. Optional since phase 3: the backend assigns
    # AGENT_SVC_WEBHOOK_URL; only legacy/admin callers still pass an URL.
    n8n_webhook: Optional[str] = None
    is_default: bool = False


class AgentRead(AgentPersonaFields):
    id_agent: int
    agent_name: str
    kind: AgentKind = "hosted"
    slug: Optional[str] = None
    n8n_webhook: str
    is_default: bool

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tool_policy_effective(self) -> dict[str, str]:
        """One value per policy group (allow | ask | off | mixed) after the
        defaults of kind and autonomy are applied: what the UI shows."""
        return agent_tools.group_policy(self.tool_policy, kind=self.kind, autonomy=self.settings.autonomy)


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
    settings: Optional[AgentSettings] = None
    tool_policy: Optional[dict[str, str]] = None
    skills: Optional[list[str]] = None
    budget_credits_month: Optional[int] = Field(default=None, ge=0, le=1_000_000)

    @field_validator("accent_color")
    @classmethod
    def _check_accent_color(cls, value: Optional[str]) -> Optional[str]:
        return _validate_accent_color(value)

    @field_validator("tool_policy")
    @classmethod
    def _check_tool_policy(cls, value: Optional[dict[str, str]]) -> Optional[dict[str, str]]:
        return None if value is None else _validate_tool_policy(value)

    @field_validator("skills")
    @classmethod
    def _check_skills(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        return None if value is None else _validate_skill_names(value)


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
        "kind": getattr(agent, "kind", None) or "hosted",
        "avatar": getattr(agent, "avatar", None) or DEFAULT_AVATAR,
        "accent_color": getattr(agent, "accent_color", None) or DEFAULT_ACCENT_COLOR,
        "avatar_url": getattr(agent, "avatar_url", None),
        "description": getattr(agent, "description", None),
        "risk_profile": getattr(agent, "risk_profile", None) or "balanced",
        "persona": persona if isinstance(persona, dict) else {},
        "settings": AgentSettings.model_validate(getattr(agent, "settings", None) or {}).model_dump(),
        # F3: the policy travels with the persona so agent-svc applies it
        # without a second lookup; the allowlist of skills likewise.
        "tool_policy": dict(getattr(agent, "tool_policy", None) or {}),
        "skills": list(getattr(agent, "skills", None) or []),
        "slug": getattr(agent, "slug", None),
    }
