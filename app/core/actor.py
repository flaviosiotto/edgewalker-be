"""Who is acting in the current request: the user, or an agent on their behalf.

Every mutation is already isolated per user (``current_user.id``); what was
missing is *attribution*: a strategy saved by Claude Code through a personal
access token bound to the agent "Analyst", an order sent by the hosted agent
"Aurelio" from a live turn, or a click in the UI all looked the same.

The auth dependencies resolve the caller and store an ``Actor`` in a
contextvar; services read it with :func:`current_actor` when they want to
record who did what (``strategies.updated_by``, ``strategy_live.started_by``,
the ``extra.actor`` of an order command). The contextvar is per request:
``get_current_user``/``get_current_active_principal`` are async dependencies,
so the value set there is visible to the endpoint and to sync endpoints run
in the threadpool (starlette copies the context into the worker thread).

Nothing here authorises anything: scopes and ownership stay where they are.
"""
from __future__ import annotations

import contextvars
from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional

ActorVia = Literal["ui", "pat", "agent", "runner", "system"]


@dataclass(frozen=True)
class Actor:
    via: ActorVia
    user_id: int
    agent_id: Optional[int] = None
    agent_name: Optional[str] = None
    agent_kind: Optional[str] = None
    pat_id: Optional[int] = None
    pat_name: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, without the ``None`` fields."""
        return {k: v for k, v in asdict(self).items() if v is not None}


_current_actor: contextvars.ContextVar[Optional[Actor]] = contextvars.ContextVar(
    "edgewalker_current_actor", default=None
)


def set_current_actor(actor: Optional[Actor]) -> contextvars.Token:
    return _current_actor.set(actor)


def reset_current_actor(token: contextvars.Token) -> None:
    _current_actor.reset(token)


def current_actor() -> Optional[Actor]:
    """The actor of the request being served, or ``None`` outside a request
    (background jobs, tests that bypass the auth dependencies)."""
    return _current_actor.get()


def current_actor_dict() -> Optional[dict[str, Any]]:
    actor = current_actor()
    return actor.as_dict() if actor is not None else None


def actor_from_claims(user_id: int, claims: dict[str, Any]) -> Actor:
    """Derive the actor from the resolved token claims.

    * UI access token (``purpose=ui_auth``) -> ``via=ui``;
    * personal access token (``type=pat``) -> ``via=pat``, with the bound agent
      when the token has one;
    * consultative / callback tokens minted for the hosted agent and the
      runner (``agent_backend_consult``, ``agent_runner_callback``,
      ``n8n_*``) -> ``via=agent``;
    * runner tokens (``runner_backend``) -> ``via=runner``.
    """
    token_type = str(claims.get("type") or "")
    purpose = str(claims.get("purpose") or "")
    if token_type == "pat":
        return Actor(
            via="pat",
            user_id=user_id,
            agent_id=claims.get("agent_id"),
            agent_name=claims.get("agent_name"),
            agent_kind=claims.get("agent_kind"),
            pat_id=claims.get("pat_id"),
            pat_name=claims.get("pat_name"),
        )
    if purpose == "ui_auth":
        return Actor(via="ui", user_id=user_id)
    if purpose == "runner_backend":
        return Actor(via="runner", user_id=user_id)
    if purpose.startswith(("agent_", "n8n_")):
        agent_id = claims.get("agent_id")
        return Actor(
            via="agent",
            user_id=user_id,
            agent_id=int(agent_id) if agent_id is not None else None,
        )
    return Actor(via="system", user_id=user_id)
