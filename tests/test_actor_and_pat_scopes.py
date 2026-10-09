"""Pure-function tests for the agent bridge F1 plumbing (no database).

* ``core.actor.actor_from_claims``: the actor derived from each token kind;
* ``pat_service.required_scope_for``: ``/users/me/*`` readable with a PAT,
  the rest of ``/users`` still banned.
Run: ``venv/bin/python -m pytest tests/test_actor_and_pat_scopes.py``.
"""
from __future__ import annotations

from app.core.actor import Actor, actor_from_claims, current_actor, reset_current_actor, set_current_actor
from app.services.pat_service import required_scope_for


def test_actor_from_ui_token():
    actor = actor_from_claims(7, {"type": "access", "purpose": "ui_auth"})
    assert actor == Actor(via="ui", user_id=7)
    assert actor.as_dict() == {"via": "ui", "user_id": 7}


def test_actor_from_pat_bound_to_agent():
    claims = {
        "type": "pat",
        "purpose": "pat_access",
        "pat_id": 3,
        "pat_name": "claude-code",
        "agent_id": 11,
        "agent_name": "Analyst",
        "agent_kind": "external",
    }
    actor = actor_from_claims(7, claims)
    assert actor.via == "pat"
    assert actor.agent_id == 11
    assert actor.agent_kind == "external"
    assert actor.as_dict()["pat_name"] == "claude-code"


def test_actor_from_pat_without_agent_has_no_agent_fields():
    actor = actor_from_claims(7, {"type": "pat", "purpose": "pat_access", "pat_id": 3, "pat_name": "cli"})
    assert actor.agent_id is None
    assert "agent_id" not in actor.as_dict()


def test_actor_from_agent_and_runner_tokens():
    assert actor_from_claims(1, {"type": "delegated", "purpose": "agent_backend_consult", "agent_id": "4"}).via == "agent"
    assert actor_from_claims(1, {"type": "delegated", "purpose": "agent_backend_consult", "agent_id": "4"}).agent_id == 4
    assert actor_from_claims(1, {"type": "delegated", "purpose": "runner_backend"}).via == "runner"
    assert actor_from_claims(1, {"type": "delegated", "purpose": "something_else"}).via == "system"


def test_current_actor_contextvar_roundtrip():
    assert current_actor() is None
    token = set_current_actor(Actor(via="ui", user_id=1))
    try:
        assert current_actor() == Actor(via="ui", user_id=1)
    finally:
        reset_current_actor(token)
    assert current_actor() is None


def test_pat_can_read_own_profile_and_sub_resources():
    assert required_scope_for("GET", "/users/me") == "read"
    assert required_scope_for("GET", "/api/users/me/actor") == "read"
    assert required_scope_for("GET", "/users/me/studio-token") == "read"


def test_pat_still_banned_from_other_users_and_writes():
    assert required_scope_for("GET", "/users/") is None
    assert required_scope_for("GET", "/users/12") is None
    assert required_scope_for("PATCH", "/users/me") is None
    assert required_scope_for("POST", "/users/me/actor") is None
    assert required_scope_for("GET", "/pats/") is None


def test_trade_surfaces_unchanged():
    assert required_scope_for("POST", "/accounts/1/orders") == "trade"
    assert required_scope_for("POST", "/live/instances") == "trade"
    assert required_scope_for("POST", "/strategies/") == "write"
    assert required_scope_for("GET", "/strategies/") == "read"
