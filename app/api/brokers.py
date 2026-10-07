"""Broker catalogue — what the frontend needs to render connection forms.

Served from the shared broker registry (``edgewalker_platform.brokers``), so
adding a broker there is enough for the UI to offer it: label, logo, auth
flow, market types and the config fields with their kinds and defaults.
Secret *values* never travel here; only which keys are secrets.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from edgewalker_platform.brokers import public_brokers

router = APIRouter(prefix="/brokers", tags=["brokers"])


@router.get("/", response_model=list[dict[str, Any]])
def list_brokers(
    include_planned: bool = Query(
        False, description="Also return brokers designed but not yet available (status=planned)."
    ),
) -> list[dict[str, Any]]:
    """Brokers a user can open a connection to, with their config schema."""
    return public_brokers(include_planned=include_planned)
