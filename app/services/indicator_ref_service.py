"""Inject user-indicator refs into a strategy definition at run launch.

At backtest/live start the definition snapshot is walked, every indicator
type is resolved against indicator-svc for the owning user, and resolved ones
get ``params.ew_hash = <content_hash>`` — the reserved ref key the runtime
loaders understand (see edgewalker.indicators.loader). Built-in/TA-Lib types
come back in ``missing`` and are left untouched.

Best effort by design: if indicator-svc is unreachable the launch proceeds
and custom indicators simply produce no values (loud in the logs); the run
must not be hostage of the catalog service.
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any, Iterator

import httpx

logger = logging.getLogger(__name__)

_SERVICE_NAME = "edgewalker-be"


def _indicator_lists(definition: dict[str, Any]) -> Iterator[list[dict[str, Any]]]:
    """Yield every list of indicator configs across known definition shapes."""
    strategy = definition.get("strategy")
    if isinstance(strategy, dict):
        charts = strategy.get("charts")
        if isinstance(charts, list):
            for chart in charts:
                if isinstance(chart, dict) and isinstance(chart.get("indicators"), list):
                    yield chart["indicators"]
        if isinstance(strategy.get("indicators"), list):
            yield strategy["indicators"]
    if isinstance(definition.get("indicators"), list):
        yield definition["indicators"]
    datasets = definition.get("datasets")
    if isinstance(datasets, list):
        for dataset in datasets:
            features = dataset.get("features") if isinstance(dataset, dict) else None
            if isinstance(features, dict) and isinstance(features.get("indicators"), list):
                yield features["indicators"]


def inject_user_indicator_refs(definition: Any, user_id: int) -> Any:
    """Return the definition with current ``ew_hash`` refs for the user's
    custom indicators; the input object is never mutated."""
    if not isinstance(definition, dict):
        return definition

    updated = copy.deepcopy(definition)
    cfgs = [
        cfg
        for cfg_list in _indicator_lists(updated)
        for cfg in cfg_list
        if isinstance(cfg, dict) and cfg.get("type")
    ]
    type_keys = sorted({str(cfg["type"]).lower() for cfg in cfgs})
    if not type_keys:
        return definition

    base_url = os.getenv("INDICATOR_SVC_URL", "http://indicator-svc:8080").rstrip("/")
    try:
        from edgewalker_platform.auth.service_token import mint_service_token

        token = mint_service_token(
            issuer=_SERVICE_NAME,
            audience="indicator-svc",
            scopes=["indicators:resolve"],
        )
        response = httpx.post(
            f"{base_url}/internal/resolve",
            json={"user_id": user_id, "type_keys": type_keys},
            headers={"Authorization": f"Bearer {token}"},
            timeout=10.0,
        )
        response.raise_for_status()
        resolved: dict[str, Any] = response.json().get("resolved", {})
    except Exception as exc:  # noqa: BLE001 - launch must not depend on the catalog
        logger.error(
            "indicator-svc resolve failed (user %s, types %s): %s — "
            "custom indicators of this run will not compute",
            user_id,
            type_keys,
            exc,
        )
        return definition

    changed = False
    for cfg in cfgs:
        entry = resolved.get(str(cfg["type"]).lower())
        if not entry:
            continue
        params = cfg.setdefault("params", {})
        if params.get("ew_hash") != entry["content_hash"]:
            params["ew_hash"] = entry["content_hash"]
            changed = True

    if changed:
        logger.info(
            "Injected user indicator refs for user %s: %s",
            user_id,
            {key: entry["content_hash"][:12] for key, entry in resolved.items()},
        )
        return updated
    return definition
