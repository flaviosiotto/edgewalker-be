"""Typed registry of the platform limits a plan can set.

A plan stores its limits as JSONB (``plan.limits``); the KEYS live here so
adding a limit is one entry in :data:`LIMIT_REGISTRY` plus the enforcement
point, with no migration. The admin console builds its form from
``GET /admin/plans/limit-keys``. ``None`` (JSON ``null`` or a missing key)
always means *unlimited*.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class LimitKind(str, Enum):
    #: how many of a resource the user may own (strategies, studios…)
    COUNT = "count"
    #: how many may be running at once (live sessions, backtests…)
    CONCURRENCY = "concurrency"
    #: cap on a single resource (indicators inside one strategy)
    PER_RESOURCE = "per_resource"
    #: periodic consumption (AI credits per month)
    QUOTA = "quota"


class LimitKey(str, Enum):
    STRATEGIES_MAX = "strategies_max"
    INDICATORS_PER_STRATEGY_MAX = "indicators_per_strategy_max"
    LIVE_CONCURRENT_MAX = "live_concurrent_max"
    BACKTEST_CONCURRENT_MAX = "backtest_concurrent_max"
    AI_CREDITS_PER_PERIOD = "ai_credits_per_period"
    STUDIOS_MAX = "studios_max"
    STUDIO_RUNS_CONCURRENT_MAX = "studio_runs_concurrent_max"
    CONNECTIONS_MAX = "connections_max"


@dataclass(frozen=True)
class LimitSpec:
    key: LimitKey
    label: str
    description: str
    kind: LimitKind
    #: where the limit is enforced — informational, shown in the console
    enforced_by: str
    #: default used when the plan does not mention the key at all
    default: Optional[int] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key.value,
            "label": self.label,
            "description": self.description,
            "kind": self.kind.value,
            "enforced_by": self.enforced_by,
            "default": self.default,
        }


LIMIT_REGISTRY: dict[LimitKey, LimitSpec] = {
    spec.key: spec
    for spec in (
        LimitSpec(
            LimitKey.STRATEGIES_MAX,
            "Strategie",
            "Numero massimo di strategie che l'utente può possedere.",
            LimitKind.COUNT,
            "backend: creazione strategia",
        ),
        LimitSpec(
            LimitKey.INDICATORS_PER_STRATEGY_MAX,
            "Indicatori per strategia",
            "Indicatori totali (tutti i chart) dentro una singola strategia.",
            LimitKind.PER_RESOURCE,
            "backend: creazione/modifica strategia",
        ),
        LimitSpec(
            LimitKey.LIVE_CONCURRENT_MAX,
            "Live concorrenti",
            "Sessioni live attive contemporaneamente (starting/running/paused).",
            LimitKind.CONCURRENCY,
            "backend: avvio live; sweeper di fine periodo",
        ),
        LimitSpec(
            LimitKey.BACKTEST_CONCURRENT_MAX,
            "Backtest concorrenti",
            "Backtest in esecuzione contemporaneamente.",
            LimitKind.CONCURRENCY,
            "backend: avvio backtest",
        ),
        LimitSpec(
            LimitKey.AI_CREDITS_PER_PERIOD,
            "Crediti AI al mese",
            "Un credito = 1.000 token ponderati dalla tariffa del modello.",
            LimitKind.QUOTA,
            "backend: chat, trigger-agent; runner: alert/ask_agent/review",
        ),
        LimitSpec(
            LimitKey.STUDIOS_MAX,
            "Studi",
            "Numero massimo di Studi (notebook) dell'utente.",
            LimitKind.COUNT,
            "studio-svc: creazione/duplicazione Studio",
        ),
        LimitSpec(
            LimitKey.STUDIO_RUNS_CONCURRENT_MAX,
            "Run Studi concorrenti",
            "Container di run degli Studi in esecuzione contemporaneamente.",
            LimitKind.CONCURRENCY,
            "studio-svc: dispatcher dei run",
        ),
        LimitSpec(
            LimitKey.CONNECTIONS_MAX,
            "Connessioni broker",
            "Connessioni a broker/datafeed configurabili.",
            LimitKind.COUNT,
            "backend: creazione connessione (non ancora applicato)",
        ),
    )
}


def limit_keys_payload() -> list[dict[str, Any]]:
    return [spec.as_dict() for spec in LIMIT_REGISTRY.values()]


def normalize_limits(raw: dict[str, Any] | None) -> dict[str, Optional[int]]:
    """Validate a limits mapping coming from the admin console.

    Unknown keys are rejected (they would silently never be enforced);
    values must be non-negative integers or ``null``.
    """
    known = {key.value for key in LIMIT_REGISTRY}
    result: dict[str, Optional[int]] = {}
    for key, value in (raw or {}).items():
        if key not in known:
            raise ValueError(f"Limite sconosciuto: {key}")
        if value is None or value == "":
            result[key] = None
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError(f"Valore non valido per {key}")
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Valore non valido per {key}") from exc
        if number < 0:
            raise ValueError(f"Il limite {key} non può essere negativo")
        result[key] = number
    return result


def limit_value(limits: dict[str, Any] | None, key: LimitKey) -> Optional[int]:
    """Effective value of one limit: explicit value, else registry default,
    with ``None`` meaning unlimited."""
    if limits and key.value in limits:
        value = limits[key.value]
        return None if value is None else int(value)
    return LIMIT_REGISTRY[key].default
