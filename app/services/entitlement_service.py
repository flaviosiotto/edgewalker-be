"""Effective limits of a user and their enforcement.

Every creation/start path in the backend asks this module before acting:

    assert_within(session, user_id, LimitKey.STRATEGIES_MAX)
    check_ai_budget(session, user_id)

A violated limit raises :class:`LimitExceeded` — HTTP 402 with a structured
``detail`` the frontend turns into an upgrade prompt and mcp-svc relays to
the agent as a tool error. Counters and concurrency are computed live from
the domain tables; AI credits are the one metered consumption and have
their own ledger (``ai_credit_ledger`` / ``ai_credit_period``).

Enforcement is server-side only: frontend counters are informational.
"""

from __future__ import annotations

import calendar
import fnmatch
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from fastapi import BackgroundTasks, HTTPException, status
from sqlalchemy import func, text
from sqlmodel import Session, select

from app.models.billing import (
    AiCreditLedger,
    AiCreditPeriod,
    AiModelRate,
    Plan,
    Subscription,
    SubscriptionStatus,
)
from app.models.strategy import BacktestResult, BacktestStatus, LiveStatus, Strategy, StrategyLive
from app.models.user import User
from app.services.limits import LIMIT_REGISTRY, LimitKey, limit_value

logger = logging.getLogger(__name__)

#: Rough tokens-per-character ratio used when a turn is accounted from text
#: length (the runner and the chat path know the prompt/response size, the
#: real token counts arrive later from the n8n usage report and replace it).
CHARS_PER_TOKEN = 4
#: Notify the user when this share of the monthly credits is used.
AI_CREDITS_LOW_THRESHOLD = Decimal("0.8")


class LimitExceeded(HTTPException):
    """HTTP 402 with a machine-readable payload."""

    def __init__(
        self,
        *,
        limit: LimitKey,
        max_value: Optional[int],
        current: int | float,
        plan_code: str,
        plan_name: str,
        message: str | None = None,
    ) -> None:
        spec = LIMIT_REGISTRY[limit]
        detail = {
            "code": "limit_exceeded",
            "limit": limit.value,
            "label": spec.label,
            "max": max_value,
            "current": current,
            "plan": plan_code,
            "plan_name": plan_name,
            "message": message
            or f"Limite del piano {plan_name} raggiunto: {spec.label.lower()} ({current}/{max_value}).",
        }
        super().__init__(status_code=status.HTTP_402_PAYMENT_REQUIRED, detail=detail)


@dataclass
class EffectiveLimits:
    plan: Plan
    subscription: Optional[Subscription]
    limits: dict[str, Any]

    def value(self, key: LimitKey) -> Optional[int]:
        return limit_value(self.limits, key)


@dataclass
class AiBudget:
    granted: Optional[Decimal]
    used: Decimal
    period_start: date
    period_end: date

    @property
    def remaining(self) -> Optional[Decimal]:
        if self.granted is None:
            return None
        return max(Decimal("0"), self.granted - self.used)

    @property
    def exhausted(self) -> bool:
        return self.granted is not None and self.used >= self.granted


# ---------------------------------------------------------------------------
# Plan resolution
# ---------------------------------------------------------------------------


def get_default_plan(session: Session) -> Optional[Plan]:
    return session.exec(select(Plan).where(Plan.is_default == True)).first()  # noqa: E712


def get_current_subscription(session: Session, user_id: int, *, for_update: bool = False) -> Optional[Subscription]:
    stmt = (
        select(Subscription)
        .where(Subscription.user_id == user_id)
        .where(Subscription.status.in_(list(SubscriptionStatus.current_values())))
    )
    if for_update:
        stmt = stmt.with_for_update()
    return session.exec(stmt).first()


def ensure_current_subscription(session: Session, user_id: int) -> Optional[Subscription]:
    """Return the user's current subscription, creating the default-plan one
    on the fly for accounts born after the seed (migration 052 covered the
    users that existed then)."""
    current = get_current_subscription(session, user_id)
    if current is not None:
        return current
    default_plan = get_default_plan(session)
    if default_plan is None:
        return None
    now = datetime.now(timezone.utc)
    current = Subscription(
        user_id=user_id,
        plan_id=default_plan.id,
        status=SubscriptionStatus.FREE.value,
        provider="none",
        current_period_start=now,
    )
    session.add(current)
    session.commit()
    session.refresh(current)
    return current


def is_unlimited_user(session: Session, user_id: int) -> bool:
    """Administrators are exempt from every plan limit (decision 05/09)."""
    user = session.get(User, user_id)
    return user is not None and user.role == "admin"


def get_effective_limits(session: Session, user_id: int) -> EffectiveLimits:
    subscription = ensure_current_subscription(session, user_id)
    plan = session.get(Plan, subscription.plan_id) if subscription is not None else None
    if plan is None:
        plan = get_default_plan(session)
    if plan is None:
        # No plan configured at all (fresh database before the seed ran):
        # behave as unlimited rather than locking everyone out.
        plan = Plan(id=None, code="unlimited", name="Nessun piano", limits={})
    # An empty mapping means "every key unlimited" (missing key -> None).
    limits = {} if is_unlimited_user(session, user_id) else dict(plan.limits or {})
    return EffectiveLimits(plan=plan, subscription=subscription, limits=limits)


# ---------------------------------------------------------------------------
# Usage counters (live from the domain tables)
# ---------------------------------------------------------------------------


def count_strategies(session: Session, user_id: int) -> int:
    return int(session.exec(select(func.count(Strategy.id)).where(Strategy.user_id == user_id)).one())


def count_active_live_sessions(session: Session, user_id: int) -> int:
    return int(
        session.exec(
            select(func.count(StrategyLive.id))
            .join(Strategy, Strategy.id == StrategyLive.strategy_id)
            .where(Strategy.user_id == user_id)
            .where(StrategyLive.status.in_(list(LiveStatus.active_values())))
        ).one()
    )


def count_running_backtests(session: Session, user_id: int) -> int:
    return int(
        session.exec(
            select(func.count(BacktestResult.id))
            .join(Strategy, Strategy.id == BacktestResult.strategy_id)
            .where(Strategy.user_id == user_id)
            .where(BacktestResult.status == BacktestStatus.RUNNING.value)
        ).one()
    )


def count_connections(session: Session, user_id: int) -> int:
    from app.models.connection import Connection

    return int(session.exec(select(func.count(Connection.id)).where(Connection.user_id == user_id)).one())


def _count_studio_table(session: Session, sql: str, user_id: int) -> int:
    """Studios are owned by studio-svc (same database). Read-only touch,
    tolerant to the table not existing in a partial environment."""
    try:
        row = session.execute(text(sql), {"uid": user_id}).first()
    except Exception as exc:  # noqa: BLE001 - informational counter only
        logger.debug("studio counter unavailable: %s", exc)
        session.rollback()
        return 0
    return int(row[0] or 0) if row else 0


def count_studios(session: Session, user_id: int) -> int:
    return _count_studio_table(
        session, "SELECT count(*) FROM studio WHERE user_id = :uid AND deleted_at IS NULL", user_id
    )


def count_running_studio_runs(session: Session, user_id: int) -> int:
    return _count_studio_table(
        session,
        "SELECT count(*) FROM studio_run r JOIN studio s ON s.id = r.studio_id "
        "WHERE s.user_id = :uid AND r.status = 'running'",
        user_id,
    )


_COUNTERS = {
    LimitKey.STRATEGIES_MAX: count_strategies,
    LimitKey.LIVE_CONCURRENT_MAX: count_active_live_sessions,
    LimitKey.BACKTEST_CONCURRENT_MAX: count_running_backtests,
    LimitKey.STUDIOS_MAX: count_studios,
    LimitKey.STUDIO_RUNS_CONCURRENT_MAX: count_running_studio_runs,
    LimitKey.CONNECTIONS_MAX: count_connections,
}


def current_usage(session: Session, user_id: int, key: LimitKey) -> Optional[int]:
    counter = _COUNTERS.get(key)
    return counter(session, user_id) if counter else None


def count_strategy_indicators(definition: Any) -> int:
    """Indicators declared in a strategy definition: top-level ``indicators``
    plus every ``charts[*].indicators`` (multi-chart DSL)."""
    if not isinstance(definition, dict):
        return 0
    total = 0
    top = definition.get("indicators")
    if isinstance(top, list):
        total += len(top)
    charts = definition.get("charts")
    if isinstance(charts, list):
        for chart in charts:
            if isinstance(chart, dict) and isinstance(chart.get("indicators"), list):
                total += len(chart["indicators"])
    return total


# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------


def assert_within(
    session: Session,
    user_id: int,
    key: LimitKey,
    *,
    requested: int = 1,
    current: Optional[int] = None,
    lock: bool = False,
) -> None:
    """Raise :class:`LimitExceeded` when ``current + requested`` would exceed
    the plan limit for ``key``.

    ``lock=True`` takes a row lock on the user's subscription for the rest
    of the transaction, so two concurrent starts cannot both slip under a
    concurrency cap of 1.
    """
    effective = get_effective_limits(session, user_id)
    max_value = effective.value(key)
    if max_value is None:
        return
    if lock and effective.subscription is not None:
        get_current_subscription(session, user_id, for_update=True)
    usage = current if current is not None else (current_usage(session, user_id, key) or 0)
    if usage + requested > max_value:
        raise LimitExceeded(
            limit=key,
            max_value=max_value,
            current=usage,
            plan_code=effective.plan.code,
            plan_name=effective.plan.name,
        )


def assert_indicator_count(session: Session, user_id: int, definition: Any) -> None:
    effective = get_effective_limits(session, user_id)
    max_value = effective.value(LimitKey.INDICATORS_PER_STRATEGY_MAX)
    if max_value is None:
        return
    count = count_strategy_indicators(definition)
    if count > max_value:
        raise LimitExceeded(
            limit=LimitKey.INDICATORS_PER_STRATEGY_MAX,
            max_value=max_value,
            current=count,
            plan_code=effective.plan.code,
            plan_name=effective.plan.name,
            message=(
                f"La strategia ha {count} indicatori, il piano {effective.plan.name} "
                f"ne consente al massimo {max_value}."
            ),
        )


# ---------------------------------------------------------------------------
# AI credits
# ---------------------------------------------------------------------------


def _shift_months(anchor_day: int, year: int, month: int, delta: int) -> date:
    month_index = (year * 12 + (month - 1)) + delta
    year, month = divmod(month_index, 12)
    month += 1
    day = min(anchor_day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def ai_period_bounds(anchor: datetime, now: datetime) -> tuple[date, date]:
    """Monthly window anchored to the subscription start day: the latest
    monthly anniversary <= today, and the next one."""
    today = now.date()
    anchor_day = anchor.day
    start = _shift_months(anchor_day, today.year, today.month, 0)
    if start > today:
        start = _shift_months(anchor_day, today.year, today.month, -1)
    end = _shift_months(anchor_day, start.year, start.month, 1)
    return start, end


def _period_anchor(subscription: Optional[Subscription]) -> datetime:
    if subscription is not None:
        return subscription.current_period_start or subscription.created_at
    return datetime.now(timezone.utc)


def get_or_create_ai_period(
    session: Session, user_id: int, effective: EffectiveLimits, *, now: datetime | None = None
) -> AiCreditPeriod:
    now = now or datetime.now(timezone.utc)
    start, end = ai_period_bounds(_period_anchor(effective.subscription), now)
    granted_raw = effective.value(LimitKey.AI_CREDITS_PER_PERIOD)
    granted = None if granted_raw is None else Decimal(granted_raw)
    period = session.get(AiCreditPeriod, (user_id, start))
    if period is None:
        period = AiCreditPeriod(user_id=user_id, period_key=start, period_end=end, granted=granted)
        session.add(period)
        session.flush()
    elif period.granted != granted:
        # Plan changed mid-period: the allowance follows the current plan
        # (top-ups are ledger rows, they survive because they add to granted).
        period.granted = granted
        session.add(period)
        session.flush()
    return period


def get_ai_budget(session: Session, user_id: int) -> AiBudget:
    effective = get_effective_limits(session, user_id)
    period = get_or_create_ai_period(session, user_id, effective)
    return AiBudget(
        granted=period.granted, used=period.used, period_start=period.period_key, period_end=period.period_end
    )


def check_ai_budget(session: Session, user_id: int) -> AiBudget:
    """Pre-check before invoking the agent: the turn may run while
    ``used < granted`` (one turn may overshoot slightly; the next is blocked)."""
    effective = get_effective_limits(session, user_id)
    period = get_or_create_ai_period(session, user_id, effective)
    budget = AiBudget(
        granted=period.granted, used=period.used, period_start=period.period_key, period_end=period.period_end
    )
    if budget.exhausted:
        raise LimitExceeded(
            limit=LimitKey.AI_CREDITS_PER_PERIOD,
            max_value=int(budget.granted) if budget.granted is not None else None,
            current=float(budget.used),
            plan_code=effective.plan.code,
            plan_name=effective.plan.name,
            message=(
                f"Crediti AI del piano {effective.plan.name} esauriti per questo periodo "
                f"(si rinnovano il {budget.period_end.strftime('%d/%m/%Y')})."
            ),
        )
    return budget


def resolve_model_rate(session: Session, model: str | None) -> tuple[Decimal, Decimal]:
    rates = list(session.exec(select(AiModelRate).where(AiModelRate.is_active == True)).all())  # noqa: E712
    default = (Decimal("1"), Decimal("1"))
    if not rates:
        return default
    name = (model or "").strip().lower()
    best: Optional[AiModelRate] = None
    for rate in rates:
        pattern = rate.model_pattern.strip().lower()
        if pattern == "*":
            if best is None:
                best = rate
            continue
        if name and (name == pattern or fnmatch.fnmatch(name, pattern)):
            # Most specific pattern wins (longest literal prefix).
            if best is None or best.model_pattern == "*" or len(pattern) > len(best.model_pattern):
                best = rate
    if best is None:
        return default
    return Decimal(best.input_per_1k), Decimal(best.output_per_1k)


def compute_credits(session: Session, *, model: str | None, tokens_input: int, tokens_output: int) -> Decimal:
    rate_in, rate_out = resolve_model_rate(session, model)
    credits = (Decimal(tokens_input) / 1000) * rate_in + (Decimal(tokens_output) / 1000) * rate_out
    return credits.quantize(Decimal("0.001"))


def estimate_tokens_from_chars(chars: int | None) -> int:
    if not chars or chars <= 0:
        return 0
    return max(1, int(chars) // CHARS_PER_TOKEN)


def record_ai_usage(
    session: Session,
    *,
    user_id: int,
    tokens_input: int,
    tokens_output: int,
    model: str | None = None,
    correlation_id: str | None = None,
    session_id: str | None = None,
    estimated: bool = False,
    reason: str = "agent_turn",
    actor_user_id: int | None = None,
    background_tasks: BackgroundTasks | None = None,
) -> AiCreditLedger | None:
    """Charge one agent turn. Idempotent per ``(correlation_id, session_id)``:
    an estimate is replaced by the real token report, a real report is never
    overwritten. Returns the ledger row, or ``None`` when nothing changed."""
    effective = get_effective_limits(session, user_id)
    period = get_or_create_ai_period(session, user_id, effective)
    credits = compute_credits(session, model=model, tokens_input=tokens_input, tokens_output=tokens_output)

    existing: AiCreditLedger | None = None
    if correlation_id:
        existing = session.exec(
            select(AiCreditLedger)
            .where(AiCreditLedger.correlation_id == correlation_id)
            .where(AiCreditLedger.session_id == session_id)
        ).first()

    if existing is not None:
        if not existing.estimated or estimated:
            return None
        delta = credits - existing.credits
        target_period = session.get(AiCreditPeriod, (user_id, existing.period_key)) or period
        target_period.used = Decimal(target_period.used) + delta
        existing.credits = credits
        existing.tokens_input = tokens_input
        existing.tokens_output = tokens_output
        existing.model = model or existing.model
        existing.estimated = False
        session.add(existing)
        session.add(target_period)
        session.commit()
        _notify_thresholds(session, user_id, target_period, background_tasks)
        return existing

    entry = AiCreditLedger(
        user_id=user_id,
        period_key=period.period_key,
        credits=credits,
        reason=reason,
        model=model,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        correlation_id=correlation_id,
        session_id=session_id,
        estimated=estimated,
        actor_user_id=actor_user_id,
    )
    period.used = Decimal(period.used) + credits
    session.add(entry)
    session.add(period)
    session.commit()
    _notify_thresholds(session, user_id, period, background_tasks)
    return entry


def grant_ai_credits(
    session: Session,
    *,
    user_id: int,
    credits: Decimal,
    actor_user_id: int | None,
    reason: str = "admin_grant",
) -> AiCreditLedger:
    """Admin top-up: adds to ``granted`` of the current period (negative
    ledger row keeps the audit trail symmetric with consumption)."""
    effective = get_effective_limits(session, user_id)
    period = get_or_create_ai_period(session, user_id, effective)
    if period.granted is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Il piano corrente ha crediti illimitati: nessun top-up necessario",
        )
    period.granted = Decimal(period.granted) + credits
    entry = AiCreditLedger(
        user_id=user_id,
        period_key=period.period_key,
        credits=-credits,
        reason=reason,
        actor_user_id=actor_user_id,
    )
    session.add(period)
    session.add(entry)
    session.commit()
    session.refresh(entry)
    return entry


def _notify_thresholds(
    session: Session, user_id: int, period: AiCreditPeriod, background_tasks: BackgroundTasks | None
) -> None:
    if period.granted is None or period.granted <= 0:
        return
    now = datetime.now(timezone.utc)
    used = Decimal(period.used)
    granted = Decimal(period.granted)
    kind: str | None = None
    if used >= granted and period.exhausted_notified_at is None:
        period.exhausted_notified_at = now
        kind = "exhausted"
    elif used >= granted * AI_CREDITS_LOW_THRESHOLD and period.low_notified_at is None:
        period.low_notified_at = now
        kind = "low"
    if kind is None:
        return
    session.add(period)
    session.commit()
    user = session.get(User, user_id)
    if user is None:
        return
    try:
        from app.services.email_service import queue_email
        from app.services.email_templates import ai_credits_exhausted_email, ai_credits_low_email

        builder = ai_credits_exhausted_email if kind == "exhausted" else ai_credits_low_email
        subject, text_body, html_body = builder(
            display_name=user.display_name,
            used=int(used),
            granted=int(granted),
            renews_on=period.period_end,
        )
        queue_email(background_tasks, to_address=user.email, subject=subject, text_body=text_body, html_body=html_body)
    except Exception:  # noqa: BLE001 - a mail failure must not break accounting
        logger.exception("AI credits %s notification failed for user %s", kind, user_id)


# ---------------------------------------------------------------------------
# Snapshot for the FE / admin console
# ---------------------------------------------------------------------------


def usage_snapshot(session: Session, user_id: int) -> dict[str, dict[str, Any]]:
    effective = get_effective_limits(session, user_id)
    snapshot: dict[str, dict[str, Any]] = {}
    for key, spec in LIMIT_REGISTRY.items():
        max_value = effective.value(key)
        if key == LimitKey.AI_CREDITS_PER_PERIOD:
            budget = get_ai_budget(session, user_id)
            snapshot[key.value] = {
                "label": spec.label,
                "kind": spec.kind.value,
                "max": None if budget.granted is None else float(budget.granted),
                "current": float(budget.used),
                "period_start": budget.period_start.isoformat(),
                "period_end": budget.period_end.isoformat(),
            }
            continue
        if key == LimitKey.INDICATORS_PER_STRATEGY_MAX:
            snapshot[key.value] = {"label": spec.label, "kind": spec.kind.value, "max": max_value, "current": None}
            continue
        snapshot[key.value] = {
            "label": spec.label,
            "kind": spec.kind.value,
            "max": max_value,
            "current": current_usage(session, user_id, key),
        }
    return snapshot


def over_limit_keys(snapshot: dict[str, dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    for key, item in snapshot.items():
        max_value = item.get("max")
        current = item.get("current")
        if max_value is None or current is None:
            continue
        if float(current) > float(max_value):
            keys.append(key)
    return keys


def period_end_for(subscription: Optional[Subscription], now: datetime | None = None) -> date:
    _, end = ai_period_bounds(_period_anchor(subscription), now or datetime.now(timezone.utc))
    return end


def add_months(value: datetime, months: int) -> datetime:
    day = value.day
    shifted = _shift_months(day, value.year, value.month, months)
    return value.replace(year=shifted.year, month=shifted.month, day=shifted.day)


__all__ = [
    "AiBudget",
    "EffectiveLimits",
    "LimitExceeded",
    "add_months",
    "assert_indicator_count",
    "assert_within",
    "check_ai_budget",
    "count_strategy_indicators",
    "ensure_current_subscription",
    "estimate_tokens_from_chars",
    "get_ai_budget",
    "get_current_subscription",
    "get_default_plan",
    "get_effective_limits",
    "grant_ai_credits",
    "is_unlimited_user",
    "over_limit_keys",
    "record_ai_usage",
    "usage_snapshot",
]

