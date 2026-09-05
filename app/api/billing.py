"""User-facing subscription endpoints + the two machine endpoints that meter
AI usage (runner pre-check, usage report).

Admin endpoints live in ``admin_billing.py``. Payment endpoints
(checkout, portal, webhook) arrive with the Stripe adapter in phase 3 and
will sit here under ``/billing``.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from sqlmodel import Session

from app.core.config import settings
from app.db.database import get_session
from app.models.billing import Plan, Subscription
from app.models.user import User
from app.schemas.billing import (
    AiBudgetRead,
    AiUsageReportRequest,
    AiUsageReportResponse,
    PlanPriceRead,
    PlanRead,
    PublicPlanRead,
    SubscriptionEventRead,
    SubscriptionRead,
    TrialStartRequest,
    UsageItem,
)
from app.services.billing.billing_service import (
    list_events,
    list_plan_prices,
    list_public_plans,
    start_trial,
    trial_available,
)
from app.services.entitlement_service import (
    check_ai_budget,
    estimate_tokens_from_chars,
    get_ai_budget,
    get_effective_limits,
    over_limit_keys,
    record_ai_usage,
    usage_snapshot,
)
from app.utils.auth_utils import (
    AuthPrincipal,
    get_current_active_principal,
    get_current_active_user,
    get_current_ai_usage_principal,
    get_current_runner_principal,
    decode_token,
    get_user_by_email,
)

router = APIRouter(tags=["Billing"])


# ---------------------------------------------------------------------------
# Serializers shared with admin_billing
# ---------------------------------------------------------------------------


def serialize_plan(plan: Plan, prices: list | None = None) -> PlanRead:
    return PlanRead(
        id=plan.id,
        code=plan.code,
        name=plan.name,
        description=plan.description,
        is_active=plan.is_active,
        is_public=plan.is_public,
        is_default=plan.is_default,
        sort_order=plan.sort_order,
        trial_days=plan.trial_days,
        limits=dict(plan.limits or {}),
        prices=[PlanPriceRead.model_validate(p, from_attributes=True) for p in (prices or [])],
        created_at=plan.created_at,
        updated_at=plan.updated_at,
    )


def serialize_subscription(
    session: Session, user: User, *, with_events: bool = True
) -> SubscriptionRead:
    effective = get_effective_limits(session, user.id)
    sub: Optional[Subscription] = effective.subscription
    prices = list_plan_prices(session, [effective.plan.id] if effective.plan.id else [])
    snapshot = usage_snapshot(session, user.id)
    trial_plan_ids = [p.id for p in list_public_plans(session) if trial_available(session, user, p)]
    events = list_events(session, user.id, limit=30) if with_events else []
    return SubscriptionRead(
        id=sub.id if sub else None,
        status=sub.status if sub else "free",
        provider=sub.provider if sub else "none",
        plan=serialize_plan(effective.plan, prices.get(effective.plan.id, [])),
        interval=sub.interval if sub else None,
        current_period_start=sub.current_period_start if sub else None,
        current_period_end=sub.current_period_end if sub else None,
        trial_end=sub.trial_end if sub else None,
        cancel_at_period_end=sub.cancel_at_period_end if sub else False,
        ends_at=sub.ends_at if sub else None,
        usage={key: UsageItem(**item) for key, item in snapshot.items()},
        over_limit=over_limit_keys(snapshot),
        trial_available_plan_ids=trial_plan_ids,
        billing_enabled=settings.BILLING_ENABLED,
        events=[SubscriptionEventRead.model_validate(e, from_attributes=True) for e in events],
    )


# ---------------------------------------------------------------------------
# Public pricing
# ---------------------------------------------------------------------------


def _optional_user(request: Request, session: Session) -> Optional[User]:
    """Best-effort identity for the public pricing page (adds
    ``trial_available`` per plan); anonymous callers get the plain list."""
    auth = request.headers.get("authorization") or ""
    if not auth.lower().startswith("bearer "):
        return None
    payload = decode_token(auth.split(" ", 1)[1].strip(), audience=settings.ACCESS_TOKEN_AUDIENCE)
    if not payload or payload.get("type") != "access":
        return None
    user = get_user_by_email(payload.get("sub") or "", session)
    return user if user is not None and user.is_active else None


@router.get("/plans", response_model=list[PublicPlanRead])
def list_plans_endpoint(request: Request, session: Session = Depends(get_session)):
    """Public and active plans with their prices: the pricing page."""
    plans = list_public_plans(session)
    prices = list_plan_prices(session, [p.id for p in plans])
    user = _optional_user(request, session)
    return [
        PublicPlanRead(
            id=plan.id,
            code=plan.code,
            name=plan.name,
            description=plan.description,
            is_default=plan.is_default,
            trial_days=plan.trial_days,
            trial_available=(trial_available(session, user, plan) if user is not None else None),
            limits=dict(plan.limits or {}),
            prices=[PlanPriceRead.model_validate(p, from_attributes=True) for p in prices.get(plan.id, [])],
        )
        for plan in plans
    ]


# ---------------------------------------------------------------------------
# Current user
# ---------------------------------------------------------------------------


@router.get("/users/me/subscription", response_model=SubscriptionRead)
def read_my_subscription(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    return serialize_subscription(session, current_user)


@router.post("/billing/trial", response_model=SubscriptionRead, status_code=status.HTTP_201_CREATED)
def start_trial_endpoint(
    payload: TrialStartRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_active_principal),
):
    """Start the free trial of a plan (no card). Interactive sessions only:
    an agent or a PAT must not enrol the account in anything."""
    if principal.claims.get("purpose") != "ui_auth":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="La prova puo' essere attivata solo da una sessione interattiva",
        )
    start_trial(session, principal.user, payload.plan_id, background_tasks=background_tasks)
    return serialize_subscription(session, principal.user)


# ---------------------------------------------------------------------------
# AI credits: pre-check (runner) + usage report (n8n / runner / chat path)
# ---------------------------------------------------------------------------


def _budget_read(budget, *, allowed: bool) -> AiBudgetRead:
    return AiBudgetRead(
        allowed=allowed,
        granted=budget.granted,
        used=budget.used,
        remaining=budget.remaining,
        period_start=budget.period_start,
        period_end=budget.period_end,
    )


@router.post("/runners/ai-credits/check", response_model=AiBudgetRead)
def runner_ai_credits_check(
    session: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_runner_principal),
):
    """Called by the strategy-runner before dispatching to the agent. A 402
    means: do not call n8n, tell the chat, mark the alert as skipped."""
    budget = check_ai_budget(session, principal.user.id)
    return _budget_read(budget, allowed=True)


@router.get("/users/me/ai-credits", response_model=AiBudgetRead)
def read_my_ai_credits(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    budget = get_ai_budget(session, current_user.id)
    return _budget_read(budget, allowed=not budget.exhausted)


@router.post("/ai-usage/report", response_model=AiUsageReportResponse)
def report_ai_usage(
    payload: AiUsageReportRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_ai_usage_principal),
):
    """Charge one agent turn to the caller's credits.

    Real token counts (``tokens_input``/``tokens_output``, from the n8n
    workflow) replace an earlier character-based estimate for the same
    ``(correlation_id, session_id)``; an estimate never overwrites a real
    report. Idempotent, so retries are harmless.
    """
    has_tokens = payload.tokens_input is not None or payload.tokens_output is not None
    if has_tokens:
        tokens_in = int(payload.tokens_input or 0)
        tokens_out = int(payload.tokens_output or 0)
        estimated = False
    else:
        tokens_in = estimate_tokens_from_chars(payload.prompt_chars)
        tokens_out = estimate_tokens_from_chars(payload.response_chars)
        estimated = True
    entry = record_ai_usage(
        session,
        user_id=principal.user.id,
        tokens_input=tokens_in,
        tokens_output=tokens_out,
        model=payload.model,
        correlation_id=payload.correlation_id,
        session_id=payload.session_id,
        estimated=estimated,
        reason=payload.reason,
        background_tasks=background_tasks,
    )
    budget = get_ai_budget(session, principal.user.id)
    return AiUsageReportResponse(
        recorded=entry is not None,
        credits=Decimal(entry.credits) if entry is not None else None,
        estimated=entry.estimated if entry is not None else None,
        used=budget.used,
        granted=budget.granted,
    )


__all__ = ["router", "serialize_plan", "serialize_subscription"]

