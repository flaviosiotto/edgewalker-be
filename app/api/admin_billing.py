"""Administrator console: plans, prices, model rates, coupons, users and
their subscriptions. Every route is admin-only (router dependency)."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Annotated, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlmodel import Session, select

from app.api.billing import serialize_plan, serialize_subscription
from app.core.config import settings
from app.db.database import get_session
from app.models.billing import (
    AiCreditPeriod,
    AiModelRate,
    Coupon,
    Plan,
    PlanPrice,
    Subscription,
    SubscriptionStatus,
)
from app.models.strategy import BacktestResult, BacktestStatus, LiveStatus, Strategy, StrategyLive
from app.models.user import User
from app.schemas.auth import MessageResponse
from app.schemas.billing import (
    AdminAssignPlanRequest,
    AdminExtendRequest,
    AdminGrantCreditsRequest,
    AdminSubscriptionPage,
    AdminSubscriptionRow,
    AdminUserSubscriptionDetail,
    AiModelRateRead,
    AiModelRateUpsert,
    CouponCreate,
    CouponRead,
    LimitKeyRead,
    PlanCreate,
    PlanRead,
    PlanUpdate,
    SubscriptionEventRead,
    SubscriptionRead,
)
from app.services.billing.billing_service import (
    assign_plan_manually,
    cancel_subscription_by_admin,
    extend_manual_subscription,
    list_events,
    list_plan_prices,
    log_event,
)
from app.services.entitlement_service import (
    ai_period_bounds,
    ensure_current_subscription,
    get_current_subscription,
    grant_ai_credits,
)
from app.services.limits import LimitKey, limit_keys_payload, limit_value, normalize_limits
from app.utils.auth_utils import get_current_admin_user

router = APIRouter(
    prefix="/admin",
    tags=["Administration"],
    dependencies=[Depends(get_current_admin_user)],
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _get_plan_or_404(session: Session, plan_id: int) -> Plan:
    plan = session.get(Plan, plan_id)
    if plan is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Piano non trovato")
    return plan


def _get_user_or_404(session: Session, user_id: int) -> User:
    user = session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Utente non trovato")
    return user


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


@router.get("/plans/limit-keys", response_model=list[LimitKeyRead])
def list_limit_keys_endpoint():
    """Registry the console builds the limits form from."""
    return limit_keys_payload()


@router.get("/plans", response_model=list[PlanRead])
def list_plans_admin_endpoint(session: Session = Depends(get_session)):
    plans = list(session.exec(select(Plan).order_by(Plan.sort_order, Plan.id)).all())
    prices = list_plan_prices(session, [p.id for p in plans], active_only=False)
    return [serialize_plan(p, prices.get(p.id, [])) for p in plans]


def _validated_limits(raw) -> dict:
    try:
        return normalize_limits(raw)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


def _clear_other_defaults(session: Session, keep_plan_id: int | None) -> None:
    for other in session.exec(select(Plan).where(Plan.is_default == True)).all():  # noqa: E712
        if other.id != keep_plan_id:
            other.is_default = False
            session.add(other)
    session.flush()


def _replace_prices(session: Session, plan: Plan, prices) -> None:
    existing = {p.interval: p for p in session.exec(select(PlanPrice).where(PlanPrice.plan_id == plan.id)).all()}
    wanted = {p.interval.value: p for p in prices}
    for interval, row in existing.items():
        if interval not in wanted:
            session.delete(row)
    for interval, payload in wanted.items():
        row = existing.get(interval)
        if row is None:
            row = PlanPrice(plan_id=plan.id, interval=interval, amount_cents=payload.amount_cents,
                            currency=payload.currency, is_active=payload.is_active)
        else:
            row.amount_cents = payload.amount_cents
            row.currency = payload.currency
            row.is_active = payload.is_active
            row.updated_at = _utcnow()
        session.add(row)
    session.flush()


@router.post("/plans", response_model=PlanRead, status_code=status.HTTP_201_CREATED)
def create_plan_endpoint(
    payload: PlanCreate,
    admin: Annotated[User, Depends(get_current_admin_user)],
    session: Session = Depends(get_session),
):
    if session.exec(select(Plan).where(Plan.code == payload.code)).first() is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Esiste gia' un piano con questo codice")
    limits = _validated_limits(payload.limits)
    plan = Plan(
        code=payload.code,
        name=payload.name,
        description=payload.description,
        is_active=payload.is_active,
        is_public=payload.is_public,
        is_default=False,
        sort_order=payload.sort_order,
        trial_days=payload.trial_days,
        limits=limits,
    )
    session.add(plan)
    session.flush()
    if payload.is_default:
        _clear_other_defaults(session, plan.id)
        plan.is_default = True
        session.add(plan)
    _replace_prices(session, plan, payload.prices)
    session.commit()
    session.refresh(plan)
    prices = list_plan_prices(session, [plan.id], active_only=False)
    return serialize_plan(plan, prices.get(plan.id, []))


@router.patch("/plans/{plan_id}", response_model=PlanRead)
def update_plan_endpoint(
    plan_id: int,
    payload: PlanUpdate,
    session: Session = Depends(get_session),
):
    plan = _get_plan_or_404(session, plan_id)
    if payload.name is not None:
        plan.name = payload.name
    if payload.description is not None:
        plan.description = payload.description
    if payload.is_active is not None:
        if not payload.is_active and plan.is_default:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Il piano di default deve restare attivo")
        plan.is_active = payload.is_active
    if payload.is_public is not None:
        plan.is_public = payload.is_public
    if payload.sort_order is not None:
        plan.sort_order = payload.sort_order
    if payload.trial_days is not None:
        plan.trial_days = payload.trial_days
    if payload.limits is not None:
        plan.limits = _validated_limits(payload.limits)
    if payload.is_default is not None:
        if payload.is_default:
            _clear_other_defaults(session, plan.id)
            plan.is_default = True
            plan.is_active = True
        elif plan.is_default:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Imposta un altro piano come default invece di togliere il flag a questo",
            )
    if payload.prices is not None:
        _replace_prices(session, plan, payload.prices)
    plan.updated_at = _utcnow()
    session.add(plan)
    session.commit()
    session.refresh(plan)
    prices = list_plan_prices(session, [plan.id], active_only=False)
    return serialize_plan(plan, prices.get(plan.id, []))


@router.delete("/plans/{plan_id}", response_model=MessageResponse)
def delete_plan_endpoint(plan_id: int, session: Session = Depends(get_session)):
    plan = _get_plan_or_404(session, plan_id)
    if plan.is_default:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Il piano di default non si puo' eliminare")
    in_use = session.exec(select(func.count(Subscription.id)).where(Subscription.plan_id == plan.id)).one()
    if in_use:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Il piano ha abbonamenti (anche storici): disattivalo invece di eliminarlo",
        )
    session.delete(plan)
    session.commit()
    return MessageResponse(message="Piano eliminato")


# ---------------------------------------------------------------------------
# AI model rates
# ---------------------------------------------------------------------------


@router.get("/ai-model-rates", response_model=list[AiModelRateRead])
def list_model_rates_endpoint(session: Session = Depends(get_session)):
    return list(session.exec(select(AiModelRate).order_by(AiModelRate.model_pattern)).all())


@router.put("/ai-model-rates", response_model=AiModelRateRead)
def upsert_model_rate_endpoint(payload: AiModelRateUpsert, session: Session = Depends(get_session)):
    pattern = payload.model_pattern.strip()
    rate = session.exec(select(AiModelRate).where(AiModelRate.model_pattern == pattern)).first()
    if rate is None:
        rate = AiModelRate(model_pattern=pattern)
    rate.input_per_1k = payload.input_per_1k
    rate.output_per_1k = payload.output_per_1k
    rate.is_active = payload.is_active
    rate.updated_at = _utcnow()
    session.add(rate)
    session.commit()
    session.refresh(rate)
    return rate


@router.delete("/ai-model-rates/{rate_id}", response_model=MessageResponse)
def delete_model_rate_endpoint(rate_id: int, session: Session = Depends(get_session)):
    rate = session.get(AiModelRate, rate_id)
    if rate is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tariffa non trovata")
    if rate.model_pattern.strip() == "*":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="La tariffa di default (*) non si elimina")
    session.delete(rate)
    session.commit()
    return MessageResponse(message="Tariffa eliminata")


# ---------------------------------------------------------------------------
# Coupons (local registry; provider mirroring arrives with phase 4)
# ---------------------------------------------------------------------------


@router.get("/coupons", response_model=list[CouponRead])
def list_coupons_endpoint(session: Session = Depends(get_session)):
    return list(session.exec(select(Coupon).order_by(Coupon.created_at.desc())).all())


@router.post("/coupons", response_model=CouponRead, status_code=status.HTTP_201_CREATED)
def create_coupon_endpoint(
    payload: CouponCreate,
    admin: Annotated[User, Depends(get_current_admin_user)],
    session: Session = Depends(get_session),
):
    if session.exec(select(Coupon).where(Coupon.code == payload.code)).first() is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Codice coupon gia' esistente")
    if payload.kind == "percent" and payload.value > 100:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Uno sconto percentuale non supera 100")
    if payload.kind == "fixed" and not payload.currency:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Uno sconto fisso richiede la valuta")
    if payload.duration == "repeating" and not payload.duration_months:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Durata 'repeating' richiede i mesi")
    coupon = Coupon(
        code=payload.code,
        kind=payload.kind,
        value=payload.value,
        currency=payload.currency.upper() if payload.currency else None,
        duration=payload.duration,
        duration_months=payload.duration_months if payload.duration == "repeating" else None,
        applies_to_plan_ids=payload.applies_to_plan_ids,
        max_redemptions=payload.max_redemptions,
        valid_from=payload.valid_from,
        valid_until=payload.valid_until,
        note=payload.note,
        created_by=admin.id,
    )
    session.add(coupon)
    session.commit()
    session.refresh(coupon)
    if settings.BILLING_ENABLED:
        # Mirror on the provider right away so the code also works when typed
        # directly on the hosted checkout page.
        from app.services.billing.checkout_service import ensure_coupon_refs
        from app.services.billing.provider import get_billing_provider

        try:
            ensure_coupon_refs(session, coupon, get_billing_provider())
        except HTTPException:
            session.delete(coupon)
            session.commit()
            raise
    return coupon


@router.post("/coupons/{coupon_id}/revoke", response_model=CouponRead)
def revoke_coupon_endpoint(coupon_id: int, session: Session = Depends(get_session)):
    coupon = session.get(Coupon, coupon_id)
    if coupon is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Coupon non trovato")
    if coupon.revoked_at is None:
        coupon.revoked_at = _utcnow()
        session.add(coupon)
        session.commit()
        session.refresh(coupon)
        from app.services.billing.checkout_service import deactivate_coupon_on_provider

        deactivate_coupon_on_provider(session, coupon)
    return coupon


# ---------------------------------------------------------------------------
# Users & subscriptions
# ---------------------------------------------------------------------------


def _counters_by_user(session: Session) -> dict[int, dict[str, int]]:
    """One GROUP BY per counter instead of N queries per user."""
    counters: dict[int, dict[str, int]] = {}

    def put(user_id: int, key: str, value: int) -> None:
        counters.setdefault(int(user_id), {})[key] = int(value)

    for user_id, count in session.exec(
        select(Strategy.user_id, func.count(Strategy.id)).group_by(Strategy.user_id)
    ).all():
        put(user_id, LimitKey.STRATEGIES_MAX.value, count)
    for user_id, count in session.exec(
        select(Strategy.user_id, func.count(StrategyLive.id))
        .join(Strategy, Strategy.id == StrategyLive.strategy_id)
        .where(StrategyLive.status.in_(list(LiveStatus.active_values())))
        .group_by(Strategy.user_id)
    ).all():
        put(user_id, LimitKey.LIVE_CONCURRENT_MAX.value, count)
    for user_id, count in session.exec(
        select(Strategy.user_id, func.count(BacktestResult.id))
        .join(Strategy, Strategy.id == BacktestResult.strategy_id)
        .where(BacktestResult.status == BacktestStatus.RUNNING.value)
        .group_by(Strategy.user_id)
    ).all():
        put(user_id, LimitKey.BACKTEST_CONCURRENT_MAX.value, count)
    try:
        from sqlalchemy import text

        for user_id, count in session.execute(
            text("SELECT user_id, count(*) FROM studio WHERE deleted_at IS NULL GROUP BY user_id")
        ).all():
            put(user_id, LimitKey.STUDIOS_MAX.value, count)
    except Exception:  # noqa: BLE001 - studio table owned by studio-svc, may be absent
        session.rollback()
    return counters


@router.get("/subscriptions", response_model=AdminSubscriptionPage)
def list_subscriptions_endpoint(
    session: Session = Depends(get_session),
    plan_id: Optional[int] = Query(default=None),
    status_filter: Optional[str] = Query(default=None, alias="status"),
    over_limit_only: bool = Query(default=False),
    ending_within_days: Optional[int] = Query(default=None, ge=1, le=365),
    search: Optional[str] = Query(default=None),
):
    """Every user with their current plan, usage counters and flags."""
    users = list(session.exec(select(User).order_by(User.created_at.desc())).all())
    plans = {p.id: p for p in session.exec(select(Plan)).all()}
    subs = {
        s.user_id: s
        for s in session.exec(
            select(Subscription).where(Subscription.status.in_(list(SubscriptionStatus.current_values())))
        ).all()
    }
    default_plan = next((p for p in plans.values() if p.is_default), None)
    counters = _counters_by_user(session)
    now = _utcnow()
    periods = {
        (p.user_id, p.period_key): p
        for p in session.exec(select(AiCreditPeriod).where(AiCreditPeriod.period_end >= now.date())).all()
    }

    rows: list[AdminSubscriptionRow] = []
    needle = (search or "").strip().lower()
    for user in users:
        sub = subs.get(user.id)
        plan = plans.get(sub.plan_id) if sub else default_plan
        if plan is None:
            continue
        if plan_id is not None and plan.id != plan_id:
            continue
        sub_status = sub.status if sub else SubscriptionStatus.FREE.value
        if status_filter and sub_status != status_filter:
            continue
        if needle and needle not in f"{user.email} {user.username} {user.display_name}".lower():
            continue
        end = sub.effective_end if sub else None
        if ending_within_days is not None:
            if end is None or (end - now).days > ending_within_days or end < now:
                continue
        user_counters = counters.get(user.id, {})
        over: list[str] = []
        effective_plan_limits = {} if user.role == "admin" else (plan.limits or {})
        for key in (LimitKey.STRATEGIES_MAX, LimitKey.LIVE_CONCURRENT_MAX,
                    LimitKey.BACKTEST_CONCURRENT_MAX, LimitKey.STUDIOS_MAX):
            cap = limit_value(effective_plan_limits, key)
            if cap is not None and user_counters.get(key.value, 0) > cap:
                over.append(key.value)
        anchor = (sub.current_period_start or sub.created_at) if sub else now
        period_start, _ = ai_period_bounds(anchor, now)
        period = periods.get((user.id, period_start))
        if over_limit_only and not over:
            continue
        rows.append(
            AdminSubscriptionRow(
                user_id=user.id,
                email=user.email,
                username=user.username,
                display_name=user.display_name,
                role=user.role,
                user_status=user.status,
                subscription_id=sub.id if sub else None,
                plan_id=plan.id,
                plan_code=plan.code,
                plan_name=plan.name,
                status=sub_status,
                provider=sub.provider if sub else "none",
                current_period_end=sub.current_period_end if sub else None,
                ends_at=sub.ends_at if sub else None,
                trial_end=sub.trial_end if sub else None,
                cancel_at_period_end=sub.cancel_at_period_end if sub else False,
                ai_credits_used=float(period.used) if period else 0.0,
                ai_credits_granted=(
                    float(period.granted) if period and period.granted is not None
                    else (float(limit_value(plan.limits, LimitKey.AI_CREDITS_PER_PERIOD))
                          if limit_value(plan.limits, LimitKey.AI_CREDITS_PER_PERIOD) is not None else None)
                ),
                counters=user_counters,
                over_limit=over,
            )
        )
    return AdminSubscriptionPage(items=rows, total=len(rows))


@router.get("/users/{user_id}/subscription", response_model=AdminUserSubscriptionDetail)
def read_user_subscription_endpoint(user_id: int, session: Session = Depends(get_session)):
    user = _get_user_or_404(session, user_id)
    ensure_current_subscription(session, user.id)
    return AdminUserSubscriptionDetail(
        subscription=serialize_subscription(session, user, with_events=False),
        events=[SubscriptionEventRead.model_validate(e, from_attributes=True) for e in list_events(session, user.id)],
    )


@router.post("/users/{user_id}/subscription", response_model=SubscriptionRead)
def assign_user_subscription_endpoint(
    user_id: int,
    payload: AdminAssignPlanRequest,
    admin: Annotated[User, Depends(get_current_admin_user)],
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
):
    """Assign a plan without payment (comp, beta tester, support case)."""
    user = _get_user_or_404(session, user_id)
    if payload.ends_at is not None and payload.ends_at <= _utcnow():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="La scadenza deve essere nel futuro")
    assign_plan_manually(
        session, user, payload.plan_id, actor=admin, ends_at=payload.ends_at, note=payload.note,
        background_tasks=background_tasks, notify=payload.notify,
    )
    return serialize_subscription(session, user)


@router.post("/users/{user_id}/subscription/extend", response_model=SubscriptionRead)
def extend_user_subscription_endpoint(
    user_id: int,
    payload: AdminExtendRequest,
    admin: Annotated[User, Depends(get_current_admin_user)],
    session: Session = Depends(get_session),
):
    user = _get_user_or_404(session, user_id)
    extend_manual_subscription(session, user, ends_at=payload.ends_at, actor=admin)
    return serialize_subscription(session, user)


@router.post("/users/{user_id}/subscription/cancel", response_model=SubscriptionRead)
def cancel_user_subscription_endpoint(
    user_id: int,
    admin: Annotated[User, Depends(get_current_admin_user)],
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
):
    """End the current subscription now: back to the default plan, live
    sessions beyond its cap are stopped, the user is notified."""
    user = _get_user_or_404(session, user_id)
    cancel_subscription_by_admin(session, user, actor=admin, background_tasks=background_tasks)
    return serialize_subscription(session, user)


@router.post("/users/{user_id}/ai-credits", response_model=SubscriptionRead)
def grant_user_credits_endpoint(
    user_id: int,
    payload: AdminGrantCreditsRequest,
    admin: Annotated[User, Depends(get_current_admin_user)],
    session: Session = Depends(get_session),
):
    """Top-up of the current period's AI credits."""
    user = _get_user_or_404(session, user_id)
    grant_ai_credits(session, user_id=user.id, credits=Decimal(payload.credits), actor_user_id=admin.id)
    sub = get_current_subscription(session, user.id)
    log_event(
        session, user_id=user.id, subscription_id=sub.id if sub else None, type="ai_credits_granted",
        payload={"credits": float(payload.credits), "note": payload.note}, actor_user_id=admin.id,
    )
    session.commit()
    return serialize_subscription(session, user)
