"""Subscription lifecycle: default plan, trials, manual assignments, plan
changes, end of period.

Single writer of ``subscription.status``: every transition goes through
here, writes a ``subscription_event`` row and queues the matching email.
Provider webhooks (phase 3) land in :func:`apply_event` with a normalized
:class:`~app.services.billing.provider.BillingEvent`.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import BackgroundTasks, HTTPException, status
from sqlalchemy import func, text
from sqlmodel import Session, select

from app.core.config import settings
from app.db.database import engine
from app.models.billing import (
    USER_EFFECTIVE_LIMITS_VIEW_SQL,
    AiModelRate,
    Plan,
    PlanPrice,
    PriceInterval,
    Subscription,
    SubscriptionEvent,
    SubscriptionStatus,
    TrialGrant,
)
from app.models.strategy import LiveStatus, Strategy, StrategyLive
from app.models.user import User
from app.services.email_service import queue_email
from app.services import email_templates
from app.services.entitlement_service import (
    add_months,
    ensure_current_subscription,
    get_current_subscription,
    get_default_plan,
    get_effective_limits,
    is_unlimited_user,
)
from app.services.limits import LimitKey, limit_value

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _email_hash(email: str) -> str:
    return hashlib.sha256(email.strip().lower().encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Bootstrap (dev databases built by create_all; prod runs migration 052)
# ---------------------------------------------------------------------------

_SEED_FREE_LIMITS = {
    "strategies_max": 3,
    "indicators_per_strategy_max": 6,
    "live_concurrent_max": 1,
    "backtest_concurrent_max": 1,
    "ai_credits_per_period": 300,
    "studios_max": 2,
    "studio_runs_concurrent_max": 1,
}
_SEED_PRO_LIMITS = {
    "strategies_max": 30,
    "indicators_per_strategy_max": 20,
    "live_concurrent_max": 5,
    "backtest_concurrent_max": 3,
    "ai_credits_per_period": 5000,
    "studios_max": 25,
    "studio_runs_concurrent_max": 3,
}
_SEED_PRO_PRICES = {"month": 2900, "quarter": 7900, "semester": 14900, "year": 27900}


def ensure_billing_schema() -> None:
    """Create the ``user_effective_limits`` view (idempotent)."""
    with engine.begin() as connection:
        connection.execute(text(USER_EFFECTIVE_LIMITS_VIEW_SQL))


def ensure_billing_seed(session: Session) -> None:
    """Seed Free (default) + Pro and the default model rate when the plan
    table is empty, then give every user without a current subscription the
    default plan. Same data as migration 052; safe to run at every startup."""
    if session.exec(select(func.count(Plan.id))).one() == 0:
        free = Plan(
            code="free",
            name="Free",
            description="Per iniziare: una strategia live, un backtest alla volta, crediti AI mensili.",
            is_default=True,
            sort_order=0,
            limits=dict(_SEED_FREE_LIMITS),
        )
        pro = Plan(
            code="pro",
            name="Pro",
            description="Per chi fa trading sistematico ogni giorno: più live, più backtest, più crediti AI.",
            sort_order=10,
            trial_days=14,
            limits=dict(_SEED_PRO_LIMITS),
        )
        session.add(free)
        session.add(pro)
        session.flush()
        for interval, cents in _SEED_PRO_PRICES.items():
            session.add(PlanPrice(plan_id=pro.id, interval=interval, amount_cents=cents, currency="EUR"))
        logger.info("Seeded default subscription plans (free, pro)")

    if session.exec(select(func.count(AiModelRate.id))).one() == 0:
        session.add(AiModelRate(model_pattern="*"))
    session.commit()

    default_plan = get_default_plan(session)
    if default_plan is None:
        return
    covered = select(Subscription.user_id).where(
        Subscription.status.in_(list(SubscriptionStatus.current_values()))
    )
    missing = list(session.exec(select(User.id).where(User.id.not_in(covered))).all())
    for user_id in missing:
        session.add(
            Subscription(
                user_id=user_id,
                plan_id=default_plan.id,
                status=SubscriptionStatus.FREE.value,
                provider="none",
                current_period_start=_utcnow(),
            )
        )
    if missing:
        session.commit()
        logger.info("Assigned the default plan to %d user(s) without a subscription", len(missing))


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def log_event(
    session: Session,
    *,
    user_id: int,
    type: str,
    subscription_id: int | None = None,
    payload: dict[str, Any] | None = None,
    actor_user_id: int | None = None,
    provider: str | None = None,
    provider_event_id: str | None = None,
) -> SubscriptionEvent:
    event = SubscriptionEvent(
        subscription_id=subscription_id,
        user_id=user_id,
        type=type,
        payload=payload,
        actor_user_id=actor_user_id,
        provider=provider,
        provider_event_id=provider_event_id,
    )
    session.add(event)
    session.flush()
    return event


def list_events(session: Session, user_id: int, *, limit: int = 50) -> list[SubscriptionEvent]:
    return list(
        session.exec(
            select(SubscriptionEvent)
            .where(SubscriptionEvent.user_id == user_id)
            .order_by(SubscriptionEvent.created_at.desc(), SubscriptionEvent.id.desc())
            .limit(limit)
        ).all()
    )


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


def list_public_plans(session: Session) -> list[Plan]:
    return list(
        session.exec(
            select(Plan)
            .where(Plan.is_public == True)  # noqa: E712
            .where(Plan.is_active == True)  # noqa: E712
            .order_by(Plan.sort_order, Plan.id)
        ).all()
    )


def list_plan_prices(session: Session, plan_ids: list[int], *, active_only: bool = True) -> dict[int, list[PlanPrice]]:
    if not plan_ids:
        return {}
    stmt = select(PlanPrice).where(PlanPrice.plan_id.in_(plan_ids))
    if active_only:
        stmt = stmt.where(PlanPrice.is_active == True)  # noqa: E712
    result: dict[int, list[PlanPrice]] = {plan_id: [] for plan_id in plan_ids}
    for price in session.exec(stmt.order_by(PlanPrice.plan_id, PlanPrice.id)).all():
        result.setdefault(price.plan_id, []).append(price)
    order = [interval.value for interval in PriceInterval]
    for prices in result.values():
        prices.sort(key=lambda p: order.index(p.interval) if p.interval in order else 99)
    return result


def _get_plan_or_404(session: Session, plan_id: int) -> Plan:
    plan = session.get(Plan, plan_id)
    if plan is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Piano non trovato")
    return plan


def trial_available(session: Session, user: User, plan: Plan) -> bool:
    if plan.trial_days <= 0 or not plan.is_active or plan.is_default:
        return False
    used = session.exec(
        select(TrialGrant.id)
        .where(TrialGrant.plan_id == plan.id)
        .where((TrialGrant.user_id == user.id) | (TrialGrant.email_hash == _email_hash(user.email)))
    ).first()
    return used is None


# ---------------------------------------------------------------------------
# Live sessions beyond the cap (end of period)
# ---------------------------------------------------------------------------


def live_sessions_in_excess(session: Session, user_id: int, limits: dict[str, Any]) -> list[StrategyLive]:
    """Active live sessions that exceed ``live_concurrent_max``: the N started
    first are kept, the rest are returned (newest last)."""
    cap = limit_value(limits, LimitKey.LIVE_CONCURRENT_MAX)
    if cap is None or is_unlimited_user(session, user_id):
        return []
    active = list(
        session.exec(
            select(StrategyLive)
            .join(Strategy, Strategy.id == StrategyLive.strategy_id)
            .where(Strategy.user_id == user_id)
            .where(StrategyLive.status.in_(list(LiveStatus.active_values())))
            .order_by(StrategyLive.started_at.asc().nullslast(), StrategyLive.id.asc())
        ).all()
    )
    return active[cap:]


def describe_live(session: Session, sl: StrategyLive) -> str:
    strategy = session.get(Strategy, sl.strategy_id)
    name = strategy.name if strategy else f"strategia {sl.strategy_id}"
    symbol = f" · {sl.symbol}" if sl.symbol else ""
    return f"{name}{symbol} (live #{sl.id})"


def stop_live_sessions_in_excess(
    session: Session, user_id: int, limits: dict[str, Any], *, reason: str
) -> list[str]:
    """Stop the live sessions beyond the new plan's cap through the same
    path as a manual stop (runner shuts down cleanly; open positions stay
    at the broker, as for a manual stop). Returns human-readable labels."""
    from app.api.live import _stop_live_instance_internal  # lazy: avoids an import cycle

    stopped: list[str] = []
    for sl in live_sessions_in_excess(session, user_id, limits):
        label = describe_live(session, sl)
        try:
            _stop_live_instance_internal(session, sl, remove=True)
            stopped.append(label)
            logger.info("Stopped live %s for user %s (%s)", sl.id, user_id, reason)
        except Exception:  # noqa: BLE001 - keep going with the other sessions
            logger.exception("Failed to stop live %s for user %s (%s)", sl.id, user_id, reason)
            session.rollback()
    return stopped


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------


def _close_subscription(session: Session, subscription: Subscription, new_status: str) -> None:
    subscription.status = new_status
    subscription.updated_at = _utcnow()
    session.add(subscription)
    session.flush()


def _open_subscription(session: Session, **fields: Any) -> Subscription:
    subscription = Subscription(**fields)
    session.add(subscription)
    session.flush()
    return subscription


def move_to_default_plan(
    session: Session,
    user: User,
    *,
    reason: str,
    actor_user_id: int | None = None,
    background_tasks: BackgroundTasks | None = None,
    notify: bool = True,
) -> Subscription:
    """End the user's current subscription and put them on the default plan,
    stopping the live sessions the new plan does not allow."""
    default_plan = get_default_plan(session)
    if default_plan is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Nessun piano di default")
    current = get_current_subscription(session, user.id, for_update=True)
    old_plan = session.get(Plan, current.plan_id) if current is not None else None
    if current is not None:
        if current.plan_id == default_plan.id and current.status == SubscriptionStatus.FREE.value:
            return current
        closed_status = (
            SubscriptionStatus.EXPIRED.value
            if current.status in {SubscriptionStatus.TRIALING.value, SubscriptionStatus.MANUAL.value}
            else SubscriptionStatus.CANCELED.value
        )
        _close_subscription(session, current, closed_status)
    now = _utcnow()
    fresh = _open_subscription(
        session,
        user_id=user.id,
        plan_id=default_plan.id,
        status=SubscriptionStatus.FREE.value,
        provider="none",
        current_period_start=now,
    )
    session.commit()

    stopped = stop_live_sessions_in_excess(session, user.id, dict(default_plan.limits or {}), reason=reason)
    log_event(
        session,
        user_id=user.id,
        subscription_id=fresh.id,
        type="plan_ended",
        payload={
            "reason": reason,
            "from_plan": old_plan.code if old_plan else None,
            "to_plan": default_plan.code,
            "stopped_live": stopped,
        },
        actor_user_id=actor_user_id,
    )
    session.commit()

    if notify:
        subject, text_body, html_body = email_templates.subscription_deactivated_email(
            display_name=user.display_name,
            old_plan_name=old_plan.name if old_plan else "precedente",
            new_plan_name=default_plan.name,
            stopped_live=stopped,
        )
        queue_email(background_tasks, to_address=user.email, subject=subject, text_body=text_body, html_body=html_body)
        if settings.BILLING_ADMIN_NOTIFY_EMAIL and stopped:
            queue_email(
                background_tasks,
                to_address=settings.BILLING_ADMIN_NOTIFY_EMAIL,
                subject=f"[EdgeWalker] Live fermate per fine piano: {user.email}",
                text_body="Live fermate:\n" + "\n".join(f"- {item}" for item in stopped),
            )
    return fresh


def start_trial(
    session: Session,
    user: User,
    plan_id: int,
    *,
    background_tasks: BackgroundTasks | None = None,
) -> Subscription:
    plan = _get_plan_or_404(session, plan_id)
    if plan.trial_days <= 0 or not plan.is_active or not plan.is_public:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Questo piano non prevede un periodo di prova")
    if not trial_available(session, user, plan):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Hai già usato il periodo di prova di questo piano",
        )
    current = ensure_current_subscription(session, user.id)
    if current is not None and current.status != SubscriptionStatus.FREE.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Hai già un abbonamento attivo: la prova è disponibile solo dal piano gratuito",
        )
    now = _utcnow()
    trial_end = now + timedelta(days=plan.trial_days)
    if current is not None:
        _close_subscription(session, current, SubscriptionStatus.EXPIRED.value)
    subscription = _open_subscription(
        session,
        user_id=user.id,
        plan_id=plan.id,
        status=SubscriptionStatus.TRIALING.value,
        provider="none",
        current_period_start=now,
        current_period_end=trial_end,
        trial_end=trial_end,
    )
    session.add(TrialGrant(user_id=user.id, plan_id=plan.id, email_hash=_email_hash(user.email)))
    log_event(
        session,
        user_id=user.id,
        subscription_id=subscription.id,
        type="trial_started",
        payload={"plan": plan.code, "trial_end": trial_end.isoformat()},
        actor_user_id=user.id,
    )
    session.commit()
    session.refresh(subscription)

    subject, text_body, html_body = email_templates.trial_started_email(
        display_name=user.display_name, plan_name=plan.name, trial_end=trial_end
    )
    queue_email(background_tasks, to_address=user.email, subject=subject, text_body=text_body, html_body=html_body)
    return subscription


def assign_plan_manually(
    session: Session,
    user: User,
    plan_id: int,
    *,
    actor: User,
    ends_at: datetime | None = None,
    note: str | None = None,
    background_tasks: BackgroundTasks | None = None,
    notify: bool = True,
) -> Subscription:
    """Admin assignment without payment (comp, beta tester). Assigning the
    default plan is the same as ending the current subscription."""
    plan = _get_plan_or_404(session, plan_id)
    if plan.is_default:
        return move_to_default_plan(
            session, user, reason="admin_assign_default", actor_user_id=actor.id,
            background_tasks=background_tasks, notify=notify,
        )
    current = get_current_subscription(session, user.id, for_update=True)
    if current is not None:
        if current.provider == "stripe" and current.status in {
            SubscriptionStatus.ACTIVE.value, SubscriptionStatus.PAST_DUE.value,
        }:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="L'utente ha un abbonamento a pagamento attivo: cancellalo dal provider prima di assegnare un piano manuale",
            )
        closed = (
            SubscriptionStatus.EXPIRED.value
            if current.status in {SubscriptionStatus.TRIALING.value, SubscriptionStatus.MANUAL.value, SubscriptionStatus.FREE.value}
            else SubscriptionStatus.CANCELED.value
        )
        _close_subscription(session, current, closed)
    now = _utcnow()
    subscription = _open_subscription(
        session,
        user_id=user.id,
        plan_id=plan.id,
        status=SubscriptionStatus.MANUAL.value,
        provider="manual",
        current_period_start=now,
        current_period_end=ends_at,
        ends_at=ends_at,
    )
    log_event(
        session,
        user_id=user.id,
        subscription_id=subscription.id,
        type="plan_assigned_by_admin",
        payload={"plan": plan.code, "ends_at": ends_at.isoformat() if ends_at else None, "note": note},
        actor_user_id=actor.id,
    )
    session.commit()
    session.refresh(subscription)

    # A smaller plan may leave live sessions above the new cap.
    stopped = stop_live_sessions_in_excess(session, user.id, dict(plan.limits or {}), reason="admin_assign")
    if stopped:
        log_event(
            session, user_id=user.id, subscription_id=subscription.id, type="live_stopped_by_plan",
            payload={"stopped_live": stopped}, actor_user_id=actor.id,
        )
        session.commit()

    if notify:
        subject, text_body, html_body = email_templates.subscription_assigned_by_admin_email(
            display_name=user.display_name, plan_name=plan.name, ends_at=ends_at, stopped_live=stopped
        )
        queue_email(background_tasks, to_address=user.email, subject=subject, text_body=text_body, html_body=html_body)
    return subscription


def extend_manual_subscription(
    session: Session, user: User, *, ends_at: datetime | None, actor: User
) -> Subscription:
    current = get_current_subscription(session, user.id, for_update=True)
    if current is None or current.status not in {SubscriptionStatus.MANUAL.value, SubscriptionStatus.TRIALING.value}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Solo un piano manuale o una prova possono essere estesi",
        )
    previous = current.effective_end
    if current.status == SubscriptionStatus.TRIALING.value:
        current.trial_end = ends_at
    current.ends_at = ends_at
    current.current_period_end = ends_at
    current.ending_notice_sent_at = None
    current.updated_at = _utcnow()
    session.add(current)
    log_event(
        session,
        user_id=user.id,
        subscription_id=current.id,
        type="plan_extended_by_admin",
        payload={
            "from": previous.isoformat() if previous else None,
            "to": ends_at.isoformat() if ends_at else None,
        },
        actor_user_id=actor.id,
    )
    session.commit()
    session.refresh(current)
    return current


def cancel_subscription_by_admin(
    session: Session, user: User, *, actor: User, background_tasks: BackgroundTasks | None = None
) -> Subscription:
    current = get_current_subscription(session, user.id)
    if current is not None and settings.BILLING_ENABLED and current.provider not in {"none", "manual"}:
        from app.services.billing.checkout_service import ENTITY_SUBSCRIPTION, get_external_ref
        from app.services.billing.provider import get_billing_provider

        provider = get_billing_provider()
        ref = get_external_ref(session, ENTITY_SUBSCRIPTION, current.id, provider.name)
        if ref is not None:
            provider.cancel_subscription(subscription_external_id=ref.external_id, at_period_end=False)
    return move_to_default_plan(
        session, user, reason="admin_cancel", actor_user_id=actor.id, background_tasks=background_tasks
    )


# ---------------------------------------------------------------------------
# End-of-period sweep (called by billing_sweeper)
# ---------------------------------------------------------------------------


def sweep_expired_subscriptions(session: Session, *, now: datetime | None = None) -> int:
    """Move to the default plan every subscription that ended on its own:
    trials past ``trial_end``, manual plans past ``ends_at``, and non-provider
    subscriptions whose scheduled cancellation is due. Provider-managed
    subscriptions end through their webhook instead."""
    now = now or _utcnow()
    candidates = list(
        session.exec(
            select(Subscription)
            .where(Subscription.status.in_(list(SubscriptionStatus.current_values())))
            .where(Subscription.provider.in_(["none", "manual"]))
        ).all()
    )
    ended = 0
    for subscription in candidates:
        end = subscription.effective_end
        if end is None or end > now:
            continue
        user = session.get(User, subscription.user_id)
        if user is None:
            continue
        try:
            move_to_default_plan(session, user, reason=f"{subscription.status}_ended")
            ended += 1
        except Exception:  # noqa: BLE001 - one user must not block the sweep
            logger.exception("Failed to end subscription %s", subscription.id)
            session.rollback()
    return ended


def sweep_ending_notices(session: Session, *, now: datetime | None = None) -> int:
    """T-N days before a subscription ends without renewal, warn the user and
    list the live sessions that will be stopped."""
    now = now or _utcnow()
    horizon = now + timedelta(days=settings.BILLING_ENDING_NOTICE_DAYS)
    default_plan = get_default_plan(session)
    candidates = list(
        session.exec(
            select(Subscription)
            .where(Subscription.status.in_(list(SubscriptionStatus.current_values())))
            .where(Subscription.ending_notice_sent_at == None)  # noqa: E711
        ).all()
    )
    sent = 0
    for subscription in candidates:
        end = subscription.effective_end
        if end is None or end > horizon or end <= now:
            continue
        user = session.get(User, subscription.user_id)
        plan = session.get(Plan, subscription.plan_id)
        if user is None or plan is None:
            continue
        to_stop = (
            [describe_live(session, sl) for sl in live_sessions_in_excess(session, user.id, dict(default_plan.limits or {}))]
            if default_plan is not None
            else []
        )
        builder = (
            email_templates.trial_ending_email
            if subscription.status == SubscriptionStatus.TRIALING.value
            else email_templates.plan_ending_soon_email
        )
        subject, text_body, html_body = builder(
            display_name=user.display_name, plan_name=plan.name, ends_at=end, live_to_stop=to_stop
        )
        queue_email(None, to_address=user.email, subject=subject, text_body=text_body, html_body=html_body)
        subscription.ending_notice_sent_at = now
        session.add(subscription)
        log_event(
            session, user_id=user.id, subscription_id=subscription.id, type="ending_notice_sent",
            payload={"ends_at": end.isoformat(), "live_to_stop": to_stop},
        )
        session.commit()
        sent += 1
    return sent


# ---------------------------------------------------------------------------
# Read models
# ---------------------------------------------------------------------------


def next_period_end(subscription: Subscription) -> Optional[datetime]:
    """Where the current billing period ends, for display."""
    if subscription.current_period_end is not None:
        return subscription.current_period_end
    if subscription.interval and subscription.current_period_start:
        return add_months(subscription.current_period_start, PriceInterval(subscription.interval).months)
    return None


__all__ = [
    "assign_plan_manually",
    "cancel_subscription_by_admin",
    "ensure_billing_schema",
    "ensure_billing_seed",
    "extend_manual_subscription",
    "get_effective_limits",
    "list_events",
    "list_plan_prices",
    "list_public_plans",
    "live_sessions_in_excess",
    "log_event",
    "move_to_default_plan",
    "next_period_end",
    "start_trial",
    "stop_live_sessions_in_excess",
    "sweep_ending_notices",
    "sweep_expired_subscriptions",
    "trial_available",
]
