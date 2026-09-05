"""Paid subscriptions through the provider: checkout, portal, coupons,
webhook events, reconciliation.

Split from ``billing_service`` (default plan, trials, manual assignments,
end of period) so the provider-facing logic — and the only place that
touches ``billing_external_ref`` — stays in one module. Everything here
speaks to the provider through :class:`BillingProvider`; nothing knows
which provider is behind it.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import BackgroundTasks, HTTPException, status
from sqlmodel import Session, select

from app.core.config import settings
from app.models.billing import (
    BillingExternalRef,
    Coupon,
    CouponRedemption,
    Plan,
    PlanPrice,
    Subscription,
    SubscriptionEvent,
    SubscriptionStatus,
)
from app.models.user import User
from app.services import email_templates
from app.services.billing.billing_service import (
    _close_subscription,
    _open_subscription,
    log_event,
    move_to_default_plan,
    stop_live_sessions_in_excess,
)
from app.services.billing.provider import (
    BillingEvent,
    BillingEventType,
    BillingProvider,
    BillingSubscriptionStatus,
    get_billing_provider,
)
from app.services.email_service import queue_email
from app.services.email_templates import build_frontend_url
from app.services.entitlement_service import get_current_subscription

logger = logging.getLogger(__name__)

ENTITY_CUSTOMER = "customer"  # entity_id = user.id
ENTITY_PLAN = "plan"  # provider product
ENTITY_PRICE = "plan_price"
ENTITY_COUPON = "coupon"  # provider promotion code
ENTITY_COUPON_BASE = "coupon_base"  # provider coupon behind the promotion code
ENTITY_SUBSCRIPTION = "subscription"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# External references
# ---------------------------------------------------------------------------


def get_external_ref(session: Session, entity_type: str, entity_id: int, provider: str) -> Optional[BillingExternalRef]:
    return session.exec(
        select(BillingExternalRef)
        .where(BillingExternalRef.entity_type == entity_type)
        .where(BillingExternalRef.entity_id == entity_id)
        .where(BillingExternalRef.provider == provider)
    ).first()


def set_external_ref(session: Session, entity_type: str, entity_id: int, provider: str, external_id: str) -> BillingExternalRef:
    ref = get_external_ref(session, entity_type, entity_id, provider)
    if ref is None:
        ref = BillingExternalRef(entity_type=entity_type, entity_id=entity_id, provider=provider, external_id=external_id)
    else:
        ref.external_id = external_id
    session.add(ref)
    session.flush()
    return ref


def find_by_external_id(session: Session, provider: str, entity_type: str, external_id: str) -> Optional[BillingExternalRef]:
    return session.exec(
        select(BillingExternalRef)
        .where(BillingExternalRef.provider == provider)
        .where(BillingExternalRef.entity_type == entity_type)
        .where(BillingExternalRef.external_id == external_id)
    ).first()


# ---------------------------------------------------------------------------
# Provider-side objects created lazily
# ---------------------------------------------------------------------------


def ensure_customer_ref(session: Session, user: User, provider: BillingProvider) -> str:
    ref = get_external_ref(session, ENTITY_CUSTOMER, user.id, provider.name)
    if ref is not None:
        return ref.external_id
    external_id = provider.ensure_customer(user_id=user.id, email=user.email, display_name=user.display_name)
    set_external_ref(session, ENTITY_CUSTOMER, user.id, provider.name, external_id)
    session.commit()
    return external_id


def ensure_price_ref(session: Session, plan: Plan, price: PlanPrice, provider: BillingProvider) -> str:
    """Provider price for a plan price, (re)created when the local amount or
    period changed since the last sync (provider prices are immutable)."""
    product_ref = get_external_ref(session, ENTITY_PLAN, plan.id, provider.name)
    price_ref = get_external_ref(session, ENTITY_PRICE, price.id, provider.name)
    product_id, price_id = provider.sync_plan_price(
        plan_code=plan.code,
        plan_name=plan.name,
        interval=price.interval,
        amount_cents=price.amount_cents,
        currency=price.currency,
        existing_product_id=product_ref.external_id if product_ref else None,
        existing_price_id=price_ref.external_id if price_ref else None,
    )
    set_external_ref(session, ENTITY_PLAN, plan.id, provider.name, product_id)
    set_external_ref(session, ENTITY_PRICE, price.id, provider.name, price_id)
    session.commit()
    return price_id


def sync_plan_prices(session: Session, plan: Plan, provider: BillingProvider) -> dict[int, str]:
    """Every active price of a plan on the provider, keyed by local price id.

    The customer portal lets a subscriber switch between the prices of a
    product, so all of them must exist on the provider, not only the one being
    bought."""
    prices = session.exec(
        select(PlanPrice).where(PlanPrice.plan_id == plan.id).where(PlanPrice.is_active == True)  # noqa: E712
    ).all()
    return {price.id: ensure_price_ref(session, plan, price, provider) for price in prices}


def sync_catalog(session: Session) -> list[dict[str, Any]]:
    """Create/refresh products and prices on the provider for every active,
    non-default plan. Admin action: lets the provider-side portal be configured
    (eligible products) before the first sale."""
    provider = get_billing_provider()
    rows: list[dict[str, Any]] = []
    plans = session.exec(select(Plan).where(Plan.is_active == True).order_by(Plan.sort_order, Plan.id)).all()  # noqa: E712
    for plan in plans:
        if plan.is_default:
            continue
        synced = sync_plan_prices(session, plan, provider)
        product_ref = get_external_ref(session, ENTITY_PLAN, plan.id, provider.name)
        for price_id, price_external_id in synced.items():
            price = session.get(PlanPrice, price_id)
            rows.append({
                "plan_code": plan.code,
                "plan_name": plan.name,
                "interval": price.interval if price else None,
                "amount_cents": price.amount_cents if price else None,
                "currency": price.currency if price else None,
                "product_external_id": product_ref.external_id if product_ref else None,
                "price_external_id": price_external_id,
            })
    return rows


def ensure_coupon_refs(session: Session, coupon: Coupon, provider: BillingProvider) -> str:
    """Provider promotion code for a local coupon (created on first use)."""
    ref = get_external_ref(session, ENTITY_COUPON, coupon.id, provider.name)
    if ref is not None:
        return ref.external_id
    product_ids: list[str] = []
    for plan_id in coupon.applies_to_plan_ids or []:
        product_ref = get_external_ref(session, ENTITY_PLAN, plan_id, provider.name)
        if product_ref is None:
            plan = session.get(Plan, plan_id)
            prices = list(session.exec(select(PlanPrice).where(PlanPrice.plan_id == plan_id)).all()) if plan else []
            if plan and prices:
                ensure_price_ref(session, plan, prices[0], provider)
                product_ref = get_external_ref(session, ENTITY_PLAN, plan_id, provider.name)
        if product_ref is not None:
            product_ids.append(product_ref.external_id)
    ids = provider.sync_coupon(
        code=coupon.code,
        kind=coupon.kind,
        value=coupon.value,
        currency=coupon.currency,
        duration=coupon.duration,
        duration_months=coupon.duration_months,
        max_redemptions=coupon.max_redemptions,
        valid_until=coupon.valid_until,
        product_external_ids=product_ids or None,
    )
    set_external_ref(session, ENTITY_COUPON, coupon.id, provider.name, ids["promotion_code"])
    set_external_ref(session, ENTITY_COUPON_BASE, coupon.id, provider.name, ids["coupon"])
    session.commit()
    return ids["promotion_code"]


def deactivate_coupon_on_provider(session: Session, coupon: Coupon) -> None:
    if not settings.BILLING_ENABLED:
        return
    provider = get_billing_provider()
    ref = get_external_ref(session, ENTITY_COUPON, coupon.id, provider.name)
    if ref is not None:
        provider.deactivate_coupon(promotion_code_external_id=ref.external_id)


# ---------------------------------------------------------------------------
# Coupons (local rules are the source of truth)
# ---------------------------------------------------------------------------


def _coupon_error(message: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail={"code": "coupon_invalid", "message": message})


def validate_coupon(session: Session, code: str, plan: Plan, user_id: int | None = None) -> Coupon:
    normalized = (code or "").strip().upper()
    if not normalized:
        raise _coupon_error("Inserisci un codice coupon")
    coupon = session.exec(select(Coupon).where(Coupon.code == normalized)).first()
    now = _utcnow()
    if coupon is None or coupon.revoked_at is not None:
        raise _coupon_error("Coupon non valido")
    if coupon.valid_from and coupon.valid_from > now:
        raise _coupon_error("Coupon non ancora valido")
    if coupon.valid_until and coupon.valid_until < now:
        raise _coupon_error("Coupon scaduto")
    if coupon.max_redemptions is not None and coupon.redeemed_count >= coupon.max_redemptions:
        raise _coupon_error("Coupon esaurito")
    if plan.is_default:
        raise _coupon_error("Il piano gratuito non ha bisogno di coupon")
    if coupon.applies_to_plan_ids and plan.id not in coupon.applies_to_plan_ids:
        raise _coupon_error(f"Il coupon non vale per il piano {plan.name}")
    if user_id is not None:
        used = session.exec(
            select(CouponRedemption.id)
            .where(CouponRedemption.coupon_id == coupon.id)
            .where(CouponRedemption.user_id == user_id)
        ).first()
        if used is not None:
            raise _coupon_error("Hai già usato questo coupon")
    return coupon


def coupon_preview(coupon: Coupon, price: PlanPrice) -> dict[str, Any]:
    if coupon.kind == "percent":
        discount = round(price.amount_cents * coupon.value / 100)
        description = f"-{coupon.value}%"
    else:
        discount = min(price.amount_cents, coupon.value * 100)
        description = f"-{coupon.value} {coupon.currency or price.currency}"
    if coupon.duration == "once":
        description += " sul primo pagamento"
    elif coupon.duration == "repeating":
        description += f" per {coupon.duration_months} mesi"
    else:
        description += " per sempre"
    return {
        "discount_cents": discount,
        "final_cents": max(0, price.amount_cents - discount),
        "currency": price.currency,
        "description": description,
    }


# ---------------------------------------------------------------------------
# Checkout / portal
# ---------------------------------------------------------------------------


def _success_url() -> str:
    return settings.BILLING_SUCCESS_URL or build_frontend_url("settings", tab="subscription", checkout="success")


def _cancel_url() -> str:
    return settings.BILLING_CANCEL_URL or build_frontend_url("pricing", checkout="cancel")


def create_checkout(session: Session, user: User, plan_price_id: int, coupon_code: str | None = None) -> str:
    """URL of the hosted checkout for one plan price. Interactive sessions
    only (the API checks ``purpose = ui_auth``)."""
    provider = get_billing_provider()
    price = session.get(PlanPrice, plan_price_id)
    plan = session.get(Plan, price.plan_id) if price is not None else None
    if price is None or plan is None or not price.is_active or not plan.is_active:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tariffa non disponibile")
    if plan.is_default:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Il piano gratuito non si acquista")

    current = get_current_subscription(session, user.id)
    if current is not None and current.provider == provider.name and current.status in {
        SubscriptionStatus.ACTIVE.value,
        SubscriptionStatus.PAST_DUE.value,
    }:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "already_subscribed",
                "message": "Hai già un abbonamento a pagamento: cambia piano o metodo di pagamento dal portale.",
            },
        )

    coupon: Coupon | None = None
    promotion_code_id: str | None = None
    if coupon_code:
        coupon = validate_coupon(session, coupon_code, plan, user.id)
        promotion_code_id = ensure_coupon_refs(session, coupon, provider)

    customer_id = ensure_customer_ref(session, user, provider)
    # All the plan's prices, so the portal can switch period later.
    price_external_id = sync_plan_prices(session, plan, provider).get(price.id) or ensure_price_ref(
        session, plan, price, provider
    )
    metadata = {
        "user_id": str(user.id),
        "plan_id": str(plan.id),
        "plan_price_id": str(price.id),
        "coupon_id": str(coupon.id) if coupon else "",
    }
    checkout = provider.create_checkout(
        customer_external_id=customer_id,
        price_external_id=price_external_id,
        promotion_code_external_id=promotion_code_id,
        success_url=_success_url(),
        cancel_url=_cancel_url(),
        metadata=metadata,
        allow_promotion_codes=settings.BILLING_ALLOW_PROMOTION_CODES,
    )
    log_event(
        session,
        user_id=user.id,
        subscription_id=current.id if current else None,
        type="checkout_started",
        payload={"plan": plan.code, "interval": price.interval, "coupon": coupon.code if coupon else None,
                 "checkout_id": checkout.external_id},
        actor_user_id=user.id,
        provider=provider.name,
    )
    session.commit()
    return checkout.url


def create_portal(session: Session, user: User) -> str:
    provider = get_billing_provider()
    ref = get_external_ref(session, ENTITY_CUSTOMER, user.id, provider.name)
    if ref is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Nessun abbonamento a pagamento associato a questo account",
        )
    return provider.create_portal(
        customer_external_id=ref.external_id,
        return_url=build_frontend_url("settings", tab="subscription"),
    )


# ---------------------------------------------------------------------------
# Provider events
# ---------------------------------------------------------------------------


def _local_subscription_for(session: Session, event: BillingEvent) -> Optional[Subscription]:
    if not event.subscription_external_id:
        return None
    ref = find_by_external_id(session, event.provider, ENTITY_SUBSCRIPTION, event.subscription_external_id)
    return session.get(Subscription, ref.entity_id) if ref is not None else None


def _plan_price_for_external(session: Session, provider: str, price_external_id: str | None) -> Optional[PlanPrice]:
    if not price_external_id:
        return None
    ref = find_by_external_id(session, provider, ENTITY_PRICE, price_external_id)
    return session.get(PlanPrice, ref.entity_id) if ref is not None else None


def _send(builder, user: User, plan: Plan, period_end, background_tasks: BackgroundTasks | None) -> None:
    subject, text_body, html_body = builder(display_name=user.display_name, plan_name=plan.name, period_end=period_end)
    queue_email(background_tasks, to_address=user.email, subject=subject, text_body=text_body, html_body=html_body)


def _apply_checkout_completed(session: Session, event: BillingEvent, background_tasks: BackgroundTasks | None) -> bool:
    provider = get_billing_provider()
    if _local_subscription_for(session, event) is not None:
        return False  # already linked (duplicate delivery)
    try:
        user_id = int(event.metadata.get("user_id") or 0)
        plan_id = int(event.metadata.get("plan_id") or 0)
        plan_price_id = int(event.metadata.get("plan_price_id") or 0)
    except ValueError:
        user_id = plan_id = plan_price_id = 0
    user = session.get(User, user_id) if user_id else None
    plan = session.get(Plan, plan_id) if plan_id else None
    price = session.get(PlanPrice, plan_price_id) if plan_price_id else None
    if user is None or plan is None:
        logger.warning("checkout.completed %s without usable metadata: %s", event.event_id, event.metadata)
        return False
    coupon_id_raw = event.metadata.get("coupon_id") or ""
    coupon = session.get(Coupon, int(coupon_id_raw)) if coupon_id_raw.isdigit() else None

    remote = provider.fetch_subscription(subscription_external_id=event.subscription_external_id or "")
    current = get_current_subscription(session, user.id, for_update=True)
    if current is not None:
        closed = (
            SubscriptionStatus.EXPIRED.value
            if current.status in {SubscriptionStatus.FREE.value, SubscriptionStatus.TRIALING.value, SubscriptionStatus.MANUAL.value}
            else SubscriptionStatus.CANCELED.value
        )
        _close_subscription(session, current, closed)
    local_status = remote.status if remote.status in {
        BillingSubscriptionStatus.ACTIVE.value, BillingSubscriptionStatus.TRIALING.value, BillingSubscriptionStatus.PAST_DUE.value,
    } else SubscriptionStatus.ACTIVE.value
    subscription = _open_subscription(
        session,
        user_id=user.id,
        plan_id=plan.id,
        plan_price_id=price.id if price else None,
        status=local_status,
        interval=price.interval if price else None,
        provider=provider.name,
        current_period_start=remote.period_start or _utcnow(),
        current_period_end=remote.period_end,
        trial_end=None,
        cancel_at_period_end=remote.cancel_at_period_end,
        coupon_id=coupon.id if coupon else None,
    )
    set_external_ref(session, ENTITY_SUBSCRIPTION, subscription.id, provider.name, remote.external_id)
    if event.customer_external_id:
        set_external_ref(session, ENTITY_CUSTOMER, user.id, provider.name, event.customer_external_id)
    if coupon is not None:
        session.add(CouponRedemption(coupon_id=coupon.id, user_id=user.id, subscription_id=subscription.id))
        coupon.redeemed_count = int(coupon.redeemed_count or 0) + 1
        session.add(coupon)
    log_event(
        session, user_id=user.id, subscription_id=subscription.id, type=event.type.value,
        payload={**event.raw, "plan": plan.code, "interval": subscription.interval, "coupon": coupon.code if coupon else None},
        provider=event.provider, provider_event_id=event.event_id,
    )
    session.commit()

    stopped = stop_live_sessions_in_excess(session, user.id, dict(plan.limits or {}), reason="checkout_plan_change")
    if stopped:
        log_event(session, user_id=user.id, subscription_id=subscription.id, type="live_stopped_by_plan",
                  payload={"stopped_live": stopped})
        session.commit()
    _send(email_templates.subscription_activated_email, user, plan, subscription.current_period_end, background_tasks)
    return True


def _apply_subscription_updated(session: Session, subscription: Subscription, event: BillingEvent,
                                background_tasks: BackgroundTasks | None) -> None:
    user = session.get(User, subscription.user_id)
    plan = session.get(Plan, subscription.plan_id)
    was_cancel_scheduled = subscription.cancel_at_period_end
    previous_plan_id = subscription.plan_id

    if event.status in {BillingSubscriptionStatus.CANCELED.value, BillingSubscriptionStatus.EXPIRED.value}:
        log_event(session, user_id=subscription.user_id, subscription_id=subscription.id, type=event.type.value,
                  payload=event.raw, provider=event.provider, provider_event_id=event.event_id)
        session.commit()
        if user is not None:
            move_to_default_plan(session, user, reason="provider_ended", background_tasks=background_tasks)
        return

    if event.status:
        subscription.status = event.status
    if event.period_start:
        subscription.current_period_start = event.period_start
    if event.period_end:
        subscription.current_period_end = event.period_end
    if event.cancel_at_period_end is not None:
        subscription.cancel_at_period_end = event.cancel_at_period_end
        if not event.cancel_at_period_end:
            subscription.ending_notice_sent_at = None
    new_price = _plan_price_for_external(session, event.provider, event.price_external_id)
    if new_price is not None and new_price.id != subscription.plan_price_id:
        subscription.plan_price_id = new_price.id
        subscription.plan_id = new_price.plan_id
        subscription.interval = new_price.interval
    subscription.updated_at = _utcnow()
    session.add(subscription)
    log_event(session, user_id=subscription.user_id, subscription_id=subscription.id, type=event.type.value,
              payload=event.raw, provider=event.provider, provider_event_id=event.event_id)
    session.commit()

    if user is None:
        return
    new_plan = session.get(Plan, subscription.plan_id)
    if new_plan is not None and subscription.plan_id != previous_plan_id:
        stopped = stop_live_sessions_in_excess(session, user.id, dict(new_plan.limits or {}), reason="provider_plan_change")
        if stopped:
            log_event(session, user_id=user.id, subscription_id=subscription.id, type="live_stopped_by_plan",
                      payload={"stopped_live": stopped})
            session.commit()
        subject, text_body, html_body = email_templates.subscription_changed_email(
            display_name=user.display_name, old_plan_name=plan.name if plan else "precedente",
            new_plan_name=new_plan.name, period_end=subscription.current_period_end,
        )
        queue_email(background_tasks, to_address=user.email, subject=subject, text_body=text_body, html_body=html_body)
    elif subscription.cancel_at_period_end and not was_cancel_scheduled and new_plan is not None:
        _send(email_templates.cancel_scheduled_email, user, new_plan, subscription.current_period_end, background_tasks)


def apply_event(session: Session, event: BillingEvent, *, background_tasks: BackgroundTasks | None = None) -> bool:
    """Apply one normalized provider event. Idempotent on
    ``(provider, event_id)``; returns False when nothing was done."""
    duplicate = session.exec(
        select(SubscriptionEvent.id)
        .where(SubscriptionEvent.provider == event.provider)
        .where(SubscriptionEvent.provider_event_id == event.event_id)
    ).first()
    if duplicate is not None:
        return False

    if event.type == BillingEventType.CHECKOUT_COMPLETED:
        return _apply_checkout_completed(session, event, background_tasks)

    subscription = _local_subscription_for(session, event)
    if subscription is None:
        # Events for a subscription we have not linked yet (checkout.completed
        # may arrive after subscription.created): the provider retries and
        # the daily reconciliation covers the rest.
        logger.info("Billing event %s/%s for unknown subscription %s", event.provider, event.event_id,
                    event.subscription_external_id)
        return False

    if event.type == BillingEventType.SUBSCRIPTION_UPDATED:
        _apply_subscription_updated(session, subscription, event, background_tasks)
        return True

    user = session.get(User, subscription.user_id)
    plan = session.get(Plan, subscription.plan_id)
    now = _utcnow()

    if event.type == BillingEventType.SUBSCRIPTION_ENDED:
        log_event(session, user_id=subscription.user_id, subscription_id=subscription.id, type=event.type.value,
                  payload=event.raw, provider=event.provider, provider_event_id=event.event_id)
        session.commit()
        if user is not None:
            move_to_default_plan(session, user, reason="provider_ended", background_tasks=background_tasks)
        return True

    builder = None
    if event.type in {BillingEventType.SUBSCRIPTION_ACTIVATED, BillingEventType.SUBSCRIPTION_RENEWED}:
        try:
            remote = get_billing_provider().fetch_subscription(subscription_external_id=event.subscription_external_id or "")
            subscription.current_period_start = remote.period_start or subscription.current_period_start
            subscription.current_period_end = remote.period_end or subscription.current_period_end
            subscription.cancel_at_period_end = remote.cancel_at_period_end
        except HTTPException:
            logger.warning("Could not refresh subscription %s after %s", subscription.id, event.type.value)
        subscription.status = SubscriptionStatus.ACTIVE.value
        subscription.updated_at = now
        session.add(subscription)
        # The activation email went out with checkout.completed; renewals get theirs.
        builder = email_templates.subscription_renewed_email if event.type == BillingEventType.SUBSCRIPTION_RENEWED else None
    elif event.type == BillingEventType.PAYMENT_FAILED:
        subscription.status = SubscriptionStatus.PAST_DUE.value
        subscription.updated_at = now
        session.add(subscription)
        builder = email_templates.payment_failed_email
    elif event.type == BillingEventType.SUBSCRIPTION_CANCELED:
        subscription.cancel_at_period_end = True
        if event.period_end:
            subscription.current_period_end = event.period_end
        subscription.updated_at = now
        session.add(subscription)
        builder = email_templates.cancel_scheduled_email

    log_event(session, user_id=subscription.user_id, subscription_id=subscription.id, type=event.type.value,
              payload=event.raw, provider=event.provider, provider_event_id=event.event_id)
    session.commit()

    if builder is not None and user is not None and plan is not None:
        _send(builder, user, plan, subscription.current_period_end, background_tasks)
        if event.type == BillingEventType.PAYMENT_FAILED and settings.BILLING_ADMIN_NOTIFY_EMAIL:
            queue_email(
                background_tasks,
                to_address=settings.BILLING_ADMIN_NOTIFY_EMAIL,
                subject=f"[EdgeWalker] Pagamento fallito: {user.email}",
                text_body=f"Pagamento fallito per {user.email}, piano {plan.name}, abbonamento #{subscription.id}.",
            )
    return True


# ---------------------------------------------------------------------------
# Reconciliation (drift between provider and local state)
# ---------------------------------------------------------------------------


def reconcile_provider_subscriptions(session: Session) -> int:
    """Daily: refresh status / period / cancel flag of every provider-managed
    subscription; ended ones go back to the default plan."""
    if not settings.BILLING_ENABLED:
        return 0
    provider = get_billing_provider()
    rows = list(
        session.exec(
            select(Subscription)
            .where(Subscription.provider == provider.name)
            .where(Subscription.status.in_(list(SubscriptionStatus.current_values())))
        ).all()
    )
    changed = 0
    for subscription in rows:
        ref = get_external_ref(session, ENTITY_SUBSCRIPTION, subscription.id, provider.name)
        if ref is None:
            continue
        try:
            remote = provider.fetch_subscription(subscription_external_id=ref.external_id)
        except HTTPException as exc:
            logger.warning("Reconcile: cannot fetch subscription %s: %s", subscription.id, exc.detail)
            continue
        user = session.get(User, subscription.user_id)
        if remote.status in {BillingSubscriptionStatus.CANCELED.value, BillingSubscriptionStatus.EXPIRED.value}:
            if user is not None:
                move_to_default_plan(session, user, reason="reconcile_ended")
                changed += 1
            continue
        dirty = False
        if remote.status != subscription.status:
            subscription.status = remote.status
            dirty = True
        if remote.period_end and remote.period_end != subscription.current_period_end:
            subscription.current_period_end = remote.period_end
            dirty = True
        if remote.period_start and remote.period_start != subscription.current_period_start:
            subscription.current_period_start = remote.period_start
            dirty = True
        if remote.cancel_at_period_end != subscription.cancel_at_period_end:
            subscription.cancel_at_period_end = remote.cancel_at_period_end
            dirty = True
        new_price = _plan_price_for_external(session, provider.name, remote.price_external_id)
        if new_price is not None and new_price.id != subscription.plan_price_id:
            subscription.plan_price_id = new_price.id
            subscription.plan_id = new_price.plan_id
            subscription.interval = new_price.interval
            dirty = True
        if dirty:
            subscription.updated_at = _utcnow()
            session.add(subscription)
            log_event(session, user_id=subscription.user_id, subscription_id=subscription.id,
                      type="reconciled", payload={"status": remote.status})
            session.commit()
            changed += 1
    return changed


__all__ = [
    "apply_event",
    "coupon_preview",
    "create_checkout",
    "create_portal",
    "deactivate_coupon_on_provider",
    "ensure_coupon_refs",
    "get_external_ref",
    "reconcile_provider_subscriptions",
    "validate_coupon",
]

