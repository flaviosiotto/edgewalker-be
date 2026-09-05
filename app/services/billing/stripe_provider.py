"""Stripe adapter of :class:`~app.services.billing.provider.BillingProvider`.

Hosted Checkout for purchases, Customer Portal for card / invoices /
cancellation (no Stripe.js in the SPA, no card data in the backend), signed
webhooks translated into the normalized :class:`BillingEvent` vocabulary.
PayPal is enabled as a Stripe payment method from the dashboard: we never
pass ``payment_method_types`` so the dashboard configuration applies.

Nothing outside this module imports ``stripe``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import stripe
from fastapi import HTTPException, status

from app.core.config import settings
from app.services.billing.provider import (
    BillingEvent,
    BillingEventType,
    BillingSubscriptionStatus,
    CheckoutSession,
    ProviderSubscription,
)

logger = logging.getLogger(__name__)

PROVIDER_NAME = "stripe"

#: our billing intervals -> Stripe recurring settings
_RECURRING = {
    "month": {"interval": "month", "interval_count": 1},
    "quarter": {"interval": "month", "interval_count": 3},
    "semester": {"interval": "month", "interval_count": 6},
    "year": {"interval": "year", "interval_count": 1},
}

_STATUS_MAP = {
    "trialing": BillingSubscriptionStatus.TRIALING,
    "active": BillingSubscriptionStatus.ACTIVE,
    "past_due": BillingSubscriptionStatus.PAST_DUE,
    "unpaid": BillingSubscriptionStatus.PAST_DUE,
    "incomplete": BillingSubscriptionStatus.PAST_DUE,
    "paused": BillingSubscriptionStatus.PAST_DUE,
    "canceled": BillingSubscriptionStatus.CANCELED,
    "incomplete_expired": BillingSubscriptionStatus.EXPIRED,
}


def _ts(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Stripe objects behave like dicts; plain dicts arrive from tests."""
    if obj is None:
        return default
    try:
        value = obj[key]
    except (KeyError, TypeError, IndexError):
        value = getattr(obj, key, default)
    return default if value is None else value


def _provider_error(exc: Exception) -> HTTPException:
    message = getattr(exc, "user_message", None) or str(exc)
    logger.warning("Stripe error: %s", message)
    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail={"code": "billing_provider_error", "message": f"Stripe: {message}"},
    )


def map_subscription(obj: Any) -> ProviderSubscription:
    """Normalize a Stripe Subscription object (webhook payload or retrieve).

    Recent API versions moved ``current_period_*`` from the subscription to
    its items; both shapes are read.
    """
    items = _get(_get(obj, "items"), "data", []) or []
    first_item = items[0] if items else None
    price = _get(first_item, "price")
    period_start = _get(obj, "current_period_start") or _get(first_item, "current_period_start")
    period_end = _get(obj, "current_period_end") or _get(first_item, "current_period_end")
    raw_status = str(_get(obj, "status", "") or "")
    customer = _get(obj, "customer")
    return ProviderSubscription(
        external_id=str(_get(obj, "id")),
        status=_STATUS_MAP.get(raw_status, BillingSubscriptionStatus.PAST_DUE).value,
        period_start=_ts(period_start),
        period_end=_ts(period_end),
        cancel_at_period_end=bool(_get(obj, "cancel_at_period_end", False)),
        price_external_id=str(_get(price, "id")) if price is not None else None,
        customer_external_id=str(customer if isinstance(customer, str) else _get(customer, "id")) if customer else None,
    )


def _invoice_subscription_id(invoice: Any) -> Optional[str]:
    """``invoice.subscription`` (classic) or ``invoice.parent.subscription_details.subscription``."""
    direct = _get(invoice, "subscription")
    if direct:
        return direct if isinstance(direct, str) else str(_get(direct, "id"))
    details = _get(_get(invoice, "parent"), "subscription_details")
    nested = _get(details, "subscription")
    if nested:
        return nested if isinstance(nested, str) else str(_get(nested, "id"))
    return None


def map_event(event: Any) -> list[BillingEvent]:
    """Pure mapping of one Stripe event to zero or more normalized events."""
    event_type = str(_get(event, "type", "") or "")
    event_id = str(_get(event, "id", "") or "")
    obj = _get(_get(event, "data"), "object")
    raw = {"type": event_type, "object_id": _get(obj, "id")}

    if event_type == "checkout.session.completed":
        if str(_get(obj, "mode", "")) != "subscription":
            return []
        subscription = _get(obj, "subscription")
        customer = _get(obj, "customer")
        metadata = dict(_get(obj, "metadata", {}) or {})
        return [
            BillingEvent(
                type=BillingEventType.CHECKOUT_COMPLETED,
                provider=PROVIDER_NAME,
                event_id=event_id,
                subscription_external_id=subscription if isinstance(subscription, str) else str(_get(subscription, "id")),
                customer_external_id=customer if isinstance(customer, str) else str(_get(customer, "id")),
                metadata={str(k): str(v) for k, v in metadata.items()},
                raw=raw,
            )
        ]

    if event_type in {"customer.subscription.updated", "customer.subscription.created"}:
        sub = map_subscription(obj)
        return [
            BillingEvent(
                type=BillingEventType.SUBSCRIPTION_UPDATED,
                provider=PROVIDER_NAME,
                event_id=event_id,
                subscription_external_id=sub.external_id,
                customer_external_id=sub.customer_external_id,
                price_external_id=sub.price_external_id,
                status=sub.status,
                period_start=sub.period_start,
                period_end=sub.period_end,
                cancel_at_period_end=sub.cancel_at_period_end,
                metadata={str(k): str(v) for k, v in (dict(_get(obj, "metadata", {}) or {})).items()},
                raw=raw,
            )
        ]

    if event_type == "customer.subscription.deleted":
        sub = map_subscription(obj)
        return [
            BillingEvent(
                type=BillingEventType.SUBSCRIPTION_ENDED,
                provider=PROVIDER_NAME,
                event_id=event_id,
                subscription_external_id=sub.external_id,
                customer_external_id=sub.customer_external_id,
                status=sub.status,
                raw=raw,
            )
        ]

    if event_type in {"invoice.paid", "invoice.payment_succeeded"}:
        subscription_id = _invoice_subscription_id(obj)
        if not subscription_id:
            return []
        reason = str(_get(obj, "billing_reason", "") or "")
        kind = (
            BillingEventType.SUBSCRIPTION_ACTIVATED
            if reason == "subscription_create"
            else BillingEventType.SUBSCRIPTION_RENEWED
        )
        return [
            BillingEvent(
                type=kind,
                provider=PROVIDER_NAME,
                event_id=event_id,
                subscription_external_id=subscription_id,
                reason=reason,
                raw=raw,
            )
        ]

    if event_type == "invoice.payment_failed":
        subscription_id = _invoice_subscription_id(obj)
        if not subscription_id:
            return []
        return [
            BillingEvent(
                type=BillingEventType.PAYMENT_FAILED,
                provider=PROVIDER_NAME,
                event_id=event_id,
                subscription_external_id=subscription_id,
                reason=str(_get(obj, "billing_reason", "") or ""),
                raw=raw,
            )
        ]

    if event_type == "customer.subscription.trial_will_end":
        return [
            BillingEvent(
                type=BillingEventType.TRIAL_WILL_END,
                provider=PROVIDER_NAME,
                event_id=event_id,
                subscription_external_id=str(_get(obj, "id")),
                raw=raw,
            )
        ]

    return []


class StripeProvider:
    name = PROVIDER_NAME

    def __init__(self) -> None:
        if not settings.STRIPE_SECRET_KEY:
            raise RuntimeError("STRIPE_SECRET_KEY is required when BILLING_PROVIDER=stripe")
        stripe.api_key = settings.STRIPE_SECRET_KEY
        if settings.STRIPE_API_VERSION:
            stripe.api_version = settings.STRIPE_API_VERSION
        stripe.max_network_retries = 2

    # -- customers ---------------------------------------------------------

    def ensure_customer(self, *, user_id: int, email: str, display_name: str) -> str:
        try:
            customer = stripe.Customer.create(
                email=email,
                name=display_name,
                metadata={"user_id": str(user_id)},
            )
        except stripe.StripeError as exc:
            raise _provider_error(exc) from exc
        return str(customer["id"])

    # -- checkout / portal ---------------------------------------------------

    def create_checkout(
        self,
        *,
        customer_external_id: str,
        price_external_id: str,
        promotion_code_external_id: Optional[str],
        success_url: str,
        cancel_url: str,
        metadata: dict[str, str],
        allow_promotion_codes: bool,
    ) -> CheckoutSession:
        params: dict[str, Any] = {
            "mode": "subscription",
            "customer": customer_external_id,
            "line_items": [{"price": price_external_id, "quantity": 1}],
            "success_url": success_url,
            "cancel_url": cancel_url,
            "metadata": metadata,
            "subscription_data": {"metadata": metadata},
            "customer_update": {"address": "auto", "name": "auto"},
        }
        if promotion_code_external_id:
            params["discounts"] = [{"promotion_code": promotion_code_external_id}]
        elif allow_promotion_codes:
            params["allow_promotion_codes"] = True
        if settings.STRIPE_AUTOMATIC_TAX:
            params["automatic_tax"] = {"enabled": True}
            params["billing_address_collection"] = "required"
            params["tax_id_collection"] = {"enabled": True}
        try:
            session = stripe.checkout.Session.create(**params)
        except stripe.StripeError as exc:
            raise _provider_error(exc) from exc
        return CheckoutSession(url=str(session["url"]), external_id=str(session["id"]))

    def create_portal(self, *, customer_external_id: str, return_url: str) -> str:
        try:
            session = stripe.billing_portal.Session.create(
                customer=customer_external_id,
                return_url=return_url,
            )
        except stripe.StripeError as exc:
            raise _provider_error(exc) from exc
        return str(session["url"])

    # -- catalogue -----------------------------------------------------------

    def sync_plan_price(
        self,
        *,
        plan_code: str,
        plan_name: str,
        interval: str,
        amount_cents: int,
        currency: str,
        existing_product_id: Optional[str],
        existing_price_id: Optional[str],
    ) -> tuple[str, str]:
        recurring = _RECURRING.get(interval)
        if recurring is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Periodo non supportato: {interval}")
        try:
            product_id = existing_product_id
            if product_id:
                product = stripe.Product.retrieve(product_id)
                if product.get("name") != plan_name or not product.get("active", True):
                    stripe.Product.modify(product_id, name=plan_name, active=True)
            else:
                product = stripe.Product.create(name=plan_name, metadata={"plan_code": plan_code})
                product_id = str(product["id"])

            if existing_price_id:
                price = stripe.Price.retrieve(existing_price_id)
                price_recurring = price.get("recurring") or {}
                same = (
                    int(price.get("unit_amount") or -1) == int(amount_cents)
                    and str(price.get("currency", "")).lower() == currency.lower()
                    and price_recurring.get("interval") == recurring["interval"]
                    and int(price_recurring.get("interval_count") or 0) == recurring["interval_count"]
                    and price.get("product") == product_id
                )
                if same:
                    if not price.get("active", True):
                        stripe.Price.modify(existing_price_id, active=True)
                    return product_id, existing_price_id
                # Prices are immutable: archive the old one, create the new one.
                stripe.Price.modify(existing_price_id, active=False)

            price = stripe.Price.create(
                product=product_id,
                unit_amount=int(amount_cents),
                currency=currency.lower(),
                recurring=recurring,
                metadata={"plan_code": plan_code, "interval": interval},
            )
        except stripe.StripeError as exc:
            raise _provider_error(exc) from exc
        return product_id, str(price["id"])

    def sync_coupon(
        self,
        *,
        code: str,
        kind: str,
        value: int,
        currency: Optional[str],
        duration: str,
        duration_months: Optional[int],
        max_redemptions: Optional[int],
        valid_until: Optional[datetime],
        product_external_ids: Optional[list[str]],
    ) -> dict[str, str]:
        params: dict[str, Any] = {"name": code, "duration": duration, "metadata": {"code": code}}
        if kind == "percent":
            params["percent_off"] = value
        else:
            params["amount_off"] = int(value) * 100
            params["currency"] = (currency or "EUR").lower()
        if duration == "repeating":
            params["duration_in_months"] = int(duration_months or 1)
        if max_redemptions:
            params["max_redemptions"] = int(max_redemptions)
        if valid_until:
            params["redeem_by"] = int(valid_until.timestamp())
        if product_external_ids:
            params["applies_to"] = {"products": product_external_ids}
        try:
            coupon = stripe.Coupon.create(**params)
            promotion = stripe.PromotionCode.create(
                coupon=str(coupon["id"]),
                code=code,
                **({"max_redemptions": int(max_redemptions)} if max_redemptions else {}),
                **({"expires_at": int(valid_until.timestamp())} if valid_until else {}),
            )
        except stripe.StripeError as exc:
            raise _provider_error(exc) from exc
        return {"coupon": str(coupon["id"]), "promotion_code": str(promotion["id"])}

    def deactivate_coupon(self, *, promotion_code_external_id: str) -> None:
        try:
            stripe.PromotionCode.modify(promotion_code_external_id, active=False)
        except stripe.StripeError as exc:
            logger.warning("Stripe promotion code deactivation failed: %s", exc)

    # -- subscriptions -------------------------------------------------------

    def cancel_subscription(self, *, subscription_external_id: str, at_period_end: bool) -> None:
        try:
            if at_period_end:
                stripe.Subscription.modify(subscription_external_id, cancel_at_period_end=True)
            else:
                stripe.Subscription.cancel(subscription_external_id)
        except stripe.StripeError as exc:
            raise _provider_error(exc) from exc

    def fetch_subscription(self, *, subscription_external_id: str) -> ProviderSubscription:
        try:
            obj = stripe.Subscription.retrieve(subscription_external_id)
        except stripe.StripeError as exc:
            raise _provider_error(exc) from exc
        return map_subscription(obj)

    # -- webhooks ------------------------------------------------------------

    def parse_webhook(self, *, body: bytes, headers: dict[str, str]) -> list[BillingEvent]:
        secret = settings.STRIPE_WEBHOOK_SECRET
        if not secret:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="STRIPE_WEBHOOK_SECRET not configured")
        signature = headers.get("stripe-signature") or headers.get("Stripe-Signature") or ""
        try:
            event = stripe.Webhook.construct_event(body, signature, secret)
        except (ValueError, stripe.SignatureVerificationError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid Stripe webhook: {exc}") from exc
        return map_event(event)
