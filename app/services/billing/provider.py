"""Payment-provider abstraction.

The domain (plans, coupons, trials, subscription state) lives in our
database; a provider only collects money and tells us "paid / not paid /
canceled". Everything provider-specific stays inside an adapter implementing
:class:`BillingProvider`; the service layer speaks the normalized
:class:`BillingEvent` vocabulary only. Provider identifiers of local
entities are stored in ``billing_external_ref`` (see
``app.models.billing.BillingExternalRef``), never on the domain tables.

Adapters: :class:`NullProvider` (no payments: trials, free and manual plans
work end to end) and ``stripe_provider.StripeProvider`` (phase 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Protocol

from fastapi import HTTPException, status


class BillingEventType(str, Enum):
    #: hosted checkout finished: a provider subscription now exists for a user
    CHECKOUT_COMPLETED = "checkout_completed"
    #: first invoice of a subscription paid
    SUBSCRIPTION_ACTIVATED = "subscription_activated"
    #: a renewal invoice paid
    SUBSCRIPTION_RENEWED = "subscription_renewed"
    #: the provider changed status / period / price / cancel flag
    SUBSCRIPTION_UPDATED = "subscription_updated"
    PAYMENT_FAILED = "payment_failed"
    #: cancellation scheduled at period end
    SUBSCRIPTION_CANCELED = "subscription_canceled"
    #: the provider subscription no longer exists
    SUBSCRIPTION_ENDED = "subscription_ended"
    TRIAL_WILL_END = "trial_will_end"


class BillingSubscriptionStatus(str, Enum):
    """Provider status normalized to our vocabulary."""

    TRIALING = "trialing"
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    EXPIRED = "expired"


@dataclass
class BillingEvent:
    """Provider-agnostic event handed to ``billing_service.apply_event``."""

    type: BillingEventType
    provider: str
    event_id: str
    #: provider id of the subscription (resolved through billing_external_ref)
    subscription_external_id: Optional[str] = None
    customer_external_id: Optional[str] = None
    price_external_id: Optional[str] = None
    status: Optional[str] = None
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    cancel_at_period_end: Optional[bool] = None
    #: metadata we attached at checkout (user_id, plan_id, plan_price_id, coupon_id)
    metadata: dict[str, str] = field(default_factory=dict)
    #: e.g. invoice billing reason ("subscription_create" / "subscription_cycle")
    reason: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckoutSession:
    url: str
    external_id: str


@dataclass
class ProviderSubscription:
    external_id: str
    status: str  # BillingSubscriptionStatus value
    period_start: Optional[datetime]
    period_end: Optional[datetime]
    cancel_at_period_end: bool
    price_external_id: Optional[str] = None
    customer_external_id: Optional[str] = None


class BillingProvider(Protocol):
    name: str

    def ensure_customer(self, *, user_id: int, email: str, display_name: str) -> str: ...

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
    ) -> CheckoutSession: ...

    def create_portal(self, *, customer_external_id: str, return_url: str) -> str: ...

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
        """Returns ``(product_external_id, price_external_id)``; a changed
        amount yields a new price (provider prices are immutable)."""
        ...

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
        """Returns ``{"coupon": <id>, "promotion_code": <id>}``."""
        ...

    def deactivate_coupon(self, *, promotion_code_external_id: str) -> None: ...

    def cancel_subscription(self, *, subscription_external_id: str, at_period_end: bool) -> None: ...

    def fetch_subscription(self, *, subscription_external_id: str) -> ProviderSubscription: ...

    def parse_webhook(self, *, body: bytes, headers: dict[str, str]) -> list[BillingEvent]: ...


def _payments_disabled() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={
            "code": "billing_disabled",
            "message": "I pagamenti non sono abilitati su questa installazione.",
        },
    )


class NullProvider:
    """No payment provider configured: paid plans cannot be bought, everything
    else (default plan, trials, admin assignments, limits) works."""

    name = "none"

    def ensure_customer(self, **kwargs: Any) -> str:
        raise _payments_disabled()

    def create_checkout(self, **kwargs: Any) -> CheckoutSession:
        raise _payments_disabled()

    def create_portal(self, **kwargs: Any) -> str:
        raise _payments_disabled()

    def sync_plan_price(self, **kwargs: Any) -> tuple[str, str]:
        raise _payments_disabled()

    def sync_coupon(self, **kwargs: Any) -> dict[str, str]:
        raise _payments_disabled()

    def deactivate_coupon(self, **kwargs: Any) -> None:
        return None

    def cancel_subscription(self, **kwargs: Any) -> None:
        return None

    def fetch_subscription(self, **kwargs: Any) -> ProviderSubscription:
        raise _payments_disabled()

    def parse_webhook(self, **kwargs: Any) -> list[BillingEvent]:
        raise _payments_disabled()


_provider: BillingProvider | None = None


def get_billing_provider() -> BillingProvider:
    """Adapter selected by ``BILLING_ENABLED`` / ``BILLING_PROVIDER``."""
    global _provider
    if _provider is not None:
        return _provider
    from app.core.config import settings

    if not settings.BILLING_ENABLED:
        _provider = NullProvider()
        return _provider
    name = (settings.BILLING_PROVIDER or "").strip().lower()
    if name == "stripe":
        from app.services.billing.stripe_provider import StripeProvider

        _provider = StripeProvider()
        return _provider
    raise RuntimeError(f"Unknown BILLING_PROVIDER: {settings.BILLING_PROVIDER!r}")


def billing_enabled() -> bool:
    from app.core.config import settings

    return bool(settings.BILLING_ENABLED)
