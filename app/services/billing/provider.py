"""Payment-provider abstraction.

The domain (plans, coupons, trials, subscription state) lives in our
database; a provider only collects money and tells us "paid / not paid /
canceled". Everything provider-specific stays inside an adapter implementing
:class:`BillingProvider`; the service layer speaks the normalized
:class:`BillingEvent` vocabulary only. Provider identifiers of local
entities are stored in ``billing_external_ref`` (see
``app.models.billing.BillingExternalRef``), never on the domain tables.

Phase 1 ships the :class:`NullProvider` (no payments: trials, free and
manual plans work end to end). Phase 3 adds ``stripe_provider.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Protocol

from fastapi import HTTPException, status


class BillingEventType(str, Enum):
    SUBSCRIPTION_ACTIVATED = "subscription_activated"
    SUBSCRIPTION_RENEWED = "subscription_renewed"
    PAYMENT_FAILED = "payment_failed"
    SUBSCRIPTION_CANCELED = "subscription_canceled"  # cancel scheduled at period end
    SUBSCRIPTION_ENDED = "subscription_ended"
    TRIAL_WILL_END = "trial_will_end"


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
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    cancel_at_period_end: Optional[bool] = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckoutSession:
    url: str
    external_id: str


@dataclass
class ProviderSubscription:
    external_id: str
    status: str  # normalized: trialing | active | past_due | canceled | expired
    period_start: Optional[datetime]
    period_end: Optional[datetime]
    cancel_at_period_end: bool
    price_external_id: Optional[str] = None


class BillingProvider(Protocol):
    name: str

    def ensure_customer(self, *, user_id: int, email: str, display_name: str) -> str: ...

    def create_checkout(
        self,
        *,
        customer_external_id: str,
        price_external_id: str,
        coupon_external_id: Optional[str],
        success_url: str,
        cancel_url: str,
        metadata: dict[str, str],
    ) -> CheckoutSession: ...

    def create_portal(self, *, customer_external_id: str, return_url: str) -> str: ...

    def sync_plan_price(
        self, *, plan_code: str, plan_name: str, interval: str, amount_cents: int, currency: str,
        existing_external_id: Optional[str],
    ) -> str: ...

    def sync_coupon(self, *, code: str, kind: str, value: int, currency: Optional[str], duration: str,
                    duration_months: Optional[int], max_redemptions: Optional[int],
                    valid_until: Optional[datetime]) -> str: ...

    def deactivate_coupon(self, *, coupon_external_id: str) -> None: ...

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

    def ensure_customer(self, *, user_id: int, email: str, display_name: str) -> str:
        raise _payments_disabled()

    def create_checkout(self, **kwargs: Any) -> CheckoutSession:
        raise _payments_disabled()

    def create_portal(self, **kwargs: Any) -> str:
        raise _payments_disabled()

    def sync_plan_price(self, **kwargs: Any) -> str:
        raise _payments_disabled()

    def sync_coupon(self, **kwargs: Any) -> str:
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
        from app.services.billing.stripe_provider import StripeProvider  # phase 3

        _provider = StripeProvider()
        return _provider
    raise RuntimeError(f"Unknown BILLING_PROVIDER: {settings.BILLING_PROVIDER!r}")
