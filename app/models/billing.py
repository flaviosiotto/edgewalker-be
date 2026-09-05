"""Subscription plans, limits and AI-credit accounting (migration 052).

The database is the source of truth for plans, prices, coupons, trials and
the subscription state. The payment provider (Stripe, phase 3) sits behind
``app.services.billing.provider.BillingProvider`` and its identifiers live
ONLY in :class:`BillingExternalRef` — never as columns of the domain tables —
so switching provider is a new adapter plus new reference rows.

Limits are a JSONB on the plan whose keys are the typed registry in
``app.services.limits`` (``null`` = unlimited). Counters and concurrency are
computed live from the existing tables; only AI credits have their own
ledger because they are a consumption, not a state.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

from sqlalchemy import (
    ARRAY,
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PriceInterval(str, Enum):
    MONTH = "month"
    QUARTER = "quarter"
    SEMESTER = "semester"
    YEAR = "year"

    @property
    def months(self) -> int:
        return {"month": 1, "quarter": 3, "semester": 6, "year": 12}[self.value]


class SubscriptionStatus(str, Enum):
    TRIALING = "trialing"
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    EXPIRED = "expired"
    FREE = "free"
    MANUAL = "manual"

    @classmethod
    def current_values(cls) -> frozenset[str]:
        """Statuses that make a subscription the user's *current* one (the
        partial unique index ``uq_subscription_current_user`` mirrors this)."""
        return frozenset(
            {cls.TRIALING.value, cls.ACTIVE.value, cls.PAST_DUE.value, cls.FREE.value, cls.MANUAL.value}
        )


class BillingProviderKind(str, Enum):
    NONE = "none"
    MANUAL = "manual"
    STRIPE = "stripe"


class Plan(SQLModel, table=True):
    __tablename__ = "plan"
    __table_args__ = (
        Index("uq_plan_default", "is_default", unique=True, postgresql_where=text("is_default")),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    code: str = Field(sa_column=Column(String(40), nullable=False, unique=True))
    name: str = Field(sa_column=Column(String(120), nullable=False))
    description: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    is_active: bool = Field(default=True, sa_column=Column(Boolean, nullable=False, server_default=text("TRUE")))
    is_public: bool = Field(default=True, sa_column=Column(Boolean, nullable=False, server_default=text("TRUE")))
    is_default: bool = Field(default=False, sa_column=Column(Boolean, nullable=False, server_default=text("FALSE")))
    sort_order: int = Field(default=0, sa_column=Column(Integer, nullable=False, server_default="0"))
    trial_days: int = Field(default=0, sa_column=Column(Integer, nullable=False, server_default="0"))
    limits: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    )
    created_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))
    updated_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))


class PlanPrice(SQLModel, table=True):
    __tablename__ = "plan_price"
    __table_args__ = (UniqueConstraint("plan_id", "interval", name="uq_plan_price_plan_interval"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    plan_id: int = Field(sa_column=Column(Integer, ForeignKey("plan.id", ondelete="CASCADE"), nullable=False))
    interval: str = Field(sa_column=Column(String(16), nullable=False))
    amount_cents: int = Field(sa_column=Column(Integer, nullable=False))
    currency: str = Field(default="EUR", sa_column=Column(String(3), nullable=False, server_default="EUR"))
    is_active: bool = Field(default=True, sa_column=Column(Boolean, nullable=False, server_default=text("TRUE")))
    created_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))
    updated_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))


class BillingExternalRef(SQLModel, table=True):
    """Provider-side identifier of a local entity (customer, plan_price,
    coupon, subscription). The only place provider ids are stored."""

    __tablename__ = "billing_external_ref"
    __table_args__ = (
        UniqueConstraint("entity_type", "entity_id", "provider", name="uq_billing_external_ref_entity"),
        UniqueConstraint("provider", "external_id", name="uq_billing_external_ref_external"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    entity_type: str = Field(sa_column=Column(String(32), nullable=False))
    entity_id: int = Field(sa_column=Column(Integer, nullable=False))
    provider: str = Field(sa_column=Column(String(32), nullable=False))
    external_id: str = Field(sa_column=Column(String(255), nullable=False))
    created_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))


class Coupon(SQLModel, table=True):
    __tablename__ = "coupon"

    id: Optional[int] = Field(default=None, primary_key=True)
    code: str = Field(sa_column=Column(String(40), nullable=False, unique=True))
    kind: str = Field(sa_column=Column(String(16), nullable=False))  # percent | fixed
    value: int = Field(sa_column=Column(Integer, nullable=False))
    currency: Optional[str] = Field(default=None, sa_column=Column(String(3), nullable=True))
    duration: str = Field(default="once", sa_column=Column(String(16), nullable=False, server_default="once"))
    duration_months: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))
    applies_to_plan_ids: Optional[list[int]] = Field(default=None, sa_column=Column(ARRAY(Integer), nullable=True))
    max_redemptions: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))
    redeemed_count: int = Field(default=0, sa_column=Column(Integer, nullable=False, server_default="0"))
    valid_from: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True), nullable=True))
    valid_until: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True), nullable=True))
    revoked_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True), nullable=True))
    note: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    created_by: Optional[int] = Field(
        default=None, sa_column=Column(Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True)
    )
    created_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))


class TrialGrant(SQLModel, table=True):
    """One trial per plan per person: keyed by user AND by (hashed) email so a
    deleted-and-recreated account does not get a second trial."""

    __tablename__ = "trial_grant"
    __table_args__ = (
        UniqueConstraint("user_id", "plan_id", name="uq_trial_grant_user_plan"),
        UniqueConstraint("email_hash", "plan_id", name="uq_trial_grant_email_plan"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(sa_column=Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False))
    plan_id: int = Field(sa_column=Column(Integer, ForeignKey("plan.id", ondelete="CASCADE"), nullable=False))
    email_hash: str = Field(sa_column=Column(String(64), nullable=False))
    granted_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))


class Subscription(SQLModel, table=True):
    __tablename__ = "subscription"
    __table_args__ = (
        Index(
            "uq_subscription_current_user",
            "user_id",
            unique=True,
            postgresql_where=text("status IN ('trialing', 'active', 'past_due', 'free', 'manual')"),
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(
        sa_column=Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    plan_id: int = Field(sa_column=Column(Integer, ForeignKey("plan.id", ondelete="RESTRICT"), nullable=False))
    plan_price_id: Optional[int] = Field(
        default=None, sa_column=Column(Integer, ForeignKey("plan_price.id", ondelete="SET NULL"), nullable=True)
    )
    status: str = Field(sa_column=Column(String(16), nullable=False, index=True))
    interval: Optional[str] = Field(default=None, sa_column=Column(String(16), nullable=True))
    current_period_start: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True), nullable=True))
    current_period_end: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True), nullable=True))
    trial_end: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True), nullable=True))
    cancel_at_period_end: bool = Field(
        default=False, sa_column=Column(Boolean, nullable=False, server_default=text("FALSE"))
    )
    # Manual (admin) assignments: hard end date, NULL = until revoked.
    ends_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True), nullable=True))
    provider: str = Field(default="none", sa_column=Column(String(32), nullable=False, server_default="none"))
    coupon_id: Optional[int] = Field(
        default=None, sa_column=Column(Integer, ForeignKey("coupon.id", ondelete="SET NULL"), nullable=True)
    )
    # T-3 "your plan ends soon" email, sent once per subscription.
    ending_notice_sent_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    created_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))
    updated_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))

    @property
    def effective_end(self) -> Optional[datetime]:
        """When this subscription stops on its own, if ever."""
        if self.status == SubscriptionStatus.TRIALING.value:
            return self.trial_end
        if self.status == SubscriptionStatus.MANUAL.value:
            return self.ends_at
        if self.cancel_at_period_end:
            return self.current_period_end
        return None


class CouponRedemption(SQLModel, table=True):
    __tablename__ = "coupon_redemption"

    id: Optional[int] = Field(default=None, primary_key=True)
    coupon_id: int = Field(
        sa_column=Column(Integer, ForeignKey("coupon.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    user_id: int = Field(sa_column=Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False))
    subscription_id: Optional[int] = Field(
        default=None, sa_column=Column(Integer, ForeignKey("subscription.id", ondelete="SET NULL"), nullable=True)
    )
    redeemed_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))


class SubscriptionEvent(SQLModel, table=True):
    """Audit trail of every subscription change (admin actions, sweeper
    transitions, provider webhooks). ``provider_event_id`` deduplicates
    webhook deliveries."""

    __tablename__ = "subscription_event"
    __table_args__ = (
        UniqueConstraint("provider", "provider_event_id", name="uq_subscription_event_provider"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    subscription_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("subscription.id", ondelete="SET NULL"), nullable=True, index=True),
    )
    user_id: int = Field(
        sa_column=Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    type: str = Field(sa_column=Column(String(48), nullable=False))
    payload: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSONB, nullable=True))
    provider: Optional[str] = Field(default=None, sa_column=Column(String(32), nullable=True))
    provider_event_id: Optional[str] = Field(default=None, sa_column=Column(String(255), nullable=True))
    actor_user_id: Optional[int] = Field(
        default=None, sa_column=Column(Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True)
    )
    created_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))


class AiModelRate(SQLModel, table=True):
    """Credits per 1k tokens by model. ``model_pattern`` is an fnmatch glob
    (``openai/gpt-4*``); the row ``*`` is the default."""

    __tablename__ = "ai_model_rate"

    id: Optional[int] = Field(default=None, primary_key=True)
    model_pattern: str = Field(sa_column=Column(String(120), nullable=False, unique=True))
    input_per_1k: Decimal = Field(default=Decimal("1.0"), sa_column=Column(Numeric(8, 3), nullable=False))
    output_per_1k: Decimal = Field(default=Decimal("1.0"), sa_column=Column(Numeric(8, 3), nullable=False))
    is_active: bool = Field(default=True, sa_column=Column(Boolean, nullable=False, server_default=text("TRUE")))
    updated_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))


class AiCreditPeriod(SQLModel, table=True):
    """Running counter of one user's monthly credit window. ``granted`` is
    NULL for unlimited plans (usage is still recorded)."""

    __tablename__ = "ai_credit_period"

    user_id: Optional[int] = Field(
        default=None, sa_column=Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), primary_key=True)
    )
    period_key: Optional[date] = Field(default=None, sa_column=Column(Date, primary_key=True))
    period_end: date = Field(sa_column=Column(Date, nullable=False))
    granted: Optional[Decimal] = Field(default=None, sa_column=Column(Numeric(12, 3), nullable=True))
    used: Decimal = Field(default=Decimal("0"), sa_column=Column(Numeric(12, 3), nullable=False, server_default="0"))
    low_notified_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True), nullable=True))
    exhausted_notified_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )


class AiCreditLedger(SQLModel, table=True):
    __tablename__ = "ai_credit_ledger"
    __table_args__ = (
        Index("ix_ai_credit_ledger_user_period", "user_id", "period_key"),
        Index(
            "uq_ai_credit_ledger_turn",
            "correlation_id",
            "session_id",
            unique=True,
            postgresql_where=text("correlation_id IS NOT NULL"),
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(sa_column=Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False))
    period_key: date = Field(sa_column=Column(Date, nullable=False))
    credits: Decimal = Field(sa_column=Column(Numeric(12, 3), nullable=False))
    reason: str = Field(sa_column=Column(String(32), nullable=False))
    model: Optional[str] = Field(default=None, sa_column=Column(String(120), nullable=True))
    tokens_input: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))
    tokens_output: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))
    correlation_id: Optional[str] = Field(default=None, sa_column=Column(String(100), nullable=True))
    session_id: Optional[str] = Field(default=None, sa_column=Column(String(100), nullable=True))
    estimated: bool = Field(default=False, sa_column=Column(Boolean, nullable=False, server_default=text("FALSE")))
    actor_user_id: Optional[int] = Field(
        default=None, sa_column=Column(Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True)
    )
    created_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))


#: Same statement as migration 052; executed at startup so a dev database
#: bootstrapped by ``create_all`` (which knows nothing about views) matches
#: prod. ``CREATE OR REPLACE`` keeps it idempotent.
USER_EFFECTIVE_LIMITS_VIEW_SQL = """
CREATE OR REPLACE VIEW user_effective_limits AS
SELECT
    u.id AS user_id,
    p.id AS plan_id,
    p.code AS plan_code,
    p.limits AS limits,
    s.id AS subscription_id,
    s.status AS status,
    s.current_period_end AS current_period_end
FROM "user" u
LEFT JOIN subscription s
    ON s.user_id = u.id
   AND s.status IN ('trialing', 'active', 'past_due', 'free', 'manual')
LEFT JOIN plan dp ON dp.is_default
JOIN plan p ON p.id = COALESCE(s.plan_id, dp.id)
"""
