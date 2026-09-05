from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from app.models.billing import PriceInterval


# ---------------------------------------------------------------------------
# Plans (public + admin)
# ---------------------------------------------------------------------------


class PlanPriceRead(BaseModel):
    id: int
    interval: str
    amount_cents: int
    currency: str
    is_active: bool


class PlanRead(BaseModel):
    id: int
    code: str
    name: str
    description: Optional[str] = None
    is_active: bool
    is_public: bool
    is_default: bool
    sort_order: int
    trial_days: int
    limits: dict[str, Optional[int]]
    prices: list[PlanPriceRead] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class PublicPlanRead(BaseModel):
    """What the pricing page sees."""

    id: int
    code: str
    name: str
    description: Optional[str] = None
    is_default: bool
    trial_days: int
    trial_available: Optional[bool] = None  # only when the caller is authenticated
    limits: dict[str, Optional[int]]
    prices: list[PlanPriceRead] = Field(default_factory=list)


class PlanPriceUpsert(BaseModel):
    interval: PriceInterval
    amount_cents: int = Field(ge=0)
    currency: str = Field(default="EUR", min_length=3, max_length=3)
    is_active: bool = True

    @field_validator("currency")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.upper()


class PlanCreate(BaseModel):
    code: str = Field(min_length=1, max_length=40, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    name: str = Field(min_length=1, max_length=120)
    description: Optional[str] = None
    is_active: bool = True
    is_public: bool = True
    is_default: bool = False
    sort_order: int = 0
    trial_days: int = Field(default=0, ge=0, le=365)
    limits: dict[str, Any] = Field(default_factory=dict)
    prices: list[PlanPriceUpsert] = Field(default_factory=list)


class PlanUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    description: Optional[str] = None
    is_active: Optional[bool] = None
    is_public: Optional[bool] = None
    is_default: Optional[bool] = None
    sort_order: Optional[int] = None
    trial_days: Optional[int] = Field(default=None, ge=0, le=365)
    limits: Optional[dict[str, Any]] = None
    # Full replacement of the price list when provided.
    prices: Optional[list[PlanPriceUpsert]] = None


class LimitKeyRead(BaseModel):
    key: str
    label: str
    description: str
    kind: str
    enforced_by: str
    default: Optional[int] = None


# ---------------------------------------------------------------------------
# Subscription (user side)
# ---------------------------------------------------------------------------


class UsageItem(BaseModel):
    label: str
    kind: str
    max: Optional[float] = None
    current: Optional[float] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None


class SubscriptionEventRead(BaseModel):
    id: int
    type: str
    payload: Optional[dict[str, Any]] = None
    created_at: datetime


class SubscriptionRead(BaseModel):
    id: Optional[int] = None
    status: str
    provider: str
    plan: PlanRead
    interval: Optional[str] = None
    current_period_start: Optional[datetime] = None
    current_period_end: Optional[datetime] = None
    trial_end: Optional[datetime] = None
    cancel_at_period_end: bool = False
    ends_at: Optional[datetime] = None
    usage: dict[str, UsageItem]
    over_limit: list[str] = Field(default_factory=list)
    trial_available_plan_ids: list[int] = Field(default_factory=list)
    billing_enabled: bool = False
    events: list[SubscriptionEventRead] = Field(default_factory=list)


class TrialStartRequest(BaseModel):
    plan_id: int


# ---------------------------------------------------------------------------
# AI usage
# ---------------------------------------------------------------------------


class AiUsageReportRequest(BaseModel):
    """Usage of one agent turn. Either real token counts (from the n8n
    workflow) or character counts (estimate, from the runner/chat path)."""

    correlation_id: str = Field(min_length=1, max_length=100)
    session_id: Optional[str] = Field(default=None, max_length=100)
    model: Optional[str] = Field(default=None, max_length=120)
    tokens_input: Optional[int] = Field(default=None, ge=0)
    tokens_output: Optional[int] = Field(default=None, ge=0)
    prompt_chars: Optional[int] = Field(default=None, ge=0)
    response_chars: Optional[int] = Field(default=None, ge=0)
    reason: str = Field(default="agent_turn", max_length=32)


class AiUsageReportResponse(BaseModel):
    recorded: bool
    credits: Optional[Decimal] = None
    estimated: Optional[bool] = None
    used: Decimal
    granted: Optional[Decimal] = None


class AiBudgetRead(BaseModel):
    allowed: bool
    granted: Optional[Decimal] = None
    used: Decimal
    remaining: Optional[Decimal] = None
    period_start: date
    period_end: date


class AiModelRateRead(BaseModel):
    id: int
    model_pattern: str
    input_per_1k: Decimal
    output_per_1k: Decimal
    is_active: bool
    updated_at: datetime


class AiModelRateUpsert(BaseModel):
    model_pattern: str = Field(min_length=1, max_length=120)
    input_per_1k: Decimal = Field(ge=0)
    output_per_1k: Decimal = Field(ge=0)
    is_active: bool = True


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------


class AdminSubscriptionRow(BaseModel):
    user_id: int
    email: str
    username: str
    display_name: str
    role: str
    user_status: str
    subscription_id: Optional[int] = None
    plan_id: int
    plan_code: str
    plan_name: str
    status: str
    provider: str
    current_period_end: Optional[datetime] = None
    ends_at: Optional[datetime] = None
    trial_end: Optional[datetime] = None
    cancel_at_period_end: bool = False
    ai_credits_used: Optional[float] = None
    ai_credits_granted: Optional[float] = None
    counters: dict[str, int] = Field(default_factory=dict)
    over_limit: list[str] = Field(default_factory=list)


class AdminSubscriptionPage(BaseModel):
    items: list[AdminSubscriptionRow]
    total: int


class AdminAssignPlanRequest(BaseModel):
    plan_id: int
    ends_at: Optional[datetime] = None
    note: Optional[str] = Field(default=None, max_length=500)
    notify: bool = True


class AdminExtendRequest(BaseModel):
    ends_at: Optional[datetime] = None


class AdminGrantCreditsRequest(BaseModel):
    credits: Decimal = Field(gt=0)
    note: Optional[str] = Field(default=None, max_length=500)


class AdminUserSubscriptionDetail(BaseModel):
    subscription: SubscriptionRead
    events: list[SubscriptionEventRead]


class CouponRead(BaseModel):
    id: int
    code: str
    kind: str
    value: int
    currency: Optional[str] = None
    duration: str
    duration_months: Optional[int] = None
    applies_to_plan_ids: Optional[list[int]] = None
    max_redemptions: Optional[int] = None
    redeemed_count: int
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    note: Optional[str] = None
    created_at: datetime


class CouponCreate(BaseModel):
    code: str = Field(min_length=3, max_length=40, pattern=r"^[A-Za-z0-9_-]+$")
    kind: str = Field(pattern=r"^(percent|fixed)$")
    value: int = Field(ge=0)
    currency: Optional[str] = Field(default=None, min_length=3, max_length=3)
    duration: str = Field(default="once", pattern=r"^(once|repeating|forever)$")
    duration_months: Optional[int] = Field(default=None, ge=1, le=36)
    applies_to_plan_ids: Optional[list[int]] = None
    max_redemptions: Optional[int] = Field(default=None, ge=1)
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    note: Optional[str] = Field(default=None, max_length=500)

    @field_validator("code")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.upper()


# ---------------------------------------------------------------------------
# Payments (provider adapter)
# ---------------------------------------------------------------------------


class BillingConfigRead(BaseModel):
    enabled: bool
    provider: str
    automatic_tax: bool
    allow_promotion_codes: bool


class CheckoutRequest(BaseModel):
    plan_price_id: int
    coupon_code: Optional[str] = Field(default=None, max_length=40)


class CheckoutResponse(BaseModel):
    url: str


class PortalResponse(BaseModel):
    url: str


class CouponValidateRequest(BaseModel):
    code: str = Field(min_length=1, max_length=40)
    plan_price_id: int


class CouponValidateResponse(BaseModel):
    valid: bool
    code: Optional[str] = None
    description: Optional[str] = None
    discount_cents: Optional[int] = None
    final_cents: Optional[int] = None
    currency: Optional[str] = None
    message: Optional[str] = None


class CatalogSyncRow(BaseModel):
    plan_code: str
    plan_name: str
    interval: Optional[str] = None
    amount_cents: Optional[int] = None
    currency: Optional[str] = None
    product_external_id: Optional[str] = None
    price_external_id: Optional[str] = None
