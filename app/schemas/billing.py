from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal, Optional

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
    wallet: Optional["WalletRead"] = None


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
    # Informational split (agent-svc): reasoning is already inside
    # tokens_output, cached is a subset of tokens_input.
    tokens_reasoning: Optional[int] = Field(default=None, ge=0)
    tokens_cached: Optional[int] = Field(default=None, ge=0)
    prompt_chars: Optional[int] = Field(default=None, ge=0)
    response_chars: Optional[int] = Field(default=None, ge=0)
    reason: str = Field(default="agent_turn", max_length=32)
    # Real provider cost of the turn (data only, never used to charge).
    provider: Optional[str] = Field(default=None, max_length=40)
    cost: Optional[Decimal] = Field(default=None, ge=0)
    cost_currency: Optional[str] = Field(default=None, min_length=3, max_length=3)


class AiUsageReportResponse(BaseModel):
    recorded: bool
    credits: Optional[Decimal] = None
    estimated: Optional[bool] = None
    used: Decimal
    granted: Optional[Decimal] = None
    #: cents charged to the platform credit for this turn (0 = all from the plan)
    wallet_cents: Optional[int] = None
    #: where the next turn is paid from: plan | wallet | none
    source: str = "plan"
    wallet_balance_cents: int = 0


class AiBudgetRead(BaseModel):
    allowed: bool
    granted: Optional[Decimal] = None
    used: Decimal
    remaining: Optional[Decimal] = None
    period_start: date
    period_end: date
    source: str = "plan"
    wallet_enabled: bool = False
    wallet_usable: bool = False
    wallet_balance_cents: int = 0


class AiModelRateRead(BaseModel):
    id: int
    model_pattern: str
    input_per_1k: Decimal
    output_per_1k: Decimal
    is_active: bool
    updated_at: datetime


ReasoningTier = Literal["quick", "balanced", "deep"]
ReasoningEffort = Literal["low", "medium", "high"]


class AiModelPolicyRead(BaseModel):
    id: int
    plan_code: str
    tier: ReasoningTier
    provider: str
    model: str
    reasoning_effort: Optional[ReasoningEffort] = None
    max_iterations: int
    history_window: int
    is_active: bool
    notes: Optional[str] = None
    updated_at: datetime


class AiModelPolicyUpsert(BaseModel):
    """Keyed on (plan_code, tier); ``plan_code`` ``*`` = every plan."""

    plan_code: str = Field(default="*", min_length=1, max_length=40)
    tier: ReasoningTier
    provider: str = Field(default="openrouter", min_length=1, max_length=40)
    model: str = Field(min_length=1, max_length=120)
    reasoning_effort: Optional[ReasoningEffort] = None
    max_iterations: int = Field(default=15, ge=1, le=60)
    history_window: int = Field(default=12, ge=0, le=60)
    is_active: bool = True
    notes: Optional[str] = Field(default=None, max_length=500)


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
    wallet_balance_cents: int = 0


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


# ---------------------------------------------------------------------------
# Platform credit (wallet) — docs/credito-piattaforma-studio.md
# ---------------------------------------------------------------------------


class CreditPackRead(BaseModel):
    id: int
    name: str
    amount_cents: int
    credit_cents: int
    currency: str
    is_active: bool
    sort_order: int


class CreditPackInput(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    amount_cents: int = Field(gt=0)
    credit_cents: int = Field(gt=0)
    currency: str = Field(default="EUR", min_length=3, max_length=3)
    is_active: bool = True
    sort_order: int = 0


class WalletRead(BaseModel):
    enabled: bool
    currency: str
    balance_cents: int
    auto_use_for_ai: bool
    price_per_ai_credit_cents: Decimal
    low_balance_cents: int
    min_topup_cents: int
    packs: list[CreditPackRead] = Field(default_factory=list)


class WalletUpdateRequest(BaseModel):
    auto_use_for_ai: bool


class WalletLedgerRead(BaseModel):
    id: int
    user_id: int
    amount_cents: int
    balance_after_cents: int
    kind: str
    ai_credits: Optional[Decimal] = None
    ai_ledger_id: Optional[int] = None
    topup_id: Optional[int] = None
    note: Optional[str] = None
    actor_user_id: Optional[int] = None
    created_at: datetime
    # admin listing only
    email: Optional[str] = None


class WalletLedgerPage(BaseModel):
    items: list[WalletLedgerRead]
    total: int


class TopupRequest(BaseModel):
    pack_id: int


class PlatformCreditSettingsRead(BaseModel):
    enabled: bool
    currency: str
    price_per_ai_credit_cents: Decimal
    low_balance_cents: int
    min_topup_cents: int
    display_fx_eur_per_usd: Optional[Decimal] = None
    updated_at: datetime


class PlatformCreditSettingsUpdate(BaseModel):
    enabled: Optional[bool] = None
    price_per_ai_credit_cents: Optional[Decimal] = Field(default=None, ge=0)
    low_balance_cents: Optional[int] = Field(default=None, ge=0)
    min_topup_cents: Optional[int] = Field(default=None, ge=0)
    display_fx_eur_per_usd: Optional[Decimal] = Field(default=None, gt=0)
    clear_display_fx: bool = False


class AdminWalletAdjustRequest(BaseModel):
    amount_cents: int = Field(description="positivo = accredito, negativo = addebito")
    note: Optional[str] = Field(default=None, max_length=500)
    notify: bool = True
    kind: str = Field(default="admin_adjust", pattern="^(admin_adjust|refund)$")


class AdminUserWalletRead(BaseModel):
    wallet: WalletRead
    ledger: list[WalletLedgerRead]


class WalletSummaryRead(BaseModel):
    currency: str
    days: int
    outstanding_cents: int
    wallets_with_balance: int
    users_low_balance: int
    topups_cents: int
    topups_count: int
    overage_cents: int
    overage_count: int
    overage_credits: float
    admin_adjust_cents: int
    admin_adjust_count: int


class PackSyncRow(BaseModel):
    pack_id: int
    name: str
    amount_cents: int
    credit_cents: int
    currency: str
    product_external_id: Optional[str] = None
    price_external_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Provider token cost report
# ---------------------------------------------------------------------------


class AiCostRow(BaseModel):
    group: str
    calls: int
    calls_with_cost: int
    tokens_input: int
    tokens_output: int
    tokens_cached: int
    tokens_reasoning: int
    cost: Optional[Decimal] = None
    cost_currency: Optional[str] = None
    avg_cost_per_call: Optional[Decimal] = None
    cost_per_1k_tokens: Optional[Decimal] = None
    cached_share: Optional[float] = None
    credits: Decimal
    credits_per_call: Optional[Decimal] = None
    wallet_cents: int = 0


class AiCostReport(BaseModel):
    group_by: str
    since: datetime
    until: datetime
    rows: list[AiCostRow]
    total: AiCostRow
    display_fx_eur_per_usd: Optional[Decimal] = None
