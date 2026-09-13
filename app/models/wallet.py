"""Platform credit (prepaid wallet) — migration 059, docs/credito-piattaforma-studio.md.

A per-user balance in integer cents with a ledger of movements. The database
is the only source of truth: the payment provider credits it through a
webhook (``wallet_topup``) and is never read for the balance. When the AI
credits of the period are exhausted, the overage of a turn is charged here at
the admin-defined price per credit.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional

from sqlalchemy import BigInteger, Boolean, Column, Date, DateTime, ForeignKey, Index, Integer, Numeric, SmallInteger, String, Text, text
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class WalletLedgerKind(str, Enum):
    TOPUP = "topup"
    ADMIN_ADJUST = "admin_adjust"
    AI_OVERAGE = "ai_overage"
    REFUND = "refund"


class TopupStatus(str, Enum):
    PENDING = "pending"
    PAID = "paid"
    CANCELED = "canceled"


class PlatformCreditSettings(SQLModel, table=True):
    """Single row (id = 1)."""

    __tablename__ = "platform_credit_settings"

    id: Optional[int] = Field(default=1, sa_column=Column(SmallInteger, primary_key=True))
    enabled: bool = Field(default=True, sa_column=Column(Boolean, nullable=False, server_default=text("TRUE")))
    currency: str = Field(default="EUR", sa_column=Column(String(3), nullable=False, server_default="EUR"))
    #: price of one AI credit in cents (1.0 = 1 cent = 100 credits per EUR)
    price_per_ai_credit_cents: Decimal = Field(
        default=Decimal("1.0"), sa_column=Column(Numeric(10, 4), nullable=False, server_default="1.0")
    )
    low_balance_cents: int = Field(default=100, sa_column=Column(Integer, nullable=False, server_default="100"))
    min_topup_cents: int = Field(default=500, sa_column=Column(Integer, nullable=False, server_default="500"))
    #: display-only rate for the cost report (EUR per 1 USD); never used to charge
    display_fx_eur_per_usd: Optional[Decimal] = Field(default=None, sa_column=Column(Numeric(10, 6), nullable=True))
    updated_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))
    updated_by: Optional[int] = Field(
        default=None, sa_column=Column(Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True)
    )


class CreditPack(SQLModel, table=True):
    """A top-up option: the user pays ``amount_cents`` and receives
    ``credit_cents`` (price + bonus). Mirrored on the payment provider as a
    one-off product/price through ``billing_external_ref``."""

    __tablename__ = "credit_pack"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(sa_column=Column(String(80), nullable=False))
    amount_cents: int = Field(sa_column=Column(Integer, nullable=False))
    credit_cents: int = Field(sa_column=Column(Integer, nullable=False))
    currency: str = Field(default="EUR", sa_column=Column(String(3), nullable=False, server_default="EUR"))
    is_active: bool = Field(default=True, sa_column=Column(Boolean, nullable=False, server_default=text("TRUE")))
    sort_order: int = Field(default=0, sa_column=Column(Integer, nullable=False, server_default="0"))
    created_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))
    updated_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))


class UserWallet(SQLModel, table=True):
    __tablename__ = "user_wallet"

    user_id: Optional[int] = Field(
        default=None, sa_column=Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), primary_key=True)
    )
    balance_cents: int = Field(default=0, sa_column=Column(BigInteger, nullable=False, server_default="0"))
    currency: str = Field(default="EUR", sa_column=Column(String(3), nullable=False, server_default="EUR"))
    auto_use_for_ai: bool = Field(default=True, sa_column=Column(Boolean, nullable=False, server_default=text("TRUE")))
    low_notified_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True), nullable=True))
    exhausted_notified_period: Optional[date] = Field(default=None, sa_column=Column(Date, nullable=True))
    updated_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))


class WalletTopup(SQLModel, table=True):
    """A top-up order: ``pending`` at checkout, ``paid`` when the provider
    confirms (idempotent on ``checkout_external_id``)."""

    __tablename__ = "wallet_topup"
    __table_args__ = (Index("ix_wallet_topup_user", "user_id", "created_at"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(sa_column=Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False))
    pack_id: Optional[int] = Field(
        default=None, sa_column=Column(Integer, ForeignKey("credit_pack.id", ondelete="SET NULL"), nullable=True)
    )
    amount_cents: int = Field(sa_column=Column(Integer, nullable=False))
    credit_cents: int = Field(sa_column=Column(Integer, nullable=False))
    currency: str = Field(default="EUR", sa_column=Column(String(3), nullable=False, server_default="EUR"))
    status: str = Field(default=TopupStatus.PENDING.value, sa_column=Column(String(16), nullable=False, server_default="pending"))
    provider: str = Field(sa_column=Column(String(20), nullable=False))
    checkout_external_id: str = Field(sa_column=Column(String(120), nullable=False, unique=True))
    payment_external_id: Optional[str] = Field(default=None, sa_column=Column(String(120), nullable=True))
    created_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))
    paid_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True), nullable=True))


class WalletLedger(SQLModel, table=True):
    __tablename__ = "wallet_ledger"
    __table_args__ = (
        Index("ix_wallet_ledger_user_created", "user_id", "created_at"),
        Index("uq_wallet_ledger_ai_ledger", "ai_ledger_id", unique=True, postgresql_where=text("ai_ledger_id IS NOT NULL")),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(sa_column=Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False))
    amount_cents: int = Field(sa_column=Column(Integer, nullable=False))
    balance_after_cents: int = Field(sa_column=Column(BigInteger, nullable=False))
    kind: str = Field(sa_column=Column(String(20), nullable=False))
    ai_credits: Optional[Decimal] = Field(default=None, sa_column=Column(Numeric(12, 3), nullable=True))
    ai_ledger_id: Optional[int] = Field(
        default=None, sa_column=Column(Integer, ForeignKey("ai_credit_ledger.id", ondelete="SET NULL"), nullable=True)
    )
    topup_id: Optional[int] = Field(
        default=None, sa_column=Column(Integer, ForeignKey("wallet_topup.id", ondelete="SET NULL"), nullable=True)
    )
    note: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    actor_user_id: Optional[int] = Field(
        default=None, sa_column=Column(Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True)
    )
    created_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))
