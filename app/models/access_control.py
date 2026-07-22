"""Registration-side tables: external identities, invite allowlist, 2FA.

These are separate from :mod:`app.models.user` because they are all optional
satellites of an account rather than part of it.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


class AuthProvider(str, Enum):
    """Supported external identity providers."""

    GOOGLE = "google"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UserIdentity(SQLModel, table=True):
    """An external login bound to a local account."""

    __tablename__ = "user_identity"
    __table_args__ = (
        UniqueConstraint("provider", "provider_subject", name="uq_user_identity_provider_subject"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(
        sa_column=Column(
            Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True
        )
    )
    provider: str = Field(sa_column=Column(String(32), nullable=False))
    # The provider's stable subject id. Identity lookup keys on this, never on
    # the email, which users can change at the provider.
    provider_subject: str = Field(sa_column=Column(String(255), nullable=False))
    email: Optional[str] = Field(default=None, sa_column=Column(String(320), nullable=True))
    raw_profile: Optional[dict[str, Any]] = Field(
        default=None, sa_column=Column(JSONB, nullable=True)
    )
    created_at: datetime = Field(
        default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )
    last_login_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )


class AccessAllowlist(SQLModel, table=True):
    """An email pre-authorised to register without administrator approval."""

    __tablename__ = "access_allowlist"

    id: Optional[int] = Field(default=None, primary_key=True)
    # Always stored lowercased so lookups can be exact; a DB CHECK enforces it.
    email: str = Field(sa_column=Column(String(320), nullable=False, unique=True))
    note: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    invited_by: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True),
    )
    created_at: datetime = Field(
        default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )
    consumed_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    consumed_by_user_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True),
    )


class EmailVerificationToken(SQLModel, table=True):
    """One-time token proving control of the registered address."""

    __tablename__ = "email_verification_token"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(
        sa_column=Column(
            Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True
        )
    )
    token_hash: str = Field(
        sa_column=Column(String(128), nullable=False, unique=True, index=True)
    )
    expires_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, index=True)
    )
    used_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    created_at: datetime = Field(
        default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )


class UserTotp(SQLModel, table=True):
    """TOTP enrolment. Unconfirmed rows must not gate login."""

    __tablename__ = "user_totp"

    user_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer, ForeignKey("user.id", ondelete="CASCADE"), primary_key=True
        ),
    )
    secret: str = Field(sa_column=Column(String(255), nullable=False))
    confirmed_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    created_at: datetime = Field(
        default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )


class UserRecoveryCode(SQLModel, table=True):
    """Single-use backup code for when the authenticator is unavailable."""

    __tablename__ = "user_recovery_code"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(
        sa_column=Column(
            Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True
        )
    )
    code_hash: str = Field(sa_column=Column(String(128), nullable=False))
    used_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    created_at: datetime = Field(
        default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )
