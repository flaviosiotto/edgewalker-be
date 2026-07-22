from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlmodel import Field, SQLModel


class UserStatus(str, Enum):
    """Account lifecycle.

    ``is_active`` remains the enforcement point checked by the auth
    dependencies; this column records *why* an account is or is not usable.
    Keep the two in sync through ``apply_status`` rather than by hand.
    """

    PENDING_EMAIL = "pending_email"
    PENDING_APPROVAL = "pending_approval"
    ACTIVE = "active"
    REJECTED = "rejected"
    SUSPENDED = "suspended"


#: Statuses that grant access. Everything else must not be able to obtain a token.
ACTIVE_STATUSES = {UserStatus.ACTIVE.value}


class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    username: str = Field(index=True, unique=True)
    role: str = Field(
        default="user",
        sa_column=Column(String(32), nullable=False, server_default="user", index=True),
    )
    # NULL for accounts that only ever authenticate through an external
    # provider. Every password path must guard against it.
    hashed_password: Optional[str] = Field(default=None, nullable=True)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    first_name: Optional[str] = Field(
        default=None, sa_column=Column(String(120), nullable=True)
    )
    last_name: Optional[str] = Field(
        default=None, sa_column=Column(String(120), nullable=True)
    )
    status: str = Field(
        default=UserStatus.ACTIVE.value,
        sa_column=Column(
            String(32), nullable=False, server_default=UserStatus.ACTIVE.value, index=True
        ),
    )
    email_verified_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    approved_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    approved_by: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True
        ),
    )
    last_login_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )

    @property
    def display_name(self) -> str:
        full_name = " ".join(part for part in (self.first_name, self.last_name) if part)
        return full_name or self.username

    @property
    def has_usable_password(self) -> bool:
        return bool(self.hashed_password)


def apply_status(user: User, status: UserStatus) -> None:
    """Set the lifecycle status and keep ``is_active`` consistent with it.

    Every status transition must go through here. ``is_active`` is what the auth
    dependencies actually check, so writing ``status`` alone would let a rejected
    or unverified account keep working.
    """
    user.status = status.value
    user.is_active = status.value in ACTIVE_STATUSES
