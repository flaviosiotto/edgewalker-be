from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


class PersonalAccessToken(SQLModel, table=True):
    """Long-lived credential for machine clients (MCP server, scripts).

    The raw token is returned once at creation and never stored: lookup is by
    SHA-256 hex digest. ``token_prefix`` keeps the first characters of the raw
    token purely for display ("ewp_a1b2c3d4…"). Revocation is soft via
    ``revoked_at`` so a revoked token still shows up in the user's list.
    """

    __tablename__ = "personal_access_token"

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("user.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )
    name: str = Field(sa_column=Column(String(120), nullable=False))
    token_hash: str = Field(sa_column=Column(String(64), nullable=False, unique=True, index=True))
    token_prefix: str = Field(sa_column=Column(String(16), nullable=False))
    scopes: list = Field(
        default_factory=lambda: ["read"],
        sa_column=Column(JSONB, nullable=False),
    )
    expires_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    last_used_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    revoked_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
