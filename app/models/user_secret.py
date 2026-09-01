from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column, Text, UniqueConstraint
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UserSecret(SQLModel, table=True):
    """Secret di piattaforma per-utente (es. OPENROUTER_API_KEY).

    Meccanismo cross: gestito qui (UI Settings), consumato dai runtime — oggi
    dagli Studi (env dei run schedulati, ew_studio.get_secret nel Lab via
    studio-svc, che legge questa stessa tabella con la chiave condivisa
    SECRETS_ENCRYPTION_KEY). Valore cifrato at-rest (Fernet), mai riesposto
    in chiaro dalle API di gestione.
    """

    __tablename__ = "user_secret"
    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_user_secret_user_name"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True, foreign_key="user.id")
    name: str
    value_encrypted: str = Field(sa_column=Column(Text, nullable=False))
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
