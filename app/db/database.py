from contextlib import contextmanager
from typing import Generator

from sqlmodel import SQLModel, Session
from edgewalker_platform.db import create_db_engine, session_scope
from app.core.config import settings


DATABASE_URL = settings.DATABASE_URL

if DATABASE_URL.startswith("sqlite"):
    raise RuntimeError(
        "SQLite is not supported by this backend because the schema uses PostgreSQL-specific "
        "types such as JSONB. Set DATABASE_URL to a PostgreSQL connection string before startup."
    )

# Pool policy (sizing, fail-fast checkout, server-side idle-in-transaction
# kill, application_name, held-connection watchdog) is centralised in the
# platform kit — see edgewalker_platform/db.py.
engine = create_db_engine(
    DATABASE_URL,
    service_name="backend",
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_timeout=settings.DB_POOL_TIMEOUT,
    pool_recycle=settings.DB_POOL_RECYCLE,
)

_MIGRATION_MANAGED_TABLES = {"orders", "fills", "account_positions", "trades"}


def create_db_and_tables():
    from app.models.user import User  # noqa: F401
    from app.models.agent import Agent, Chat  # noqa: F401
    from app.models.n8n_chat_history import N8nChatHistory  # noqa: F401
    from app.models.password_reset_token import PasswordResetToken  # noqa: F401
    from app.models.personal_access_token import PersonalAccessToken  # noqa: F401
    from app.models.access_control import (  # noqa: F401
        AccessAllowlist,
        EmailVerificationToken,
        UserIdentity,
        UserRecoveryCode,
        UserTotp,
    )
    from app.models.strategy import Strategy, BacktestResult, BacktestTrade  # noqa: F401
    from app.models.connection import Connection, Account  # noqa: F401
    from app.models.live_trading import LiveOrder, LiveFill, LivePosition, LiveTrade  # noqa: F401
    from app.models.marketdata import SymbolCache, SymbolSyncLog  # noqa: F401
    from app.models.agent_call import AgentCall  # noqa: F401
    from app.models.billing import (  # noqa: F401
        AiCreditLedger,
        AiCreditPeriod,
        AiModelRate,
        BillingExternalRef,
        Coupon,
        CouponRedemption,
        Plan,
        PlanPrice,
        Subscription,
        SubscriptionEvent,
        TrialGrant,
    )

    bootstrap_tables = [
        table
        for table in SQLModel.metadata.sorted_tables
        if table.name not in _MIGRATION_MANAGED_TABLES
    ]
    SQLModel.metadata.create_all(engine, tables=bootstrap_tables)


def get_session():
    """Dependency for FastAPI endpoints.

    Commits on clean exit / rolls back on exception, so a request can never
    leave its connection "idle in transaction" past its own lifetime.
    ``expire_on_commit=False`` keeps ORM objects readable after a mid-request
    ``session.commit()`` releases the connection early (the auth dependency
    relies on this).
    """
    with Session(engine, expire_on_commit=False) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise


@contextmanager
def get_session_context() -> Generator[Session, None, None]:
    """Context manager for background tasks and non-FastAPI code."""
    with session_scope(engine) as session:
        yield session
