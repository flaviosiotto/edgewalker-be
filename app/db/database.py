from contextlib import contextmanager
from typing import Generator

from sqlmodel import SQLModel, Session, create_engine
from app.core.config import settings


DATABASE_URL = settings.DATABASE_URL

if DATABASE_URL.startswith("sqlite"):
    raise RuntimeError(
        "SQLite is not supported by this backend because the schema uses PostgreSQL-specific "
        "types such as JSONB. Set DATABASE_URL to a PostgreSQL connection string before startup."
    )

connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True
)

_MIGRATION_MANAGED_TABLES = {"orders", "fills", "account_positions"}


def create_db_and_tables():
    from app.models.user import User  # noqa: F401
    from app.models.agent import Agent, Chat  # noqa: F401
    from app.models.n8n_chat_history import N8nChatHistory  # noqa: F401
    from app.models.password_reset_token import PasswordResetToken  # noqa: F401
    from app.models.strategy import Strategy, BacktestResult, BacktestTrade  # noqa: F401
    from app.models.connection import Connection, Account  # noqa: F401
    from app.models.live_trading import LiveOrder, LiveFill, LivePosition  # noqa: F401
    from app.models.marketdata import SymbolCache, SymbolSyncLog  # noqa: F401

    bootstrap_tables = [
        table
        for table in SQLModel.metadata.sorted_tables
        if table.name not in _MIGRATION_MANAGED_TABLES
    ]
    SQLModel.metadata.create_all(engine, tables=bootstrap_tables)


def get_session():
    """Dependency for FastAPI endpoints."""
    with Session(engine) as session:
        yield session


@contextmanager
def get_session_context() -> Generator[Session, None, None]:
    """Context manager for background tasks and non-FastAPI code."""
    session = Session(engine)
    try:
        yield session
    finally:
        session.close()
