from sqlmodel import SQLModel, Session, create_engine
from sqlalchemy import text
from app.core.config import settings


DATABASE_URL = settings.DATABASE_URL

connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True
)


def create_db_and_tables():
    if not DATABASE_URL.startswith("sqlite"):
        with engine.begin() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS operations"))

    from app.models.user import User  # noqa: F401
    from app.models.agent import Agent, Chat  # noqa: F401
    from app.models.n8n_chat_history import N8nChatHistory  # noqa: F401

    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
