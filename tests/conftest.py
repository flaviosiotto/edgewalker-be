"""Test bootstrap.

``DATABASE_URL`` must exist before any ``app.*`` import (``app.db.database``
builds the engine at import time). Tests that only exercise pure functions get
a dummy URL; the ``pg`` fixture starts an embedded Postgres (``pgserver``,
``pip install pgserver`` in the venv) and points the app engine at it.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

SCRATCH = Path(os.environ.get("EW_TEST_SCRATCH", "/tmp")) / "edgewalker-be-tests"

# Pure-function modules import the app tree: give them a syntactically valid
# Postgres URL that is never connected to.
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@127.0.0.1:1/test")
os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("ALGORITHM", "HS256")


@pytest.fixture(scope="session")
def pg_url() -> str:
    pgserver = pytest.importorskip("pgserver")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    server = pgserver.get_server(str(SCRATCH / "pgdata"))
    return server.get_uri()


@pytest.fixture(scope="session")
def app_engine(pg_url):
    """The app engine rebound to the embedded Postgres, schema created the
    way the lifespan does it (tables + billing view + plan seed)."""
    from sqlalchemy.pool import NullPool
    from sqlmodel import create_engine

    import app.db.database as database

    engine = create_engine(pg_url, poolclass=NullPool)
    database.engine = engine
    # Modules that captured ``engine`` at import time.
    import app.services.billing.billing_service as billing

    billing.engine = engine
    database.create_db_and_tables()
    billing.ensure_billing_schema()
    with database.get_session_context() as session:
        billing.ensure_billing_seed(session)
    return engine
