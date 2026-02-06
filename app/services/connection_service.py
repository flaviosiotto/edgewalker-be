"""
Connection & Account service layer.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlmodel import Session, select

from app.models.connection import Connection, Account

logger = logging.getLogger(__name__)


# =============================================================================
# Connection CRUD
# =============================================================================

def list_connections(session: Session, *, active_only: bool = False) -> list[Connection]:
    stmt = select(Connection)
    if active_only:
        stmt = stmt.where(Connection.is_active == True)  # noqa: E712
    stmt = stmt.order_by(Connection.name)
    return list(session.exec(stmt).all())


def get_connection(session: Session, connection_id: int) -> Connection | None:
    return session.get(Connection, connection_id)


def create_connection(session: Session, *, name: str, broker_type: str, config: dict, is_active: bool = True) -> Connection:
    conn = Connection(
        name=name,
        broker_type=broker_type,
        config=config,
        is_active=is_active,
    )
    session.add(conn)
    session.commit()
    session.refresh(conn)
    logger.info("Created connection %s (id=%s, broker=%s)", conn.name, conn.id, conn.broker_type)
    return conn


def update_connection(session: Session, connection_id: int, **fields) -> Connection | None:
    conn = session.get(Connection, connection_id)
    if conn is None:
        return None
    for key, value in fields.items():
        if value is not None:
            setattr(conn, key, value)
    conn.updated_at = datetime.now(timezone.utc)
    session.add(conn)
    session.commit()
    session.refresh(conn)
    return conn


def delete_connection(session: Session, connection_id: int) -> bool:
    conn = session.get(Connection, connection_id)
    if conn is None:
        return False
    session.delete(conn)
    session.commit()
    logger.info("Deleted connection id=%s", connection_id)
    return True


# =============================================================================
# Account CRUD
# =============================================================================

def list_accounts(session: Session, connection_id: int, *, active_only: bool = False) -> list[Account]:
    stmt = select(Account).where(Account.connection_id == connection_id)
    if active_only:
        stmt = stmt.where(Account.is_active == True)  # noqa: E712
    stmt = stmt.order_by(Account.account_id)
    return list(session.exec(stmt).all())


def list_all_accounts(session: Session, *, active_only: bool = False) -> list[Account]:
    """List accounts across all connections."""
    stmt = select(Account)
    if active_only:
        stmt = stmt.where(Account.is_active == True)  # noqa: E712
    stmt = stmt.order_by(Account.account_id)
    return list(session.exec(stmt).all())


def get_account(session: Session, account_id: int) -> Account | None:
    return session.get(Account, account_id)


def create_account(
    session: Session,
    connection_id: int,
    *,
    account_id: str,
    display_name: str | None = None,
    account_type: str | None = None,
    currency: str = "USD",
    is_active: bool = True,
    extra: dict | None = None,
) -> Account:
    account = Account(
        connection_id=connection_id,
        account_id=account_id,
        display_name=display_name,
        account_type=account_type,
        currency=currency,
        is_active=is_active,
        extra=extra,
    )
    session.add(account)
    session.commit()
    session.refresh(account)
    logger.info("Created account %s (id=%s, connection=%s)", account.account_id, account.id, connection_id)
    return account


def update_account(session: Session, account_id: int, **fields) -> Account | None:
    account = session.get(Account, account_id)
    if account is None:
        return None
    for key, value in fields.items():
        if value is not None:
            setattr(account, key, value)
    account.updated_at = datetime.now(timezone.utc)
    session.add(account)
    session.commit()
    session.refresh(account)
    return account


def delete_account(session: Session, account_id: int) -> bool:
    account = session.get(Account, account_id)
    if account is None:
        return False
    session.delete(account)
    session.commit()
    logger.info("Deleted account id=%s", account_id)
    return True
