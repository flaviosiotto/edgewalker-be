from typing import Any

from fastapi import HTTPException


def is_provisional_account(account: Any) -> bool:
    extra = getattr(account, "extra", None)
    return isinstance(extra, dict) and extra.get("onboarding_provisional") is True


def require_configured_account(account: Any) -> None:
    if account is None or is_provisional_account(account):
        raise HTTPException(
            status_code=409,
            detail="Completa il collegamento del conto cTrader prima di avviare un'esecuzione.",
        )

def archived_account_message(account: Any, connection_name: str | None = None) -> str:
    """Why an archived account cannot be traded, with the number the user knows."""
    label = getattr(account, "display_name", None) or getattr(account, "account_id", None)
    where = f" on connection '{connection_name}'" if connection_name else ""
    return (
        f"Account {label} is no longer available{where}. "
        "Copy the strategy to an active account to trade it."
    )


def require_active_account(account: Any, connection_name: str | None = None) -> None:
    """Archived = the gateway no longer lists it (accounts.archived_at)."""
    if account is not None and getattr(account, "archived_at", None) is not None:
        raise HTTPException(status_code=409, detail=archived_account_message(account, connection_name))
