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