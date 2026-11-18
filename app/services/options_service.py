"""Option chain reads, proxied to the connection's gateway.

The backend does not interpret option data: chain, quotes and readiness are
produced by the gateway adapter (contract in the devops doc
``valutazione-opzioni.md`` §11) and returned to the caller untouched. What the
backend adds is the tenancy boundary — the connection (and, for readiness,
the account) must belong to the caller — and the registry gate: a broker that
does not trade options is answered without bothering its gateway.

Orders on options need nothing here: they travel through the single order
write path (``POST /accounts/{id}/orders``) with the canonical symbol.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from edgewalker_platform.brokers import supports_options
from edgewalker_platform.options import CODE_UNSUPPORTED

from app.db.database import get_session_context
from app.services.connection_service import get_account, get_connection
from app.services.gateway_client import (
    GatewayClient,
    GatewayError,
    GatewayRejectedError,
    GatewayUnavailableError,
    format_gateway_detail,
)

logger = logging.getLogger(__name__)

# Chain and readiness come from gateway caches or a couple of broker requests;
# a quotes snapshot waits for every line of the window, which takes seconds.
CHAIN_TIMEOUT = 15.0
QUOTES_TIMEOUT = 20.0
READINESS_TIMEOUT = 20.0


class OptionsNotFoundError(LookupError):
    """The connection or the account does not exist for this user."""


class OptionsRequestError(ValueError):
    """The request is inconsistent (e.g. account of another connection)."""


class OptionsUnsupportedError(Exception):
    """The connection's broker does not trade options (registry)."""

    def __init__(self, broker_type: str) -> None:
        super().__init__(f"Broker '{broker_type}' does not support options")
        self.broker_type = broker_type

    @property
    def detail(self) -> dict[str, str]:
        return {"code": CODE_UNSUPPORTED, "message": str(self)}


@dataclass(frozen=True)
class OptionsTarget:
    """Plain values resolved from the DB, so no session outlives the lookup."""

    connection_id: int
    broker_type: str
    broker_account: str | None = None


def resolve_options_target(
    *,
    user_id: int,
    connection_id: int,
    account_id: int | None = None,
) -> OptionsTarget:
    """Check ownership and capability; translate the internal account id.

    Raises ``OptionsNotFoundError`` (connection/account not the user's),
    ``OptionsRequestError`` (account of another connection) or
    ``OptionsUnsupportedError`` (broker without options).
    """
    with get_session_context() as session:
        connection = get_connection(session, connection_id, user_id)
        if connection is None:
            raise OptionsNotFoundError("Connection not found")
        broker_type = str(connection.broker_type or "")
        broker_account: str | None = None
        if account_id is not None:
            account = get_account(session, account_id, user_id)
            if account is None:
                raise OptionsNotFoundError("Account not found")
            if account.connection_id != connection.id:
                raise OptionsRequestError(
                    f"Account {account_id} does not belong to connection {connection_id}"
                )
            broker_account = str(account.account_id) if account.account_id else None
        resolved_id = int(connection.id)
    # Session closed here: the gateway call that follows can take seconds and
    # must not hold a pooled DB connection.
    if not supports_options(broker_type):
        raise OptionsUnsupportedError(broker_type)
    return OptionsTarget(connection_id=resolved_id, broker_type=broker_type, broker_account=broker_account)


async def _gateway_read(target: OptionsTarget, path: str, params: dict[str, Any], timeout: float) -> Any:
    client = GatewayClient(target.connection_id, broker_type=target.broker_type, timeout=timeout)
    try:
        return await client.get_options(path, params=params)
    except GatewayError:
        raise
    except Exception as exc:  # noqa: BLE001 - e.g. a non-JSON body from a stale gateway
        raise GatewayUnavailableError(f"Gateway options read failed: {exc}") from exc


async def get_option_chain(target: OptionsTarget, *, underlying: str) -> Any:
    return await _gateway_read(target, "/options/chain", {"underlying": underlying}, CHAIN_TIMEOUT)


async def get_option_quotes(
    target: OptionsTarget,
    *,
    underlying: str,
    expiry: str | None = None,
    strikes_around_atm: int | None = None,
    symbols: str | None = None,
) -> Any:
    params: dict[str, Any] = {"underlying": underlying}
    if expiry:
        params["expiry"] = expiry
    if strikes_around_atm is not None:
        params["strikes_around_atm"] = strikes_around_atm
    if symbols:
        params["symbols"] = symbols
    return await _gateway_read(target, "/options/quotes", params, QUOTES_TIMEOUT)


async def get_option_readiness(target: OptionsTarget, *, underlying: str) -> Any:
    params: dict[str, Any] = {"underlying": underlying}
    if target.broker_account:
        params["account"] = target.broker_account
    return await _gateway_read(target, "/options/readiness", params, READINESS_TIMEOUT)


def http_error_for(exc: GatewayError) -> tuple[int, Any]:
    """``(status, detail)`` the API answers for a gateway failure.

    - 409 stays 409 with the gateway's own ``detail`` (``OPTIONS_*`` codes);
    - 400/422 (invalid parameters) → 400 with the gateway's reason;
    - anything else — unreachable, timeout, 5xx/503 (broker not connected),
      or a 404 from a gateway that predates the options routes — → 424.
      Never 502/504: the edge proxy replaces them with its own page.
    """
    if isinstance(exc, GatewayRejectedError):
        if exc.status_code == 409:
            return 409, exc.detail if exc.detail is not None else str(exc)
        if exc.status_code == 400:
            return 400, exc.detail if exc.detail is not None else str(exc)
        if exc.status_code == 422:
            return 400, format_gateway_detail(exc.detail) or str(exc)
        if exc.status_code == 404:
            return 424, f"The connection's gateway does not serve options yet: {exc}"
    return 424, str(exc) or "Gateway unavailable"
