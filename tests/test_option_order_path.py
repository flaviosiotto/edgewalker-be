"""Options F2: a single-leg option order uses the existing write path.

``POST /accounts/{id}/orders`` → ``order_command_service.place_account_order``
→ ``GatewayClient.place_order`` must deliver the canonical option symbol and
``extra.asset_type`` to the gateway exactly as sent: no normalisation, no
whitelist, no length limit below the 32 characters of the ledger columns.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import httpx

from app.services import gateway_client as gc

OPTION_SYMBOL = "MNQZ6-261218-21500-C"


def test_option_order_reaches_the_gateway_unchanged(monkeypatch):
    from app.services import order_command_service as svc

    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/orders"
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"order_id": "42", "status": "submitted"})

    real = httpx.AsyncClient
    monkeypatch.setattr(gc.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))

    class FakeSession:
        def get(self, model: Any, key: Any) -> Any:
            if model is svc.Connection:
                return SimpleNamespace(id=7, broker_type="ibkr")
            return None

    account = SimpleNamespace(id=12, connection_id=7, account_id="U1234567")
    result = asyncio.run(svc.place_account_order(
        FakeSession(), account, symbol=OPTION_SYMBOL, side="BUY", order_type="LIMIT",
        quantity=1, limit_price=412.5, extra={"asset_type": "futures_option"},
    ))

    (payload,) = seen
    assert payload["symbol"] == OPTION_SYMBOL
    assert payload["extra"]["asset_type"] == "futures_option"
    assert payload["account"] == "U1234567"
    assert payload["side"] == "buy" and payload["order_type"] == "limit" and payload["limit_price"] == 412.5
    assert result["result"] == {"order_id": "42", "status": "submitted"}


def test_order_request_schema_keeps_the_option_symbol_and_extra():
    from app.api.accounts import AccountPlaceOrderRequest

    body = AccountPlaceOrderRequest(
        symbol=OPTION_SYMBOL, side="buy", order_type="limit", quantity=1, limit_price=412.5,
        extra={"asset_type": "futures_option"},
    )
    assert body.symbol == OPTION_SYMBOL
    assert body.extra == {"asset_type": "futures_option"}


def test_option_symbols_fit_the_ledger_symbol_columns():
    from edgewalker_platform.options import MAX_SYMBOL_LENGTH

    from app.models.live_trading import LiveFill, LiveOrder, LivePosition, LiveTrade

    for model in (LiveOrder, LiveFill, LiveTrade, LivePosition):
        assert model.__table__.c.symbol.type.length >= MAX_SYMBOL_LENGTH
