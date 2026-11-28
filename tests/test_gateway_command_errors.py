"""Trading commands surface the gateway's reason as typed errors.

A broker rejection used to come back as ``RuntimeError("... 400 Bad Request for
url ...")`` → HTTP 502 with no reason, which the edge proxy then replaced with
its own page: the UI showed "Network Error". Now the gateway's ``detail`` is
the message and the HTTP layer answers 400 (rejected) / 424 (unreachable).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from app.services import gateway_client as gc
from app.services.gateway_client import (
    GatewayClient,
    GatewayError,
    GatewayRejectedError,
    GatewayUnavailableError,
    format_gateway_detail,
)


def _client_with(handler, monkeypatch) -> GatewayClient:
    transport = httpx.MockTransport(handler)
    real = httpx.AsyncClient

    def factory(**kwargs: Any) -> httpx.AsyncClient:
        return real(transport=transport, **kwargs)

    monkeypatch.setattr(gc.httpx, "AsyncClient", factory)
    return GatewayClient(21, broker_type="okx")


def test_format_gateway_detail_flattens_broker_and_validation_payloads():
    assert format_gateway_detail("limit_price required for limit orders") == "limit_price required for limit orders"
    assert format_gateway_detail({"message": "Insufficient USDT margin", "code": "51008", "broker": "okx"}) == "Insufficient USDT margin (code 51008)"
    assert format_gateway_detail({"message": "unknown account"}) == "unknown account"
    assert format_gateway_detail([{"loc": ["body", "quantity"], "msg": "value is not a valid float"}]) == "value is not a valid float"
    assert format_gateway_detail(None) == ""


def test_broker_rejection_is_a_rejected_error_carrying_the_reason(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/orders"
        return httpx.Response(400, json={"detail": {"message": "Insufficient USDT margin", "code": "51008", "broker": "okx"}})

    client = _client_with(handler, monkeypatch)
    with pytest.raises(GatewayRejectedError) as info:
        asyncio.run(client.place_order(symbol="BTC-USDT-SWAP", side="buy", order_type="limit", quantity=0.02, limit_price=83000))
    assert str(info.value) == "Insufficient USDT margin (code 51008)"
    assert info.value.status_code == 400 and info.value.detail["code"] == "51008"
    assert isinstance(info.value, RuntimeError)  # existing ``except RuntimeError`` handlers still catch it


def test_broker_unreachable_and_gateway_failures_are_unavailable_errors(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/close"):
            return httpx.Response(503, json={"detail": "Cannot reach OKX API at https://eea.okx.com"})
        return httpx.Response(500, text="boom")

    client = _client_with(handler, monkeypatch)
    with pytest.raises(GatewayUnavailableError) as info:
        asyncio.run(client.close_position("p1", quantity=1))
    assert "Cannot reach OKX API" in str(info.value) and info.value.status_code == 503
    with pytest.raises(GatewayUnavailableError) as info:
        asyncio.run(client.amend_position_protection("p1", stop_loss_price=1.0))
    assert info.value.status_code == 500 and "boom" in str(info.value)


def test_transport_errors_are_unavailable_errors(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _client_with(handler, monkeypatch)
    with pytest.raises(GatewayUnavailableError) as info:
        asyncio.run(client.cancel_order("abc", symbol="BTC-USDT-SWAP"))
    assert "unreachable" in str(info.value) and info.value.status_code is None


def test_read_paths_keep_raising_httpx_errors(monkeypatch):
    """Only the commands are translated: status/accounts callers handle httpx themselves."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = _client_with(handler, monkeypatch)
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(client.get_status())


def test_place_account_order_lets_the_typed_error_through(monkeypatch):
    from app.services import order_command_service as svc

    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None: ...

        async def place_order(self, **kwargs: Any) -> dict:
            raise GatewayRejectedError("Insufficient USDT margin (code 51008)", status_code=400)

    monkeypatch.setattr(svc, "GatewayClient", FakeClient)

    class FakeSession:
        def get(self, model: Any, key: Any) -> Any:
            if model is svc.Connection:
                return SimpleNamespace(id=21, broker_type="okx")
            if model is svc.StrategyLive:
                return SimpleNamespace(id=530, strategy_id=64)
            return None

    account = SimpleNamespace(id=121, connection_id=21, account_id="378169067619623190")
    with pytest.raises(GatewayRejectedError) as info:
        asyncio.run(svc.place_account_order(
            FakeSession(), account, symbol="BTC-USDT-SWAP", side="buy", order_type="limit",
            quantity=0.02, limit_price=83000.0, strategy_live_id=530,
        ))
    assert str(info.value) == "Insufficient USDT margin (code 51008)"
    assert isinstance(info.value, GatewayError)


def test_snapshot_read_before_a_live_start_carries_the_gateway_reason(monkeypatch):
    """The read that stopped a live start with "Client error '400 Bad Request'
    for url ...": the gateway had said why, the client dropped it."""
    reason = "cTrader account 47494755 is not the account of this connection (connected on 17258923, id 49713817)"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/positions":
            return httpx.Response(400, json={"detail": reason})
        return httpx.Response(200, json={"orders": []})

    client = _client_with(handler, monkeypatch)
    assert asyncio.run(client.list_open_orders()) == []
    with pytest.raises(GatewayRejectedError) as info:
        asyncio.run(client.list_positions(account="47494755"))
    assert str(info.value) == reason

    from app.api.live import _gateway_start_error

    answer = _gateway_start_error(info.value)
    assert answer.status_code == 400 and reason in answer.detail
    assert _gateway_start_error(GatewayUnavailableError("Gateway http://gw-11:8080 unreachable")).status_code == 424


def test_agent_risk_policy_uses_current_actor_and_database_not_request_metadata():
    from app.core.actor import Actor, reset_current_actor, set_current_actor
    from app.services.order_command_service import agent_sizing_policy

    session = SimpleNamespace(get=lambda model, key: SimpleNamespace(persona={"max_risk_per_trade_pct": 0.5}))
    token = set_current_actor(Actor(via="agent", user_id=1, agent_id=2))
    try:
        sizing, extra = agent_sizing_policy(session, 10, None, {"risk_limit_percent": 99, "actor": {"agent_id": 999}})
    finally:
        reset_current_actor(token)
    assert sizing == {"mode": "quantity", "value": 10}
    assert extra["risk_limit_percent"] == 0.5
