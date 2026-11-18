"""``GET /marketdata/options/{chain,quotes,readiness}`` (options F2).

The backend is a proxy with a tenancy boundary: the connection (and the
account, for readiness) must be the caller's, a broker without options is
answered 409 ``OPTIONS_UNSUPPORTED`` without a gateway call, the gateway body
comes back untouched, and gateway failures map to 409/400/424 — never 502/504,
which the edge proxy would replace with its own page.
Run: ``venv/bin/python -m pytest tests/test_options_proxy_db.py``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import Session, select

import app.models.agent_lesson  # noqa: E402,F401
import app.models.agent_turn  # noqa: E402,F401
import app.models.connection  # noqa: E402,F401
import app.models.live_trading  # noqa: E402,F401
import app.models.strategy  # noqa: E402,F401
import app.models.strategy_template  # noqa: E402,F401
import app.models.user_event  # noqa: E402,F401
from app.services import gateway_client as gc

pytest.importorskip("pgserver")

CHAIN = {
    "underlying": {"symbol": "MNQZ6", "asset_type": "futures", "exchange": "CME", "currency": "USD",
                   "broker_instrument_id": "123"},
    "option_type": "futures_option",
    "expiries": [{"expiry": "2026-12-18", "trading_class": "MNQ", "exchange": "CME", "multiplier": 2.0,
                  "exercise_style": None, "strikes": [21000, 21100, 21200]}],
    "as_of": "2026-11-18T16:00:00Z",
}
QUOTES = {
    "underlying": "MNQZ6", "underlying_price": 21510.25, "expiry": "2026-12-18", "market_data": "not_entitled",
    "quotes": [{"symbol": "MNQZ6-261218-21500-C", "expiry": "2026-12-18", "strike": 21500, "right": "C",
                "bid": None, "ask": None, "last": None, "mark": None, "volume": None, "open_interest": None,
                "iv": None, "delta": None, "gamma": None, "theta": None, "vega": None,
                "greeks_source": None, "multiplier": 2.0}],
    "warnings": [{"code": "OPTIONS_DATA_NOT_ENTITLED", "message": "no subscription"}],
    "as_of": "2026-11-18T16:00:00Z",
}
READINESS = {
    "underlying": "MNQZ6", "option_type": "futures_option",
    "requirements": {
        "chain": {"state": "ok", "code": None, "message": None, "broker_code": None},
        "market_data": {"state": "delayed", "code": "OPTIONS_DATA_DELAYED", "message": "delayed", "broker_code": 10167},
        "trading": {"state": "unknown", "code": None, "message": "not checked", "broker_code": None},
    },
    "as_of": "2026-11-18T16:00:00Z",
}


@pytest.fixture
def session(app_engine):
    with Session(app_engine, expire_on_commit=False) as s:
        yield s


@pytest.fixture
def tenant(session):
    """Two users; the first owns an IBKR connection (two accounts), an OKX
    connection (no options) and a second IBKR connection with one account."""
    from app.models.connection import Account, Connection
    from app.models.user import User

    now = datetime.now(timezone.utc)
    for email in ("opt@example.com", "opt-other@example.com"):
        stale = session.exec(select(User).where(User.email == email)).first()
        if stale is not None:
            session.delete(stale)
            session.commit()
    user = User(email="opt@example.com", username="user_opt", hashed_password="x")
    other = User(email="opt-other@example.com", username="user_opt_other", hashed_password="x")
    session.add(user)
    session.add(other)
    session.flush()

    def conn(owner, name, broker):
        row = Connection(user_id=owner.id, name=name, broker_type=broker, status="connected",
                         created_at=now, updated_at=now)
        session.add(row)
        session.flush()
        return row

    def acc(connection, code):
        row = Account(connection_id=connection.id, account_id=code, currency="USD", created_at=now, updated_at=now)
        session.add(row)
        session.flush()
        return row

    ibkr = conn(user, "IBKR", "ibkr")
    okx = conn(user, "OKX", "okx")
    ibkr2 = conn(user, "IBKR 2", "ibkr")
    foreign = conn(other, "IBKR other", "ibkr")
    ns = SimpleNamespace(
        user=user, other=other, ibkr=ibkr, okx=okx, ibkr2=ibkr2, foreign=foreign,
        account=acc(ibkr, "U1234567"), account2=acc(ibkr2, "U7654321"), foreign_account=acc(foreign, "U0000001"),
    )
    session.commit()
    yield ns
    session.delete(user)
    session.delete(other)
    session.commit()


@pytest.fixture
def gateway(monkeypatch):
    """Records the gateway requests; ``gateway.reply`` decides the answer."""
    calls: list[httpx.Request] = []
    state = SimpleNamespace(calls=calls, reply=lambda request: httpx.Response(200, json={}))

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return state.reply(request)

    real = httpx.AsyncClient

    def factory(**kwargs: Any) -> httpx.AsyncClient:
        return real(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(gc.httpx, "AsyncClient", factory)
    return state


def _client(user) -> TestClient:
    from app.api.marketdata import router
    from app.utils.auth_utils import get_current_active_user

    api = FastAPI()
    api.include_router(router)
    api.dependency_overrides[get_current_active_user] = lambda: SimpleNamespace(id=user.id, is_active=True)
    return TestClient(api)


def test_chain_passes_the_gateway_body_through_untouched(tenant, gateway):
    gateway.reply = lambda request: httpx.Response(200, json=CHAIN)
    resp = _client(tenant.user).get(
        "/marketdata/options/chain", params={"connection_id": tenant.ibkr.id, "underlying": "MNQZ6"}
    )
    assert resp.status_code == 200
    assert resp.json() == CHAIN
    (request,) = gateway.calls
    assert request.url.host == f"gw-{tenant.ibkr.id}"
    assert request.url.path == "/options/chain"
    assert dict(request.url.params) == {"underlying": "MNQZ6"}


def test_quotes_forwards_the_window_parameters(tenant, gateway):
    gateway.reply = lambda request: httpx.Response(200, json=QUOTES)
    resp = _client(tenant.user).get("/marketdata/options/quotes", params={
        "connection_id": tenant.ibkr.id, "underlying": "MNQZ6", "expiry": "2026-12-18",
        "strikes_around_atm": 5, "symbols": "MNQZ6-261218-21500-C,MNQZ6-270115-21000-P",
    })
    assert resp.status_code == 200
    assert resp.json() == QUOTES
    (request,) = gateway.calls
    assert request.url.path == "/options/quotes"
    assert dict(request.url.params) == {
        "underlying": "MNQZ6", "expiry": "2026-12-18", "strikes_around_atm": "5",
        "symbols": "MNQZ6-261218-21500-C,MNQZ6-270115-21000-P",
    }


def test_quotes_rejects_a_window_beyond_the_gateway_maximum(tenant, gateway):
    resp = _client(tenant.user).get("/marketdata/options/quotes", params={
        "connection_id": tenant.ibkr.id, "underlying": "MNQZ6", "strikes_around_atm": 26,
    })
    assert resp.status_code == 422
    assert gateway.calls == []


def test_readiness_translates_the_account_to_its_broker_code(tenant, gateway):
    gateway.reply = lambda request: httpx.Response(200, json=READINESS)
    client = _client(tenant.user)
    resp = client.get("/marketdata/options/readiness", params={
        "connection_id": tenant.ibkr.id, "underlying": "MNQZ6", "account_id": tenant.account.id,
    })
    assert resp.status_code == 200
    assert resp.json() == READINESS
    assert dict(gateway.calls[-1].url.params) == {"underlying": "MNQZ6", "account": "U1234567"}

    # Without an account the gateway gets no ``account`` parameter.
    resp = client.get("/marketdata/options/readiness", params={"connection_id": tenant.ibkr.id, "underlying": "MNQZ6"})
    assert resp.status_code == 200
    assert dict(gateway.calls[-1].url.params) == {"underlying": "MNQZ6"}


def test_readiness_account_of_another_connection_is_a_bad_request(tenant, gateway):
    resp = _client(tenant.user).get("/marketdata/options/readiness", params={
        "connection_id": tenant.ibkr.id, "underlying": "MNQZ6", "account_id": tenant.account2.id,
    })
    assert resp.status_code == 400
    assert "does not belong" in resp.json()["detail"]
    assert gateway.calls == []


def test_readiness_account_of_another_user_is_not_found(tenant, gateway):
    resp = _client(tenant.user).get("/marketdata/options/readiness", params={
        "connection_id": tenant.ibkr.id, "underlying": "MNQZ6", "account_id": tenant.foreign_account.id,
    })
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Account not found"
    assert gateway.calls == []


@pytest.mark.parametrize("path", ["chain", "quotes", "readiness"])
def test_connection_of_another_user_is_not_found(tenant, gateway, path):
    resp = _client(tenant.user).get(
        f"/marketdata/options/{path}", params={"connection_id": tenant.foreign.id, "underlying": "MNQZ6"}
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Connection not found"
    assert gateway.calls == []


@pytest.mark.parametrize("path", ["chain", "quotes", "readiness"])
def test_broker_without_options_is_409_without_a_gateway_call(tenant, gateway, path):
    resp = _client(tenant.user).get(
        f"/marketdata/options/{path}", params={"connection_id": tenant.okx.id, "underlying": "BTC-USD"}
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["code"] == "OPTIONS_UNSUPPORTED" and "okx" in detail["message"]
    assert gateway.calls == []


def test_gateway_409_detail_is_passed_through(tenant, gateway):
    detail = {"code": "OPTIONS_CHAIN_UNAVAILABLE", "message": "No option chain for XYZ"}
    gateway.reply = lambda request: httpx.Response(409, json={"detail": detail})
    resp = _client(tenant.user).get(
        "/marketdata/options/chain", params={"connection_id": tenant.ibkr.id, "underlying": "XYZ"}
    )
    assert resp.status_code == 409
    assert resp.json() == {"detail": detail}


def test_gateway_400_stays_400(tenant, gateway):
    gateway.reply = lambda request: httpx.Response(400, json={"detail": "invalid expiry '2026-13-01'"})
    resp = _client(tenant.user).get("/marketdata/options/quotes", params={
        "connection_id": tenant.ibkr.id, "underlying": "MNQZ6", "expiry": "2026-13-01",
    })
    assert resp.status_code == 400
    assert resp.json() == {"detail": "invalid expiry '2026-13-01'"}


@pytest.mark.parametrize("reply", [
    lambda request: httpx.Response(503, json={"detail": "Broker not connected"}),
    lambda request: httpx.Response(500, text="boom"),
    lambda request: httpx.Response(404, json={"detail": "Not Found"}),
    lambda request: httpx.Response(200, text="<html>not json</html>"),
])
def test_gateway_failures_become_424(tenant, gateway, reply):
    gateway.reply = reply
    resp = _client(tenant.user).get(
        "/marketdata/options/chain", params={"connection_id": tenant.ibkr.id, "underlying": "MNQZ6"}
    )
    assert resp.status_code == 424
    assert isinstance(resp.json()["detail"], str) and resp.json()["detail"]


@pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ReadTimeout])
def test_unreachable_or_slow_gateway_is_424(tenant, gateway, error):
    def reply(request):
        raise error("gateway down", request=request)

    gateway.reply = reply
    resp = _client(tenant.user).get(
        "/marketdata/options/quotes", params={"connection_id": tenant.ibkr.id, "underlying": "MNQZ6"}
    )
    assert resp.status_code == 424
    assert "unreachable" in resp.json()["detail"]


def test_quotes_uses_the_long_timeout(tenant, gateway, monkeypatch):
    seen: dict[str, Any] = {}
    real_init = gc.GatewayClient.__init__

    def spy(self, *args, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(gc.GatewayClient, "__init__", spy)
    gateway.reply = lambda request: httpx.Response(200, json=QUOTES)
    _client(tenant.user).get("/marketdata/options/quotes", params={"connection_id": tenant.ibkr.id, "underlying": "MNQZ6"})
    assert seen["timeout"] == 20.0
