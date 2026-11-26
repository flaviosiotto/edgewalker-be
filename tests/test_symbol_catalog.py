"""Symbol catalogue cache (``app.services.symbol_catalog_service``): fresh copy
served without the gateway, stale copy when the gateway is down, brokers
without a catalogue, quotes from the ticker or from the last closed bars.
Run: ``venv/bin/python -m pytest tests/test_symbol_catalog.py``.
"""
from __future__ import annotations

import json

import pytest

from app.services import symbol_catalog_service as svc


class FakeRedis:
    def __init__(self):
        self.data: dict[str, str] = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None):
        self.data[key] = value

    def delete(self, key):
        self.data.pop(key, None)


@pytest.fixture
def redis(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(svc, "get_redis", lambda: fake)
    return fake


@pytest.fixture
def gateway(monkeypatch):
    """Scripted gateway: ``calls`` records the paths, ``answers`` maps a path
    to a payload or to an exception to raise."""
    state = {"calls": [], "answers": {}}

    def fake_request(connection_id, broker_type, method, path, *, params=None, body=None, timeout):
        state["calls"].append((method, path, params, body))
        answer = state["answers"][path]
        if isinstance(answer, Exception):
            raise answer
        return answer(params, body) if callable(answer) else answer

    monkeypatch.setattr(svc, "_gateway_request", fake_request)
    return state


PRICED = {"as_of": 1000, "broker_type": "okx", "market_type": "swap", "rows": [
    {"symbol": "BTC-USDT-SWAP", "name": "BTC", "asset_type": "crypto_futures", "last": 100.0, "change_24h_pct": 1.5},
    {"symbol": "ETH-USDT-SWAP", "name": "ETH", "asset_type": "crypto_futures", "last": 10.0, "change_24h_pct": -2.0},
]}
STATIC = {"as_of": 2000, "broker_type": "ctrader", "market_type": "demo", "rows": [
    {"symbol": "US500.cash", "name": "US 500", "asset_type": "index"},
]}


def test_fresh_catalog_is_served_without_the_gateway(redis, gateway):
    gateway["answers"]["/universe/snapshot"] = PRICED
    first = svc.get_catalog(7, "okx")
    second = svc.get_catalog(7, "okx")
    assert first["count"] == 2 and first["has_prices"] is True and first["stale"] is False
    assert second["rows"] == first["rows"]
    assert len(gateway["calls"]) == 1


def test_refresh_reads_the_gateway_again(redis, gateway):
    gateway["answers"]["/universe/snapshot"] = PRICED
    svc.get_catalog(7, "okx")
    svc.get_catalog(7, "okx", refresh=True)
    assert len(gateway["calls"]) == 2


def test_expired_catalog_falls_back_to_the_stale_copy_when_the_gateway_is_down(redis, gateway):
    gateway["answers"]["/universe/snapshot"] = PRICED
    svc.get_catalog(7, "okx")
    entry = json.loads(redis.data["symbols:catalog:7"])
    entry["fetched_at"] -= svc.CATALOG_FRESH_PRICED_S + 5
    redis.data["symbols:catalog:7"] = json.dumps(entry)
    gateway["answers"]["/universe/snapshot"] = svc.CatalogUnavailableError("down")

    stale = svc.get_catalog(7, "okx")
    assert stale["stale"] is True and stale["count"] == 2 and stale["as_of"] == 1000


def test_gateway_down_with_nothing_cached_raises(redis, gateway):
    gateway["answers"]["/universe/snapshot"] = svc.CatalogUnavailableError("down")
    with pytest.raises(svc.CatalogUnavailableError):
        svc.get_catalog(7, "okx")


def test_broker_without_catalogue_is_remembered(redis, gateway):
    gateway["answers"]["/universe/snapshot"] = svc._Unsupported("/universe/snapshot")
    first = svc.get_catalog(3, "ibkr")
    second = svc.get_catalog(3, "ibkr")
    assert first["supported"] is False and first["rows"] == []
    assert second["supported"] is False
    assert len(gateway["calls"]) == 1


def test_static_catalogue_stays_fresh_longer_than_a_priced_one(redis, gateway):
    gateway["answers"]["/universe/snapshot"] = STATIC
    svc.get_catalog(9, "ctrader")
    entry = json.loads(redis.data["symbols:catalog:9"])
    entry["fetched_at"] -= svc.CATALOG_FRESH_PRICED_S + 5
    redis.data["symbols:catalog:9"] = json.dumps(entry)
    again = svc.get_catalog(9, "ctrader")
    assert again["has_prices"] is False and len(gateway["calls"]) == 1


def test_without_redis_every_read_reaches_the_gateway(monkeypatch, gateway):
    monkeypatch.setattr(svc, "get_redis", lambda: None)
    gateway["answers"]["/universe/snapshot"] = PRICED
    svc.get_catalog(7, "okx")
    svc.get_catalog(7, "okx")
    assert len(gateway["calls"]) == 2


def test_symbol_info_is_the_exact_match_with_its_market_row(redis, gateway):
    gateway["answers"]["/universe/snapshot"] = PRICED
    gateway["answers"]["/search"] = {"symbols": [
        {"symbol": "BTC-USDT-SWAP", "name": "BTC", "asset_type": "crypto_futures", "exchange": "okx", "currency": "USDT", "tick_size": 0.1},
        {"symbol": "BTC-USDT-SWAP-OLD", "name": "x", "asset_type": "crypto_futures"},
    ]}
    info = svc.get_symbol_info(7, "okx", "btc-usdt-swap")
    assert info["symbol"] == "BTC-USDT-SWAP"
    assert info["extra_data"] == {"tick_size": 0.1}
    assert info["market"]["last"] == 100.0

    # Second read: description from cache, no second search.
    svc.get_symbol_info(7, "okx", "BTC-USDT-SWAP")
    assert [c[1] for c in gateway["calls"]].count("/search") == 1


def test_symbol_info_unknown_symbol_is_none(redis, gateway):
    gateway["answers"]["/search"] = {"symbols": [{"symbol": "BTC-USDT-SWAP"}]}
    assert svc.get_symbol_info(7, "okx", "BTC") is None


def test_quotes_come_from_the_ticker_when_the_venue_has_one(redis, gateway):
    gateway["answers"]["/universe/snapshot"] = PRICED
    out = svc.get_quotes(7, "okx", ["ETH-USDT-SWAP", "MISSING"])
    assert out["supported"] is True
    assert set(out["quotes"]) == {"ETH-USDT-SWAP"}
    assert out["quotes"]["ETH-USDT-SWAP"]["last"] == 10.0
    assert [c[1] for c in gateway["calls"]] == ["/universe/snapshot"]


def test_quotes_come_from_closed_bars_when_there_is_no_ticker(redis, gateway):
    gateway["answers"]["/universe/snapshot"] = STATIC
    bars = [[i * 3_600_000, 100.0, 101.0 + i, 99.0 - i, 100.0 + i, 5.0] for i in range(svc.QUOTE_BARS)]
    gateway["answers"]["/universe/bars"] = {"as_of": 5000, "bars": {"US500.cash": bars}, "errors": {}}

    out = svc.get_quotes(9, "ctrader", ["US500.cash"])
    quote = out["quotes"]["US500.cash"]
    assert quote["last_from_bar"] is True
    assert quote["last"] == 124.0
    assert quote["change_24h_pct"] == pytest.approx(24.0)
    assert quote["high_24h"] == 125.0 and quote["low_24h"] == 75.0

    svc.get_quotes(9, "ctrader", ["US500.cash"])
    assert [c[1] for c in gateway["calls"]].count("/universe/bars") == 1


def test_short_bar_history_gives_a_last_without_a_change(redis, gateway):
    assert svc._quote_from_bars([[0, 1, 2, 0.5, 1.5, 3]]) == {"last": 1.5, "ts": 0, "last_from_bar": True}
    assert svc._quote_from_bars([]) is None


def test_quotes_of_a_broker_without_catalogue_are_unsupported(redis, gateway):
    gateway["answers"]["/universe/snapshot"] = svc._Unsupported("/universe/snapshot")
    out = svc.get_quotes(3, "ibkr", ["MNQZ6"])
    assert out == {"connection_id": 3, "supported": False, "as_of": None, "stale": False, "quotes": {}}
