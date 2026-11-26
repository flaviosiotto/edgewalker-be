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
    first = svc.get_catalog(3, "binance")
    second = svc.get_catalog(3, "binance")
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
    gateway["answers"]["/universe/snapshot"] = PRICED
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
    out = svc.get_quotes(3, "binance", ["BTCUSDT"])
    assert out == {"connection_id": 3, "supported": False, "as_of": None, "stale": False, "quotes": {}}


# ── Brokers without a catalogue of their own: public listings (IBKR) ──

NASDAQ_FILE = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N
QQQ|Invesco QQQ Trust, Series 1|G|N|N|100|Y|N
ZZZT|Test issue|G|Y|N|100|N|N
ABCDW|Some Warrant|G|N|N|100|N|N
File Creation Time: 1125202621:32|||||||
"""
OTHER_FILE = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
BRK.B|Berkshire Hathaway Inc. New Common Stock|N|BRK.B|N|40|N|BRK.B
SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY
BAC$B|Bank of America Preferred|N|BACpB|N|100|N|BAC-B
AAPL|Duplicate of a Nasdaq ticker|N|AAPL|N|100|N|AAPL
File Creation Time: 1125202621:32||||||
"""


def test_symbol_directory_rows_are_spelled_the_way_ibkr_does():
    from app.services.symbol_listings import parse_symbol_directory

    rows = parse_symbol_directory(NASDAQ_FILE, OTHER_FILE)
    by_symbol = {row["symbol"]: row for row in rows}
    # test issues, preferreds and five-letter-plus suffixes are left to the broker search
    assert list(by_symbol) == ["AAPL", "ABCDW", "BRK B", "QQQ", "SPY"]
    assert by_symbol["AAPL"] == {
        "symbol": "AAPL", "name": "Apple Inc. - Common Stock", "asset_type": "stock",
        "exchange": "NASDAQ", "currency": "USD",
    }
    assert by_symbol["QQQ"]["asset_type"] == "etf"
    assert by_symbol["SPY"]["exchange"] == "ARCA" and by_symbol["SPY"]["asset_type"] == "etf"
    assert by_symbol["BRK B"]["exchange"] == "NYSE"


@pytest.fixture
def directory(monkeypatch):
    from app.services import symbol_listings as listings

    state = {"downloads": 0, "fail": False}

    def fake_download(name):
        state["downloads"] += 1
        if state["fail"]:
            raise RuntimeError("no route")
        return NASDAQ_FILE if name == "nasdaqlisted" else OTHER_FILE

    monkeypatch.setattr(listings, "_download", fake_download)
    return state


FUTURE = {"symbol": "MNQZ6", "name": "Micro Nasdaq", "asset_type": "futures", "exchange": "CME", "currency": "USD", "con_id": 7, "expiry": "20261218", "multiplier": "2"}


def test_ibkr_catalogue_is_futures_plus_us_listings_and_says_it_is_partial(redis, gateway, directory):
    gateway["answers"]["/search"] = lambda params, body: {"symbols": [FUTURE] if params["query"] == "MNQ" else []}

    catalog = svc.get_catalog(4, "ibkr")
    assert catalog["supported"] is True and catalog["complete"] is False and catalog["has_prices"] is False
    symbols = [row["symbol"] for row in catalog["rows"]]
    assert symbols == ["MNQZ6", "AAPL", "ABCDW", "BRK B", "QQQ", "SPY"]
    assert catalog["rows"][0]["extra_data"] == {"expiry": "20261218", "multiplier": "2"}
    assert catalog["rows"][0]["con_id"] == 7
    # never the gateway's universe: IBKR has none
    assert all(call[1] == "/search" for call in gateway["calls"])

    calls, downloads = len(gateway["calls"]), directory["downloads"]
    svc.get_catalog(4, "ibkr")
    assert len(gateway["calls"]) == calls and directory["downloads"] == downloads == 2


def test_ibkr_catalogue_survives_a_silent_gateway_and_a_failed_download(redis, gateway, directory):
    gateway["answers"]["/search"] = svc.CatalogUnavailableError("down")
    only_stocks = svc.get_catalog(4, "ibkr")
    assert [row["symbol"] for row in only_stocks["rows"]] == ["AAPL", "ABCDW", "BRK B", "QQQ", "SPY"]
    # one attempt, not one per root
    assert len(gateway["calls"]) == 1

    redis.data.clear()
    directory["fail"] = True
    nothing = svc.get_catalog(5, "ibkr")
    assert nothing["supported"] is False and nothing["rows"] == []


def test_ibkr_symbol_info_prefers_the_listing_currency_and_reuses_a_futures_row(redis, gateway, directory):
    def search(params, body):
        if params["query"] == "MNQ":
            return {"symbols": [FUTURE]}
        if params["query"] == "AAPL":
            return {"symbols": [
                {"symbol": "AAPL", "name": "APPLE MXN", "asset_type": "stock", "exchange": "MEXI", "currency": "MXN", "con_id": 1},
                {"symbol": "AAPL", "name": "APPLE INC", "asset_type": "stock", "exchange": "NASDAQ", "currency": "USD", "con_id": 265598},
            ]}
        return {"symbols": []}

    gateway["answers"]["/search"] = search
    stock = svc.get_symbol_info(4, "ibkr", "AAPL")
    assert stock["con_id"] == 265598 and stock["currency"] == "USD" and stock["market"] is None

    before = len(gateway["calls"])
    future = svc.get_symbol_info(4, "ibkr", "MNQZ6")
    assert future["con_id"] == 7 and future["extra_data"]["expiry"] == "20261218"
    assert len(gateway["calls"]) == before


def test_ibkr_quotes_are_unsupported_without_asking_the_gateway_for_bars(redis, gateway, directory):
    gateway["answers"]["/search"] = {"symbols": []}
    out = svc.get_quotes(4, "ibkr", ["AAPL"])
    assert out["supported"] is False and out["quotes"] == {}
    assert all(call[1] != "/universe/bars" for call in gateway["calls"])
