"""``POST /marketdata/options/payoff`` (options F5).

Pure computation: the model is the kit's Black-76
(``edgewalker_platform.option_pricing``); the endpoint adds the bookkeeping —
IV source per leg (broker, else implied from the mark), premium source, legs
excluded with a reason, validation. No gateway, no database.
Run: ``venv/bin/python -m pytest tests/test_options_payoff.py``.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from edgewalker_platform import option_pricing as pricing

URL = "/marketdata/options/payoff"
# Far enough that no test depends on the day it runs.
DEC = "MNQZ9-291221-21500-C"
DEC_PUT = "MNQZ9-291221-21500-P"
DEC_HIGH = "MNQZ9-291221-22000-C"
DEC_CUT = datetime(2029, 12, 21, 21, 0, tzinfo=timezone.utc)   # 16:00 New York, EST
JAN = "MNQZ9-300118-21500-C"
JAN_CUT = datetime(2030, 1, 18, 21, 0, tzinfo=timezone.utc)
SPOT = 21510.25
GRID = [20500.0, 21000.0, 21500.0, 22000.0, 22500.0]
VALUATION = DEC_CUT - timedelta(days=30)


@pytest.fixture
def client() -> TestClient:
    from app.api.marketdata import router
    from app.utils.auth_utils import get_current_active_user

    api = FastAPI()
    api.include_router(router)
    api.dependency_overrides[get_current_active_user] = lambda: SimpleNamespace(id=1, is_active=True)
    return TestClient(api)


def leg(symbol=DEC, side="buy", quantity=1, price=400.0, multiplier=2.0, iv=0.2, mark=None):
    return {"symbol": symbol, "side": side, "quantity": quantity, "price": price,
            "multiplier": multiplier, "iv": iv, "mark": mark}


def body(legs, **extra):
    return {"legs": legs, "underlying_price": SPOT, "underlying_prices": GRID,
            "valuation_at": VALUATION.isoformat(), **extra}


def years(cut: datetime, at: datetime = VALUATION) -> float:
    return (cut - at).total_seconds() / (365 * 86_400)


def test_values_and_net_greeks_of_a_vertical_spread(client):
    resp = client.post(URL, json=body([
        leg(DEC, "buy", 2, price=400.0, iv=0.20),
        leg(DEC_HIGH, "sell", 2, price=180.0, iv=0.19),
    ]))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    t = years(DEC_CUT)
    assert t == pytest.approx(30 / 365)

    def spread(f):
        return 2 * 2.0 * (
            (pricing.black76_price(f, 21500, t, 0.20, "C") - 400.0)
            - (pricing.black76_price(f, 22000, t, 0.19, "C") - 180.0)
        )

    assert data["model"] == "black76"
    assert data["underlying"] == "MNQZ9"
    assert data["underlying_prices"] == GRID
    assert data["values"] == pytest.approx([spread(f) for f in GRID])
    assert data["pnl"] == pytest.approx(spread(SPOT))
    assert data["excluded"] == []
    assert datetime.fromisoformat(data["valuation_at"]) == VALUATION
    assert data["iv_shift"] == 0 and data["rate"] == 0

    low = pricing.black76_greeks(SPOT, 21500, t, 0.20, "C")
    high = pricing.black76_greeks(SPOT, 22000, t, 0.19, "C")
    # Position greeks: × quantity × side × multiplier.
    for name in ("delta", "gamma", "theta", "vega"):
        assert data["net_greeks"][name] == pytest.approx(4.0 * (getattr(low, name) - getattr(high, name))), name
    assert data["units"]["theta"] == "currency per calendar day"
    assert data["units"]["vega"] == "currency per volatility point (1%)"

    first, second = data["per_leg"]
    assert first == {
        "index": 0, "symbol": DEC, "expires_at": first["expires_at"], "expired": False, "time_to_expiry": pytest.approx(t),
        "iv": 0.2, "iv_source": "broker", "iv_used": 0.2,
        "model_price": pytest.approx(low.price), "premium": 400.0, "premium_source": "price",
    }
    assert datetime.fromisoformat(first["expires_at"]) == DEC_CUT
    assert datetime.fromisoformat(data["nearest_expiry_at"]) == DEC_CUT
    assert second["symbol"] == DEC_HIGH and second["model_price"] == pytest.approx(high.price)


def test_iv_is_implied_from_the_mark_when_the_broker_gives_none(client):
    now = datetime.now(timezone.utc)
    t_now = years(DEC_CUT, now)
    mark = pricing.black76_price(SPOT, 21500, t_now, 0.27, "C")
    resp = client.post(URL, json=body([leg(iv=None, mark=mark)], as_of=now.isoformat()))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    (row,) = data["per_leg"]
    assert row["iv_source"] == "implied"
    assert row["iv"] == pytest.approx(0.27, rel=1e-5)
    assert row["iv_used"] == pytest.approx(0.27, rel=1e-5)
    assert data["values"][2] == pytest.approx(
        2.0 * (pricing.black76_price(21500, 21500, years(DEC_CUT), 0.27, "C") - 400.0), rel=1e-4
    )
    # A broker IV that is not a usable number (0, negative, absurd) falls back to the mark too.
    for bad in (0, -0.2, 50):
        again = client.post(URL, json=body([leg(iv=bad, mark=mark)], as_of=now.isoformat())).json()
        assert again["per_leg"][0]["iv_source"] == "implied", bad


def test_leg_without_price_uses_its_mark_and_says_so(client):
    data = client.post(URL, json=body([leg(price=None, mark=412.5)])).json()
    (row,) = data["per_leg"]
    assert row["premium"] == 412.5 and row["premium_source"] == "mark"
    assert row["iv_source"] == "broker"      # the IV given still wins over the mark
    t = years(DEC_CUT)
    assert data["values"][0] == pytest.approx(2.0 * (pricing.black76_price(20500, 21500, t, 0.2, "C") - 412.5))


def test_legs_that_cannot_be_valued_are_excluded_with_the_reason(client):
    data = client.post(URL, json=body([
        leg(DEC, iv=0.2, price=400.0),                                # valued
        leg(DEC_PUT, iv=None, mark=None, price=300.0),                # no IV, no mark
        leg(DEC_HIGH, iv=None, mark=None, price=None),                # no premium at all
        leg("MNQZ9-291221-20000-C", iv=None, mark=1000.0, price=1.0), # mark below intrinsic (1510.25)
    ])).json()
    assert [row["index"] for row in data["per_leg"]] == [0]
    assert [(row["index"], row["symbol"], row["reason"]) for row in data["excluded"]] == [
        (1, DEC_PUT, "iv_unavailable"),
        (2, DEC_HIGH, "price_unavailable"),
        (3, "MNQZ9-291221-20000-C", "iv_not_implied"),
    ]
    assert all(row["message"] for row in data["excluded"])
    t = years(DEC_CUT)
    assert data["values"] == pytest.approx([2.0 * (pricing.black76_price(f, 21500, t, 0.2, "C") - 400.0) for f in GRID])


def test_no_leg_valued_is_an_answer_not_an_error(client):
    resp = client.post(URL, json=body([leg(iv=None, mark=None)]))
    assert resp.status_code == 200
    data = resp.json()
    assert data["values"] == [] and data["net_greeks"] is None and data["pnl"] is None
    assert data["per_leg"] == [] and data["excluded"][0]["reason"] == "iv_unavailable"


def test_iv_shift_is_in_volatility_points_and_floored(client):
    t = years(DEC_CUT)
    up = client.post(URL, json=body([leg()], iv_shift=5)).json()
    assert up["iv_shift"] == 5
    assert up["per_leg"][0]["iv"] == 0.2 and up["per_leg"][0]["iv_used"] == pytest.approx(0.25)
    assert up["values"][2] == pytest.approx(2.0 * (pricing.black76_price(21500, 21500, t, 0.25, "C") - 400.0))

    base = client.post(URL, json=body([leg()])).json()
    # One point of shift moves the P&L by about the net vega (per volatility point).
    one = client.post(URL, json=body([leg()], iv_shift=1)).json()
    assert one["pnl"] - base["pnl"] == pytest.approx(base["net_greeks"]["vega"], rel=0.01)

    floored = client.post(URL, json=body([leg()], iv_shift=-60)).json()
    assert floored["per_leg"][0]["iv_used"] == pytest.approx(0.001)
    assert all(v == v for v in floored["values"])   # finite, no NaN

    for out_of_bounds in (-101, 100.5):
        assert client.post(URL, json=body([leg()], iv_shift=out_of_bounds)).status_code == 422


def test_theta_matches_one_calendar_day_of_decay(client):
    base = client.post(URL, json=body([leg()])).json()
    later = client.post(URL, json=body([leg()], valuation_at=(VALUATION + timedelta(days=1)).isoformat())).json()
    assert base["net_greeks"]["theta"] < 0
    assert later["pnl"] - base["pnl"] == pytest.approx(base["net_greeks"]["theta"], rel=0.03)


def test_legs_expired_at_valuation_contribute_their_intrinsic_value(client):
    after_december = DEC_CUT + timedelta(days=2)
    data = client.post(URL, json=body([
        leg(DEC, "sell", 1, price=300.0, iv=None, mark=None),   # expired: no IV needed
        leg(JAN, "buy", 1, price=450.0, iv=0.2),
    ], valuation_at=after_december.isoformat())).json()
    assert data["excluded"] == []
    expired, alive = data["per_leg"]
    assert expired["expired"] is True and expired["time_to_expiry"] == 0
    # The nearest expiry is reported even when it is already behind the valuation instant.
    assert datetime.fromisoformat(data["nearest_expiry_at"]) == DEC_CUT
    assert datetime.fromisoformat(alive["expires_at"]) == JAN_CUT
    assert expired["iv"] is None and expired["iv_source"] is None and expired["iv_used"] is None
    assert expired["model_price"] == pytest.approx(SPOT - 21500)
    assert alive["expired"] is False
    t = years(JAN_CUT, after_december)
    expected = [
        2.0 * ((pricing.black76_price(f, 21500, t, 0.2, "C") - 450.0) - (max(f - 21500, 0.0) - 300.0))
        for f in GRID
    ]
    assert data["values"] == pytest.approx(expected)


def test_valuation_exactly_at_the_cut_off_is_the_expiry_payoff(client):
    data = client.post(URL, json=body([leg(DEC, "buy", 1, price=400.0), leg(DEC_PUT, "buy", 1, price=380.0)],
                                      valuation_at=DEC_CUT.isoformat())).json()
    assert data["values"] == pytest.approx([2.0 * (abs(f - 21500) - 780.0) for f in GRID])
    assert data["net_greeks"]["gamma"] == 0 and data["net_greeks"]["vega"] == 0 and data["net_greeks"]["theta"] == 0


def test_valuation_defaults_to_now_and_rejects_the_past(client):
    before = datetime.now(timezone.utc)
    payload = body([leg()])
    del payload["valuation_at"]
    data = client.post(URL, json=payload).json()
    after = datetime.now(timezone.utc)
    assert before <= datetime.fromisoformat(data["valuation_at"]) <= after
    assert before <= datetime.fromisoformat(data["as_of"]) <= after

    # A small lag (client clock, request built a moment ago) is tolerated …
    ok = client.post(URL, json=body([leg()], valuation_at=(before - timedelta(minutes=2)).isoformat()))
    assert ok.status_code == 200
    # … an instant really in the past is not; neither is a quote time in the future.
    past = client.post(URL, json=body([leg()], valuation_at=(before - timedelta(hours=1)).isoformat()))
    assert past.status_code == 400 and "in the past" in past.json()["detail"]
    future = client.post(URL, json=body([leg()], as_of=(before + timedelta(hours=1)).isoformat()))
    assert future.status_code == 400 and "in the future" in future.json()["detail"]
    # A naive timestamp is read as UTC.
    naive = client.post(URL, json=body([leg()], valuation_at=VALUATION.replace(tzinfo=None).isoformat())).json()
    assert datetime.fromisoformat(naive["valuation_at"]) == VALUATION


def test_legs_on_different_underlyings_are_a_bad_request(client):
    resp = client.post(URL, json=body([leg(DEC), leg("MESZ9-291221-5000-C")]))
    assert resp.status_code == 400
    assert "same underlying" in resp.json()["detail"] and "MESZ9" in resp.json()["detail"]


@pytest.mark.parametrize("mutate, where", [
    (lambda b: b.update(legs=[]), "legs"),
    (lambda b: b.update(legs=[leg()] * 51), "legs"),
    (lambda b: b["legs"][0].update(symbol="MNQZ9"), "symbol"),
    (lambda b: b["legs"][0].update(symbol="MNQZ9-291321-21500-C"), "symbol"),
    (lambda b: b["legs"][0].update(side="hold"), "side"),
    (lambda b: b["legs"][0].update(quantity=0), "quantity"),
    (lambda b: b["legs"][0].update(quantity=-1), "quantity"),
    (lambda b: b["legs"][0].update(multiplier=0), "multiplier"),
    (lambda b: b["legs"][0].update(price=-1), "price"),
    (lambda b: b["legs"][0].pop("multiplier"), "multiplier"),
    (lambda b: b.update(underlying_price=0), "underlying_price"),
    (lambda b: b.pop("underlying_price"), "underlying_price"),
    (lambda b: b.update(underlying_prices=[21500.0]), "underlying_prices"),
    (lambda b: b.update(underlying_prices=[float(20000 + i) for i in range(402)]), "underlying_prices"),
    (lambda b: b.update(underlying_prices=[21500.0, 0.0]), "underlying_prices"),
    (lambda b: b.update(underlying_prices=[21500.0, -3.0]), "underlying_prices"),
    (lambda b: b.update(valuation_at="tomorrow"), "valuation_at"),
])
def test_invalid_requests_are_rejected_with_the_field_named(client, mutate, where):
    payload = body([leg()])
    mutate(payload)
    resp = client.post(URL, json=payload)
    assert resp.status_code == 422, resp.text
    assert any(where in [str(part) for part in error["loc"]] for error in resp.json()["detail"]), resp.text


def test_grid_bounds_are_inclusive_and_the_symbol_is_normalised(client):
    grid = [20000.0 + 5 * i for i in range(401)]
    payload = body([leg(symbol=DEC.lower())])
    payload["underlying_prices"] = grid
    data = client.post(URL, json=payload).json()
    assert len(data["values"]) == 401 and data["per_leg"][0]["symbol"] == DEC
    payload["underlying_prices"] = [21000.0, 22000.0]
    assert len(client.post(URL, json=payload).json()["values"]) == 2


def test_route_needs_the_authenticated_user_and_stays_clear_of_the_edge_prefixes():
    from app.api.marketdata import router

    (route,) = [r for r in router.routes if getattr(r, "path", "") == URL]
    assert route.methods == {"POST"}
    dependencies = {dep.call.__name__ for dep in route.dependant.dependencies}
    assert "get_current_active_user" in dependencies
    # Prefixes Traefik sends to another service.
    for prefix in ("/marketdata/ohlc-history", "/marketdata/coverage", "/marketdata/fetch-status", "/marketdata/history"):
        assert not URL.startswith(prefix)

    # Without the override the route asks for credentials.
    api = FastAPI()
    api.include_router(router)
    assert TestClient(api).post(URL, json=body([leg()])).status_code == 401
