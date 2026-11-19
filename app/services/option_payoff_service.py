"""Model valuation of a set of option legs before expiry (options F5).

Pure computation on the request: no gateway call, no database. The maths is
the kit's (``edgewalker_platform.option_pricing``, Black-76), so every layer
that values an option gets the same number.

What this module decides is only the bookkeeping around the model:

- which volatility a leg is priced with: the broker's ``iv`` when the request
  carries one, else the IV implied from the leg's ``mark`` at the current
  underlying price (``as_of``); a leg with neither is *excluded and listed*,
  never guessed;
- which premium the P&L is measured against: the leg's ``price``, else its
  ``mark`` — and the answer says which;
- a leg already expired at ``valuation_at`` is worth its intrinsic value and
  needs no volatility.

Everything returned is a model value and is labelled as such
(``model: "black76"``, ``iv_source`` per leg).
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

from edgewalker_platform.option_pricing import (
    IV_SEARCH_MAX,
    MODEL_BLACK76,
    PositionLeg,
    black76_price,
    expiry_instant,
    implied_volatility,
    position_greeks,
    position_pnl,
    year_fraction,
)
from edgewalker_platform.options import OptionContract

from app.schemas.options import OptionPayoffLeg, OptionPayoffRequest

# ``valuation_at`` may trail the server clock by this much (client clock skew,
# a request built a moment ago); ``as_of`` may lead it by the same amount.
CLOCK_TOLERANCE = timedelta(minutes=5)
# Floor of the volatility a leg is priced with, after the shift (0.1 %).
IV_FLOOR = 0.001
# Flat rate of the model. Zero in this phase: premiums of futures options are
# quoted against the future itself and the discounting is second order.
RATE = 0.0


class OptionPayoffError(ValueError):
    """The request is well-formed but cannot be valued (answered 400)."""


def _utc(value: datetime | None, default: datetime) -> datetime:
    if value is None:
        return default
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _usable(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0


def _excluded(index: int, leg: OptionPayoffLeg, reason: str, message: str) -> dict[str, Any]:
    return {"index": index, "symbol": leg.symbol, "reason": reason, "message": message}


def compute_payoff(request: OptionPayoffRequest, *, now: datetime | None = None) -> dict[str, Any]:
    """Value the legs at ``valuation_at`` over the price grid.

    Raises ``OptionPayoffError`` for legs on different underlyings, a
    ``valuation_at`` in the past or an ``as_of`` in the future.
    """
    now = _utc(now, datetime.now(timezone.utc))
    valuation_at = _utc(request.valuation_at, now)
    as_of = _utc(request.as_of, now)
    if valuation_at < now - CLOCK_TOLERANCE:
        raise OptionPayoffError(
            f"valuation_at {valuation_at.isoformat()} is in the past (server time {now.isoformat()})"
        )
    if as_of > now + CLOCK_TOLERANCE:
        raise OptionPayoffError(f"as_of {as_of.isoformat()} is in the future (server time {now.isoformat()})")

    contracts = [OptionContract.from_symbol(leg.symbol) for leg in request.legs]
    underlyings = sorted({contract.underlying for contract in contracts})
    if len(underlyings) > 1:
        raise OptionPayoffError(
            "all legs must be on the same underlying, got: " + ", ".join(underlyings)
        )

    spot = request.underlying_price
    shift = request.iv_shift / 100.0
    priced: list[PositionLeg] = []
    per_leg: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    for index, (leg, contract) in enumerate(zip(request.legs, contracts)):
        if leg.price is not None:
            premium, premium_source = leg.price, "price"
        elif _usable(leg.mark):
            premium, premium_source = float(leg.mark), "mark"  # type: ignore[arg-type]
        else:
            excluded.append(_excluded(
                index, leg, "price_unavailable",
                "no entry price and no mark: the premium of the leg is unknown",
            ))
            continue

        remaining = year_fraction(valuation_at, contract.expiry)
        expired = remaining <= 0
        iv: float | None = None
        iv_source: str | None = None
        iv_used: float | None = None
        if not expired:
            if _usable(leg.iv) and leg.iv <= IV_SEARCH_MAX:  # type: ignore[operator]
                iv, iv_source = float(leg.iv), "broker"  # type: ignore[arg-type]
            elif _usable(leg.mark):
                iv = implied_volatility(
                    float(leg.mark), spot, float(contract.strike),  # type: ignore[arg-type]
                    year_fraction(as_of, contract.expiry), contract.right, rate=RATE,
                )
                if iv is None:
                    excluded.append(_excluded(
                        index, leg, "iv_not_implied",
                        f"no volatility reproduces the mark {leg.mark} at underlying price {spot} "
                        "(mark at or below intrinsic value, or out of the model's range)",
                    ))
                    continue
                iv_source = "implied"
            else:
                excluded.append(_excluded(
                    index, leg, "iv_unavailable",
                    "no implied volatility and no mark to derive it from",
                ))
                continue
            iv_used = max(IV_FLOOR, iv + shift)

        priced.append(PositionLeg(
            contract=contract, side=leg.side, quantity=leg.quantity, multiplier=leg.multiplier,
            iv=iv_used, entry_price=premium,
        ))
        per_leg.append({
            "index": index,
            "symbol": leg.symbol,
            "expires_at": expiry_instant(contract.expiry),
            "expired": expired,
            "time_to_expiry": remaining,
            "iv": iv,
            "iv_source": iv_source,
            "iv_used": iv_used,
            "model_price": black76_price(spot, float(contract.strike), remaining, iv_used or 0.0, contract.right, rate=RATE),
            "premium": premium,
            "premium_source": premium_source,
        })

    values: list[float] = []
    pnl: float | None = None
    net_greeks: dict[str, float] | None = None
    if priced:
        values = position_pnl(priced, request.underlying_prices, valuation_at, rate=RATE)
        pnl = position_pnl(priced, [spot], valuation_at, rate=RATE)[0]
        greeks = position_greeks(priced, spot, valuation_at, rate=RATE)
        net_greeks = {"delta": greeks.delta, "gamma": greeks.gamma, "theta": greeks.theta, "vega": greeks.vega}

    return {
        "model": MODEL_BLACK76,
        "underlying": underlyings[0],
        "underlying_price": spot,
        "valuation_at": valuation_at,
        "as_of": as_of,
        "nearest_expiry_at": min(expiry_instant(contract.expiry) for contract in contracts),
        "iv_shift": request.iv_shift,
        "rate": RATE,
        "underlying_prices": list(request.underlying_prices),
        "values": values,
        "pnl": pnl,
        "net_greeks": net_greeks,
        "per_leg": per_leg,
        "excluded": excluded,
    }
