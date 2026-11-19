"""Schemas of ``POST /marketdata/options/payoff`` (options F5).

The chain/quotes/readiness reads have no schema here: their body is the
gateway's, forwarded untouched.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from edgewalker_platform.options import OptionSymbolError, OptionContract

# Bounds of the request, also read by the service and the tests.
MAX_LEGS = 50
MIN_GRID_POINTS = 2
MAX_GRID_POINTS = 401
MAX_IV_SHIFT_POINTS = 100.0


class OptionPayoffLeg(BaseModel):
    model_config = ConfigDict(extra="ignore")

    symbol: str = Field(..., max_length=32, description="Canonical option symbol, e.g. MNQZ6-261218-21500-C")
    side: Literal["buy", "sell"]
    quantity: float = Field(..., gt=0, allow_inf_nan=False)
    price: Optional[float] = Field(
        None, ge=0, allow_inf_nan=False,
        description="Entry premium per unit (paid on buy, received on sell). Null: the mark is used instead",
    )
    multiplier: float = Field(..., gt=0, allow_inf_nan=False, description="Contract multiplier")
    iv: Optional[float] = Field(
        None, allow_inf_nan=False,
        description="Implied volatility as a fraction (0.18 = 18 %), as quoted by the broker. Null: implied from the mark",
    )
    mark: Optional[float] = Field(None, allow_inf_nan=False, description="Current premium per unit")

    @field_validator("symbol")
    @classmethod
    def _canonical(cls, value: str) -> str:
        try:
            return OptionContract.from_symbol(value).symbol
        except OptionSymbolError as exc:
            raise ValueError(f"not a canonical option symbol: {value!r}") from exc


class OptionPayoffRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    legs: list[OptionPayoffLeg] = Field(..., min_length=1, max_length=MAX_LEGS)
    underlying_price: float = Field(
        ..., gt=0, allow_inf_nan=False,
        description="Current underlying price: where the net greeks are computed and IVs are implied from marks",
    )
    underlying_prices: list[float] = Field(
        ..., min_length=MIN_GRID_POINTS, max_length=MAX_GRID_POINTS,
        description="Underlying prices the P&L is computed at (the x grid)",
    )
    valuation_at: Optional[datetime] = Field(
        None, description="Instant the set is valued at (default: now). Not in the past",
    )
    iv_shift: float = Field(
        0.0, ge=-MAX_IV_SHIFT_POINTS, le=MAX_IV_SHIFT_POINTS, allow_inf_nan=False,
        description="Parallel shift of every IV, in volatility points (1 = +1 %)",
    )
    as_of: Optional[datetime] = Field(
        None, description="Instant the marks and the underlying price refer to (default: now); used to imply IVs",
    )

    @field_validator("underlying_prices")
    @classmethod
    def _positive_grid(cls, values: list[float]) -> list[float]:
        for value in values:
            if not (value > 0) or value == float("inf"):
                raise ValueError("underlying_prices must be positive finite numbers")
        return values


class OptionPayoffGreeks(BaseModel):
    """Position greeks: already multiplied by side, quantity and multiplier."""

    delta: float
    gamma: float
    theta: float
    vega: float


class OptionPayoffLegResult(BaseModel):
    index: int
    symbol: str
    expires_at: datetime
    expired: bool
    time_to_expiry: float
    iv: Optional[float]
    iv_source: Optional[Literal["broker", "implied"]]
    iv_used: Optional[float]
    model_price: float
    premium: float
    premium_source: Literal["price", "mark"]


class OptionPayoffExcluded(BaseModel):
    index: int
    symbol: str
    reason: Literal["iv_unavailable", "iv_not_implied", "price_unavailable"]
    message: str


class OptionPayoffUnits(BaseModel):
    values: str = "currency"
    delta: str = "currency per 1 point of underlying"
    gamma: str = "delta change per 1 point of underlying"
    theta: str = "currency per calendar day"
    vega: str = "currency per volatility point (1%)"


class OptionPayoffResponse(BaseModel):
    model: Literal["black76"] = "black76"
    underlying: str
    underlying_price: float
    valuation_at: datetime
    as_of: datetime
    # Cut-off instant of the nearest expiry among ALL the legs of the request:
    # the upper bound of a "value at date" control.
    nearest_expiry_at: datetime
    iv_shift: float
    rate: float
    underlying_prices: list[float]
    # Empty / null when no leg could be valued (every leg is in ``excluded``).
    values: list[float]
    pnl: Optional[float]
    net_greeks: Optional[OptionPayoffGreeks]
    per_leg: list[OptionPayoffLegResult]
    excluded: list[OptionPayoffExcluded]
    units: OptionPayoffUnits = Field(default_factory=OptionPayoffUnits)
