"""Timeframe strings ("1m", "4h", "1d") → seconds, and the backtest simulation driver choice."""
from __future__ import annotations

import re

from fastapi import HTTPException, status

TF_SECONDS: dict[str, int] = {
    "1s": 1, "5s": 5, "10s": 10, "15s": 15, "30s": 30,
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "4h": 14400, "1d": 86400,
}
_DYNAMIC_RE = re.compile(r"^(\d+)(s|m|h|d)$")
_UNIT = {"s": 1, "m": 60, "h": 3600, "d": 86400}

# Clocks a backtest can replay on. 1m is the floor every historical adapter
# serves (cTrader M1, IBKR "1 min"); sub-minute needs IBKR-only work.
SIMULATION_TIMEFRAMES: tuple[str, ...] = ("1m", "5m", "15m", "30m", "1h", "4h", "1d")


def parse_tf_seconds(tf: str) -> int:
    tf = str(tf or "").strip()
    if tf in TF_SECONDS:
        return TF_SECONDS[tf]
    m = _DYNAMIC_RE.match(tf)
    if m:
        return int(m.group(1)) * _UNIT[m.group(2)]
    raise ValueError(f"Unsupported timeframe: {tf!r}")


def normalize_simulation_timeframe(requested: str | None, chart_timeframe: str | None) -> str | None:
    """The simulation driver to store for a backtest, or None.

    None/empty means "the primary chart is the clock". A driver equal to the
    chart timeframe is the same thing and is stored as None; a coarser one is
    refused (the clock can never be slower than the bars the rules read).
    """
    tf = str(requested or "").strip()
    if not tf:
        return None
    if tf not in SIMULATION_TIMEFRAMES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"simulation_timeframe must be one of {', '.join(SIMULATION_TIMEFRAMES)}",
        )
    try:
        chart_s = parse_tf_seconds(chart_timeframe or "")
    except ValueError:
        return tf
    driver_s = parse_tf_seconds(tf)
    if driver_s == chart_s:
        return None
    if driver_s > chart_s:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"simulation_timeframe {tf} is coarser than the strategy timeframe {chart_timeframe}",
        )
    return tf
