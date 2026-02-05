"""
Indicator Registry Service - Dynamic listing of all available indicators.

Uses TA-Lib's abstract API to dynamically list all available indicators
with their parameters, input types, and output names.
Also includes custom indicators from edgewalker library.
"""
from __future__ import annotations

import logging
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import talib
from talib import abstract

logger = logging.getLogger(__name__)

# US Eastern timezone for trading session calculation
ET_TZ = ZoneInfo("America/New_York")

# CME futures session boundary: session starts at 18:00 ET
# Data at/after 18:00 ET belongs to the NEXT calendar day's session
CME_SESSION_START_HOUR = 18


def _compute_trading_session_date(ts: pd.Timestamp) -> pd.Timestamp:
    """Compute trading session date for a timestamp.
    
    CME futures have a daily session that starts at 18:00 ET the prior calendar day.
    For example:
    - Sunday 18:00 ET to Monday 17:00 ET = Monday's session
    - Monday 18:00 ET to Tuesday 17:00 ET = Tuesday's session
    
    This aligns pivot calculations with TradingView behavior.
    
    Args:
        ts: Timestamp (timezone-aware UTC or naive assumed UTC)
        
    Returns:
        Date representing the trading session
    """
    # Convert to ET
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts_et = ts.tz_convert(ET_TZ)
    
    # If at or after 18:00 ET, it belongs to the next calendar day's session
    if ts_et.hour >= CME_SESSION_START_HOUR:
        return (ts_et + pd.Timedelta(days=1)).normalize()
    else:
        return ts_et.normalize()


def _calculate_pivot_levels(h: float, l: float, c: float, o: float, kind: str) -> dict[str, float]:
    """Calculate pivot levels based on the specified method.
    
    Based on TradingView's Pivot Points Standard indicator formulas.
    
    Args:
        h: Previous period high
        l: Previous period low
        c: Previous period close
        o: Previous period open
        kind: Method type (traditional, fibonacci, woodie, classic, dm, camarilla)
    
    Returns:
        Dictionary with keys: p, r1, r2, r3, r4, r5, s1, s2, s3, s4, s5
        (not all levels are set for all kinds)
    """
    levels = {}
    
    if kind == "traditional" or kind == "classic":
        # Traditional (same as Classic in TradingView)
        p = (h + l + c) / 3
        levels["p"] = p
        levels["r1"] = 2 * p - l
        levels["s1"] = 2 * p - h
        levels["r2"] = p + (h - l)
        levels["s2"] = p - (h - l)
        levels["r3"] = p + 2 * (h - l)
        levels["s3"] = p - 2 * (h - l)
        levels["r4"] = p + 3 * (h - l)
        levels["s4"] = p - 3 * (h - l)
        if kind == "traditional":
            # Traditional has R5/S5
            levels["r5"] = p + 4 * (h - l)
            levels["s5"] = p - 4 * (h - l)
            
    elif kind == "fibonacci":
        p = (h + l + c) / 3
        diff = h - l
        levels["p"] = p
        levels["r1"] = p + 0.382 * diff
        levels["s1"] = p - 0.382 * diff
        levels["r2"] = p + 0.618 * diff
        levels["s2"] = p - 0.618 * diff
        levels["r3"] = p + 1.0 * diff
        levels["s3"] = p - 1.0 * diff
        
    elif kind == "woodie":
        p = (h + l + 2 * c) / 4
        levels["p"] = p
        levels["r1"] = 2 * p - l
        levels["s1"] = 2 * p - h
        levels["r2"] = p + (h - l)
        levels["s2"] = p - (h - l)
        levels["r3"] = h + 2 * (p - l)
        levels["s3"] = l - 2 * (h - p)
        levels["r4"] = levels["r3"] + (h - l)
        levels["s4"] = levels["s3"] - (h - l)
        
    elif kind == "dm":
        # Demark pivot points
        if c < o:
            x = h + 2 * l + c
        elif c > o:
            x = 2 * h + l + c
        else:
            x = h + l + 2 * c
        p = x / 4
        levels["p"] = p
        levels["r1"] = x / 2 - l
        levels["s1"] = x / 2 - h
        
    elif kind == "camarilla":
        diff = h - l
        levels["p"] = (h + l + c) / 3
        levels["r1"] = c + diff * 1.1 / 12
        levels["s1"] = c - diff * 1.1 / 12
        levels["r2"] = c + diff * 1.1 / 6
        levels["s2"] = c - diff * 1.1 / 6
        levels["r3"] = c + diff * 1.1 / 4
        levels["s3"] = c - diff * 1.1 / 4
        levels["r4"] = c + diff * 1.1 / 2
        levels["s4"] = c - diff * 1.1 / 2
        levels["r5"] = (h / l) * c
        levels["s5"] = c - (levels["r5"] - c)
    
    else:
        # Default to traditional
        p = (h + l + c) / 3
        levels["p"] = p
        levels["r1"] = 2 * p - l
        levels["s1"] = 2 * p - h
    
    return levels


# Custom indicators from edgewalker (not in TA-Lib)
CUSTOM_INDICATORS = {
    "vwap": {
        "name": "VWAP",
        "display_name": "Volume Weighted Average Price",
        "group": "Custom",
        "description": "Volume Weighted Average Price (resets daily)",
        "overlay": True,
        "inputs": {"ohlcv": ["open", "high", "low", "close", "volume"]},
        "parameters": {},
        "outputs": ["value"],
    },
    "pivot": {
        "name": "PIVOT",
        "display_name": "Pivot Points Standard",
        "group": "Custom",
        "description": "Pivot points calculated from previous day's HLC values (multi-output)",
        "overlay": True,
        "inputs": {"ohlc": ["open", "high", "low", "close"]},
        "parameters": {
            "kind": {
                "type": "string",
                "default": "traditional",
                "options": ["traditional", "fibonacci", "woodie", "classic", "dm", "camarilla"],
                "description": "Pivot point calculation method",
            },
        },
        "outputs": ["p", "r1", "r2", "r3", "r4", "r5", "s1", "s2", "s3", "s4", "s5"],
        "output_descriptions": {
            "p": "Pivot Point",
            "r1": "Resistance 1",
            "r2": "Resistance 2",
            "r3": "Resistance 3",
            "r4": "Resistance 4",
            "r5": "Resistance 5",
            "s1": "Support 1",
            "s2": "Support 2",
            "s3": "Support 3",
            "s4": "Support 4",
            "s5": "Support 5",
        },
    },
    "session_high": {
        "name": "SESSION_HIGH",
        "display_name": "Running Session High",
        "group": "Custom",
        "description": "Running high price within the session",
        "overlay": True,
        "inputs": {"high": ["high"]},
        "parameters": {},
        "outputs": ["value"],
    },
    "session_low": {
        "name": "SESSION_LOW",
        "display_name": "Running Session Low",
        "group": "Custom",
        "description": "Running low price within the session",
        "overlay": True,
        "inputs": {"low": ["low"]},
        "parameters": {},
        "outputs": ["value"],
    },
}


# Groups that should overlay on price chart (vs separate panel)
OVERLAY_GROUPS = {"Overlap Studies", "Price Transform", "Custom"}

# Map TA-Lib output types to chart-friendly flags
OUTPUT_TYPE_MAP = {
    "Line": "line",
    "Dashed Line": "dashed_line",
    "Histogram": "histogram",
    "Dotted Line": "dotted_line",
}

# Map TA-Lib output names to standardized frontend-friendly names
OUTPUT_NAME_MAP = {
    # BBANDS
    "upperband": "upper",
    "middleband": "middle",
    "lowerband": "lower",
    # MACD
    "macdsignal": "signal",
    "macdhist": "histogram",
    # STOCH - keep as-is (slowk, slowd)
}


def _get_talib_indicator_info(func_name: str) -> dict[str, Any] | None:
    """Get detailed info about a TA-Lib indicator using the abstract API."""
    try:
        func = abstract.Function(func_name)
        info = func.info
        
        # Parse input names
        inputs = {}
        for input_name, price_type in info.get("input_names", {}).items():
            if isinstance(price_type, str):
                inputs[input_name] = [price_type]
            elif isinstance(price_type, (list, tuple)):
                inputs[input_name] = list(price_type)
            else:
                inputs[input_name] = ["close"]
        
        # Parse parameters with their defaults
        parameters = {}
        for param_name, default_value in info.get("parameters", {}).items():
            param_info = {
                "type": "integer" if isinstance(default_value, int) else "number",
                "default": default_value,
            }
            # Add typical min/max constraints
            if "period" in param_name.lower():
                param_info["min"] = 1
                param_info["max"] = 1000
            parameters[param_name] = param_info
        
        # Parse outputs
        output_flags = info.get("output_flags", {})
        outputs = []
        output_types = {}
        for output_name, flags in output_flags.items():
            outputs.append(output_name)
            if flags:
                output_types[output_name] = OUTPUT_TYPE_MAP.get(flags[0], "line")
            else:
                output_types[output_name] = "line"
        
        if not outputs:
            outputs = info.get("output_names", ["value"])
        
        # Determine if this overlays on price
        group = info.get("group", "Other")
        overlay = group in OVERLAY_GROUPS
        
        return {
            "name": func_name.upper(),
            "display_name": info.get("display_name", func_name),
            "group": group,
            "description": info.get("display_name", func_name),
            "overlay": overlay,
            "inputs": inputs,
            "parameters": parameters,
            "outputs": outputs,
            "output_types": output_types,
        }
    except Exception as e:
        logger.warning(f"Failed to get info for TA-Lib function {func_name}: {e}")
        return None


def get_all_indicators() -> list[dict[str, Any]]:
    """Get information about all available indicators.
    
    Returns a list of indicator definitions including:
    - name: Technical name (e.g., 'SMA', 'MACD')
    - display_name: Human-readable name
    - group: Category (e.g., 'Overlap Studies', 'Momentum Indicators')
    - description: Brief description
    - overlay: Whether it overlays on price chart
    - inputs: Required input data (e.g., {'price': ['close']})
    - parameters: Configurable parameters with types and defaults
    - outputs: List of output names
    """
    indicators = []
    
    # Get all TA-Lib indicators by group
    groups = talib.get_function_groups()
    
    for group_name, func_names in groups.items():
        for func_name in func_names:
            info = _get_talib_indicator_info(func_name)
            if info:
                indicators.append(info)
    
    # Add custom indicators from edgewalker
    for key, info in CUSTOM_INDICATORS.items():
        indicators.append(info)
    
    # Sort by group, then by name
    indicators.sort(key=lambda x: (x.get("group", ""), x.get("name", "")))
    
    return indicators


def get_indicator_by_name(name: str) -> dict[str, Any] | None:
    """Get info for a specific indicator by name.
    
    Args:
        name: Indicator name (e.g., 'SMA', 'MACD', 'vwap')
    
    Returns:
        Indicator info dict or None if not found.
    """
    name_upper = name.upper()
    name_lower = name.lower()
    
    # Check custom indicators first
    if name_lower in CUSTOM_INDICATORS:
        return CUSTOM_INDICATORS[name_lower]
    
    # Try TA-Lib
    try:
        return _get_talib_indicator_info(name_upper)
    except Exception:
        return None


def get_indicator_groups() -> dict[str, list[str]]:
    """Get all indicator groups with their function names.
    
    Returns:
        Dictionary mapping group name to list of indicator names.
    """
    groups = dict(talib.get_function_groups())
    
    # Add custom indicators group
    groups["Custom"] = list(CUSTOM_INDICATORS.keys())
    
    return groups


def compute_indicator(
    indicator_type: str,
    params: dict[str, Any],
    data: dict[str, Any],
) -> dict[str, list[float]] | list[float]:
    """Compute an indicator on OHLCV data.
    
    This uses the same logic as edgewalker to ensure consistency.
    
    Args:
        indicator_type: Indicator type (e.g., 'SMA', 'MACD', 'vwap')
        params: Parameters dict (e.g., {'timeperiod': 20})
        data: OHLCV data dict with keys: open, high, low, close, volume
    
    Returns:
        Computed indicator values (single array or dict of arrays for multi-output)
    """
    import numpy as np
    
    # Normalize common param names: period -> timeperiod (ta-lib convention)
    params = dict(params)
    if "period" in params and "timeperiod" not in params:
        params["timeperiod"] = params.pop("period")
    
    # Normalize std_dev -> nbdevup/nbdevdn for BBANDS
    if "std_dev" in params:
        std_dev = params.pop("std_dev")
        if "nbdevup" not in params:
            params["nbdevup"] = std_dev
        if "nbdevdn" not in params:
            params["nbdevdn"] = std_dev
    
    type_lower = indicator_type.lower()
    type_upper = indicator_type.upper()
    
    # Handle custom indicators
    if type_lower == "vwap":
        typical_price = (np.array(data["high"]) + np.array(data["low"]) + np.array(data["close"])) / 3
        volume = np.array(data["volume"])
        cum_tp_vol = np.cumsum(typical_price * volume)
        cum_vol = np.cumsum(volume)
        return cum_tp_vol / np.where(cum_vol > 0, cum_vol, 1)
    
    if type_lower == "pivot":
        # Pivot Points Standard - multi-output, session-based calculation
        # Uses previous TRADING SESSION's H/L/C to calculate pivot levels for current session
        # Trading session: 18:00 ET (prior day) to 17:00 ET (current day)
        # This aligns with TradingView behavior for futures
        kind = params.get("kind", "traditional").lower()
        h = np.array(data["high"])
        l = np.array(data["low"])
        c = np.array(data["close"])
        o = np.array(data.get("open", c))
        n = len(c)
        
        # Initialize all output arrays
        output_keys = ["p", "r1", "r2", "r3", "r4", "r5", "s1", "s2", "s3", "s4", "s5"]
        result = {key: np.full(n, np.nan).tolist() for key in output_keys}
        
        # Get timestamps if available for session aggregation
        timestamps = data.get("timestamps")
        if timestamps is not None:
            # Convert to trading session dates (not calendar dates!)
            # This ensures Sunday evening data belongs to Monday's session
            ts_series = pd.to_datetime(timestamps, utc=True)
            
            # Compute trading session date for each timestamp
            session_dates = np.array([
                _compute_trading_session_date(ts).date() 
                for ts in ts_series
            ])
            unique_sessions = sorted(set(session_dates))
            
            # Build session H/L/C for each trading session
            session_hlc = {}
            for session_date in unique_sessions:
                mask = session_dates == session_date
                indices = np.where(mask)[0]
                if len(indices) > 0:
                    session_hlc[session_date] = {
                        "high": float(np.max(h[mask])),
                        "low": float(np.min(l[mask])),
                        "close": float(c[indices[-1]]),  # Last close of the session
                        "open": float(o[indices[0]]),     # First open of the session
                    }
            
            # Calculate pivots for each bar based on PREVIOUS SESSION's HLC
            prev_session = None
            prev_hlc = None
            
            for i in range(n):
                current_session = session_dates[i]
                
                # Find previous trading session's HLC (only update on session change)
                if current_session != prev_session:
                    prev_sessions = [s for s in unique_sessions if s < current_session]
                    if prev_sessions:
                        prev_trading_session = max(prev_sessions)
                        prev_hlc = session_hlc.get(prev_trading_session)
                    prev_session = current_session
                
                if prev_hlc is None:
                    continue
                
                # Calculate pivot levels from previous session's data
                levels = _calculate_pivot_levels(
                    prev_hlc["high"], prev_hlc["low"], prev_hlc["close"], prev_hlc["open"], kind
                )
                
                for key, value in levels.items():
                    if key in result:
                        result[key][i] = value
        else:
            # Fallback: bar-by-bar (for daily timeframe)
            for i in range(1, n):
                levels = _calculate_pivot_levels(h[i-1], l[i-1], c[i-1], o[i-1], kind)
                for key, value in levels.items():
                    if key in result:
                        result[key][i] = value
        
        return result
    
    if type_lower == "session_high":
        return np.maximum.accumulate(np.array(data["high"]))
    
    if type_lower == "session_low":
        return np.minimum.accumulate(np.array(data["low"]))
    
    # Handle TA-Lib indicators
    talib_fn = getattr(talib, type_upper, None)
    if talib_fn is None:
        raise ValueError(f"Unknown indicator type: {indicator_type}")
    
    # Get indicator info to understand inputs
    info = _get_talib_indicator_info(type_upper)
    if not info:
        raise ValueError(f"Could not get info for indicator: {indicator_type}")
    
    # Prepare data arrays
    close = np.array(data.get("close", []), dtype=float)
    high = np.array(data.get("high", []), dtype=float)
    low = np.array(data.get("low", []), dtype=float)
    open_ = np.array(data.get("open", []), dtype=float)
    volume = np.array(data.get("volume", []), dtype=float)
    
    # Try different argument signatures based on typical TA-Lib patterns
    signatures = [
        lambda: talib_fn(close, **params),
        lambda: talib_fn(high, low, close, **params),
        lambda: talib_fn(open_, high, low, close, **params),
        lambda: talib_fn(close, volume, **params),
        lambda: talib_fn(high, low, **params),
        lambda: talib_fn(high, **params),
        lambda: talib_fn(low, **params),
    ]
    
    last_error = None
    for sig in signatures:
        try:
            result = sig()
            # Multi-output indicators return tuple
            if isinstance(result, tuple):
                output_names = info.get("outputs", [f"output_{i}" for i in range(len(result))])
                # Apply output name mapping for frontend consistency
                mapped_names = [OUTPUT_NAME_MAP.get(name, name) for name in output_names]
                return {name: arr.tolist() for name, arr in zip(mapped_names, result)}
            return result.tolist() if hasattr(result, "tolist") else result
        except (TypeError, Exception) as e:
            last_error = e
            continue
    
    raise ValueError(f"Could not compute talib.{type_upper}: {last_error}")
