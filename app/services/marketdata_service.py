"""
Market Data Service - OHLCV data provider with indicators.

Loads historical bar data from parquet files and computes technical indicators.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

# Path to market data directory (configurable via MARKETDATA_DIR env var)
EDGEWALKER_DATA_DIR = Path(settings.MARKETDATA_DIR)


class MarketDataError(Exception):
    """Error loading or processing market data."""
    pass


import re

def _extract_date_range_from_filename(filename: str) -> tuple[date | None, date | None]:
    """Extract date range from filename patterns like:
    - QQQ_STK_SMART_USD_1min_2024-01-01_2025-12-31.parquet
    - QQQ_5m_2025-12-01_2025-12-10.parquet
    
    Returns (start_date, end_date) or (None, None) if not found.
    """
    # Pattern to find YYYY-MM-DD dates
    date_pattern = r'(\d{4}-\d{2}-\d{2})'
    dates = re.findall(date_pattern, filename)
    
    if len(dates) >= 2:
        try:
            start = date.fromisoformat(dates[0])
            end = date.fromisoformat(dates[1])
            return start, end
        except ValueError:
            return None, None
    return None, None


def _find_parquet_file(
    symbol: str, 
    timeframe: str = "1min",
    start_date: date | None = None,
    end_date: date | None = None,
) -> Path:
    """Find the parquet file for a given symbol, timeframe, and date range.
    
    Selection priority:
    1. Files with matching timeframe AND containing requested date range
    2. Files with matching timeframe (largest by size)
    3. Files with finer granularity (1min) that can be resampled up
    4. Fallback to any matching symbol file
    
    Args:
        symbol: Trading symbol (e.g., 'QQQ')
        timeframe: Target timeframe (1min, 5min, 15min, 1h, 1d)
        start_date: Start date of requested data
        end_date: End date of requested data
    """
    symbol_upper = symbol.upper()
    
    # Normalize timeframe patterns to search for
    tf_patterns = {
        "1min": ["_1min", "_1m_", "_1m."],
        "1m": ["_1min", "_1m_", "_1m."],
        "5min": ["_5min", "_5m_", "_5m.", "_5mins"],
        "5m": ["_5min", "_5m_", "_5m.", "_5mins"],
        "15min": ["_15min", "_15m_", "_15m."],
        "15m": ["_15min", "_15m_", "_15m."],
        "30min": ["_30min", "_30m_", "_30m."],
        "30m": ["_30min", "_30m_", "_30m."],
        "1h": ["_60min", "_60m", "_1h_", "_1h."],
        "1hour": ["_60min", "_60m", "_1h_", "_1h."],
        "1d": ["_1d_", "_1d.", "_daily"],
        "1day": ["_1d_", "_1d.", "_daily"],
        "daily": ["_1d_", "_1d.", "_daily"],
    }
    
    # Get all matching symbol files
    all_matches = list(EDGEWALKER_DATA_DIR.glob(f"{symbol_upper}*.parquet"))
    if not all_matches:
        raise MarketDataError(f"No parquet file found for symbol '{symbol}' in {EDGEWALKER_DATA_DIR}")
    
    # Try to find files matching the requested timeframe
    patterns = tf_patterns.get(timeframe.lower(), [])
    tf_matches = []
    for path in all_matches:
        name_lower = path.name.lower()
        if any(pat in name_lower for pat in patterns):
            tf_matches.append(path)
    
    def _file_covers_date_range(path: Path, check_actual_content: bool = False) -> bool:
        """Check if file's date range covers the requested range.
        
        Args:
            path: Path to the parquet file
            check_actual_content: If True and filename doesn't have date info,
                                  read the file to check actual date range
        """
        if start_date is None and end_date is None:
            return True  # No date filter, any file is ok
        
        file_start, file_end = _extract_date_range_from_filename(path.name)
        
        # If can't determine from filename and check_actual_content is True,
        # read the file to get actual date range
        if (file_start is None or file_end is None) and check_actual_content:
            try:
                df = pd.read_parquet(path, columns=["ts"])
                df["ts"] = pd.to_datetime(df["ts"], utc=True)
                file_start = df["ts"].min().date()
                file_end = df["ts"].max().date()
                logger.debug(f"Read actual date range from {path.name}: {file_start} to {file_end}")
            except Exception as e:
                logger.warning(f"Could not read date range from {path.name}: {e}")
                return True  # Can't determine, assume it might work
        
        if file_start is None or file_end is None:
            return True  # Can't determine, assume it might work
        
        # Check if file range overlaps with requested range
        req_start = start_date or date(1900, 1, 1)
        req_end = end_date or date(2100, 12, 31)
        
        # File covers range if its range overlaps with requested range
        return file_start <= req_end and file_end >= req_start
    
    if tf_matches:
        # Filter by date range coverage
        date_filtered = [p for p in tf_matches if _file_covers_date_range(p)]
        
        if date_filtered:
            # Sort by:
            # 1. Files whose end date is closest to/past the requested end date (more recent data)
            # 2. Then by file size (more data is better)
            def score_file(p: Path) -> tuple:
                file_start, file_end = _extract_date_range_from_filename(p.name)
                # Higher end date = higher score
                end_score = file_end.toordinal() if file_end else 0
                size_score = p.stat().st_size
                return (end_score, size_score)
            
            best = max(date_filtered, key=score_file)
            logger.info(f"Selected file by timeframe + date range: {best.name}")
            return best
        else:
            # No date match, use largest by size
            best = max(tf_matches, key=lambda p: p.stat().st_size)
            logger.info(f"Selected file by timeframe (no date match): {best.name}")
            return best
    
    # Fallback: prefer 1min files (can be resampled) over coarser files
    one_min_patterns = ["_1min", "_1m_", "_1m."]
    one_min_matches = [p for p in all_matches if any(pat in p.name.lower() for pat in one_min_patterns)]
    
    if one_min_matches:
        # Also filter by date range (from filename)
        date_filtered = [p for p in one_min_matches if _file_covers_date_range(p)]
        
        if date_filtered:
            # Check if the best candidate actually has data for the end date
            # If not, we'll try other files
            def score_file_1min(p: Path) -> tuple:
                file_start, file_end = _extract_date_range_from_filename(p.name)
                end_score = file_end.toordinal() if file_end else 0
                size_score = p.stat().st_size
                return (end_score, size_score)
            
            best_1min = max(date_filtered, key=score_file_1min)
            best_start, best_end = _extract_date_range_from_filename(best_1min.name)
            
            # Check if this file actually covers the requested end date
            if end_date and best_end and best_end >= end_date:
                logger.info(f"Selected 1min file for resampling: {best_1min.name}")
                return best_1min
            else:
                # The 1min files don't cover the requested end date
                # Check if other files (without timeframe pattern) might have newer data
                logger.info(f"Best 1min file {best_1min.name} ends at {best_end}, need data until {end_date}")
        else:
            best_1min = max(one_min_matches, key=lambda p: p.stat().st_size)
    else:
        best_1min = None
    
    # Check remaining files (without explicit timeframe pattern) for actual date coverage
    # This handles files like "QQQ_live.parquet" that don't have date/timeframe in name
    all_tf_patterns = ["_1min", "_1m_", "_1m.", "_5min", "_5m_", "_5m.", "_15min", "_15m", 
                       "_30min", "_30m", "_60min", "_1h", "_1d", "_daily"]
    remaining_files = [p for p in all_matches 
                       if not any(pat in p.name.lower() for pat in all_tf_patterns)]
    if remaining_files and (start_date or end_date):
        # Check actual content for date coverage
        covering_files = [p for p in remaining_files if _file_covers_date_range(p, check_actual_content=True)]
        if covering_files:
            best = max(covering_files, key=lambda p: p.stat().st_size)
            logger.info(f"Selected file by actual content date range: {best.name}")
            return best
    
    # If we had a 1min candidate, use it as final fallback
    if best_1min:
        logger.info(f"Falling back to 1min file: {best_1min.name}")
        return best_1min
    
    # Final fallback: largest file (most data)
    best = max(all_matches, key=lambda p: p.stat().st_size)
    logger.warning(f"No timeframe match found, using largest file: {best.name}")
    return best


def load_ohlcv(
    symbol: str,
    start_date: date | None = None,
    end_date: date | None = None,
    timeframe: str = "1min",
) -> pd.DataFrame:
    """Load OHLCV data from parquet file.
    
    Args:
        symbol: Trading symbol (e.g., 'QQQ', 'SPY')
        start_date: Start date filter (inclusive)
        end_date: End date filter (inclusive)
        timeframe: Target timeframe for resampling (1min, 5min, 15min, 1h, 1d)
    
    Returns:
        DataFrame with columns: ts, open, high, low, close, volume
    """
    parquet_path = _find_parquet_file(symbol, timeframe, start_date, end_date)
    logger.info(f"Loading OHLCV from {parquet_path} for timeframe {timeframe}")
    
    df = pd.read_parquet(parquet_path)
    
    # Normalize column names
    df.columns = [c.lower() for c in df.columns]
    
    # Ensure ts column exists and is datetime
    if "ts" not in df.columns:
        if "date" in df.columns:
            df = df.rename(columns={"date": "ts"})
        elif "datetime" in df.columns:
            df = df.rename(columns={"datetime": "ts"})
        else:
            raise MarketDataError(f"No timestamp column found in {parquet_path}")
    
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts")
    
    # Filter by date range
    if start_date:
        start_dt = pd.Timestamp(start_date, tz="UTC")
        df = df[df["ts"] >= start_dt]
    if end_date:
        # Include full end date
        end_dt = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
        df = df[df["ts"] < end_dt]
    
    # Resample if needed
    df = _resample_ohlcv(df, timeframe)
    
    # Ensure required columns
    required = ["ts", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise MarketDataError(f"Missing columns: {missing}")
    
    return df[required].reset_index(drop=True)


def _resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLCV data to target timeframe."""
    # Map timeframe strings to pandas resample rules
    tf_map = {
        "1min": "1min",
        "1m": "1min",
        "5min": "5min",
        "5m": "5min",
        "15min": "15min",
        "15m": "15min",
        "30min": "30min",
        "30m": "30min",
        "1h": "1h",
        "1hour": "1h",
        "1d": "1D",
        "1day": "1D",
        "daily": "1D",
    }
    
    rule = tf_map.get(timeframe.lower())
    if not rule:
        logger.warning(f"Unknown timeframe '{timeframe}', returning original data")
        return df
    
    # If data is already at target resolution or coarser, return as-is
    if len(df) < 2:
        return df
    
    # Set index for resampling
    df = df.set_index("ts")
    
    # Resample OHLCV
    resampled = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    
    return resampled.reset_index()


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

def compute_indicators(
    df: pd.DataFrame,
    indicators: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Compute technical indicators on OHLCV data.
    
    Args:
        df: DataFrame with columns: ts, open, high, low, close, volume
        indicators: List of indicator specs (e.g., ['sma_20', 'ema_50', 'rsi_14', 'bbands_20_2'])
    
    Returns:
        Dictionary mapping indicator name to list of {time, value} dicts
        For multi-line indicators (e.g., BBands), returns {time, upper, middle, lower}
    """
    result = {}
    
    for spec in indicators:
        try:
            name, values = _compute_single_indicator(df, spec)
            result[name] = values
        except Exception as e:
            logger.warning(f"Failed to compute indicator '{spec}': {e}")
            continue
    
    return result


def _compute_single_indicator(
    df: pd.DataFrame,
    spec: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Compute a single indicator from spec string.
    
    Supported formats:
    - sma_20: Simple Moving Average with period 20
    - ema_50: Exponential Moving Average with period 50
    - rsi_14: Relative Strength Index with period 14
    - atr_14: Average True Range with period 14
    - bbands_20_2: Bollinger Bands with period 20 and 2 std devs
    - macd_12_26_9: MACD with fast=12, slow=26, signal=9
    - wma_20: Weighted Moving Average with period 20
    - dema_20: Double EMA with period 20
    - tema_20: Triple EMA with period 20
    - stoch_14_3_3: Stochastic with fastk=14, slowk=3, slowd=3
    - adx_14: Average Directional Index with period 14
    - cci_20: Commodity Channel Index with period 20
    - vwap: Volume Weighted Average Price (daily)
    
    Also supports talib.XXX format for compatibility with edgewalker:
    - talib.ema_20: Same as ema_20
    """
    # Handle talib.XXX format
    if spec.lower().startswith("talib."):
        spec = spec[6:]  # Remove "talib." prefix
    
    parts = spec.lower().split("_")
    indicator_type = parts[0]
    
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    timestamps = df["ts"].values
    
    if indicator_type == "sma":
        period = int(parts[1]) if len(parts) > 1 else 20
        values = _sma(close, period)
        return spec, _to_line_series(timestamps, values)
    
    elif indicator_type == "ema":
        period = int(parts[1]) if len(parts) > 1 else 20
        values = _ema(close, period)
        return spec, _to_line_series(timestamps, values)
    
    elif indicator_type == "rsi":
        period = int(parts[1]) if len(parts) > 1 else 14
        values = _rsi(close, period)
        return spec, _to_line_series(timestamps, values)
    
    elif indicator_type == "atr":
        period = int(parts[1]) if len(parts) > 1 else 14
        values = _atr(high, low, close, period)
        return spec, _to_line_series(timestamps, values)
    
    elif indicator_type == "wma":
        period = int(parts[1]) if len(parts) > 1 else 20
        values = _wma(close, period)
        return spec, _to_line_series(timestamps, values)
    
    elif indicator_type == "dema":
        period = int(parts[1]) if len(parts) > 1 else 20
        values = _dema(close, period)
        return spec, _to_line_series(timestamps, values)
    
    elif indicator_type == "tema":
        period = int(parts[1]) if len(parts) > 1 else 20
        values = _tema(close, period)
        return spec, _to_line_series(timestamps, values)
    
    elif indicator_type == "bbands":
        period = int(parts[1]) if len(parts) > 1 else 20
        std_dev = float(parts[2]) if len(parts) > 2 else 2.0
        upper, middle, lower = _bbands(close, period, std_dev)
        return spec, _to_bands_series(timestamps, upper, middle, lower)
    
    elif indicator_type == "macd":
        fast = int(parts[1]) if len(parts) > 1 else 12
        slow = int(parts[2]) if len(parts) > 2 else 26
        signal = int(parts[3]) if len(parts) > 3 else 9
        macd_line, signal_line, histogram = _macd(close, fast, slow, signal)
        return spec, _to_macd_series(timestamps, macd_line, signal_line, histogram)
    
    elif indicator_type == "stoch":
        fastk = int(parts[1]) if len(parts) > 1 else 14
        slowk = int(parts[2]) if len(parts) > 2 else 3
        slowd = int(parts[3]) if len(parts) > 3 else 3
        k, d = _stoch(high, low, close, fastk, slowk, slowd)
        return spec, _to_stoch_series(timestamps, k, d)
    
    elif indicator_type == "adx":
        period = int(parts[1]) if len(parts) > 1 else 14
        values = _adx(high, low, close, period)
        return spec, _to_line_series(timestamps, values)
    
    elif indicator_type == "cci":
        period = int(parts[1]) if len(parts) > 1 else 20
        values = _cci(high, low, close, period)
        return spec, _to_line_series(timestamps, values)
    
    elif indicator_type == "vwap":
        values = _vwap(df)
        return spec, _to_line_series(timestamps, values)
    
    else:
        raise ValueError(f"Unknown indicator type: {indicator_type}")


def _to_line_series(timestamps: np.ndarray, values: np.ndarray) -> list[dict[str, Any]]:
    """Convert to lightweight-charts line series format."""
    result = []
    for ts, val in zip(timestamps, values):
        if pd.notna(val):
            result.append({
                "time": int(pd.Timestamp(ts).timestamp()),
                "value": float(val),
            })
    return result


def _to_bands_series(
    timestamps: np.ndarray,
    upper: np.ndarray,
    middle: np.ndarray,
    lower: np.ndarray,
) -> list[dict[str, Any]]:
    """Convert to bands series format (for Bollinger Bands, etc.)."""
    result = []
    for ts, u, m, l in zip(timestamps, upper, middle, lower):
        if pd.notna(m):
            result.append({
                "time": int(pd.Timestamp(ts).timestamp()),
                "upper": float(u) if pd.notna(u) else None,
                "middle": float(m),
                "lower": float(l) if pd.notna(l) else None,
            })
    return result


def _to_macd_series(
    timestamps: np.ndarray,
    macd_line: np.ndarray,
    signal_line: np.ndarray,
    histogram: np.ndarray,
) -> list[dict[str, Any]]:
    """Convert to MACD series format."""
    result = []
    for ts, m, s, h in zip(timestamps, macd_line, signal_line, histogram):
        if pd.notna(m):
            result.append({
                "time": int(pd.Timestamp(ts).timestamp()),
                "macd": float(m),
                "signal": float(s) if pd.notna(s) else None,
                "histogram": float(h) if pd.notna(h) else None,
            })
    return result


def _to_stoch_series(
    timestamps: np.ndarray,
    slowk: np.ndarray,
    slowd: np.ndarray,
) -> list[dict[str, Any]]:
    """Convert to Stochastic series format."""
    result = []
    for ts, k, d in zip(timestamps, slowk, slowd):
        if pd.notna(k):
            result.append({
                "time": int(pd.Timestamp(ts).timestamp()),
                "k": float(k),
                "d": float(d) if pd.notna(d) else None,
            })
    return result


# =============================================================================
# INDICATOR CALCULATIONS (using TA-Lib for consistency with edgewalker)
# =============================================================================

import talib


def _sma(data: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average using TA-Lib."""
    return talib.SMA(data.astype(float), timeperiod=period)


def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average using TA-Lib."""
    return talib.EMA(data.astype(float), timeperiod=period)


def _rsi(data: np.ndarray, period: int) -> np.ndarray:
    """Relative Strength Index using TA-Lib."""
    return talib.RSI(data.astype(float), timeperiod=period)


def _bbands(
    data: np.ndarray,
    period: int,
    std_dev: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands using TA-Lib."""
    upper, middle, lower = talib.BBANDS(
        data.astype(float),
        timeperiod=period,
        nbdevup=std_dev,
        nbdevdn=std_dev,
        matype=0  # SMA
    )
    return upper, middle, lower


def _macd(
    data: np.ndarray,
    fast: int,
    slow: int,
    signal: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD using TA-Lib."""
    macd_line, signal_line, histogram = talib.MACD(
        data.astype(float),
        fastperiod=fast,
        slowperiod=slow,
        signalperiod=signal
    )
    return macd_line, signal_line, histogram


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Average True Range using TA-Lib."""
    return talib.ATR(
        high.astype(float),
        low.astype(float),
        close.astype(float),
        timeperiod=period
    )


def _wma(data: np.ndarray, period: int) -> np.ndarray:
    """Weighted Moving Average using TA-Lib."""
    return talib.WMA(data.astype(float), timeperiod=period)


def _dema(data: np.ndarray, period: int) -> np.ndarray:
    """Double Exponential Moving Average using TA-Lib."""
    return talib.DEMA(data.astype(float), timeperiod=period)


def _tema(data: np.ndarray, period: int) -> np.ndarray:
    """Triple Exponential Moving Average using TA-Lib."""
    return talib.TEMA(data.astype(float), timeperiod=period)


def _stoch(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
           fastk_period: int, slowk_period: int, slowd_period: int) -> tuple[np.ndarray, np.ndarray]:
    """Stochastic Oscillator using TA-Lib."""
    slowk, slowd = talib.STOCH(
        high.astype(float),
        low.astype(float),
        close.astype(float),
        fastk_period=fastk_period,
        slowk_period=slowk_period,
        slowk_matype=0,
        slowd_period=slowd_period,
        slowd_matype=0
    )
    return slowk, slowd


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Average Directional Index using TA-Lib."""
    return talib.ADX(
        high.astype(float),
        low.astype(float),
        close.astype(float),
        timeperiod=period
    )


def _cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Commodity Channel Index using TA-Lib."""
    return talib.CCI(
        high.astype(float),
        low.astype(float),
        close.astype(float),
        timeperiod=period
    )


def _vwap(df: pd.DataFrame) -> np.ndarray:
    """Volume Weighted Average Price (resets daily)."""
    result = np.full(len(df), np.nan)
    
    typical_price = (df["high"].values + df["low"].values + df["close"].values) / 3
    volume = df["volume"].values
    
    # Group by date and calculate VWAP
    df_temp = df.copy()
    df_temp["date"] = pd.to_datetime(df_temp["ts"]).dt.date
    df_temp["tp"] = typical_price
    df_temp["vol"] = volume
    
    for date_val, group in df_temp.groupby("date"):
        indices = group.index.values
        tp = group["tp"].values
        vol = group["vol"].values
        
        cum_tp_vol = np.cumsum(tp * vol)
        cum_vol = np.cumsum(vol)
        
        vwap = np.where(cum_vol > 0, cum_tp_vol / cum_vol, np.nan)
        result[indices] = vwap
    
    return result


# =============================================================================
# HIGH-LEVEL API
# =============================================================================

def get_ohlc_history(
    symbol: str,
    start_date: date | None = None,
    end_date: date | None = None,
    timeframe: str = "1min",
    indicators: list[dict[str, Any]] | list[str] | None = None,
) -> dict[str, Any]:
    """Get OHLC history with optional indicators from cached files.
    
    Args:
        symbol: Trading symbol
        start_date: Start date filter
        end_date: End date filter
        timeframe: Bar timeframe (1min, 5min, 15min, 1h, 1d)
        indicators: List of indicator configs or legacy string specs
    
    Returns:
        {
            "symbol": "QQQ",
            "timeframe": "5min",
            "candles": [{"time": 1234567890, "open": 100, ...}, ...],
            "indicators": {"SMA_20": [...], ...}
        }
    """
    df = load_ohlcv(symbol, start_date, end_date, timeframe)
    
    # Convert to lightweight-charts candlestick format
    candles = []
    for _, row in df.iterrows():
        candles.append({
            "time": int(pd.Timestamp(row["ts"]).timestamp()),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]) if pd.notna(row["volume"]) else 0,
        })
    
    # Compute indicators if requested
    computed_indicators = {}
    if indicators:
        computed_indicators = compute_indicators_dynamic(df, indicators)
    
    return {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "count": len(candles),
        "candles": candles,
        "indicators": computed_indicators,
    }


def get_ohlc_history_with_fetch(
    symbol: str,
    source: str,
    asset_type: str,
    start_date: date | None = None,
    end_date: date | None = None,
    timeframe: str = "5min",
    indicators: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fetch fresh OHLC data from source, then return with indicators.
    
    Uses the edgewalker library's fetch function to download data,
    then returns in the same format as get_ohlc_history.
    
    Args:
        symbol: Trading symbol
        source: Data source ('yahoo' or 'ibkr')
        asset_type: Asset type ('stock', 'futures', 'index', 'etf')
        start_date: Start date
        end_date: End date  
        timeframe: Bar timeframe
        indicators: List of indicator configs
    
    Returns:
        Same format as get_ohlc_history
    """
    import sys
    import os
    
    # Add edgewalker to path if needed
    edgewalker_path = os.environ.get("EDGEWALKER_PATH", "/home/flavio/playground/edgewalker")
    if edgewalker_path not in sys.path:
        sys.path.insert(0, edgewalker_path)
    
    try:
        from edgewalker.marketdata.fetch import FetchRequest, fetch as edgewalker_fetch
        
        # Map timeframe to interval format
        interval_map = {
            "1min": "1m",
            "5min": "5m", 
            "15min": "15m",
            "30min": "30m",
            "1h": "60m",
            "1d": "1d",
        }
        interval = interval_map.get(timeframe.lower(), timeframe)
        
        # Determine output path based on source and asset type
        output_dir = EDGEWALKER_DATA_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build filename with metadata
        start_str = start_date.isoformat() if start_date else "auto"
        end_str = end_date.isoformat() if end_date else "now"
        filename = f"{symbol.upper()}_{interval}_{start_str}_{end_str}.parquet"
        output_path = str(output_dir / filename)
        
        # Create fetch request
        req = FetchRequest(
            source=source,
            symbol=symbol,
            start=start_date.isoformat() if start_date else None,
            end=end_date.isoformat() if end_date else None,
            interval=interval,
            output=output_path,
            update=True,  # Allow incremental updates
        )
        
        # Execute fetch
        logger.info(f"Fetching {symbol} from {source} ({start_date} to {end_date})")
        result = edgewalker_fetch(req)
        
        if not result.success:
            raise MarketDataError(f"Fetch failed: {result.error}")
        
        logger.info(f"Fetched {result.rows_fetched} rows, total {result.rows_total}")
        
        # Now load the data and compute indicators
        return get_ohlc_history(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            indicators=indicators,
        )
        
    except ImportError as e:
        logger.error(f"Could not import edgewalker: {e}")
        raise MarketDataError(
            f"Fetching requires edgewalker library. Error: {e}"
        )
    except Exception as e:
        logger.exception(f"Error fetching {symbol} from {source}")
        raise MarketDataError(f"Fetch error: {e}")


def compute_indicators_dynamic(
    df: pd.DataFrame,
    indicators: list[dict[str, Any]] | list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Compute technical indicators dynamically using indicator registry.
    
    Supports both new format (list of dicts) and legacy format (list of strings).
    
    Args:
        df: DataFrame with ts, open, high, low, close, volume
        indicators: List of {"type": str, "params": dict} or legacy strings
    
    Returns:
        Dict mapping indicator name to time series data
    """
    from app.services.indicator_registry import compute_indicator
    
    result = {}
    timestamps = df["ts"].values
    
    # Prepare data dict for indicator computation
    data = {
        "open": df["open"].values,
        "high": df["high"].values,
        "low": df["low"].values,
        "close": df["close"].values,
        "volume": df["volume"].values,
    }
    
    for spec in indicators:
        try:
            # Handle legacy string format
            if isinstance(spec, str):
                name, values = _compute_single_indicator(df, spec)
                result[name] = values
                continue
            
            # New format: {"type": "SMA", "params": {"timeperiod": 20}}
            ind_type = spec.get("type", "")
            params = spec.get("params", {})
            custom_name = spec.get("name")
            
            # Generate name from type and params
            if custom_name:
                name = custom_name
            else:
                param_str = "_".join(str(v) for v in params.values())
                name = f"{ind_type}_{param_str}" if param_str else ind_type
            
            # Compute indicator
            values = compute_indicator(ind_type, params, data)
            
            # Format for output
            if isinstance(values, dict):
                # Multi-output indicator (e.g., MACD, BBANDS)
                formatted = _format_multi_output(timestamps, values)
            else:
                # Single-output indicator
                formatted = _to_line_series(timestamps, np.array(values))
            
            result[name] = formatted
            
        except Exception as e:
            logger.warning(f"Failed to compute indicator {spec}: {e}")
            continue
    
    return result


def _format_multi_output(
    timestamps: np.ndarray,
    values: dict[str, list],
) -> list[dict[str, Any]]:
    """Format multi-output indicator for API response."""
    # Get length from first output
    first_key = next(iter(values.keys()))
    length = len(values[first_key])
    
    result = []
    for i in range(length):
        point = {"time": int(pd.Timestamp(timestamps[i]).timestamp())}
        has_value = False
        for key, arr in values.items():
            val = arr[i] if i < len(arr) else None
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                point[key] = float(val)
                has_value = True
            else:
                point[key] = None
        if has_value:
            result.append(point)
    
    return result
