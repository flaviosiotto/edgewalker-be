"""
Market Data Service - OHLCV data provider with indicators.

Loads historical bar data from partitioned parquet datasets and computes technical indicators.

Partitioned dataset structure:
    data/<source>-ohlcv/<symbol>/<timeframe>/<date>/part-0000.parquet
    
Example:
    data/ibkr-ohlcv/QQQ/5m/2026-01-26/part-0000.parquet
    data/yahoo-ohlcv/SPY/5m/2026-01-26/part-0000.parquet
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
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


# =============================================================================
# PARTITIONED DATASET READER
# =============================================================================

def _normalize_timeframe(tf: str) -> str:
    """Normalize timeframe to canonical form for directory names."""
    tf_lower = tf.strip().lower().replace(" ", "")
    aliases = {
        "1min": "1m", "2min": "2m", "5min": "5m", "15min": "15m", "30min": "30m",
        "60min": "1h", "1hour": "1h", "1day": "1d",
        "5mins": "5m", "1 min": "1m", "5 mins": "5m",
    }
    return aliases.get(tf_lower, tf_lower)


def _is_monthly_timeframe(tf: str) -> bool:
    """Determine if timeframe uses monthly partitions (tf >= 1h)."""
    tf_lower = tf.strip().lower()
    if re.match(r"^\d+[hdw]$", tf_lower):
        return True
    if re.match(r"^\d+\s*(hour|day|week)", tf_lower):
        return True
    m = re.match(r"^(\d+)m$", tf_lower)
    if m and int(m.group(1)) >= 60:
        return True
    return False


def _iter_partition_values(start_date: date, end_date: date, monthly: bool) -> list[str]:
    """Generate partition values for a date range."""
    values = []
    if monthly:
        current = date(start_date.year, start_date.month, 1)
        while current <= end_date:
            values.append(current.strftime("%Y-%m"))
            if current.month == 12:
                current = date(current.year + 1, 1, 1)
            else:
                current = date(current.year, current.month + 1, 1)
    else:
        current = start_date
        while current <= end_date:
            values.append(current.isoformat())
            current += timedelta(days=1)
    return values


def _find_partitioned_data(
    symbol: str,
    timeframe: str,
    start_date: date | None = None,
    end_date: date | None = None,
    source: str | None = None,
) -> list[Path]:
    """Find partitioned parquet files for a symbol/timeframe/date range.
    
    Directory structure: data/<source>-ohlcv/<symbol>/<timeframe>/<date>/part-0000.parquet
    
    Args:
        symbol: Trading symbol (e.g., 'QQQ')
        timeframe: Target timeframe (1m, 5m, 15m, 1h, 1d)
        start_date: Start date filter
        end_date: End date filter
        source: Data source (ibkr, yahoo). If None, searches all sources.
        
    Returns:
        List of parquet file paths sorted by date
    """
    symbol_upper = symbol.upper()
    tf_canonical = _normalize_timeframe(timeframe)
    monthly = _is_monthly_timeframe(tf_canonical)
    
    # Find available sources
    sources_to_check = []
    if source:
        sources_to_check = [source.lower()]
    else:
        # Auto-detect: check ibkr first, then yahoo
        for src in ["ibkr", "yahoo"]:
            src_dir = EDGEWALKER_DATA_DIR / f"{src}-ohlcv"
            if src_dir.exists():
                sources_to_check.append(src)
    
    parquet_files = []
    
    for src in sources_to_check:
        tf_dir = EDGEWALKER_DATA_DIR / f"{src}-ohlcv" / symbol_upper / tf_canonical
        
        if not tf_dir.exists():
            continue
        
        # List partition directories
        for entry in tf_dir.iterdir():
            if not entry.is_dir():
                continue
            
            partition_value = entry.name
            
            # Validate partition format and filter by date
            if monthly:
                if not re.match(r"^\d{4}-\d{2}$", partition_value):
                    continue
                partition_date = date(int(partition_value[:4]), int(partition_value[5:7]), 1)
            else:
                if not re.match(r"^\d{4}-\d{2}-\d{2}$", partition_value):
                    continue
                try:
                    partition_date = date.fromisoformat(partition_value)
                except ValueError:
                    continue
            
            # Apply date filter
            if start_date and partition_date < start_date:
                if monthly:
                    # For monthly, check if month end is before start
                    if partition_date.month == 12:
                        month_end = date(partition_date.year + 1, 1, 1) - timedelta(days=1)
                    else:
                        month_end = date(partition_date.year, partition_date.month + 1, 1) - timedelta(days=1)
                    if month_end < start_date:
                        continue
                else:
                    continue
            
            if end_date and partition_date > end_date:
                continue
            
            # Find parquet file in partition
            parquet_file = entry / "part-0000.parquet"
            if parquet_file.exists():
                parquet_files.append(parquet_file)
        
        # If we found data from this source, use it
        if parquet_files:
            break
    
    return sorted(parquet_files)


def load_ohlcv_partitioned(
    symbol: str,
    start_date: date | None = None,
    end_date: date | None = None,
    timeframe: str = "5m",
    source: str | None = None,
) -> pd.DataFrame:
    """Load OHLCV data from partitioned parquet files.
    
    Args:
        symbol: Trading symbol (e.g., 'QQQ', 'SPY')
        start_date: Start date filter (inclusive)
        end_date: End date filter (inclusive)
        timeframe: Target timeframe (1m, 5m, 15m, 1h, 1d)
        source: Data source (ibkr, yahoo). If None, auto-detects.
    
    Returns:
        DataFrame with columns: ts, open, high, low, close, volume
    """
    parquet_files = _find_partitioned_data(symbol, timeframe, start_date, end_date, source)
    
    if not parquet_files:
        raise MarketDataError(
            f"No partitioned data found for symbol '{symbol}' timeframe '{timeframe}' "
            f"in {EDGEWALKER_DATA_DIR}. "
            f"Expected structure: <source>-ohlcv/{symbol.upper()}/{_normalize_timeframe(timeframe)}/<date>/part-0000.parquet"
        )
    
    logger.info(f"Loading {len(parquet_files)} partition files for {symbol}/{timeframe}")
    
    # Read and concatenate all partitions
    dfs = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to read {pf}: {e}")
            continue
    
    if not dfs:
        raise MarketDataError(f"Failed to read any partition files for {symbol}")
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Normalize column names
    df.columns = [c.lower() for c in df.columns]
    
    # Ensure ts column exists and is datetime
    if "ts" not in df.columns:
        if "date" in df.columns:
            df = df.rename(columns={"date": "ts"})
        elif "datetime" in df.columns:
            df = df.rename(columns={"datetime": "ts"})
        else:
            raise MarketDataError(f"No timestamp column found in partition files")
    
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").drop_duplicates(subset=["ts"])
    
    # Filter by exact date range (partitions may include extra data)
    if start_date:
        start_dt = pd.Timestamp(start_date, tz="UTC")
        df = df[df["ts"] >= start_dt]
    if end_date:
        end_dt = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
        df = df[df["ts"] < end_dt]
    
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
    timeframe: str = "5m",
    indicators: list[dict[str, Any]] | list[str] | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Get OHLC history with optional indicators from partitioned dataset.
    
    Reads from partitioned parquet structure:
        data/<source>-ohlcv/<symbol>/<timeframe>/<date>/part-0000.parquet
    
    Args:
        symbol: Trading symbol
        start_date: Start date filter
        end_date: End date filter
        timeframe: Bar timeframe (1m, 5m, 15m, 1h, 1d)
        indicators: List of indicator configs or legacy string specs
        source: Data source (ibkr, yahoo). If None, auto-detects.
    
    Returns:
        {
            "symbol": "QQQ",
            "timeframe": "5m",
            "candles": [{"time": 1234567890, "open": 100, ...}, ...],
            "indicators": {"SMA_20": [...], ...}
        }
    """
    df = load_ohlcv_partitioned(symbol, start_date, end_date, timeframe, source)
    
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
        
        # Normalize timeframe
        tf_normalized = _normalize_timeframe(timeframe)
        
        # Determine asset for IBKR
        asset = "fut" if asset_type == "futures" else "stk"
        
        # Create fetch request - all params required
        req = FetchRequest(
            source=source,
            symbol=symbol,
            start=start_date.isoformat() if start_date else datetime.now().date().isoformat(),
            end=end_date.isoformat() if end_date else datetime.now().date().isoformat(),
            timeframe=tf_normalized,
            rth="true",  # Default to RTH
            asset=asset,
            ibkr_config="configs/ibkr.yaml",
            expiry="auto",
            exchange="SMART" if asset == "stk" else "CME",
            currency="USD",
            chunk_days=30,
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
            source=source,
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
        "timestamps": timestamps,  # Pass timestamps for daily aggregations
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
