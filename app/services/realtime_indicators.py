"""
Real-time Indicator Calculator Service.

Maintains OHLCV buffers per symbol/timeframe and calculates indicators
when new bars arrive. This ensures indicator calculation is centralized
and uses the same TA-Lib code as backtesting.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Import TA-Lib (same as used in backtesting)
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    logger.warning("TA-Lib not available, real-time indicators will be disabled")
    TALIB_AVAILABLE = False


@dataclass
class OHLCVBar:
    """Single OHLCV bar."""
    time: int  # Unix timestamp (seconds)
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class IndicatorConfig:
    """Configuration for an indicator to calculate."""
    type: str  # e.g., "SMA", "EMA", "RSI", "MACD", "BBANDS"
    name: str  # Unique name for this indicator instance
    params: dict[str, Any] = field(default_factory=dict)


class SymbolBuffer:
    """
    Rolling buffer of OHLCV bars for a single symbol/timeframe.
    
    Provides numpy arrays for TA-Lib calculations.
    """
    
    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self._bars: deque[OHLCVBar] = deque(maxlen=max_size)
    
    def add_or_update(self, bar: OHLCVBar) -> bool:
        """
        Add or update a bar in the buffer.
        
        Returns:
            True if a new bar was added, False if existing bar was updated.
        """
        # Check if this is an update to the last bar
        if self._bars and self._bars[-1].time == bar.time:
            self._bars[-1] = bar
            return False
        
        # Add new bar
        self._bars.append(bar)
        return True
    
    def get_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get numpy arrays for TA-Lib calculations.
        
        Returns:
            Tuple of (open, high, low, close, volume) arrays.
        """
        if not self._bars:
            empty = np.array([], dtype=np.float64)
            return empty, empty, empty, empty, empty
        
        opens = np.array([b.open for b in self._bars], dtype=np.float64)
        highs = np.array([b.high for b in self._bars], dtype=np.float64)
        lows = np.array([b.low for b in self._bars], dtype=np.float64)
        closes = np.array([b.close for b in self._bars], dtype=np.float64)
        volumes = np.array([b.volume for b in self._bars], dtype=np.float64)
        
        return opens, highs, lows, closes, volumes
    
    def get_last_time(self) -> int | None:
        """Get the timestamp of the last bar."""
        return self._bars[-1].time if self._bars else None
    
    def __len__(self) -> int:
        return len(self._bars)


class RealtimeIndicatorCalculator:
    """
    Calculates indicators in real-time as new bars arrive.
    
    Uses TA-Lib for indicator calculations - same library used in backtesting.
    """
    
    # Buffer size per symbol/timeframe
    BUFFER_SIZE = 200
    
    def __init__(self):
        # symbol -> SymbolBuffer
        self._buffers: dict[str, SymbolBuffer] = {}
        # symbol -> list of configured indicators
        self._indicators: dict[str, list[IndicatorConfig]] = {}
    
    def configure_indicators(self, symbol: str, indicators: list[dict[str, Any]]) -> None:
        """
        Configure indicators for a symbol.
        
        Args:
            symbol: Trading symbol
            indicators: List of indicator configs, e.g.:
                [{"type": "SMA", "name": "sma_20", "params": {"period": 20}}]
        """
        configs = []
        for ind in indicators:
            configs.append(IndicatorConfig(
                type=ind.get("type", "").upper(),
                name=ind.get("name", ind.get("type", "unknown")),
                params=ind.get("params", {}),
            ))
        
        self._indicators[symbol] = configs
        logger.info(f"Configured {len(configs)} indicators for {symbol}")
    
    def process_bar(self, symbol: str, bar_data: dict[str, Any]) -> dict[str, Any]:
        """
        Process a new bar and calculate indicators.
        
        Args:
            symbol: Trading symbol
            bar_data: Bar data dict with timestamp, open, high, low, close, volume
            
        Returns:
            Indicator values dict, e.g.: {"sma_20": 123.45, "rsi_14": 65.2}
        """
        if not TALIB_AVAILABLE:
            logger.warning("TA-Lib not available, skipping indicator calculation")
            return {}
        
        # Ensure we have a buffer for this symbol
        if symbol not in self._buffers:
            self._buffers[symbol] = SymbolBuffer(self.BUFFER_SIZE)
        
        buffer = self._buffers[symbol]
        
        # Create bar from data
        bar = OHLCVBar(
            time=int(bar_data.get("timestamp", 0) / 1000),  # ms to seconds
            open=float(bar_data.get("open", 0)),
            high=float(bar_data.get("high", 0)),
            low=float(bar_data.get("low", 0)),
            close=float(bar_data.get("close", 0)),
            volume=float(bar_data.get("volume", 0)),
        )
        
        # Add to buffer
        is_new = buffer.add_or_update(bar)
        buffer_size = len(buffer)
        logger.debug(f"process_bar: symbol={symbol}, buffer_size={buffer_size}, is_new={is_new}")
        
        # Get configured indicators for this symbol
        indicators = self._indicators.get(symbol, [])
        if not indicators:
            logger.debug(f"No indicators configured for {symbol}")
            return {}
        
        # Calculate indicators
        result = {}
        opens, highs, lows, closes, volumes = buffer.get_arrays()
        
        for config in indicators:
            try:
                values = self._calculate_indicator(
                    config, opens, highs, lows, closes, volumes
                )
                if values:
                    result.update(values)
            except Exception as e:
                logger.warning(f"Error calculating {config.name}: {e}")
        
        return result
    
    def _calculate_indicator(
        self,
        config: IndicatorConfig,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
    ) -> dict[str, float]:
        """
        Calculate a single indicator using TA-Lib dynamically.
        
        Uses the same approach as indicator_registry.compute_indicator()
        for consistency between historical and real-time data.
        
        Returns:
            Dict mapping output names to last values.
        """
        from app.services.indicator_registry import (
            compute_indicator as registry_compute,
            OUTPUT_NAME_MAP,
        )
        
        name = config.name
        ind_type = config.type.upper()
        params = dict(config.params)
        
        # Normalize common param names to TA-Lib convention
        if "period" in params and "timeperiod" not in params:
            params["timeperiod"] = params.pop("period")
        if "std_dev" in params:
            std_dev = params.pop("std_dev")
            params.setdefault("nbdevup", std_dev)
            params.setdefault("nbdevdn", std_dev)
        if "stdDev" in params:
            std_dev = params.pop("stdDev")
            params.setdefault("nbdevup", std_dev)
            params.setdefault("nbdevdn", std_dev)
        
        # Build data dict for registry function
        data = {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
        
        try:
            # Use the registry's compute function
            values = registry_compute(ind_type, params, data)
            
            result = {}
            
            if isinstance(values, dict):
                # Multi-output indicator (MACD, BBANDS, STOCH, etc.)
                for output_key, arr in values.items():
                    if arr and len(arr) > 0:
                        last_val = arr[-1]
                        if last_val is not None and not (isinstance(last_val, float) and np.isnan(last_val)):
                            result[f"{name}_{output_key}"] = float(last_val)
            else:
                # Single-output indicator (SMA, EMA, RSI, etc.)
                if values is not None and len(values) > 0:
                    last_val = values[-1] if hasattr(values, '__getitem__') else values
                    if last_val is not None and not (isinstance(last_val, float) and np.isnan(last_val)):
                        result[name] = float(last_val)
            
            return result
            
        except Exception as e:
            logger.debug(f"Failed to compute {ind_type} for {name}: {e}")
            return {}
    
    def prefill_buffer(self, symbol: str, bars: list[dict[str, Any]]) -> None:
        """
        Pre-fill the buffer with historical bars for indicator warm-up.
        
        Args:
            symbol: Trading symbol
            bars: List of bar dicts with timestamp, open, high, low, close, volume
                  Bars should be sorted oldest first.
        """
        if symbol not in self._buffers:
            self._buffers[symbol] = SymbolBuffer(self.BUFFER_SIZE)
        
        buffer = self._buffers[symbol]
        
        for bar_data in bars:
            bar = OHLCVBar(
                time=int(bar_data.get("timestamp", bar_data.get("time", 0))),
                open=float(bar_data.get("open", 0)),
                high=float(bar_data.get("high", 0)),
                low=float(bar_data.get("low", 0)),
                close=float(bar_data.get("close", 0)),
                volume=float(bar_data.get("volume", 0)),
            )
            buffer.add_or_update(bar)
        
        logger.info(f"Pre-filled buffer for {symbol} with {len(bars)} bars, buffer size: {len(buffer)}")
    
    def clear_buffer(self, symbol: str) -> None:
        """Clear the buffer for a symbol."""
        if symbol in self._buffers:
            del self._buffers[symbol]
    
    def get_buffer_size(self, symbol: str) -> int:
        """Get the current buffer size for a symbol."""
        return len(self._buffers.get(symbol, []))


# Global calculator instance
indicator_calculator = RealtimeIndicatorCalculator()
