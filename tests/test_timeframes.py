import pytest
from fastapi import HTTPException

from app.utils.timeframes import normalize_simulation_timeframe, parse_tf_seconds


def test_parse():
    assert parse_tf_seconds("1m") == 60 and parse_tf_seconds("4h") == 14400 and parse_tf_seconds("3m") == 180
    with pytest.raises(ValueError):
        parse_tf_seconds("1w")


def test_driver_finer_than_chart_is_kept():
    assert normalize_simulation_timeframe("1m", "4h") == "1m"
    assert normalize_simulation_timeframe("15m", "1h") == "15m"


def test_same_or_empty_means_no_driver():
    assert normalize_simulation_timeframe("4h", "4h") is None
    assert normalize_simulation_timeframe("", "4h") is None
    assert normalize_simulation_timeframe(None, "4h") is None


def test_coarser_or_unknown_is_refused():
    with pytest.raises(HTTPException):
        normalize_simulation_timeframe("1h", "5m")
    with pytest.raises(HTTPException):
        normalize_simulation_timeframe("2m", "5m")
