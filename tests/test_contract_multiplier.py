"""Pure parts of the backtest contract-multiplier resolver (no DB)."""
from __future__ import annotations

from app.services.contract_multiplier import (
    _futures_root,
    multiplier_from_config,
    multiplier_from_definition,
)


def test_config_pin():
    assert multiplier_from_config({"backtest": {"contract_multiplier": 2}}) == 2.0
    assert multiplier_from_config({"backtest": {"contract_multiplier": "2.0"}}) == 2.0
    assert multiplier_from_config({"backtest": {"broker_type": "ibkr"}}) is None
    assert multiplier_from_config({"backtest": {"contract_multiplier": 0}}) is None
    assert multiplier_from_config(None) is None


def test_definition_symbol_picker_metadata():
    definition = {
        "strategy": {
            "symbol": "MNQZ6",
            "extra_data": {"expiry": "20261218", "multiplier": "2"},
            "charts": [
                {"chart_id": "primary", "symbol": "MNQZ6", "extra_data": {"multiplier": 2}},
                {"chart_id": "es", "symbol": "ESZ6", "extra_data": {"multiplier": 50}},
            ],
        }
    }
    assert multiplier_from_definition(definition, "MNQZ6") == 2.0
    assert multiplier_from_definition(definition, "ESZ6") == 50.0
    assert multiplier_from_definition(definition, "NQZ6") is None
    assert multiplier_from_definition(definition) == 2.0
    assert multiplier_from_definition({"strategy": {"symbol": "AAPL"}}, "AAPL") is None
    # backtest.config is the same DSL document wrapped with a "backtest" block
    assert multiplier_from_definition({"backtest": {}, **definition}, "MNQZ6") == 2.0


def test_futures_root():
    assert _futures_root("MNQZ6") == "MNQ"
    assert _futures_root("ESH27") == "ES"
    assert _futures_root("6EZ6") == "6E"
    assert _futures_root("AAPL") == "AAPL"
