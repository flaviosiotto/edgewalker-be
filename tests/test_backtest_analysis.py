"""Backtest detail payload: scalar metrics + fixed-size analysis."""
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from app.services.backtest_analysis import analyze_backtest, scalar_metrics

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
HOUR_MS = 3_600_000


def _trade(i, pnl, *, direction="long", exit_reason="take_profit"):
    entry = T0 + timedelta(hours=i)
    return SimpleNamespace(pnl=pnl, entry_time=entry, exit_time=entry + timedelta(minutes=30), direction=direction, exit_reason=exit_reason)


def _metrics(equities):
    start = T0.timestamp() * 1000
    return {
        "return_pct": 1.5,
        "risk_ratio_basis": "daily",
        "orders": 12,
        "positions_summary": {"positions": [1, 2, 3]},
        "runner_snapshot_history": [{}],
        "ledger": {
            "orders": [{}],
            "equity_snapshots": [
                {"ts": start + i * HOUR_MS, "equity": eq, "position_qty": 1 if i % 2 else 0, "positions": [{"x": 1}]}
                for i, eq in enumerate(equities)
            ],
        },
    }


def test_scalar_metrics_drops_nested_sections():
    assert scalar_metrics(_metrics([100.0])) == {"return_pct": 1.5, "risk_ratio_basis": "daily", "orders": 12}
    assert scalar_metrics(None) is None


def test_trade_statistics():
    trades = [_trade(0, 100.0), _trade(1, -50.0, exit_reason="stop_loss"), _trade(2, -30.0, direction="short", exit_reason="stop_loss"), _trade(3, 60.0)]
    stats = analyze_backtest(None, trades)["trades"]
    assert stats["trades"] == 4 and stats["winners"] == 2 and stats["losers"] == 2
    assert stats["net_pnl"] == 80.0 and stats["expectancy_per_trade"] == 20.0
    assert stats["payoff_ratio"] == 2.0
    assert stats["max_consecutive_losses"] == 2 and stats["max_consecutive_wins"] == 1
    assert stats["avg_duration_hours"] == 0.5
    assert stats["by_direction"]["short"]["trades"] == 1
    assert stats["by_exit_reason"]["stop_loss"]["net_pnl"] == -80.0


def test_open_trades_are_ignored():
    trades = [_trade(0, 10.0), SimpleNamespace(pnl=None, entry_time=T0, exit_time=None, direction="long", exit_reason=None)]
    assert analyze_backtest(None, trades)["trades"]["trades"] == 1


def test_drawdowns_exposure_and_underwater():
    # peak 110 -> trough 99 (-10%) -> recovered at 110; then 120 -> 114 (-5%), never recovered
    analysis = analyze_backtest(_metrics([100, 110, 99, 105, 110, 120, 114]), [])
    worst, second = analysis["drawdowns"]
    assert worst["depth_pct"] == -10.0 and worst["recovered_ts"] is not None
    assert second["depth_pct"] == -5.0 and second["recovered_ts"] is None
    assert analysis["longest_underwater_days"] == round(3 / 24, 2)
    assert analysis["exposure_pct"] == round(3 / 7 * 100, 3)


def test_segments_cover_the_run_and_chain_returns():
    equities = [100.0 + i for i in range(41)]  # 40 hours, +1 per hour
    trades = [_trade(i, 1.0) for i in range(0, 40, 4)]
    segments = analyze_backtest(_metrics(equities), trades)["segments"]
    assert len(segments) == 4
    assert sum(s["trades"] for s in segments) == len(trades)
    growth = 1.0
    for s in segments:
        growth *= 1 + s["return_pct"] / 100
    assert round(growth * 100, 1) == 140.0


def test_concentration():
    trades = [_trade(i, pnl) for i, pnl in enumerate([500.0, 20.0, 10.0, 10.0, -100.0])]
    conc = analyze_backtest(None, trades)["concentration"]
    assert conc["top_trades"] == 3
    assert conc["top_trades_share_of_gross_profit_pct"] == round(530 / 540 * 100, 3)
    assert conc["net_pnl_without_top_trades"] == -90.0


def test_nothing_to_analyze():
    assert analyze_backtest(None, []) is None
    assert analyze_backtest({"ledger": {"equity_snapshots": []}}, []) is None


def test_size_does_not_grow_with_the_run():
    def size(bars, n_trades):
        equities = [100_000 + (i % 97) * 10 - (i % 13) * 25 for i in range(bars)]
        trades = [_trade(i, (i % 7) - 3.0, exit_reason=f"reason {i % 40}") for i in range(n_trades)]
        return len(json.dumps(analyze_backtest(_metrics(equities), trades)))

    small, large = size(200, 20), size(100_000, 5_000)
    assert large < small * 1.5
    assert large < 8_000
