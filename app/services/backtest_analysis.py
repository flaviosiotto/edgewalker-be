"""Fixed-size analysis of a finished backtest.

``strategy_backtests.metrics`` embeds the full ledger of the run (one equity
snapshot per bar, every order and fill): megabytes that grow with the length
of the backtest. Callers that judge a run (the performance panel, the agent's
final evaluation, MCP clients) need its statistics, not its rows.

Everything here is computed from the ledger and the closed trades and has a
size that does not depend on how long the run was: no per-bar, per-trade or
per-calendar-period lists.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable

TOP_DRAWDOWNS = 3
SEGMENTS = 4
TOP_TRADES = 3
MAX_EXIT_REASONS = 8


def scalar_metrics(metrics: Any) -> dict[str, Any] | None:
    """The scalar indicators of ``metrics``; nested sections (ledger,
    positions, runner snapshots) grow with the run and are left out."""
    if not isinstance(metrics, dict):
        return None
    return {k: v for k, v in metrics.items() if v is None or isinstance(v, (str, int, float, bool))}


def _r(value: float | None, digits: int = 2) -> float | None:
    return round(value, digits) if value is not None else None


def _pct(num: float, den: float) -> float | None:
    return round(num / den * 100, 3) if den else None


def _ms(value: Any) -> float | None:
    """Epoch milliseconds of a datetime (naive = UTC) or of an ISO string."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.timestamp() * 1000
    return None


def _iso(ts_ms: float | None) -> str | None:
    if ts_ms is None:
        return None
    try:
        return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def _closed_trades(trades: Iterable[Any]) -> list[dict[str, Any]]:
    out = []
    for t in trades:
        pnl = getattr(t, "pnl", None)
        if pnl is None:
            continue
        out.append(
            {
                "pnl": float(pnl),
                "entry_ms": _ms(getattr(t, "entry_time", None)),
                "exit_ms": _ms(getattr(t, "exit_time", None)),
                "direction": getattr(t, "direction", None) or "unknown",
                "exit_reason": getattr(t, "exit_reason", None) or "unknown",
            }
        )
    return out


def _equity_points(metrics: Any) -> list[tuple[float, float, bool]]:
    """(ts_ms, equity, in_market) of every usable ledger snapshot."""
    ledger = metrics.get("ledger") if isinstance(metrics, dict) else None
    snapshots = ledger.get("equity_snapshots") if isinstance(ledger, dict) else None
    points = []
    for snap in snapshots or []:
        if not isinstance(snap, dict):
            continue
        ts, equity = snap.get("ts"), snap.get("equity")
        if not isinstance(ts, (int, float)) or not isinstance(equity, (int, float)):
            continue
        points.append((float(ts), float(equity), bool(snap.get("position_qty") or 0)))
    return points


def _group_stats(trades: list[dict[str, Any]]) -> dict[str, Any]:
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    return {
        "trades": len(pnls),
        "win_rate_pct": _pct(len(wins), len(pnls)),
        "net_pnl": _r(sum(pnls)),
        "avg_pnl": _r(sum(pnls) / len(pnls)) if pnls else None,
        "avg_win": _r(sum(wins) / len(wins)) if wins else None,
        "avg_loss": _r(sum(losses) / len(losses)) if losses else None,
    }


def _by_exit_reason(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Exit reasons may be free text: keep the most frequent, merge the rest."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in trades:
        groups[t["exit_reason"]].append(t)
    ranked = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    out = {reason: _group_stats(rows) for reason, rows in ranked[:MAX_EXIT_REASONS]}
    rest = [t for _, rows in ranked[MAX_EXIT_REASONS:] for t in rows]
    if rest:
        out["other"] = _group_stats(rest)
    return out


def _trade_stats(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {"trades": 0}
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_profit = sum(wins)
    gross_loss = -sum(losses)

    max_win_streak = max_loss_streak = win_streak = loss_streak = 0
    for p in pnls:
        win_streak = win_streak + 1 if p > 0 else 0
        loss_streak = loss_streak + 1 if p < 0 else 0
        max_win_streak = max(max_win_streak, win_streak)
        max_loss_streak = max(max_loss_streak, loss_streak)

    hours = [
        (t["exit_ms"] - t["entry_ms"]) / 3_600_000
        for t in trades
        if t["entry_ms"] is not None and t["exit_ms"] is not None
    ]
    best = max(trades, key=lambda t: t["pnl"])
    worst = min(trades, key=lambda t: t["pnl"])
    by_direction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in trades:
        by_direction[t["direction"]].append(t)

    return {
        "trades": len(trades),
        "winners": len(wins),
        "losers": len(losses),
        "gross_profit": _r(gross_profit),
        "gross_loss": _r(-gross_loss),
        "net_pnl": _r(sum(pnls)),
        "expectancy_per_trade": _r(sum(pnls) / len(pnls)),
        "avg_win": _r(gross_profit / len(wins)) if wins else None,
        "avg_loss": _r(-gross_loss / len(losses)) if losses else None,
        "payoff_ratio": _r((gross_profit / len(wins)) / (gross_loss / len(losses)), 3) if wins and losses else None,
        "largest_win": {"pnl": _r(best["pnl"]), "entry_time": _iso(best["entry_ms"])},
        "largest_loss": {"pnl": _r(worst["pnl"]), "entry_time": _iso(worst["entry_ms"])},
        "max_consecutive_wins": max_win_streak,
        "max_consecutive_losses": max_loss_streak,
        "avg_duration_hours": _r(sum(hours) / len(hours)) if hours else None,
        "max_duration_hours": _r(max(hours)) if hours else None,
        "by_direction": {k: _group_stats(v) for k, v in by_direction.items()},
        "by_exit_reason": _by_exit_reason(trades),
    }


def _drawdowns(points: list[tuple[float, float, bool]]) -> tuple[list[dict[str, Any]], float | None]:
    """Worst peak→trough→recovery episodes and the longest time under water (days)."""
    episodes: list[dict[str, Any]] = []
    longest_ms = 0.0
    peak_eq = peak_ts = trough_eq = trough_ts = None

    def close(recovered_ts: float | None, end_ts: float) -> None:
        nonlocal longest_ms
        if trough_eq is None or not peak_eq or trough_eq >= peak_eq:
            return
        longest_ms = max(longest_ms, end_ts - peak_ts)
        episodes.append(
            {
                "depth_pct": _pct(trough_eq - peak_eq, peak_eq),
                "peak_ts": _iso(peak_ts),
                "trough_ts": _iso(trough_ts),
                # None: still under water at the end of the run
                "recovered_ts": _iso(recovered_ts),
            }
        )

    for ts, equity, _ in points:
        if peak_eq is None or equity >= peak_eq:
            close(ts, ts)
            peak_eq, peak_ts, trough_eq, trough_ts = equity, ts, None, None
        elif trough_eq is None or equity < trough_eq:
            trough_eq, trough_ts = equity, ts
    if points:
        close(None, points[-1][0])
    episodes.sort(key=lambda e: e["depth_pct"] or 0)
    return episodes[:TOP_DRAWDOWNS], (_r(longest_ms / 86_400_000) if episodes else None)


def _segments(points: list[tuple[float, float, bool]], trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The run cut in SEGMENTS equal time slices: is the result spread over
    the period or made in one stretch? Trades belong to the slice they closed in."""
    exits = [t["exit_ms"] for t in trades if t["exit_ms"] is not None]
    if points:
        start, end = points[0][0], points[-1][0]
    elif exits:
        start, end = min(exits), max(exits)
    else:
        return []
    if end <= start:
        return []
    width = (end - start) / SEGMENTS
    edges = [start + width * i for i in range(SEGMENTS)] + [end]

    def slot(ts: float) -> int:
        return min(SEGMENTS - 1, max(0, int((ts - start) / width)))

    buckets: list[list[dict[str, Any]]] = [[] for _ in range(SEGMENTS)]
    for t in trades:
        if t["exit_ms"] is not None:
            buckets[slot(t["exit_ms"])].append(t)

    # Equity at the close of each slice; a slice without snapshots keeps the previous close.
    closes: list[float | None] = [None] * SEGMENTS
    for ts, equity, _ in points:
        closes[slot(ts)] = equity

    out = []
    prev = points[0][1] if points else None
    for i in range(SEGMENTS):
        pnls = [t["pnl"] for t in buckets[i]]
        close_eq = closes[i] if closes[i] is not None else prev
        out.append(
            {
                "from": _iso(edges[i]),
                "to": _iso(edges[i + 1]),
                "return_pct": _pct(close_eq - prev, prev) if prev and close_eq is not None else None,
                "trades": len(pnls),
                "net_pnl": _r(sum(pnls)),
                "win_rate_pct": _pct(sum(1 for p in pnls if p > 0), len(pnls)),
            }
        )
        prev = close_eq
    return out


def _concentration(trades: list[dict[str, Any]]) -> dict[str, Any] | None:
    """How much of the result hangs on the few best trades."""
    if not trades:
        return None
    pnls = sorted((t["pnl"] for t in trades), reverse=True)
    top = [p for p in pnls[:TOP_TRADES] if p > 0]
    gross_profit = sum(p for p in pnls if p > 0)
    return {
        "top_trades": len(top),
        "top_trades_share_of_gross_profit_pct": _pct(sum(top), gross_profit),
        "net_pnl_without_top_trades": _r(sum(pnls) - sum(top)),
    }


def analyze_backtest(metrics: Any, trades: Iterable[Any]) -> dict[str, Any] | None:
    """Fixed-size statistics of a run, or None when it left nothing to analyze."""
    closed = _closed_trades(trades)
    points = _equity_points(metrics)
    if not closed and not points:
        return None
    analysis: dict[str, Any] = {"trades": _trade_stats(closed)}
    if points:
        drawdowns, longest_days = _drawdowns(points)
        analysis["drawdowns"] = drawdowns
        analysis["longest_underwater_days"] = longest_days
        analysis["exposure_pct"] = _pct(sum(1 for p in points if p[2]), len(points))
    analysis["segments"] = _segments(points, closed)
    analysis["concentration"] = _concentration(closed)
    return analysis
