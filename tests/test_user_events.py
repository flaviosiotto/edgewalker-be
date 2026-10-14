"""The closed catalogue of task outcomes (migr. 070): pure classification.
Run: ``venv/bin/python -m pytest tests/test_user_events.py``.
"""
from __future__ import annotations

import pytest

from app.services.user_events import ENTITY_OF, SEVERITY, UserEventType, catalog, classify

T = UserEventType


@pytest.mark.parametrize(
    "table, old, new, expected",
    [
        # connections: outcome of an attempt, fall of a healthy one, recovery
        ("connections", "connecting", "connected", T.CONNECTION_CONNECTED),
        ("connections", "awaiting_auth", "connected", T.CONNECTION_CONNECTED),
        ("connections", "degraded", "connected", T.CONNECTION_CONNECTED),
        ("connections", "connecting", "error", T.CONNECTION_FAILED),
        ("connections", "connecting", "disconnected", T.CONNECTION_FAILED),
        ("connections", "awaiting_auth", "disconnected", T.CONNECTION_FAILED),
        ("connections", "connected", "disconnected", T.CONNECTION_LOST),
        ("connections", "connected", "degraded", T.CONNECTION_LOST),
        ("connections", "degraded", "error", T.CONNECTION_LOST),
        # the health-loop cleanup after a failed attempt is not news
        ("connections", "error", "disconnected", None),
        ("connections", "disconnected", "error", None),
        ("connections", "disconnected", "connecting", None),
        ("connections", "connected", "connected", None),
        # no transition info (067 payload / tests): new status decides
        ("connections", None, "error", T.CONNECTION_FAILED),
        ("connections", None, "disconnected", T.CONNECTION_LOST),
        ("connections", None, "connected", T.CONNECTION_CONNECTED),
        ("connections", None, "connecting", None),
        # live
        ("strategy_live", "starting", "running", T.LIVE_STARTED),
        ("strategy_live", None, "running", T.LIVE_STARTED),
        ("strategy_live", "paused", "running", None),
        ("strategy_live", "starting", "error", T.LIVE_FAILED),
        ("strategy_live", "running", "error", T.LIVE_FAILED),
        ("strategy_live", "stopping", "stopped", T.LIVE_STOPPED),
        ("strategy_live", "running", "stopped", T.LIVE_STOPPED),
        ("strategy_live", "starting", "stopped", None),
        ("strategy_live", "running", "paused", None),
        ("strategy_live", "running", "stopping", None),
        # backtest
        ("strategy_backtests", "running", "completed", T.BACKTEST_COMPLETED),
        ("strategy_backtests", "running", "failed", T.BACKTEST_FAILED),
        ("strategy_backtests", "pending", "error", T.BACKTEST_FAILED),
        ("strategy_backtests", "pending", "running", None),
        # studio
        ("studio_run", "running", "succeeded", T.STUDIO_RUN_SUCCEEDED),
        ("studio_run", "running", "failed", T.STUDIO_RUN_FAILED),
        ("studio_run", "queued", "cancelled", T.STUDIO_RUN_FAILED),
        ("studio_run", "queued", "running", None),
        # unknown table, same status, missing status
        ("trades", None, "x", None),
        ("connections", "Connected", "CONNECTED", None),
        ("connections", "connected", None, None),
    ],
)
def test_classify(table, old, new, expected):
    assert classify(table, old, new) is expected


def test_catalog_is_closed_and_complete():
    rows = catalog()
    assert {r["type"] for r in rows} == {t.value for t in UserEventType}
    assert len(rows) == 10
    for t in UserEventType:
        assert SEVERITY[t] in {"info", "success", "warning", "error"}
        assert ENTITY_OF[t] in {"connection", "backtest", "live", "studio_run"}
