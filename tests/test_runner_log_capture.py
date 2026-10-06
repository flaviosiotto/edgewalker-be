"""The reaper saves the runner log tail on the backtest row before removing the container."""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import app.db.database as database
from app.services.backtest_runner_service import BacktestRunnerService


class _Container:
    def __init__(self, backtest_id, exit_code, text="line1\nline2\n"):
        self.name = f"edgewalker-backtest-runner-{backtest_id}"
        self.labels = {"edgewalker.backtest_id": str(backtest_id)}
        self.attrs = {"State": {"ExitCode": exit_code}}
        self._text = text

    def reload(self):
        pass

    def logs(self, tail, timestamps):
        return self._text.encode()


class _Session:
    def __init__(self, backtest):
        self.backtest, self.added = backtest, []

    def get(self, model, pk):
        return self.backtest

    def add(self, obj):
        self.added.append(obj)


def _run(monkeypatch, backtest, container):
    session = _Session(backtest)

    @contextmanager
    def ctx():
        yield session

    monkeypatch.setattr(database, "get_session_context", ctx)
    BacktestRunnerService.__new__(BacktestRunnerService)._capture_runner_log(container)
    return session


def test_failed_backtest_gets_the_log_tail(monkeypatch):
    bt = SimpleNamespace(status="failed", runner_log_tail=None)
    s = _run(monkeypatch, bt, _Container(201, 0))
    assert bt.runner_log_tail.startswith("[edgewalker-backtest-runner-201 exit_code=0")
    assert bt.runner_log_tail.endswith("line1\nline2\n") and s.added == [bt]


def test_abnormal_exit_is_captured_even_when_completed(monkeypatch):
    bt = SimpleNamespace(status="completed", runner_log_tail=None)
    _run(monkeypatch, bt, _Container(7, 137))
    assert "exit_code=137" in bt.runner_log_tail


def test_clean_completed_runner_is_not_captured(monkeypatch):
    bt = SimpleNamespace(status="completed", runner_log_tail=None)
    s = _run(monkeypatch, bt, _Container(7, 0))
    assert bt.runner_log_tail is None and s.added == []


def test_existing_tail_is_kept(monkeypatch):
    bt = SimpleNamespace(status="failed", runner_log_tail="old")
    _run(monkeypatch, bt, _Container(7, 1))
    assert bt.runner_log_tail == "old"
