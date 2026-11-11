"""Live runner monitor: a healthy, just-started runner is never reported as
terminated, and a real termination says why.

Regression for live 547 (11/11/2026): ``GET /strategies/`` read the container
status while the live row existed but the container did not yet, the
"not_found" answer stayed cached for 5 s, and the monitor tick that followed
the start marked the runner "terminated unexpectedly" while it was running.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from types import SimpleNamespace

from docker.errors import NotFound

import app.services.live_runner_monitor as monitor_module
from app.models.strategy import LiveStatus
from app.services.live_runner_service import LiveRunnerService


class _Container:
    def __init__(self, status="running", state=None, logs=""):
        self.status, self.short_id, self.name, self.labels = status, "abc123", "edgewalker-live-547", {}
        self.attrs = {"Created": "2026-11-11T17:03:37Z", "State": state or {"StartedAt": "2026-11-11T17:03:37Z"}}
        self._logs = logs

    def reload(self):
        pass

    def logs(self, **kwargs):
        return self._logs.encode()


class _Docker:
    """Just enough of the Docker client: one optional container."""

    def __init__(self):
        self.container: _Container | None = None
        self.containers = SimpleNamespace(get=self._get, list=lambda **kw: [])

    def _get(self, name):
        if self.container is None:
            raise NotFound("no such container")
        return self.container


def _service() -> tuple[LiveRunnerService, _Docker]:
    service = LiveRunnerService()
    docker = _Docker()
    service._client = docker
    return service, docker


def test_cached_not_found_from_before_the_start_is_not_served_to_a_fresh_read():
    service, docker = _service()
    assert service.get_live_instance_status(547)["status"] == "not_found"  # GET /strategies/ before the container exists
    docker.container = _Container()                                         # the runner starts
    assert service.get_live_instance_status(547)["status"] == "not_found"  # the 5 s cache still says so...
    assert service.get_live_instance_status(547, fresh=True)["status"] == "running"  # ...a deciding caller does not use it
    assert service.get_live_instance_status(547)["status"] == "running"    # and the fresh answer replaced the stale one


def test_termination_message_says_how_the_runner_ended():
    service, docker = _service()
    log = "\n".join([
        json.dumps({"level": "info", "message": "PERIODIC STATS"}),
        json.dumps({"level": "error", "message": "Error in consumer loop: Timeout reading from redis:6379"}),
        json.dumps({"level": "info", "message": "shutting down"}),
    ])
    docker.container = _Container("exited", {"ExitCode": 1, "OOMKilled": False, "Error": "", "FinishedAt": "2026-11-11T17:20:05.123Z"}, log)
    message = service.describe_termination(service.get_live_instance_status(547, fresh=True))
    assert "crashed (exit code 1)" in message and "2026-11-11T17:20:05Z" in message
    assert "Last log: Error in consumer loop: Timeout reading from redis:6379" in message

    docker.container = _Container("exited", {"ExitCode": 137, "OOMKilled": True, "FinishedAt": "2026-11-11T17:20:05Z"})
    assert "exceeding memory (OOM)" in service.describe_termination(service.get_live_instance_status(547, fresh=True))

    docker.container = None
    assert "no longer exists" in service.describe_termination(service.get_live_instance_status(547, fresh=True))


class _Session:
    def __init__(self, rows):
        self.rows, self.merged = rows, []

    def exec(self, stmt):
        return SimpleNamespace(all=lambda: [SimpleNamespace(**vars(r)) for r in self.rows.values()])

    def get(self, model, pk):
        return self.rows.get(pk)

    def merge(self, obj):
        self.merged.append(obj)
        self.rows[obj.id] = obj

    def commit(self):
        pass


def _run_monitor(monkeypatch, rows, service, *, during=None):
    session = _Session(rows)

    @contextmanager
    def ctx():
        yield session

    calls = {"n": 0}
    real = service.get_live_instance_status

    def status(live_id, *, fresh=False):
        calls["n"] += 1
        assert fresh, "the monitor must never decide on a cached status"
        if during:
            during()
        return real(live_id, fresh=fresh)

    monkeypatch.setattr(monitor_module, "get_session_context", ctx)
    monkeypatch.setattr(monitor_module, "live_runner_service", SimpleNamespace(
        get_live_instance_status=status, describe_termination=service.describe_termination,
    ))
    updated = asyncio.run(monitor_module.LiveRunnerMonitor().sync_once())
    return updated, session


def _live(status=LiveStatus.RUNNING.value):
    return SimpleNamespace(id=547, status=status, container_id="abc123", error_message=None, started_at=None, created_at=None, stopped_at=None, updated_at=None)


def test_monitor_leaves_a_running_runner_alone_despite_a_stale_cache(monkeypatch):
    service, docker = _service()
    service.get_live_instance_status(547)   # stale "not_found" cached before the start
    docker.container = _Container()
    updated, session = _run_monitor(monkeypatch, {547: _live()}, service)
    assert updated == 0 and session.rows[547].status == LiveStatus.RUNNING.value


def test_monitor_reports_a_real_exit_with_its_reason(monkeypatch):
    service, docker = _service()
    docker.container = _Container("exited", {"ExitCode": 3, "FinishedAt": "2026-11-11T17:20:05Z"}, "Traceback: boom\n")
    updated, session = _run_monitor(monkeypatch, {547: _live()}, service)
    row = session.rows[547]
    assert updated == 1 and row.status == LiveStatus.ERROR.value
    assert "exit code 3" in row.error_message and "Last log: Traceback: boom" in row.error_message


def test_monitor_does_not_overwrite_a_row_changed_while_it_was_reading_docker(monkeypatch):
    service, docker = _service()   # no container: the monitor would say "no longer exists"
    rows = {547: _live()}

    def user_stops_meanwhile():
        rows[547] = _live(LiveStatus.STOPPING.value)

    updated, session = _run_monitor(monkeypatch, rows, service, during=user_stops_meanwhile)
    assert updated == 0 and session.rows[547].status == LiveStatus.STOPPING.value
