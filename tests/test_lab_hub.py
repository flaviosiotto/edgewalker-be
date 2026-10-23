"""Sessione Lab via API REST di JupyterHub (app/services/lab_hub): funzioni
pure contro un hub finto (httpx.MockTransport), niente DB."""
from __future__ import annotations

import json

import httpx
import pytest

from app.services import lab_hub


def _hub(routes):
    """routes: {(method, path): handler(request) -> Response}."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path, request.content))
        fn = routes.get((request.method, request.url.path))
        if fn is None:
            return httpx.Response(404, json={"message": "no route"})
        return fn(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://hub/hub/api")
    return client, calls


def test_session_from_user_model():
    assert lab_hub.session_from_user({"servers": {}}).state == "stopped"
    assert lab_hub.session_from_user({"servers": {"": {"ready": True, "pending": None}}}).state == "ready"
    assert lab_hub.session_from_user({"servers": {"": {"ready": False, "pending": "spawn"}}}).state == "pending"
    assert lab_hub.session_from_user({"servers": {"": {"ready": False, "pending": "stop"}}}).state == "stopping"


@pytest.mark.asyncio
async def test_start_creates_user_sets_auth_state_and_spawns():
    state = {"exists": False, "auth_state": None, "spawned": False}

    def get_user(_):
        if not state["exists"]:
            return httpx.Response(404, json={"message": "No such user"})
        servers = {"": {"ready": True, "pending": None}} if state["spawned"] else {}
        return httpx.Response(200, json={"name": "7", "servers": servers})

    def create_user(_):
        state["exists"] = True
        return httpx.Response(201, json={"name": "7"})

    def patch_user(request):
        state["auth_state"] = json.loads(request.content)["auth_state"]
        return httpx.Response(200, json={"name": "7"})

    def start(_):
        state["spawned"] = True
        return httpx.Response(202, text="")

    client, calls = _hub({
        ("GET", "/hub/api/users/7"): get_user,
        ("POST", "/hub/api/users/7"): create_user,
        ("PATCH", "/hub/api/users/7"): patch_user,
        ("POST", "/hub/api/users/7/server"): start,
    })
    async with client:
        await lab_hub.ensure_user(client, 7)
        assert (await lab_hub.fetch_session(client, 7)).state == "stopped"
        await lab_hub.set_auth_state(client, 7, {"uid": 7, "studio_token": "t", "lab_theme": "dark"})
        assert (await lab_hub.start_server(client, 7)).state == "pending"
        assert (await lab_hub.fetch_session(client, 7)).state == "ready"
    assert state["auth_state"] == {"uid": 7, "studio_token": "t", "lab_theme": "dark"}
    assert [c[0] + " " + c[1] for c in calls][:3] == ["GET /hub/api/users/7", "POST /hub/api/users/7", "GET /hub/api/users/7"]


@pytest.mark.asyncio
async def test_start_when_already_running_reads_state_and_failure_is_reported():
    client, _ = _hub({
        ("POST", "/hub/api/users/7/server"): lambda _: httpx.Response(400, json={"message": "7 is already running"}),
        ("GET", "/hub/api/users/7"): lambda _: httpx.Response(200, json={"servers": {"": {"ready": True, "pending": None}}}),
    })
    async with client:
        assert (await lab_hub.start_server(client, 7)).state == "ready"

    client, _ = _hub({
        ("POST", "/hub/api/users/7/server"): lambda _: httpx.Response(500, json={"message": "image not found"}),
    })
    async with client:
        failed = await lab_hub.start_server(client, 7)
    assert failed.state == "failed" and failed.message == "image not found"


@pytest.mark.asyncio
async def test_stop_waits_until_gone(monkeypatch):
    ticks = {"n": 0}

    def get_user(_):
        ticks["n"] += 1
        servers = {"": {"ready": False, "pending": "stop"}} if ticks["n"] < 2 else {}
        return httpx.Response(200, json={"servers": servers})

    async def no_sleep(_):
        return None

    monkeypatch.setattr(lab_hub.asyncio, "sleep", no_sleep)
    client, calls = _hub({
        ("DELETE", "/hub/api/users/7/server"): lambda _: httpx.Response(202, text=""),
        ("GET", "/hub/api/users/7"): get_user,
    })
    async with client:
        await lab_hub.stop_server(client, 7, wait_seconds=5)
    assert ticks["n"] == 2 and calls[0][0] == "DELETE"
