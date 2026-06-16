"""Backtest runner orchestration.

The backend no longer spawns ``strategy-backtest`` per run. That service is
always on. For each backtest the backend spawns one ``strategy-runner``
container in ``BACKTEST_MODE=true``; the runner consumes ``bars:backtest-{id}``
and routes orders back to the always-on strategy-backtest service.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import docker
import httpx
from docker.errors import APIError, NotFound
from docker.models.containers import Container

logger = logging.getLogger(__name__)

DOCKER_NETWORK = os.getenv("DOCKER_NETWORK", "edgewalker-devops_default")
TRAEFIK_DOCKER_NETWORK = os.getenv("TRAEFIK_DOCKER_NETWORK", "").strip()
EDGEWALKER_PATH = os.getenv("EDGEWALKER_PATH", "/home/flavio/playground/edgewalker")
RUNTIME_PATH = os.getenv("RUNTIME_PATH", "/home/flavio/playground/edgewalker-runtime")
SPAWN_CODE_MOUNTS = os.getenv("SPAWN_CODE_MOUNTS", "false").lower() == "true"
JWT_KEY_DIR = os.getenv("JWT_KEY_DIR", "").strip()
RUNNER_IMAGE = os.getenv("RUNNER_IMAGE", "").strip()
BACKTEST_SERVICE_URL = os.getenv("BACKTEST_SERVICE_URL", "http://strategy-backtest:8080").strip().rstrip("/")
BACKTEST_DEBUG_HOLD_SECONDS = os.getenv("BACKTEST_DEBUG_HOLD_SECONDS", "300")
CONTAINER_PREFIX = "edgewalker-backtest-runner-"

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_URL = os.getenv("REDIS_URL", "")
REDIS_USERNAME = os.getenv("REDIS_USERNAME", "")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

N8N_INTERNAL_URL = os.getenv("N8N_INTERNAL_URL", "http://n8n:5678")
N8N_PATH_PREFIX = os.getenv("N8N_PATH_PREFIX", "/n8n")

RUNNER_CORS_ALLOWED_ORIGINS = os.getenv(
    "RUNNER_CORS_ALLOWED_ORIGINS",
    os.getenv("BACKEND_CORS_ORIGINS", ""),
)
RUNNER_CORS_ALLOW_CREDENTIALS = os.getenv("RUNNER_CORS_ALLOW_CREDENTIALS", "true")
RUNNER_TRAEFIK_ENTRYPOINTS = os.getenv("RUNNER_TRAEFIK_ENTRYPOINTS", "").strip()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")


def _normalize_root_path(value: str | None) -> str:
    if not value or value == "/":
        return ""

    normalized = value.strip()
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    return normalized.rstrip("/")


API_ROOT_PATH = _normalize_root_path(os.getenv("API_ROOT_PATH", "/api"))


def _public_runner_route(*segments: str) -> str:
    suffix = "/".join(segment.strip("/") for segment in segments if segment)
    if not suffix:
        return API_ROOT_PATH or "/"
    return f"{API_ROOT_PATH}/{suffix}" if API_ROOT_PATH else f"/{suffix}"


def _docker_runtime_requirements() -> str:
    return (
        "backend must have Docker Engine access and the Docker network "
        f"'{DOCKER_NETWORK}' must exist; host paths EDGEWALKER_PATH={EDGEWALKER_PATH} "
        f"and RUNTIME_PATH={RUNTIME_PATH} must be valid absolute paths on the Docker host"
    )


def _required_runner_image() -> str:
    if RUNNER_IMAGE:
        return RUNNER_IMAGE
    raise RuntimeError("Missing RUNNER_IMAGE for backend-spawned backtest runners")


def _rewrite_webhook_for_docker(url: str) -> str:
    from urllib.parse import urlparse, urlunparse

    parsed = urlparse(url)
    path = parsed.path or ""
    if not path.startswith(N8N_PATH_PREFIX):
        return url

    new_path = path[len(N8N_PATH_PREFIX):]
    if not new_path.startswith("/"):
        new_path = "/" + new_path

    internal = urlparse(N8N_INTERNAL_URL)
    return urlunparse((
        internal.scheme,
        internal.netloc,
        new_path,
        parsed.params,
        parsed.query,
        parsed.fragment,
    ))


class BacktestRunnerService:
    """Spawn and manage strategy-runner containers for backtests."""

    def __init__(self) -> None:
        self._client: docker.DockerClient | None = None

    @property
    def client(self) -> docker.DockerClient:
        if self._client is None:
            try:
                self._client = docker.from_env()
            except Exception as exc:
                raise RuntimeError(f"Docker is not available; {_docker_runtime_requirements()}") from exc
        return self._client

    def _container_name(self, backtest_id: int) -> str:
        return f"{CONTAINER_PREFIX}{backtest_id}"

    def _get_container(self, backtest_id: int) -> Container | None:
        try:
            return self.client.containers.get(self._container_name(backtest_id))
        except NotFound:
            return None

    def start_backtest(
        self,
        backtest_id: int,
        connection_id: int | str,
        strategy_id: int,
        strategy_config: dict[str, Any],
        symbol: str,
        timeframe: str,
        *,
        manager_webhook_url: str | None = None,
        backend_auth_token: str | None = None,
        manager_webhook_auth_token: str | None = None,
        manager_chat_session_id: str | None = None,
        owner_user_id: int | str | None = None,
    ) -> dict[str, Any]:
        """Start a strategy-runner container in backtest mode."""
        try:
            with httpx.Client(timeout=5.0) as client:
                client.get(f"{BACKTEST_SERVICE_URL}/health").raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"Backtest coordinator is not reachable at {BACKTEST_SERVICE_URL}: {exc}"
            ) from exc

        container_name = self._container_name(backtest_id)
        existing = self._get_container(backtest_id)
        if existing:
            existing.reload()
            if existing.status == "running":
                return {
                    "status": "already_running",
                    "container_id": existing.short_id,
                    "container_name": container_name,
                    "stream_id": f"backtest-{backtest_id}",
                }
            existing.remove(force=True)

        stream_id = f"backtest-{backtest_id}"
        runner_prefix = _public_runner_route("runners", stream_id)
        backend_url = os.getenv("BACKEND_URL", "http://backend:8000")
        env: dict[str, str] = {
            "BACKTEST_MODE": "true",
            "BACKTEST_ID": str(backtest_id),
            "BACKTEST_STREAM_ID": stream_id,
            "BACKTEST_SERVICE_URL": BACKTEST_SERVICE_URL,
            "BACKTEST_DEBUG_HOLD_SECONDS": BACKTEST_DEBUG_HOLD_SECONDS,
            "REAL_DATA_CONNECTION_ID": str(connection_id),
            "CONNECTION_ID": str(connection_id),
            "REDIS_HOST": REDIS_HOST,
            "REDIS_PORT": REDIS_PORT,
            "STRATEGY_ID": str(strategy_id),
            "OWNER_USER_ID": str(owner_user_id or ""),
            "STRATEGY_CONFIG_JSON": json.dumps(strategy_config),
            "SYMBOL": symbol,
            "STREAMS": f"bars:{stream_id}",
            "CONSUMER_GROUP": f"cg:backtest-runner:{backtest_id}",
            "CONSUMER_ID": f"backtest-runner-{backtest_id}",
            "EVAL_IN_PROGRESS": "false",
            "DEBUG_RULES": os.getenv("BACKTEST_DEBUG_RULES", os.getenv("DEBUG_RULES", "true")),
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
            "PYTHONPATH": "/app",
            "RUNNER_INTERNAL_URL": f"http://{container_name}:8080",
            "BACKEND_URL": backend_url,
            "RUNNER_AGENT_TOKEN_URL": f"{backend_url}/runners/agent-token",
            "DATABASE_URL": os.getenv("DATABASE_URL", ""),
            "ALGORITHM": os.getenv("ALGORITHM", "RS256"),
            "JWT_ISSUER": os.getenv("JWT_ISSUER", "edgewalker-backend"),
            "JWT_PUBLIC_KEY_PATH": os.getenv("JWT_PUBLIC_KEY_PATH", "/run/secrets/edgewalker-jwt/public.pem"),
            "ACCESS_TOKEN_AUDIENCE": os.getenv("ACCESS_TOKEN_AUDIENCE", "edgewalker-ui"),
            "RUNNER_TOKEN_AUDIENCE": os.getenv("RUNNER_TOKEN_AUDIENCE", "edgewalker-runner"),
            "AGENT_TOKEN_AUDIENCE": os.getenv("AGENT_TOKEN_AUDIENCE", "edgewalker-agent"),
            "CORS_ALLOWED_ORIGINS": RUNNER_CORS_ALLOWED_ORIGINS,
            "CORS_ALLOW_CREDENTIALS": RUNNER_CORS_ALLOW_CREDENTIALS,
            "BROKER_SYNC_ENABLED": "false",
            "LIVE_RECONCILIATION_ENABLED": "false",
            "POSITION_SYNC_INTERVAL_SECONDS": "0",
            "OTEL_EXPORTER_OTLP_ENDPOINT": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317"),
            "OTEL_SERVICE_NAME": f"strategy-runner-backtest-{backtest_id}",
        }
        if PUBLIC_BASE_URL:
            env["RUNNER_PUBLIC_BASE_URL"] = f"{PUBLIC_BASE_URL}{runner_prefix}"
        if REDIS_URL:
            env["REDIS_URL"] = REDIS_URL
        if REDIS_USERNAME:
            env["REDIS_USERNAME"] = REDIS_USERNAME
        if REDIS_PASSWORD:
            env["REDIS_PASSWORD"] = REDIS_PASSWORD
        if backend_auth_token:
            env["BACKEND_AUTH_TOKEN"] = backend_auth_token
        if manager_webhook_url:
            env["MANAGER_WEBHOOK_URL"] = _rewrite_webhook_for_docker(manager_webhook_url)
        if manager_webhook_auth_token:
            env["MANAGER_WEBHOOK_AUTH_TOKEN"] = manager_webhook_auth_token
        if manager_chat_session_id:
            env["MANAGER_CHAT_SESSION_ID"] = manager_chat_session_id

        labels = {
            "traefik.enable": "true",
            f"traefik.http.routers.backtest-runner-{backtest_id}.rule": f"PathPrefix(`{runner_prefix}`)",
            f"traefik.http.routers.backtest-runner-{backtest_id}.priority": "200",
            f"traefik.http.routers.backtest-runner-{backtest_id}.service": f"backtest-runner-{backtest_id}",
            f"traefik.http.middlewares.backtest-runner-{backtest_id}-strip.stripprefix.prefixes": runner_prefix,
            f"traefik.http.middlewares.backtest-runner-{backtest_id}-cors.headers.accesscontrolalloworiginlist": RUNNER_CORS_ALLOWED_ORIGINS,
            f"traefik.http.middlewares.backtest-runner-{backtest_id}-cors.headers.accesscontrolallowmethods": "GET,POST,PUT,PATCH,DELETE,OPTIONS",
            f"traefik.http.middlewares.backtest-runner-{backtest_id}-cors.headers.accesscontrolallowheaders": "*",
            f"traefik.http.middlewares.backtest-runner-{backtest_id}-cors.headers.accesscontrolallowcredentials": RUNNER_CORS_ALLOW_CREDENTIALS.lower(),
            f"traefik.http.middlewares.backtest-runner-{backtest_id}-cors.headers.addvaryheader": "true",
            f"traefik.http.routers.backtest-runner-{backtest_id}.middlewares": f"backtest-runner-{backtest_id}-strip,backtest-runner-{backtest_id}-cors",
            f"traefik.http.services.backtest-runner-{backtest_id}.loadbalancer.server.port": "8080",
            "edgewalker.type": "strategy-runner-backtest",
            "edgewalker.backtest_id": str(backtest_id),
            "edgewalker.strategy_id": str(strategy_id),
            "edgewalker.symbol": symbol,
            "edgewalker.stream_id": stream_id,
        }
        if TRAEFIK_DOCKER_NETWORK:
            labels["traefik.docker.network"] = TRAEFIK_DOCKER_NETWORK
        if RUNNER_TRAEFIK_ENTRYPOINTS:
            labels[f"traefik.http.routers.backtest-runner-{backtest_id}.entrypoints"] = RUNNER_TRAEFIK_ENTRYPOINTS

        volumes = {
            f"{EDGEWALKER_PATH}/strategies": {"bind": "/opt/edgewalker/strategies", "mode": "ro"},
            f"{EDGEWALKER_PATH}/artifacts": {"bind": "/opt/edgewalker/artifacts", "mode": "ro"},
        }
        if SPAWN_CODE_MOUNTS:
            volumes.update({
                f"{RUNTIME_PATH}/strategy-runner/app": {"bind": "/app/app", "mode": "ro"},
                f"{EDGEWALKER_PATH}/edgewalker": {"bind": "/opt/edgewalker/edgewalker", "mode": "ro"},
                f"{RUNTIME_PATH}/shared": {"bind": "/app/shared", "mode": "ro"},
            })
        if JWT_KEY_DIR:
            volumes[JWT_KEY_DIR] = {"bind": "/run/secrets/edgewalker-jwt", "mode": "ro"}

        image_name = _required_runner_image()
        try:
            container = self.client.containers.run(
                image=image_name,
                name=container_name,
                environment=env,
                labels=labels,
                volumes=volumes,
                network=DOCKER_NETWORK,
                detach=True,
                auto_remove=False,
                restart_policy={"Name": "no"},
                extra_hosts={"host.docker.internal": "host-gateway"},
            )
            if TRAEFIK_DOCKER_NETWORK and TRAEFIK_DOCKER_NETWORK != DOCKER_NETWORK:
                self.client.networks.get(TRAEFIK_DOCKER_NETWORK).connect(container)
            logger.info("Started backtest runner: %s (%s)", container_name, container.short_id)
            return {
                "status": "started",
                "container_id": container.short_id,
                "container_name": container_name,
                "stream_id": stream_id,
                "image": image_name,
            }
        except APIError as exc:
            logger.error("Failed to start backtest runner %s: %s", container_name, exc)
            raise RuntimeError(f"Failed to start backtest runner: {exc}") from exc

    def stop_backtest(self, backtest_id: int, remove: bool = True) -> dict[str, Any]:
        service_result: dict[str, Any] = {}
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.post(f"{BACKTEST_SERVICE_URL}/backtests/{backtest_id}/cancel")
                service_result = resp.json() if resp.content else {"status": resp.status_code}
        except Exception as exc:
            service_result = {
                "status": "service_cancel_failed",
                "service_url": BACKTEST_SERVICE_URL,
                "error": str(exc),
            }

        container = self._get_container(backtest_id)
        if not container:
            return {"status": "not_found", "service": service_result}
        try:
            if container.status == "running":
                container.stop(timeout=30)
            if remove:
                container.remove(force=True)
                return {"status": "stopped_and_removed", "service": service_result}
            return {"status": "stopped", "service": service_result}
        except APIError as exc:
            raise RuntimeError(f"Failed to stop backtest runner: {exc}") from exc

    def get_backtest_status(self, backtest_id: int) -> dict[str, Any]:
        container = self._get_container(backtest_id)
        if not container:
            container_status: dict[str, Any] = {"status": "not_found", "running": False}
        else:
            container.reload()
            health = container.attrs.get("State", {}).get("Health", {})
            container_status = {
                "status": container.status,
                "running": container.status == "running",
                "container_id": container.short_id,
                "container_name": container.name,
                "health_status": health.get("Status") if health else None,
                "labels": container.labels,
            }

        service_status: dict[str, Any] = {}
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{BACKTEST_SERVICE_URL}/backtests/{backtest_id}/status")
                service_status = resp.json()
        except Exception as exc:
            service_status = {
                "status": "unavailable",
                "service_url": BACKTEST_SERVICE_URL,
                "error": str(exc),
            }

        service_status.setdefault("service_url", BACKTEST_SERVICE_URL)
        stream_status = self._get_stream_diagnostics(backtest_id)
        runner_status = self._get_runner_diagnostics(container) if container and container.status == "running" else {
            "status": "not_running",
        }

        return {
            "container": container_status,
            "service": service_status,
            "stream": stream_status,
            "runner": runner_status,
        }

    def control_backtest_playback(self, backtest_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.post(f"{BACKTEST_SERVICE_URL}/backtests/{backtest_id}/control", json=payload)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as exc:
            detail: Any
            try:
                detail = exc.response.json()
            except Exception:
                detail = exc.response.text
            raise RuntimeError(f"Backtest playback control failed: {detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"Backtest coordinator is not reachable at {BACKTEST_SERVICE_URL}: {exc}") from exc

    def list_backtest_orders(self, backtest_id: int, *, active_only: bool = False) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{BACKTEST_SERVICE_URL}/backtests/{backtest_id}/orders")
                if resp.status_code == 404:
                    return {"orders": []}
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            detail: Any
            try:
                detail = exc.response.json()
            except Exception:
                detail = exc.response.text
            raise RuntimeError(f"Backtest orders fetch failed: {detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"Backtest coordinator is not reachable at {BACKTEST_SERVICE_URL}: {exc}") from exc

        if isinstance(data, dict):
            raw_orders = data.get("orders") or []
        elif isinstance(data, list):
            raw_orders = data
        else:
            raw_orders = []
        orders = list(raw_orders)
        if active_only:
            active_statuses = {"pending", "submitted", "partially_filled"}
            orders = [order for order in orders if str(order.get("status") or "").lower() in active_statuses]
        return {"orders": orders}

    def get_backtest_position(self, backtest_id: int) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{BACKTEST_SERVICE_URL}/backtests/{backtest_id}/positions")
                if resp.status_code == 404:
                    return {"side": "flat", "quantity": 0}
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as exc:
            detail: Any
            try:
                detail = exc.response.json()
            except Exception:
                detail = exc.response.text
            raise RuntimeError(f"Backtest position fetch failed: {detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"Backtest coordinator is not reachable at {BACKTEST_SERVICE_URL}: {exc}") from exc

    def _create_redis_client(self):
        import redis as sync_redis

        if REDIS_URL:
            return sync_redis.from_url(REDIS_URL, decode_responses=True)

        kwargs: dict[str, Any] = {
            "host": REDIS_HOST,
            "port": int(REDIS_PORT),
            "decode_responses": True,
        }
        if REDIS_USERNAME:
            kwargs["username"] = REDIS_USERNAME
        if REDIS_PASSWORD:
            kwargs["password"] = REDIS_PASSWORD
        return sync_redis.Redis(**kwargs)

    def _get_stream_diagnostics(self, backtest_id: int) -> dict[str, Any]:
        stream_key = f"bars:backtest-{backtest_id}"
        consumer_group = f"cg:backtest-runner:{backtest_id}"
        try:
            redis_client = self._create_redis_client()
            try:
                exists = bool(redis_client.exists(stream_key))
                diagnostics: dict[str, Any] = {
                    "stream_key": stream_key,
                    "consumer_group": consumer_group,
                    "exists": exists,
                }
                if exists:
                    diagnostics["stream"] = redis_client.xinfo_stream(stream_key)
                    groups = redis_client.xinfo_groups(stream_key)
                    diagnostics["groups"] = groups
                    diagnostics["runner_group"] = next(
                        (group for group in groups if group.get("name") == consumer_group),
                        None,
                    )
                    try:
                        diagnostics["consumers"] = redis_client.xinfo_consumers(stream_key, consumer_group)
                    except Exception as exc:
                        diagnostics["consumers_error"] = str(exc)
                return diagnostics
            finally:
                redis_client.close()
        except Exception as exc:
            return {
                "stream_key": stream_key,
                "consumer_group": consumer_group,
                "status": "unavailable",
                "error": str(exc),
            }

    @staticmethod
    def _container_env(container: Container, key: str) -> str | None:
        prefix = f"{key}="
        for item in container.attrs.get("Config", {}).get("Env", []) or []:
            if item.startswith(prefix):
                return item[len(prefix):]
        return None

    def _get_runner_diagnostics(self, container: Container) -> dict[str, Any]:
        base_url = f"http://{container.name}:8080"
        token = self._container_env(container, "BACKEND_AUTH_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        diagnostics: dict[str, Any] = {"base_url": base_url, "auth": "set" if token else "missing"}

        endpoints = {
            "health": "/health",
            "metrics": "/metrics",
            "state": "/state",
            "config": "/config",
            "debug": "/debug",
        }
        try:
            with httpx.Client(base_url=base_url, timeout=2.0, headers=headers) as client:
                for name, path in endpoints.items():
                    try:
                        resp = client.get(path)
                        content_type = resp.headers.get("content-type", "")
                        diagnostics[name] = resp.json() if content_type.startswith("application/json") else resp.text
                        diagnostics[f"{name}_status_code"] = resp.status_code
                    except Exception as exc:
                        diagnostics[name] = {"status": "unavailable", "error": str(exc)}
        except Exception as exc:
            diagnostics["status"] = "unavailable"
            diagnostics["error"] = str(exc)
        return diagnostics

    def get_container_logs(self, backtest_id: int, tail: int = 200) -> str:
        container = self._get_container(backtest_id)
        if not container:
            raise ValueError(f"Runner container for backtest {backtest_id} not found")
        return container.logs(tail=tail, timestamps=True).decode("utf-8")

    def cleanup_finished(self) -> int:
        containers = self.client.containers.list(
            all=True,
            filters={"label": "edgewalker.type=strategy-runner-backtest", "status": "exited"},
        )
        count = 0
        for container in containers:
            try:
                container.remove(force=True)
                count += 1
            except Exception as exc:
                logger.warning("Failed to remove container %s: %s", container.name, exc)
        return count


backtest_runner_service = BacktestRunnerService()