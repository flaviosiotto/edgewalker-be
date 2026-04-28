"""
Live Strategy Runner Service.

Manages Docker containers for live strategy execution.
Each live instance runs in its own container.

The containers:
- Use the strategy-runner image
- Are labeled for Traefik routing
- Subscribe to Redis for market data
- Execute the same logic as backtest
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import docker
from docker.errors import APIError, NotFound
from docker.models.containers import Container

logger = logging.getLogger(__name__)

# Docker network for internal communication
DOCKER_NETWORK = os.getenv("DOCKER_NETWORK", "edgewalker-devops_default")

# Host paths used for Docker bind mounts when spawning runner containers.
EDGEWALKER_PATH = os.getenv("EDGEWALKER_PATH", "/home/flavio/playground/edgewalker")
RUNTIME_PATH = os.getenv("RUNTIME_PATH", "/home/flavio/playground/edgewalker-runtime")
SPAWN_CODE_MOUNTS = os.getenv("SPAWN_CODE_MOUNTS", "false").lower() == "true"

# Strategy runner image must be explicitly configured by environment.
RUNNER_IMAGE = os.getenv("RUNNER_IMAGE", "").strip()

# Container naming convention
CONTAINER_PREFIX = "edgewalker-live-"

# Redis settings
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_URL = os.getenv("REDIS_URL", "")
REDIS_USERNAME = os.getenv("REDIS_USERNAME", "")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

# n8n internal address (Docker service name + internal port)
N8N_INTERNAL_URL = os.getenv("N8N_INTERNAL_URL", "http://n8n:5678")
# n8n path prefix used by Traefik (stripped before forwarding to n8n)
N8N_PATH_PREFIX = os.getenv("N8N_PATH_PREFIX", "/n8n")

# CORS for live runner APIs exposed directly through Traefik and the spawned runner app.
# By default reuse BACKEND_CORS_ORIGINS so the backend and runner share one source of truth.
def _parse_cors_origins(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]

    return [item.strip() for item in value.split(",") if item.strip()]


DEFAULT_CORS_ORIGINS = _parse_cors_origins(
    os.getenv("BACKEND_CORS_ORIGINS"),
    ["https://edgewalker.tech"],
)

RUNNER_CORS_ALLOWED_ORIGIN_LIST = _parse_cors_origins(
    os.getenv("RUNNER_CORS_ALLOWED_ORIGINS"),
    DEFAULT_CORS_ORIGINS,
)
RUNNER_CORS_ALLOWED_ORIGINS = ",".join(RUNNER_CORS_ALLOWED_ORIGIN_LIST)
RUNNER_CORS_ALLOW_CREDENTIALS = os.getenv("RUNNER_CORS_ALLOW_CREDENTIALS", "true").lower() in (
    "true",
    "1",
    "yes",
)


def _docker_runtime_requirements() -> str:
    return (
        "backend must have Docker Engine access (for example via /var/run/docker.sock or DOCKER_HOST) "
        f"and the Docker network '{DOCKER_NETWORK}' must exist; host paths "
        f"EDGEWALKER_PATH={EDGEWALKER_PATH} and RUNTIME_PATH={RUNTIME_PATH} must be valid absolute paths on the Docker host"
    )


def _get_required_runner_image() -> str:
    """Return configured RUNNER_IMAGE or fail with a clear error."""
    if RUNNER_IMAGE:
        return RUNNER_IMAGE
    raise RuntimeError(
        "Missing required environment variable RUNNER_IMAGE for backend. "
        "Set it to the strategy runner image tag (example: edgewalker-strategy-runner:latest)."
    )


def _rewrite_webhook_for_docker(url: str) -> str:
    """Rewrite an external webhook URL to a Docker-internal URL.

    Agent webhook URLs are stored with the external host (e.g.
    ``http://localhost:8081/n8n/webhook/abc``).  Inside the Docker
    network the runner must reach n8n directly at ``n8n:5678`` and
    without the ``/n8n`` Traefik path prefix.

    The rewrite is applied when the URL contains the ``N8N_PATH_PREFIX``
    segment (default ``/n8n``).  Other URLs are returned unchanged.
    """
    from urllib.parse import urlparse, urlunparse

    parsed = urlparse(url)
    path = parsed.path

    # Only rewrite URLs that go through Traefik to n8n
    if not path.startswith(N8N_PATH_PREFIX):
        return url

    # Strip the Traefik prefix (/n8n/webhook/abc -> /webhook/abc)
    new_path = path[len(N8N_PATH_PREFIX):]
    if not new_path.startswith("/"):
        new_path = "/" + new_path

    # Build internal URL using the n8n Docker service
    internal = urlparse(N8N_INTERNAL_URL)
    rewritten = urlunparse((
        internal.scheme,
        internal.netloc,
        new_path,
        parsed.params,
        parsed.query,
        parsed.fragment,
    ))
    logger.debug("Rewrote webhook URL: %s -> %s", url, rewritten)
    return rewritten


class LiveRunnerService:
    """
    Service for managing live strategy runner containers.
    
    Provides start/stop/status operations for strategy containers.
    """
    
    def __init__(self):
        self._client: docker.DockerClient | None = None
    
    @property
    def client(self) -> docker.DockerClient:
        """Get or create Docker client."""
        if self._client is None:
            try:
                self._client = docker.from_env()
            except Exception as e:
                raise RuntimeError(f"Docker is not available; {_docker_runtime_requirements()}") from e
        return self._client
    
    def _container_name(self, live_id: int) -> str:
        """Generate container name for a live instance."""
        return f"{CONTAINER_PREFIX}{live_id}"
    
    def _get_container(self, live_id: int) -> Container | None:
        """Get container for a live instance if it exists."""
        try:
            return self.client.containers.get(self._container_name(live_id))
        except NotFound:
            return None

    def _find_container_by_live_id(self, live_id: int) -> Container | None:
        """Find a container by live-id label as a fallback during migrations."""
        containers = self.client.containers.list(
            all=True,
            filters={"label": f"edgewalker.live_id={live_id}"},
        )
        return containers[0] if containers else None
    
    def start_live_instance(
        self,
        live_id: int,
        strategy_id: int,
        strategy_config: dict[str, Any],
        symbol: str,
        timeframe: str = "5s",
        eval_in_progress: bool = True,
        debug_rules: bool = False,
        account_config: dict[str, Any] | None = None,
        broker_type: str | None = None,
        manager_webhook_url: str | None = None,
        manager_chat_session_id: str | None = None,
        strategy_live_id: int | None = None,
        legacy_strategy_route: bool = False,
    ) -> dict[str, Any]:
        """
        Start a live strategy runner container.
        
        Args:
            strategy_id: Database ID of the strategy
            strategy_config: Strategy definition (YAML-like dict) - kept for reference,
                runner loads config from DB via STRATEGY_LIVE_ID
            symbol: Trading symbol to subscribe to
            timeframe: Bar timeframe for subscription (default: 5s)
            eval_in_progress: Evaluate rules on in-progress bars too (not only bar close)
            debug_rules: Enable detailed condition logging
            account_config: Broker account config (host, port, account_id, etc.)
            broker_type: Broker type identifier (e.g. 'ibkr')
            manager_webhook_url: n8n webhook URL for manager agent
            manager_chat_session_id: Chat session ID for manager agent
            strategy_live_id: Database ID of the strategy_live session
            
        Returns:
            Dict with container info and status
        """
        container_name = self._container_name(live_id)
        
        # Check if container already exists
        existing = self._get_container(live_id) or self._find_container_by_live_id(live_id)
        if existing:
            existing.reload()
            if existing.status == "running":
                return {
                    "status": "already_running",
                    "container_id": existing.short_id,
                    "container_name": container_name,
                }
            else:
                # Remove stopped container
                existing.remove(force=True)
        
        # Build environment variables
        # Connection ID determines the stream key suffix (architecture v3)
        conn_id = str(account_config["connection_id"]) if account_config and "connection_id" in account_config else ""
        conn_suffix = f":{conn_id}" if conn_id else ""
        
        env = {
            "REDIS_HOST": REDIS_HOST,
            "REDIS_PORT": REDIS_PORT,
            "STRATEGY_ID": str(strategy_id),
            "STRATEGY_LIVE_ID": str(live_id),
            "SYMBOL": symbol,
            # Single live:bars stream - runner derives from DB config if STRATEGY_LIVE_ID is set
            "STREAMS": f"live:bars:{symbol}:{timeframe}{conn_suffix}",
            "CONSUMER_GROUP": f"cg:strategy-{strategy_id}",
            "CONSUMER_ID": f"runner-{strategy_id}",
            "EVAL_IN_PROGRESS": str(eval_in_progress).lower(),
            "DEBUG_RULES": str(debug_rules).lower(),
            "LOG_LEVEL": "DEBUG" if debug_rules else "INFO",
            "PYTHONPATH": "/app",
            # Backend API URL for manager agent notifications
            "BACKEND_URL": os.getenv("BACKEND_URL", "http://backend:8000"),
            # Database URL for direct order/position persistence + config loading
            "DATABASE_URL": os.getenv("DATABASE_URL", ""),
            # OpenTelemetry - send metrics/traces to the shared OTel Collector
            "OTEL_EXPORTER_OTLP_ENDPOINT": os.getenv(
                "OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317"
            ),
            "OTEL_SERVICE_NAME": f"strategy-runner-{strategy_id}",
        }

        if REDIS_URL:
            env["REDIS_URL"] = REDIS_URL
        if REDIS_USERNAME:
            env["REDIS_USERNAME"] = REDIS_USERNAME
        if REDIS_PASSWORD:
            env["REDIS_PASSWORD"] = REDIS_PASSWORD
        env["CORS_ALLOWED_ORIGINS"] = RUNNER_CORS_ALLOWED_ORIGINS
        env["CORS_ALLOW_CREDENTIALS"] = str(RUNNER_CORS_ALLOW_CREDENTIALS).lower()

        # Manager agent webhook (so the runner can call the agent directly)
        if manager_webhook_url:
            env["MANAGER_WEBHOOK_URL"] = _rewrite_webhook_for_docker(manager_webhook_url)
        if manager_chat_session_id:
            env["MANAGER_CHAT_SESSION_ID"] = manager_chat_session_id
        
        # Add broker / account configuration if provided
        if broker_type:
            env["BROKER_TYPE"] = broker_type
        if account_config:
            env["ACCOUNT_CONFIG_JSON"] = json.dumps(account_config)
            # Also set individual vars for convenience
            if "account_id" in account_config:
                env["BROKER_ACCOUNT_ID"] = str(account_config["account_id"])
            if "db_account_id" in account_config:
                env["DB_ACCOUNT_ID"] = str(account_config["db_account_id"])
            if "host" in account_config:
                env["BROKER_HOST"] = str(account_config["host"])
            if "port" in account_config:
                env["BROKER_PORT"] = str(account_config["port"])
            if "client_id" in account_config:
                env["BROKER_CLIENT_ID"] = str(account_config["client_id"])
            if "connection_id" in account_config:
                env["CONNECTION_ID"] = str(account_config["connection_id"])
        
        # Traefik labels for routing
        # Route: /api/runners/{strategy_id}/* -> container:8080
        # Priority must be higher than the backend's generic /api route
        labels = {
            "traefik.enable": "true",
            f"traefik.http.routers.live-runner-{live_id}.rule": f"PathPrefix(`/live/instances/{live_id}/runner`)",
            f"traefik.http.routers.live-runner-{live_id}.entrypoints": "web,websecure",
            f"traefik.http.routers.live-runner-{live_id}.priority": "200",
            f"traefik.http.middlewares.live-runner-{live_id}-strip.stripprefix.prefixes": f"/live/instances/{live_id}/runner",
            f"traefik.http.middlewares.live-runner-{live_id}-cors.headers.accesscontrolalloworiginlist": RUNNER_CORS_ALLOWED_ORIGINS,
            f"traefik.http.middlewares.live-runner-{live_id}-cors.headers.accesscontrolallowmethods": "GET,POST,PUT,PATCH,DELETE,OPTIONS",
            f"traefik.http.middlewares.live-runner-{live_id}-cors.headers.accesscontrolallowheaders": "*",
            f"traefik.http.middlewares.live-runner-{live_id}-cors.headers.accesscontrolallowcredentials": str(RUNNER_CORS_ALLOW_CREDENTIALS).lower(),
            f"traefik.http.middlewares.live-runner-{live_id}-cors.headers.addvaryheader": "true",
            f"traefik.http.routers.live-runner-{live_id}.middlewares": f"live-runner-{live_id}-strip,live-runner-{live_id}-cors",
            f"traefik.http.services.live-runner-{live_id}.loadbalancer.server.port": "8080",
            # Custom labels for identification
            "edgewalker.type": "strategy-runner",
            "edgewalker.strategy_id": str(strategy_id),
            "edgewalker.live_id": str(live_id),
            "edgewalker.symbol": symbol,
        }

        if legacy_strategy_route:
            labels.update({
                f"traefik.http.routers.runner-{strategy_id}.rule": f"PathPrefix(`/runners/{strategy_id}`)",
                f"traefik.http.routers.runner-{strategy_id}.entrypoints": "web,websecure",
                f"traefik.http.routers.runner-{strategy_id}.priority": "200",
                f"traefik.http.middlewares.runner-{strategy_id}-strip.stripprefix.prefixes": f"/runners/{strategy_id}",
                f"traefik.http.middlewares.runner-{strategy_id}-cors.headers.accesscontrolalloworiginlist": RUNNER_CORS_ALLOWED_ORIGINS,
                f"traefik.http.middlewares.runner-{strategy_id}-cors.headers.accesscontrolallowmethods": "GET,POST,PUT,PATCH,DELETE,OPTIONS",
                f"traefik.http.middlewares.runner-{strategy_id}-cors.headers.accesscontrolallowheaders": "*",
                f"traefik.http.middlewares.runner-{strategy_id}-cors.headers.accesscontrolallowcredentials": str(RUNNER_CORS_ALLOW_CREDENTIALS).lower(),
                f"traefik.http.middlewares.runner-{strategy_id}-cors.headers.addvaryheader": "true",
                f"traefik.http.routers.runner-{strategy_id}.middlewares": f"runner-{strategy_id}-strip,runner-{strategy_id}-cors",
                f"traefik.http.services.runner-{strategy_id}.loadbalancer.server.port": "8080",
            })
        
        # Volume mounts
        volumes = {
            # Strategy configs and artifacts
            f"{EDGEWALKER_PATH}/strategies": {
                "bind": "/opt/edgewalker/strategies",
                "mode": "ro",
            },
            f"{EDGEWALKER_PATH}/artifacts": {
                "bind": "/opt/edgewalker/artifacts",
                "mode": "ro",
            },
        }

        if SPAWN_CODE_MOUNTS:
            volumes.update({
                # Runner application source (local dev: host-mounted for live code changes)
                f"{RUNTIME_PATH}/strategy-runner/app": {
                    "bind": "/app/app",
                    "mode": "ro",
                },
                # Edgewalker library source overlay for local development.
                f"{EDGEWALKER_PATH}/edgewalker": {
                    "bind": "/opt/edgewalker/edgewalker",
                    "mode": "ro",
                },
                # Shared schemas overlay for local development.
                f"{RUNTIME_PATH}/shared": {
                    "bind": "/app/shared",
                    "mode": "ro",
                },
            })
        
        image_name = _get_required_runner_image()

        try:
            # Create and start container
            container = self.client.containers.run(
                image=image_name,
                name=container_name,
                environment=env,
                labels=labels,
                volumes=volumes,
                network=DOCKER_NETWORK,
                detach=True,
                auto_remove=False,
                restart_policy={"Name": "unless-stopped"},
                extra_hosts={"host.docker.internal": "host-gateway"},
            )
            
            logger.info("Started live runner container: %s (%s)", container_name, container.short_id)
            
            return {
                "status": "started",
                "container_id": container.short_id,
                "container_name": container_name,
                "image": image_name,
                "symbol": symbol,
                "timeframe": timeframe,
            }
            
        except APIError as e:
            logger.error("Failed to start container %s: %s", container_name, e)
            raise RuntimeError(f"Failed to start strategy runner (image={image_name}): {e}")

    def stop_live_instance(self, live_id: int, remove: bool = True) -> dict[str, Any]:
        """
        Stop a live strategy runner container.
        
        Args:
            live_id: Database ID of the live instance
            remove: Whether to remove the container after stopping
            
        Returns:
            Dict with operation status
        """
        container = self._get_container(live_id) or self._find_container_by_live_id(live_id)
        
        if not container:
            return {"status": "not_found"}
        
        try:
            if container.status == "running":
                container.stop(timeout=30)
                logger.info(f"Stopped strategy runner: {container.name}")
            
            if remove:
                container.remove(force=True)
                logger.info(f"Removed strategy runner: {container.name}")
                return {"status": "stopped_and_removed"}
            
            return {"status": "stopped"}
            
        except APIError as e:
            logger.error(f"Failed to stop container: {e}")
            raise RuntimeError(f"Failed to stop strategy runner: {e}")
    
    def get_live_instance_status(self, live_id: int) -> dict[str, Any]:
        """
        Get status of a strategy runner container.
        
        Returns container state, health, and runtime info.
        """
        container = self._get_container(live_id) or self._find_container_by_live_id(live_id)
        
        if not container:
            return {
                "status": "not_found",
                "running": False,
            }
        
        # Refresh container state
        container.reload()
        
        health = container.attrs.get("State", {}).get("Health", {})
        
        return {
            "status": container.status,
            "running": container.status == "running",
            "container_id": container.short_id,
            "container_name": container.name,
            "created": container.attrs.get("Created"),
            "started_at": container.attrs.get("State", {}).get("StartedAt"),
            "health_status": health.get("Status") if health else None,
            "labels": container.labels,
        }
    
    def list_live_instances(self, include_all: bool = False) -> list[dict[str, Any]]:
        """
        List live strategy runner containers.
        
        Returns list of container info dicts.
        """
        filters: dict[str, Any] = {"label": "edgewalker.type=strategy-runner"}
        if not include_all:
            filters["status"] = "running"

        containers = self.client.containers.list(filters=filters)
        
        result = []
        for container in containers:
            result.append({
                "container_id": container.short_id,
                "container_name": container.name,
                "strategy_id": container.labels.get("edgewalker.strategy_id"),
                "live_id": container.labels.get("edgewalker.live_id"),
                "symbol": container.labels.get("edgewalker.symbol"),
                "status": container.status,
            })
        
        return result
    
    def get_container_logs(
        self,
        live_id: int,
        tail: int = 100,
        since: str | None = None,
    ) -> str:
        """
        Get logs from a strategy runner container.
        
        Args:
            live_id: Database ID of the live instance
            tail: Number of lines to return from end
            since: Return logs since this timestamp
            
        Returns:
            Log output as string
        """
        container = self._get_container(live_id) or self._find_container_by_live_id(live_id)
        
        if not container:
            raise ValueError(f"Container for live instance {live_id} not found")
        
        kwargs = {"tail": tail, "timestamps": True}
        if since:
            kwargs["since"] = since
        
        return container.logs(**kwargs).decode("utf-8")


# Global service instance
live_runner_service = LiveRunnerService()
