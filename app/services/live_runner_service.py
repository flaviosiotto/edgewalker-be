"""
Live Strategy Runner Service.

Manages Docker containers for live strategy execution.
Each strategy runs in its own container (1:1 mapping).

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
from docker.errors import NotFound, APIError
from docker.models.containers import Container

logger = logging.getLogger(__name__)

# Docker network for internal communication
DOCKER_NETWORK = os.getenv("DOCKER_NETWORK", "edgewalker-devops_default")

# Strategy runner image
RUNNER_IMAGE = os.getenv("RUNNER_IMAGE", "edgewalker-devops-strategy-runner:latest")

# Container naming convention
CONTAINER_PREFIX = "edgewalker-strategy-"

# Redis settings
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")

# n8n internal address (Docker service name + internal port)
N8N_INTERNAL_URL = os.getenv("N8N_INTERNAL_URL", "http://n8n:5678")
# n8n path prefix used by Traefik (stripped before forwarding to n8n)
N8N_PATH_PREFIX = os.getenv("N8N_PATH_PREFIX", "/n8n")


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
            self._client = docker.from_env()
        return self._client
    
    def _container_name(self, strategy_id: int) -> str:
        """Generate container name for a strategy."""
        return f"{CONTAINER_PREFIX}{strategy_id}"
    
    def _get_container(self, strategy_id: int) -> Container | None:
        """Get container for a strategy if it exists."""
        try:
            return self.client.containers.get(self._container_name(strategy_id))
        except NotFound:
            return None
    
    def start_strategy(
        self,
        strategy_id: int,
        strategy_config: dict[str, Any],
        symbol: str,
        timeframe: str = "5s",
        tick_eval: bool = True,
        debug_rules: bool = False,
        account_config: dict[str, Any] | None = None,
        broker_type: str | None = None,
        manager_webhook_url: str | None = None,
        manager_chat_session_id: str | None = None,
        strategy_live_id: int | None = None,
    ) -> dict[str, Any]:
        """
        Start a live strategy runner container.
        
        Args:
            strategy_id: Database ID of the strategy
            strategy_config: Strategy definition (YAML-like dict)
            symbol: Trading symbol to subscribe to
            timeframe: Bar timeframe for subscription (default: 5s)
            tick_eval: Whether to evaluate rules on every tick
            debug_rules: Enable detailed condition logging
            account_config: Broker account config (host, port, account_id, etc.)
            broker_type: Broker type identifier (e.g. 'ibkr')
            
        Returns:
            Dict with container info and status
        """
        container_name = self._container_name(strategy_id)
        
        # Check if container already exists
        existing = self._get_container(strategy_id)
        if existing:
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
        env = {
            "REDIS_HOST": REDIS_HOST,
            "REDIS_PORT": REDIS_PORT,
            "STRATEGY_ID": str(strategy_id),
            "SYMBOL": symbol,
            "TIMEFRAME": timeframe,
            # Streams to consume (market:* for reliable delivery via XREAD)
            # Format: market:bars:{symbol}:{tf}, market:ticks:{symbol}, market:quotes:{symbol}
            "STREAMS": f"market:bars:{symbol}:{timeframe},market:ticks:{symbol},market:quotes:{symbol}",
            "CONSUMER_GROUP": f"cg:strategy-{strategy_id}",
            "CONSUMER_ID": f"runner-{strategy_id}",
            # Strategy config as JSON (runner will persist if needed)
            "STRATEGY_CONFIG_JSON": json.dumps(strategy_config),
            "TICK_EVAL": str(tick_eval).lower(),
            "DEBUG_RULES": str(debug_rules).lower(),
            "LOG_LEVEL": "DEBUG" if debug_rules else "INFO",
            "PYTHONPATH": "/app",
            # Backend API URL for manager agent notifications
            "BACKEND_URL": os.getenv("BACKEND_URL", "http://backend:8000"),
            # Database URL for direct order/position persistence
            "DATABASE_URL": os.getenv("DATABASE_URL", ""),
        }

        # Strategy live session ID (for DB writes)
        if strategy_live_id:
            env["STRATEGY_LIVE_ID"] = str(strategy_live_id)

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
            f"traefik.http.routers.runner-{strategy_id}.rule": f"PathPrefix(`/api/runners/{strategy_id}`)",
            f"traefik.http.routers.runner-{strategy_id}.entrypoints": "http",
            f"traefik.http.routers.runner-{strategy_id}.priority": "200",
            f"traefik.http.middlewares.runner-{strategy_id}-strip.stripprefix.prefixes": f"/api/runners/{strategy_id}",
            f"traefik.http.routers.runner-{strategy_id}.middlewares": f"runner-{strategy_id}-strip",
            f"traefik.http.services.runner-{strategy_id}.loadbalancer.server.port": "8080",
            # Custom labels for identification
            "edgewalker.type": "strategy-runner",
            "edgewalker.strategy_id": str(strategy_id),
            "edgewalker.symbol": symbol,
        }
        
        # Volume mounts
        volumes = {
            # Shared schemas
            "/home/flavio/playground/edgewalker-runtime/shared": {
                "bind": "/app/shared",
                "mode": "ro",
            },
            # Strategy configs and artifacts
            "/home/flavio/playground/edgewalker/strategies": {
                "bind": "/opt/edgewalker/strategies",
                "mode": "ro",
            },
            "/home/flavio/playground/edgewalker/artifacts": {
                "bind": "/opt/edgewalker/artifacts",
                "mode": "ro",
            },
        }
        
        try:
            # Create and start container
            container = self.client.containers.run(
                image=RUNNER_IMAGE,
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
            
            logger.info(f"Started strategy runner container: {container_name} ({container.short_id})")
            
            return {
                "status": "started",
                "container_id": container.short_id,
                "container_name": container_name,
                "symbol": symbol,
                "timeframe": timeframe,
            }
            
        except APIError as e:
            logger.error(f"Failed to start container {container_name}: {e}")
            raise RuntimeError(f"Failed to start strategy runner: {e}")
    
    def stop_strategy(self, strategy_id: int, remove: bool = True) -> dict[str, Any]:
        """
        Stop a live strategy runner container.
        
        Args:
            strategy_id: Database ID of the strategy
            remove: Whether to remove the container after stopping
            
        Returns:
            Dict with operation status
        """
        container = self._get_container(strategy_id)
        
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
    
    def get_strategy_status(self, strategy_id: int) -> dict[str, Any]:
        """
        Get status of a strategy runner container.
        
        Returns container state, health, and runtime info.
        """
        container = self._get_container(strategy_id)
        
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
    
    def list_running_strategies(self) -> list[dict[str, Any]]:
        """
        List all running strategy runner containers.
        
        Returns list of container info dicts.
        """
        containers = self.client.containers.list(
            filters={
                "label": "edgewalker.type=strategy-runner",
                "status": "running",
            }
        )
        
        result = []
        for container in containers:
            result.append({
                "container_id": container.short_id,
                "container_name": container.name,
                "strategy_id": container.labels.get("edgewalker.strategy_id"),
                "symbol": container.labels.get("edgewalker.symbol"),
                "status": container.status,
            })
        
        return result
    
    def get_container_logs(
        self,
        strategy_id: int,
        tail: int = 100,
        since: str | None = None,
    ) -> str:
        """
        Get logs from a strategy runner container.
        
        Args:
            strategy_id: Database ID of the strategy
            tail: Number of lines to return from end
            since: Return logs since this timestamp
            
        Returns:
            Log output as string
        """
        container = self._get_container(strategy_id)
        
        if not container:
            raise ValueError(f"Container for strategy {strategy_id} not found")
        
        kwargs = {"tail": tail, "timestamps": True}
        if since:
            kwargs["since"] = since
        
        return container.logs(**kwargs).decode("utf-8")


# Global service instance
live_runner_service = LiveRunnerService()
