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
    ) -> dict[str, Any]:
        """
        Start a live strategy runner container.
        
        Args:
            strategy_id: Database ID of the strategy
            strategy_config: Strategy definition (YAML-like dict)
            symbol: Trading symbol to subscribe to
            timeframe: Bar timeframe for subscription (default: 5s)
            
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
            "LOG_LEVEL": "INFO",
            "PYTHONPATH": "/app",
        }
        
        # Traefik labels for routing
        # Route: /api/runners/{strategy_id}/* -> container:8080
        labels = {
            "traefik.enable": "true",
            f"traefik.http.routers.runner-{strategy_id}.rule": f"PathPrefix(`/api/runners/{strategy_id}`)",
            f"traefik.http.routers.runner-{strategy_id}.entrypoints": "http",
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
