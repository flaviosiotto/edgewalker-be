"""
Backtest Runner Service.

Manages Docker containers for backtest execution.
Each backtest runs in its own short-lived container (1:1 mapping).

The container:
- Reads backtest parameters from the database
- Verifies data coverage in the shared parquet datalake
- Runs the backtest using the edgewalker library
- Writes results (metrics + trades) directly to the database
- Exits automatically on completion
"""
from __future__ import annotations

import logging
import os
from typing import Any

import docker
from docker.errors import NotFound, APIError
from docker.models.containers import Container

logger = logging.getLogger(__name__)

# Docker network for internal communication
DOCKER_NETWORK = os.getenv("DOCKER_NETWORK", "edgewalker-devops_default")

# Backtest runner image (built via docker-compose build strategy-backtest)
BACKTEST_IMAGE = os.getenv("BACKTEST_IMAGE", "edgewalker-devops-strategy-backtest:latest")

# Container naming convention
CONTAINER_PREFIX = "edgewalker-backtest-"


class BacktestRunnerService:
    """Service for managing backtest runner containers.

    Provides start/stop/status operations.  Containers are one-shot:
    they run the backtest, persist results to the DB, and exit.
    """

    def __init__(self):
        self._client: docker.DockerClient | None = None

    @property
    def client(self) -> docker.DockerClient:
        """Get or create Docker client."""
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    def _container_name(self, backtest_id: int) -> str:
        """Generate container name for a backtest."""
        return f"{CONTAINER_PREFIX}{backtest_id}"

    def _get_container(self, backtest_id: int) -> Container | None:
        """Get container for a backtest if it exists."""
        try:
            return self.client.containers.get(self._container_name(backtest_id))
        except NotFound:
            return None

    def start_backtest(
        self,
        backtest_id: int,
        connection_id: int | None = None,
    ) -> dict[str, Any]:
        """Start a backtest runner container.

        Args:
            backtest_id: Database ID of the backtest.
            connection_id: Connection ID for data scoping (from strategy).

        Returns:
            Dict with container info and status.
        """
        container_name = self._container_name(backtest_id)

        # Check if container already exists
        existing = self._get_container(backtest_id)
        if existing:
            if existing.status == "running":
                return {
                    "status": "already_running",
                    "container_id": existing.short_id,
                    "container_name": container_name,
                }
            else:
                # Remove stopped/exited container from a previous run
                existing.remove(force=True)

        # Environment variables
        env = {
            "BACKTEST_ID": str(backtest_id),
            "DATABASE_URL": os.getenv("DATABASE_URL", ""),
            "DATA_DIR": "/opt/edgewalker/data",
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
            "PYTHONPATH": "/app",
        }

        if connection_id:
            env["CONNECTION_ID"] = str(connection_id)

        # Labels for identification and management
        labels = {
            "edgewalker.type": "strategy-backtest",
            "edgewalker.backtest_id": str(backtest_id),
        }

        # Volume mounts — shared data + artifacts for reports
        edgewalker_path = os.getenv(
            "EDGEWALKER_PATH", "/home/flavio/playground/edgewalker"
        )
        volumes = {
            f"{edgewalker_path}/data": {
                "bind": "/opt/edgewalker/data",
                "mode": "rw",
            },
            f"{edgewalker_path}/artifacts": {
                "bind": "/opt/edgewalker/artifacts",
                "mode": "rw",
            },
            f"{edgewalker_path}/strategies": {
                "bind": "/opt/edgewalker/strategies",
                "mode": "ro",
            },
            f"{edgewalker_path}/configs": {
                "bind": "/opt/edgewalker/configs",
                "mode": "ro",
            },
        }

        try:
            container = self.client.containers.run(
                image=BACKTEST_IMAGE,
                name=container_name,
                environment=env,
                labels=labels,
                volumes=volumes,
                network=DOCKER_NETWORK,
                detach=True,
                auto_remove=False,
                restart_policy={"Name": "no"},  # one-shot: don't restart
                extra_hosts={"host.docker.internal": "host-gateway"},
            )

            logger.info(
                "Started backtest runner: %s (%s) for backtest %d",
                container_name,
                container.short_id,
                backtest_id,
            )

            return {
                "status": "started",
                "container_id": container.short_id,
                "container_name": container_name,
            }

        except APIError as e:
            logger.error(
                "Failed to start backtest container %s: %s", container_name, e
            )
            raise RuntimeError(f"Failed to start backtest runner: {e}")

    def stop_backtest(
        self, backtest_id: int, remove: bool = True
    ) -> dict[str, Any]:
        """Stop and optionally remove a backtest runner container."""
        container = self._get_container(backtest_id)

        if not container:
            return {"status": "not_found"}

        try:
            if container.status == "running":
                container.stop(timeout=30)
                logger.info("Stopped backtest runner: %s", container.name)

            if remove:
                container.remove(force=True)
                logger.info("Removed backtest runner: %s", container.name)
                return {"status": "stopped_and_removed"}

            return {"status": "stopped"}

        except APIError as e:
            logger.error("Failed to stop backtest container: %s", e)
            raise RuntimeError(f"Failed to stop backtest runner: {e}")

    def get_backtest_status(self, backtest_id: int) -> dict[str, Any]:
        """Get status of a backtest runner container."""
        container = self._get_container(backtest_id)

        if not container:
            return {"status": "not_found", "running": False}

        container.reload()
        return {
            "status": container.status,
            "running": container.status == "running",
            "container_id": container.short_id,
            "container_name": container.name,
            "labels": container.labels,
        }

    def get_container_logs(
        self, backtest_id: int, tail: int = 200
    ) -> str:
        """Get logs from a backtest runner container."""
        container = self._get_container(backtest_id)

        if not container:
            raise ValueError(f"Container for backtest {backtest_id} not found")

        return container.logs(tail=tail, timestamps=True).decode("utf-8")

    def cleanup_finished(self) -> int:
        """Remove finished (exited) backtest containers.

        Returns the number of containers removed.
        """
        containers = self.client.containers.list(
            all=True,
            filters={
                "label": "edgewalker.type=strategy-backtest",
                "status": "exited",
            },
        )
        count = 0
        for container in containers:
            try:
                container.remove(force=True)
                count += 1
            except Exception as e:
                logger.warning(
                    "Failed to remove container %s: %s", container.name, e
                )
        return count


# Global service instance
backtest_runner_service = BacktestRunnerService()
