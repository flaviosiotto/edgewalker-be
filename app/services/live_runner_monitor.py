from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone

from sqlmodel import select

from app.db.database import get_session_context
from app.models.strategy import LiveStatus, StrategyLive
from app.services.live_runner_service import live_runner_service

logger = logging.getLogger(__name__)

RUNNER_MONITOR_INTERVAL_SECONDS = int(os.getenv("RUNNER_MONITOR_INTERVAL_SECONDS", "30"))
RUNNER_START_TIMEOUT_SECONDS = int(os.getenv("RUNNER_START_TIMEOUT_SECONDS", "120"))


class LiveRunnerMonitor:
    def __init__(self) -> None:
        self._running = False
        self._task: asyncio.Task[None] | None = None

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await self.sync_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Live runner monitor sync failed")
            await asyncio.sleep(RUNNER_MONITOR_INTERVAL_SECONDS)

    async def sync_once(self) -> int:
        with get_session_context() as session:
            live_sessions = session.exec(
                select(StrategyLive).where(
                    StrategyLive.status.in_(
                        [
                            LiveStatus.STARTING.value,
                            LiveStatus.RUNNING.value,
                            LiveStatus.STOPPING.value,
                        ]
                    )
                )
            ).all()

            updated = 0
            now = datetime.now(timezone.utc)

            for sl in live_sessions:
                container_info = live_runner_service.get_live_instance_status(sl.id)
                container_status = container_info.get("status")

                if sl.status == LiveStatus.RUNNING.value and container_status in {"not_found", "exited", "dead"}:
                    sl.status = LiveStatus.ERROR.value
                    sl.container_id = None
                    sl.error_message = "Runner container terminated unexpectedly"
                    sl.stopped_at = now
                    sl.updated_at = now
                    session.add(sl)
                    updated += 1
                    continue

                if sl.status == LiveStatus.STARTING.value:
                    if container_status == "running":
                        sl.status = LiveStatus.RUNNING.value
                        sl.container_id = container_info.get("container_id") or sl.container_id
                        sl.started_at = sl.started_at or now
                        sl.error_message = None
                        sl.updated_at = now
                        session.add(sl)
                        updated += 1
                        continue

                    started_reference = sl.started_at or sl.created_at
                    age_seconds = (now - started_reference).total_seconds()
                    if container_status in {"not_found", "exited", "dead"} and age_seconds >= RUNNER_START_TIMEOUT_SECONDS:
                        sl.status = LiveStatus.ERROR.value
                        sl.container_id = None
                        sl.error_message = "Runner container did not reach running state within timeout"
                        sl.stopped_at = now
                        sl.updated_at = now
                        session.add(sl)
                        updated += 1
                        continue

                if sl.status == LiveStatus.STOPPING.value and container_status in {"not_found", "exited", "dead"}:
                    sl.status = LiveStatus.STOPPED.value
                    sl.container_id = None
                    sl.stopped_at = sl.stopped_at or now
                    sl.updated_at = now
                    session.add(sl)
                    updated += 1
                    continue

                if container_status == "running":
                    container_id = container_info.get("container_id")
                    if container_id and sl.container_id != container_id:
                        sl.container_id = container_id
                        sl.updated_at = now
                        session.add(sl)
                        updated += 1

            if updated:
                session.commit()
                logger.info("Live runner monitor reconciled %d live instance(s)", updated)

            return updated

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        await self.sync_once()
        self._task = asyncio.create_task(self._run_loop(), name="live-runner-monitor")
        logger.info("Live runner monitor started")

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Live runner monitor stopped")


_live_runner_monitor: LiveRunnerMonitor | None = None


def get_live_runner_monitor() -> LiveRunnerMonitor:
    global _live_runner_monitor
    if _live_runner_monitor is None:
        _live_runner_monitor = LiveRunnerMonitor()
    return _live_runner_monitor


async def start_live_runner_monitor() -> None:
    await get_live_runner_monitor().start()


async def stop_live_runner_monitor() -> None:
    await get_live_runner_monitor().stop()