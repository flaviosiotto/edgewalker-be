from __future__ import annotations

import asyncio
import logging
import os

from app.services.backtest_runner_service import backtest_runner_service

logger = logging.getLogger(__name__)

BACKTEST_CLEANUP_INTERVAL_SECONDS = int(os.getenv("BACKTEST_CLEANUP_INTERVAL_SECONDS", "60"))


class BacktestRunnerMonitor:
    """Periodically reap finished backtest runner containers.

    Backtest runners self-terminate a grace period after completion (see the
    strategy-runner keepalive loop), turning into ``exited`` containers. Nothing
    else removes them, so without this monitor they accumulate forever and keep
    consuming host resources. This loop is the control-plane piece that garbage
    collects them.
    """

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
                logger.exception("Backtest runner cleanup failed")
            await asyncio.sleep(BACKTEST_CLEANUP_INTERVAL_SECONDS)

    async def sync_once(self) -> int:
        # docker-py calls are blocking; keep the event loop responsive.
        return await asyncio.to_thread(backtest_runner_service.cleanup_finished)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        try:
            await self.sync_once()
        except Exception:
            logger.exception("Initial backtest runner cleanup failed")
        self._task = asyncio.create_task(self._run_loop(), name="backtest-runner-monitor")
        logger.info("Backtest runner monitor started")

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
        logger.info("Backtest runner monitor stopped")


_backtest_runner_monitor: BacktestRunnerMonitor | None = None


def get_backtest_runner_monitor() -> BacktestRunnerMonitor:
    global _backtest_runner_monitor
    if _backtest_runner_monitor is None:
        _backtest_runner_monitor = BacktestRunnerMonitor()
    return _backtest_runner_monitor


async def start_backtest_runner_monitor() -> None:
    await get_backtest_runner_monitor().start()


async def stop_backtest_runner_monitor() -> None:
    await get_backtest_runner_monitor().stop()
