"""Periodic subscription sweep (same shape as live_runner_monitor).

Every ``BILLING_SWEEPER_INTERVAL_SECONDS``:
- subscriptions that ended on their own (trial, manual, scheduled cancel)
  go to the default plan, live sessions beyond the new cap are stopped;
- T-N days before an end without renewal the user gets a warning that
  lists the live sessions that will be stopped.

The DB work is synchronous (SQLModel sessions + Docker stops), so it runs
in a worker thread and never blocks the event loop.
"""

from __future__ import annotations

import asyncio
import logging

from app.core.config import settings
from app.db.database import get_session_context
from app.services.billing.billing_service import sweep_ending_notices, sweep_expired_subscriptions

logger = logging.getLogger(__name__)


def sweep_once_sync() -> tuple[int, int]:
    with get_session_context() as session:
        ended = sweep_expired_subscriptions(session)
    with get_session_context() as session:
        notified = sweep_ending_notices(session)
    return ended, notified


class BillingSweeper:
    def __init__(self) -> None:
        self._running = False
        self._task: asyncio.Task[None] | None = None

    async def _run_loop(self) -> None:
        while self._running:
            try:
                ended, notified = await asyncio.to_thread(sweep_once_sync)
                if ended or notified:
                    logger.info("Billing sweep: %d subscription(s) ended, %d notice(s) sent", ended, notified)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Billing sweep failed")
            await asyncio.sleep(settings.BILLING_SWEEPER_INTERVAL_SECONDS)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None


_sweeper: BillingSweeper | None = None


def get_billing_sweeper() -> BillingSweeper:
    global _sweeper
    if _sweeper is None:
        _sweeper = BillingSweeper()
    return _sweeper


async def start_billing_sweeper() -> None:
    await get_billing_sweeper().start()


async def stop_billing_sweeper() -> None:
    await get_billing_sweeper().stop()
