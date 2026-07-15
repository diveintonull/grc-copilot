"""Async task registration, cancellation, and guaranteed cleanup."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable
from typing import Any


class DuplicateTaskError(RuntimeError):
    """A live task already owns the requested identifier."""


class TaskManager:
    """Own active request tasks and remove them on every exit path."""

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._lock = asyncio.Lock()

    @property
    def active_count(self) -> int:
        return len(self._tasks)

    async def contains(self, request_id: str) -> bool:
        async with self._lock:
            task = self._tasks.get(request_id)
            return task is not None and not task.done()

    async def start(
        self,
        request_id: str,
        work: Awaitable[Any],
    ) -> asyncio.Task[Any]:
        """Register and start work under a unique request identifier."""
        if not request_id.strip():
            self._close_unstarted(work)
            raise ValueError("request_id must not be blank")

        async with self._lock:
            existing = self._tasks.get(request_id)
            if existing is not None and not existing.done():
                self._close_unstarted(work)
                raise DuplicateTaskError(
                    f"active task already exists: {request_id}"
                )

            task = asyncio.create_task(
                self._run_and_cleanup(request_id, work),
                name=f"grc-agent:{request_id}",
            )
            self._tasks[request_id] = task

        await asyncio.sleep(0)
        return task

    async def stop(self, request_id: str) -> bool:
        """Cancel one live task, await its cleanup, and report if it existed."""
        async with self._lock:
            task = self._tasks.get(request_id)
            if task is None or task.done():
                return False
            task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            await self._remove_if_current(request_id, task)
        return True

    async def cancel_all(self) -> None:
        """Cancel every active task during application shutdown."""
        async with self._lock:
            tasks = list(self._tasks.items())
            for _, task in tasks:
                if not task.done():
                    task.cancel()

        for request_id, task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            finally:
                await self._remove_if_current(request_id, task)

    async def _run_and_cleanup(
        self,
        request_id: str,
        work: Awaitable[Any],
    ) -> Any:
        try:
            return await work
        finally:
            current_task = asyncio.current_task()
            if current_task is not None:
                await self._remove_if_current(request_id, current_task)

    async def _remove_if_current(
        self,
        request_id: str,
        task: asyncio.Task[Any],
    ) -> None:
        async with self._lock:
            if self._tasks.get(request_id) is task:
                self._tasks.pop(request_id, None)

    @staticmethod
    def _close_unstarted(work: Awaitable[Any]) -> None:
        if inspect.iscoroutine(work):
            work.close()
