# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Fail-closed supervision for scheduler subprocesses."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Sequence
from enum import Enum, auto
from typing import Protocol

logger = logging.getLogger(__name__)


class _Process(Protocol):
    pid: int | None
    exitcode: int | None

    def is_alive(self) -> bool: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def join(self, timeout: float | None = None) -> None: ...


class SupervisorState(Enum):
    """Lifecycle state of a local scheduler process group."""

    CREATED = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    FAILED = auto()


class SchedulerProcessSupervisor:
    """Supervise the scheduler processes owned by one engine process.

    The supervisor starts only after every local scheduler reports ready. Any
    subsequent child exit is unexpected, including exit code zero. The first
    unexpected exit atomically marks the group failed, makes health checks
    fail, terminates the remaining children, and exits the engine process.

    Args:
        processes: Spawned scheduler or data-parallel-controller processes.
        fail_closed: Parent exit callback. The production default calls
            ``os._exit(1)``; tests may inject a non-terminating callback.
        poll_interval: Seconds between child liveness checks.
        terminate_timeout: Shared grace period for terminating all children.
    """

    def __init__(
        self,
        processes: Sequence[_Process],
        *,
        fail_closed: Callable[[int], object] | None = None,
        poll_interval: float = 0.1,
        terminate_timeout: float = 5.0,
    ) -> None:
        if not processes:
            raise ValueError("at least one scheduler process is required")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if terminate_timeout < 0:
            raise ValueError("terminate_timeout must be non-negative")

        self._processes = tuple(processes)
        self._fail_closed = os._exit if fail_closed is None else fail_closed
        self._poll_interval = poll_interval
        self._terminate_timeout = terminate_timeout
        self._state = SupervisorState.CREATED
        self._state_lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._monitor_thread: threading.Thread | None = None

    @property
    def state(self) -> SupervisorState:
        """Return the current process-group lifecycle state."""
        with self._state_lock:
            return self._state

    def start(self) -> None:
        """Begin steady-state monitoring after all children report ready."""
        with self._state_lock:
            if self._state is not SupervisorState.CREATED:
                raise RuntimeError(
                    f"cannot start supervisor in state {self._state.name}"
                )
            self._state = SupervisorState.RUNNING
            self._monitor_thread = threading.Thread(
                target=self._monitor,
                name="scheduler-process-supervisor",
                daemon=True,
            )
            self._monitor_thread.start()

    def is_healthy(self) -> bool:
        """Return whether the group is running and every child is alive."""
        with self._state_lock:
            if self._state is not SupervisorState.RUNNING:
                return False
            processes = self._processes
        return all(
            process.exitcode is None and process.is_alive() for process in processes
        )

    def begin_shutdown(self) -> None:
        """Atomically classify subsequent child exits as intentional."""
        with self._state_lock:
            if self._state in (SupervisorState.CREATED, SupervisorState.RUNNING):
                self._state = SupervisorState.STOPPING
            elif self._state in (SupervisorState.STOPPED, SupervisorState.FAILED):
                return
        self._stop_event.set()

    def shutdown(self) -> None:
        """Stop monitoring and terminate all scheduler processes."""
        self.begin_shutdown()
        try:
            self._terminate_processes()
        except Exception:
            logger.exception("Failed to clean up the local scheduler process group")

        monitor_thread = self._monitor_thread
        if (
            monitor_thread is not None
            and monitor_thread is not threading.current_thread()
        ):
            monitor_thread.join(timeout=self._terminate_timeout)

        with self._state_lock:
            if self._state is SupervisorState.STOPPING:
                self._state = SupervisorState.STOPPED

    def _monitor(self) -> None:
        while not self._stop_event.is_set():
            exited = next(
                (
                    process
                    for process in self._processes
                    if process.exitcode is not None
                ),
                None,
            )
            if exited is not None:
                with self._state_lock:
                    if self._state is not SupervisorState.RUNNING:
                        return
                    self._state = SupervisorState.FAILED
                self._stop_event.set()
                logger.critical(
                    "Scheduler process %s exited unexpectedly with code %s; "
                    "terminating the local scheduler group",
                    exited.pid,
                    exited.exitcode,
                )
                try:
                    self._terminate_processes()
                except Exception:
                    logger.exception(
                        "Failed to clean up after an unexpected scheduler exit"
                    )
                finally:
                    self._fail_closed(1)
                return
            self._stop_event.wait(self._poll_interval)

    def _terminate_processes(self) -> None:
        with self._cleanup_lock:
            for process in self._processes:
                if process.is_alive():
                    try:
                        process.terminate()
                    except (OSError, ValueError):
                        logger.exception(
                            "Failed to terminate scheduler process %s", process.pid
                        )

            deadline = time.monotonic() + self._terminate_timeout
            for process in self._processes:
                process.join(timeout=max(0.0, deadline - time.monotonic()))

            for process in self._processes:
                if process.is_alive():
                    try:
                        process.kill()
                    except (OSError, ValueError):
                        logger.exception(
                            "Failed to kill scheduler process %s", process.pid
                        )

            deadline = time.monotonic() + self._terminate_timeout
            for process in self._processes:
                process.join(timeout=max(0.0, deadline - time.monotonic()))
