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

from __future__ import annotations

import multiprocessing as mp
import threading
import time
from dataclasses import dataclass

import pytest

from tokenspeed.runtime.entrypoints.scheduler_process_supervisor import (
    SchedulerProcessSupervisor,
    SupervisorState,
)


@dataclass
class _FakeProcess:
    pid: int
    exitcode: int | None = None
    terminate_calls: int = 0
    kill_calls: int = 0
    join_calls: int = 0

    def is_alive(self) -> bool:
        return self.exitcode is None

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.exitcode = -15

    def kill(self) -> None:
        self.kill_calls += 1
        self.exitcode = -9

    def join(self, timeout: float | None = None) -> None:
        del timeout
        self.join_calls += 1


def _wait_for_release(release) -> None:
    release.wait(timeout=30)


def _run_default_fail_closed() -> None:
    supervisor = SchedulerProcessSupervisor(
        [_FakeProcess(pid=3000, exitcode=0)],
        poll_interval=0.001,
        terminate_timeout=0.01,
    )
    supervisor.start()
    time.sleep(5)


@pytest.mark.parametrize("exitcode", [0, 70])
def test_unexpected_scheduler_exit_fails_closed_and_stops_siblings(exitcode: int):
    processes = [_FakeProcess(pid=1000 + rank) for rank in range(8)]
    failed = threading.Event()
    parent_exit_codes: list[int] = []
    supervisor = SchedulerProcessSupervisor(
        processes,
        fail_closed=lambda code: (parent_exit_codes.append(code), failed.set()),
        poll_interval=0.001,
        terminate_timeout=0.01,
    )

    supervisor.start()
    processes[3].exitcode = exitcode

    assert not supervisor.is_healthy()
    assert failed.wait(timeout=1)
    assert supervisor.state is SupervisorState.FAILED
    assert parent_exit_codes == [1]
    assert processes[3].terminate_calls == 0
    assert all(proc.terminate_calls == 1 for proc in processes if proc.pid != 1003)


def test_active_shutdown_does_not_trigger_fail_closed():
    processes = [_FakeProcess(pid=2000 + rank) for rank in range(8)]
    failed = threading.Event()
    supervisor = SchedulerProcessSupervisor(
        processes,
        fail_closed=lambda _code: failed.set(),
        poll_interval=0.001,
        terminate_timeout=0.01,
    )

    supervisor.start()
    supervisor.shutdown()
    time.sleep(0.02)

    assert supervisor.state is SupervisorState.STOPPED
    assert not supervisor.is_healthy()
    assert not failed.is_set()
    assert all(proc.terminate_calls == 1 for proc in processes)


def test_default_fail_closed_exits_parent_process_with_error():
    context = mp.get_context("spawn")
    engine_process = context.Process(target=_run_default_fail_closed)

    engine_process.start()
    engine_process.join(timeout=10)
    if engine_process.is_alive():
        engine_process.kill()
        engine_process.join(timeout=2)

    assert engine_process.exitcode == 1


def test_spawned_scheduler_exit_is_observed_after_ready():
    context = mp.get_context("spawn")
    failed = threading.Event()
    exiting_scheduler_release = context.Event()
    sibling_release = context.Event()
    processes = [
        context.Process(target=_wait_for_release, args=(exiting_scheduler_release,))
    ] + [
        context.Process(target=_wait_for_release, args=(sibling_release,))
        for _ in range(7)
    ]
    for process in processes:
        process.start()

    supervisor = SchedulerProcessSupervisor(
        processes,
        fail_closed=lambda _code: failed.set(),
        poll_interval=0.005,
        terminate_timeout=1,
    )
    try:
        supervisor.start()
        exiting_scheduler_release.set()

        assert failed.wait(timeout=10)
        assert supervisor.state is SupervisorState.FAILED
        assert not supervisor.is_healthy()
        assert all(process.exitcode is not None for process in processes)
    finally:
        supervisor.shutdown()
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join(timeout=2)
