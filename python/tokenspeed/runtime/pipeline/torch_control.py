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

"""Global step consensus and bounded-failure state for torch pipelines."""

from __future__ import annotations

import hashlib
import os
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.pipeline.contracts import (
    PipelineForwardMode,
    PipelineProtocolError,
    PipelineStepAborted,
    PipelineStepDescriptor,
)
from tokenspeed.runtime.pipeline.groups import (
    PIPELINE_FAULT_DOWNSTREAM_GROUP_ROLE,
    PIPELINE_FAULT_HEARTBEAT_SECONDS,
    PIPELINE_FAULT_UPSTREAM_GROUP_ROLE,
    PIPELINE_STEP_META_GROUP_ROLE,
)

_PIPELINE_EXIT_CODE = 70
_FAULT_MAGIC = 0x5453504641554C54  # "TSPFAULT"
_FAULT_PACKET_WORDS = 8
_FAULT_PACKET_KIND = 1
_STOP_PACKET_KIND = 2
_HEARTBEAT_PACKET_KIND = 3


@dataclass(frozen=True)
class PipelineStepLease:
    descriptor: PipelineStepDescriptor
    deadline: float


class _DeadlineWatchdog:
    def __init__(
        self,
        terminate: Callable[[int], None],
        *,
        clock: Callable[[], float],
        context: str,
    ) -> None:
        self._terminate = terminate
        self._clock = clock
        self._context = context
        self._condition = threading.Condition()
        self._armed: tuple[int, float, str] | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="tokenspeed-pipeline-watchdog",
            daemon=True,
        )
        self._thread.start()

    def arm(self, step_id: int, deadline: float, reason: str) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("pipeline watchdog is closed")
            self._armed = (step_id, deadline, reason)
            self._condition.notify_all()

    def disarm(self, step_id: int) -> None:
        with self._condition:
            if self._armed is not None and self._armed[0] == step_id:
                self._armed = None
                self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._armed = None
            self._condition.notify_all()
        self._thread.join(timeout=1)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._closed and self._armed is None:
                    self._condition.wait()
                if self._closed:
                    return
                assert self._armed is not None
                step_id, deadline, reason = self._armed
                remaining = deadline - self._clock()
                if remaining > 0:
                    self._condition.wait(timeout=remaining)
                    continue
                if self._armed != (step_id, deadline, reason):
                    continue
                self._closed = True
                overdue = max(0.0, self._clock() - deadline)
            diagnostic = (
                "tokenspeed pipeline watchdog deadline expired: "
                f"{self._context} step_id={step_id} reason={reason} "
                f"overdue_seconds={overdue:.6f} exit_code={_PIPELINE_EXIT_CODE}\n"
            )
            try:
                os.write(2, diagnostic.encode("utf-8", errors="replace"))
            except Exception:
                pass
            self._terminate(_PIPELINE_EXIT_CODE)
            return


class TorchPipelineControlPlane:
    """Announce every step globally and poison the instance on any failure."""

    def __init__(
        self,
        mapping: Mapping,
        *,
        plan_digest: str,
        step_timeout_seconds: float = 300.0,
        abort_grace_seconds: float = 2.0,
        terminate: Callable[[int], None] = os._exit,
        clock: Callable[[], float] = time.monotonic,
        enable_fault_listener: bool = True,
    ) -> None:
        if mapping.pipeline.stage_count < 2:
            raise ValueError("pipeline control requires at least two stages")
        if step_timeout_seconds <= 0 or abort_grace_seconds <= 0:
            raise ValueError("pipeline control timeouts must be positive")
        self._mapping = mapping
        self._plan_digest = plan_digest
        self._step_timeout_seconds = float(step_timeout_seconds)
        self._abort_grace_seconds = float(abort_grace_seconds)
        self._clock = clock
        self._group = pg_manager.get_process_group(
            "gloo",
            mapping.world_group,
            role=PIPELINE_STEP_META_GROUP_ROLE,
        )
        self._fault_upstream_group = (
            pg_manager.get_process_group(
                "gloo",
                mapping.world_group,
                role=PIPELINE_FAULT_UPSTREAM_GROUP_ROLE,
            )
            if enable_fault_listener
            else None
        )
        self._fault_downstream_group = (
            pg_manager.get_process_group(
                "gloo",
                mapping.world_group,
                role=PIPELINE_FAULT_DOWNSTREAM_GROUP_ROLE,
            )
            if enable_fault_listener
            else None
        )
        if mapping.rank == 0:
            self._fault_in_group = self._fault_upstream_group
            self._fault_out_group = self._fault_downstream_group
        else:
            self._fault_in_group = self._fault_downstream_group
            self._fault_out_group = self._fault_upstream_group
        self._state_lock = threading.Lock()
        self._fault_send_lock = threading.Lock()
        self._closed = False
        self._fault_works: list[tuple[object, torch.Tensor]] = []
        self._watchdog = _DeadlineWatchdog(
            terminate,
            clock=clock,
            context=(
                f"global_rank={mapping.rank} "
                f"pipeline_stage={mapping.pipeline.stage_index}"
            ),
        )
        self._watchdog.arm(
            0,
            self._clock() + self._step_timeout_seconds,
            "startup-epoch-synchronize",
        )
        try:
            self._epoch = self._synchronize_epoch()
        finally:
            self._watchdog.disarm(0)
        self._next_step_id = 1
        self._active: PipelineStepLease | None = None
        self._poison: PipelineStepAborted | None = None
        self._fault_thread = None
        self._fault_heartbeat_stop = threading.Event()
        self._fault_heartbeat_thread = None
        if self._fault_in_group is not None:
            self._fault_thread = threading.Thread(
                target=self._listen_for_fault,
                name="tokenspeed-pipeline-fault-listener",
                daemon=True,
            )
            self._fault_thread.start()
            if mapping.rank in (0, 1):
                self._fault_heartbeat_thread = threading.Thread(
                    target=self._run_fault_heartbeat,
                    name="tokenspeed-pipeline-fault-heartbeat",
                    daemon=True,
                )
                self._fault_heartbeat_thread.start()

    def _synchronize_epoch(self) -> int:
        value = torch.zeros(1, dtype=torch.int64)
        if self._mapping.rank == 0:
            value[0] = secrets.randbits(60) or 1
        dist.broadcast(value, src=0, group=self._group)
        epoch = int(value.item())
        if not 0 < epoch < 1 << 60:
            raise PipelineProtocolError("pipeline epoch is outside the wire range")
        return epoch

    def begin_step(
        self,
        *,
        forward_mode_name: str,
        batch_size: int,
        input_num_tokens: int,
        num_extends: int,
        batch_fingerprint: int,
        stage_cache_fingerprint: int,
    ) -> PipelineStepLease:
        with self._state_lock:
            if self._closed:
                raise PipelineStepAborted("pipeline control plane is closed")
            poison = self._poison
            if poison is not None:
                raise poison
            if self._active is not None:
                raise PipelineProtocolError("a pipeline step is already active")
        try:
            descriptor = PipelineStepDescriptor(
                epoch=self._epoch,
                step_id=self._next_step_id,
                forward_mode=PipelineForwardMode.from_runtime_name(forward_mode_name),
                batch_size=batch_size,
                input_num_tokens=input_num_tokens,
                num_extends=num_extends,
                plan_digest=self._plan_digest,
                batch_fingerprint=batch_fingerprint,
            )
            if (
                isinstance(stage_cache_fingerprint, bool)
                or not isinstance(stage_cache_fingerprint, int)
                or not 0 <= stage_cache_fingerprint < 1 << 60
            ):
                raise ValueError(
                    "pipeline stage_cache_fingerprint must be a non-negative "
                    "60-bit integer"
                )
        except Exception as exc:
            raise self.abort_runtime("step-descriptor", exc) from exc
        lease = PipelineStepLease(
            descriptor=descriptor,
            deadline=self._clock() + self._step_timeout_seconds,
        )
        with self._state_lock:
            if self._closed:
                raise PipelineStepAborted("pipeline control plane is closed")
            if self._poison is not None:
                raise self._poison
            if self._active is not None:
                raise PipelineProtocolError("a pipeline step is already active")
            self._active = lease
            self._watchdog.arm(
                descriptor.step_id,
                lease.deadline,
                f"pipeline-step-{descriptor.forward_mode.name.lower()}",
            )

        local = torch.tensor(
            (*descriptor.wire_words(), stage_cache_fingerprint), dtype=torch.int64
        )
        gathered = [torch.empty_like(local) for _ in range(self._mapping.world_size)]
        try:
            dist.all_gather(gathered, local, group=self._group)
            divergent = [
                rank
                for rank, candidate in enumerate(gathered)
                if not torch.equal(candidate[:-1], gathered[0][:-1])
            ]
            if divergent:
                raise PipelineProtocolError(
                    f"pipeline step descriptor differs on global ranks {divergent}"
                )
            stage_world_size = self._mapping.pipeline.stage_world_size
            for stage_index in range(self._mapping.pipeline.stage_count):
                first_rank = stage_index * stage_world_size
                stage_ranks = range(first_rank, first_rank + stage_world_size)
                stage_reference = gathered[first_rank][-1]
                divergent = [
                    rank
                    for rank in stage_ranks
                    if gathered[rank][-1] != stage_reference
                ]
                if divergent:
                    raise PipelineProtocolError(
                        f"pipeline stage {stage_index} cache state differs on "
                        f"global ranks {divergent}"
                    )
        except Exception as exc:
            raise self.abort(lease, "announce", exc) from exc

        self._next_step_id += 1
        return lease

    def abort(
        self,
        lease: PipelineStepLease,
        phase: str,
        error: BaseException,
    ) -> PipelineStepAborted:
        should_report = False
        with self._state_lock:
            if self._active is not None and self._active != lease:
                return PipelineStepAborted(
                    "pipeline failure references a stale step lease"
                )
            if self._poison is None:
                self._poison = PipelineStepAborted(
                    f"pipeline step {lease.descriptor.step_id} failed during "
                    f"{phase}: {type(error).__name__}: {error}"
                )
                should_report = True
            poison = self._poison
        if should_report:
            self._watchdog.arm(
                lease.descriptor.step_id,
                self._clock() + self._abort_grace_seconds,
                f"pipeline-abort-{phase}",
            )
            self._report_fault(lease, phase, error)
        return poison

    def abort_runtime(
        self,
        phase: str,
        error: BaseException,
    ) -> PipelineStepAborted:
        """Poison the instance for a failure outside an acquired step."""

        with self._state_lock:
            active = self._active
            step_id = self._next_step_id
        if active is None:
            active = PipelineStepLease(
                descriptor=PipelineStepDescriptor(
                    epoch=self._epoch,
                    step_id=step_id,
                    forward_mode=PipelineForwardMode.IDLE,
                    batch_size=0,
                    input_num_tokens=0,
                    num_extends=0,
                    plan_digest=self._plan_digest,
                    batch_fingerprint=0,
                ),
                deadline=self._clock() + self._abort_grace_seconds,
            )
        return self.abort(active, phase, error)

    def complete_step(self, lease: PipelineStepLease) -> None:
        with self._state_lock:
            poison = self._poison
            if poison is not None:
                raise poison
            if self._active != lease:
                raise PipelineProtocolError(
                    "pipeline completion references a stale lease"
                )
            self._active = None
        self._watchdog.disarm(lease.descriptor.step_id)

    def wait_work(self, work, lease: PipelineStepLease, phase: str) -> None:
        with self._state_lock:
            poison = self._poison
        if poison is not None:
            raise poison
        remaining = lease.deadline - self._clock()
        if remaining <= 0:
            timeout = TimeoutError(
                f"pipeline step {lease.descriptor.step_id} timed out during {phase}"
            )
            raise self.abort(lease, phase, timeout) from timeout
        try:
            completed = work.wait(timeout=timedelta(seconds=remaining))
            if completed is False:
                raise TimeoutError(
                    f"pipeline step {lease.descriptor.step_id} timed out during "
                    f"{phase}"
                )
        except PipelineStepAborted:
            raise
        except Exception as exc:
            raise self.abort(lease, phase, exc) from exc
        with self._state_lock:
            poison = self._poison
        if poison is not None:
            raise poison

    def close(self, *, cancel_fatal_deadline: bool = False) -> None:
        active_to_report = None
        with self._state_lock:
            if self._closed:
                poisoned = self._poison is not None
            else:
                self._closed = True
                if self._active is not None and self._poison is None:
                    self._poison = PipelineStepAborted(
                        "pipeline control closed while a step was active"
                    )
                    active_to_report = self._active
                poisoned = self._poison is not None
        if active_to_report is not None:
            self._watchdog.arm(
                active_to_report.descriptor.step_id,
                self._clock() + self._abort_grace_seconds,
                "pipeline-abort-close-active-step",
            )
            self._report_fault(
                active_to_report,
                "close-active-step",
                self._poison,
            )
        if poisoned and not cancel_fatal_deadline:
            # Keep the independently armed fatal deadline alive while the
            # ordinary event-loop cleanup unwinds. It is the backstop if that
            # cleanup or the external supervisor stalls.
            self._fault_heartbeat_stop.set()
            return
        shutdown_deadline = self._clock() + self._abort_grace_seconds
        self._fault_heartbeat_stop.set()
        heartbeat_thread = self._fault_heartbeat_thread
        if (
            heartbeat_thread is not None
            and heartbeat_thread is not threading.current_thread()
        ):
            heartbeat_thread.join(timeout=max(0.0, shutdown_deadline - self._clock()))
        stop_sent = self._send_stop(shutdown_deadline)
        fault_thread = self._fault_thread
        if fault_thread is not None and fault_thread is not threading.current_thread():
            fault_thread.join(timeout=max(0.0, shutdown_deadline - self._clock()))
        heartbeat_stuck = heartbeat_thread is not None and heartbeat_thread.is_alive()
        if (
            not stop_sent
            or heartbeat_stuck
            or (fault_thread is not None and fault_thread.is_alive())
        ):
            error = PipelineStepAborted(
                "pipeline fault listener did not stop before the shutdown deadline"
            )
            with self._state_lock:
                if self._poison is None:
                    self._poison = error
                poison = self._poison
            self._watchdog.arm(
                self._next_step_id,
                self._clock() + self._abort_grace_seconds,
                "pipeline-abort-fault-channel-shutdown",
            )
            raise poison
        self._watchdog.close()

    def _send_stop(self, deadline: float) -> bool:
        if self._fault_out_group is None or self._fault_thread is None:
            return True
        packet = self._control_packet(_STOP_PACKET_KIND)
        if self._mapping.rank == 0:
            destinations = range(1, self._mapping.world_size)
        elif self._mapping.rank == 1:
            destinations = (0,)
        else:
            destinations = ()
        return self._send_packets(
            packet,
            destinations,
            group=self._fault_out_group,
            wait=True,
            deadline=deadline,
        )

    def _control_packet(self, kind: int) -> torch.Tensor:
        return torch.tensor(
            (
                _FAULT_MAGIC,
                kind,
                self._epoch,
                0,
                self._mapping.rank,
                self._mapping.pipeline.stage_index,
                0,
                0,
            ),
            dtype=torch.int64,
        )

    def _run_fault_heartbeat(self) -> None:
        destinations = (
            tuple(range(1, self._mapping.world_size))
            if self._mapping.rank == 0
            else (0,)
        )
        while not self._fault_heartbeat_stop.wait(PIPELINE_FAULT_HEARTBEAT_SECONDS):
            packet = self._control_packet(_HEARTBEAT_PACKET_KIND)
            deadline = self._clock() + min(5.0, PIPELINE_FAULT_HEARTBEAT_SECONDS / 2)
            if self._send_packets(
                packet,
                destinations,
                group=self._fault_out_group,
                wait=True,
                deadline=deadline,
            ):
                continue
            with self._state_lock:
                closed = self._closed
                step_id = (
                    self._active.descriptor.step_id
                    if self._active is not None
                    else self._next_step_id
                )
            if not closed:
                self._poison_from_peer(
                    PipelineStepAborted("pipeline fault heartbeat failed"),
                    step_id=step_id,
                )
            return

    @staticmethod
    def _fault_hash(value: str) -> int:
        raw = hashlib.sha256(value.encode("utf-8", errors="replace")).digest()
        return int.from_bytes(raw[:8], "big") & ((1 << 60) - 1)

    def _fault_packet(
        self,
        lease: PipelineStepLease,
        phase: str,
        error: BaseException,
    ) -> torch.Tensor:
        return torch.tensor(
            (
                _FAULT_MAGIC,
                _FAULT_PACKET_KIND,
                lease.descriptor.epoch,
                lease.descriptor.step_id,
                self._mapping.rank,
                self._mapping.pipeline.stage_index,
                self._fault_hash(phase),
                self._fault_hash(f"{type(error).__name__}: {error}"),
            ),
            dtype=torch.int64,
        )

    def _report_fault(
        self,
        lease: PipelineStepLease,
        phase: str,
        error: BaseException,
    ) -> None:
        if self._fault_out_group is None:
            return
        packet = self._fault_packet(lease, phase, error)
        destinations = (
            range(1, self._mapping.world_size) if self._mapping.rank == 0 else (0,)
        )
        self._send_packets(
            packet,
            destinations,
            group=self._fault_out_group,
            wait=True,
        )

    def _send_packets(
        self,
        packet,
        destinations,
        *,
        group,
        wait: bool,
        deadline: float | None = None,
    ) -> bool:
        submitted = []
        success = True
        with self._fault_send_lock:
            for destination in destinations:
                payload = packet.clone()
                try:
                    work = dist.isend(payload, dst=destination, group=group)
                    submitted.append((work, payload))
                except Exception:
                    # The local hard-exit deadline is already armed. Reporting
                    # is best-effort when the control network itself is broken.
                    success = False
            if not wait:
                self._fault_works.extend(submitted)
                return success
            deadline = deadline or (self._clock() + self._abort_grace_seconds)
            for work, _payload in submitted:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    return False
                try:
                    completed = work.wait(timeout=timedelta(seconds=remaining))
                except Exception:
                    success = False
                    continue
                if completed is False:
                    success = False
        return success

    def _listen_for_fault(self) -> None:
        packet = torch.zeros(_FAULT_PACKET_WORDS, dtype=torch.int64)
        source = 0 if self._mapping.rank != 0 else None
        while True:
            packet.zero_()
            try:
                sender = dist.recv(packet, src=source, group=self._fault_in_group)
            except Exception as exc:
                with self._state_lock:
                    closed = self._closed
                    step_id = (
                        self._active.descriptor.step_id
                        if self._active is not None
                        else self._next_step_id
                    )
                if not closed:
                    self._poison_from_peer(
                        PipelineStepAborted(
                            "pipeline fault channel failed: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                        step_id=step_id,
                    )
                return
            if source is not None:
                sender = 0
            words = tuple(int(value) for value in packet.tolist())
            if self._is_control_packet(
                words,
                sender=sender,
                kind=_HEARTBEAT_PACKET_KIND,
            ):
                continue
            break

        reported_rank = words[4]
        expected_stage = (
            reported_rank // self._mapping.pipeline.stage_world_size
            if 0 <= reported_rank < self._mapping.world_size
            else -1
        )
        is_valid_sender = not (
            not 0 <= reported_rank < self._mapping.world_size
            or words[5] != expected_stage
            or (
                self._mapping.rank == 0
                and sender is not None
                and reported_rank != sender
            )
        )
        if self._is_control_packet(
            words,
            sender=sender,
            kind=_STOP_PACKET_KIND,
        ):
            return
        if (
            len(words) != _FAULT_PACKET_WORDS
            or words[0] != _FAULT_MAGIC
            or words[1] != _FAULT_PACKET_KIND
            or words[2] != self._epoch
            or not 0 < words[3] < 1 << 60
            or not is_valid_sender
            or any(not 0 <= words[index] < 1 << 60 for index in (6, 7))
        ):
            error = PipelineStepAborted("pipeline fault packet is invalid")
        else:
            error = PipelineStepAborted(
                f"pipeline peer rank {words[4]} stage {words[5]} reported "
                f"failure in step {words[3]} (phase={words[6]:x}, "
                f"error={words[7]:x})"
            )
        self._poison_from_peer(error, step_id=max(words[3], 0))
        if self._mapping.rank == 0:
            destinations = (
                destination
                for destination in range(1, self._mapping.world_size)
                if destination != reported_rank
            )
            self._send_packets(
                packet,
                destinations,
                group=self._fault_out_group,
                wait=True,
            )

    def _is_control_packet(
        self,
        words: tuple[int, ...],
        *,
        sender: int | None,
        kind: int,
    ) -> bool:
        if len(words) != _FAULT_PACKET_WORDS:
            return False
        reported_rank = words[4]
        expected_stage = (
            reported_rank // self._mapping.pipeline.stage_world_size
            if 0 <= reported_rank < self._mapping.world_size
            else -1
        )
        if self._mapping.rank == 0:
            expected_rank = 1
            valid_sender = sender == expected_rank
        else:
            expected_rank = 0
            valid_sender = sender in (None, expected_rank)
        return (
            words[0] == _FAULT_MAGIC
            and words[1] == kind
            and words[2] == self._epoch
            and words[3] == 0
            and reported_rank == expected_rank
            and words[5] == expected_stage
            and words[6:] == (0, 0)
            and valid_sender
        )

    def _poison_from_peer(
        self,
        error: PipelineStepAborted,
        *,
        step_id: int,
    ) -> None:
        should_arm = False
        with self._state_lock:
            if self._closed:
                return
            if self._poison is None:
                self._poison = error
                should_arm = True
        if should_arm:
            self._watchdog.arm(
                step_id,
                self._clock() + self._abort_grace_seconds,
                "pipeline-abort-peer-fault",
            )
