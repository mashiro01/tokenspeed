import multiprocessing
import threading
import time
from datetime import timedelta

import pytest

torch = pytest.importorskip("torch")

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.pipeline.contracts import PipelinePlan, PipelineStepAborted
from tokenspeed.runtime.pipeline.groups import (
    PIPELINE_FAULT_DOWNSTREAM_GROUP_ROLE,
    PIPELINE_FAULT_UPSTREAM_GROUP_ROLE,
    PIPELINE_STEP_META_GROUP_ROLE,
)
from tokenspeed.runtime.pipeline.torch_control import (
    _FAULT_MAGIC,
    _HEARTBEAT_PACKET_KIND,
    _PIPELINE_EXIT_CODE,
    _STOP_PACKET_KIND,
    TorchPipelineControlPlane,
    _DeadlineWatchdog,
    pg_manager,
)


class _ImmediateWork:
    def wait(self, timeout=None):
        del timeout
        return True


class _DeferredWork:
    def __init__(self, submitted, expected_submissions, completed=True):
        self.submitted = submitted
        self.expected_submissions = expected_submissions
        self.completed = completed

    def wait(self, timeout=None):
        assert timeout is not None
        assert len(self.submitted) == self.expected_submissions
        return self.completed


class _StuckThread:
    def join(self, timeout=None):
        assert timeout is not None

    def is_alive(self):
        return True


def _run_idle_fault_channel(rank: int, init_file: str) -> None:
    import torch.distributed as dist

    import tokenspeed.runtime.pipeline.torch_control as torch_control

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=2),
    )
    mapping = _mapping(rank)
    roles = (
        PIPELINE_STEP_META_GROUP_ROLE,
        PIPELINE_FAULT_UPSTREAM_GROUP_ROLE,
        PIPELINE_FAULT_DOWNSTREAM_GROUP_ROLE,
    )
    try:
        for role in roles:
            timeout = 2 if role == PIPELINE_STEP_META_GROUP_ROLE else 0.4
            group = dist.new_group(
                ranks=[0, 1],
                backend="gloo",
                timeout=timedelta(seconds=timeout),
            )
            pg_manager.register_process_group(
                "gloo",
                mapping.world_group,
                group,
                role=role,
            )

        torch_control.PIPELINE_FAULT_HEARTBEAT_SECONDS = 0.05
        control = TorchPipelineControlPlane(
            mapping,
            plan_digest=PipelinePlan.single(1).digest,
            step_timeout_seconds=2,
            abort_grace_seconds=1,
        )
        time.sleep(1.0)
        assert control._poison is None
        control.close(cancel_fatal_deadline=True)
    finally:
        dist.destroy_process_group()


def _mapping(rank: int = 0) -> Mapping:
    return Mapping(
        rank=rank,
        world_size=2,
        pipeline_parallel_size=2,
        attn_tp_size=1,
        dense_tp_size=1,
        moe_tp_size=1,
    )


def _patch_groups_and_epoch(monkeypatch):
    monkeypatch.setattr(
        pg_manager,
        "get_process_group",
        lambda backend, group, **kwargs: (backend, group, kwargs.get("role")),
    )

    def broadcast(tensor, src, group):
        assert src == 0
        assert group[0] == "gloo"
        if not tensor.item():
            tensor.fill_(13)

    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)


def test_control_plane_announces_monotonic_steps_and_disarms(monkeypatch):
    _patch_groups_and_epoch(monkeypatch)

    def all_gather(gathered, local, group):
        assert group[0] == "gloo"
        for candidate in gathered:
            candidate.copy_(local)

    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)
    exits = []
    control = TorchPipelineControlPlane(
        _mapping(),
        plan_digest=PipelinePlan.single(1).digest,
        step_timeout_seconds=30,
        terminate=exits.append,
        enable_fault_listener=False,
    )
    try:
        first = control.begin_step(
            forward_mode_name="decode",
            batch_size=2,
            input_num_tokens=2,
            num_extends=0,
            batch_fingerprint=17,
            stage_cache_fingerprint=101,
        )
        control.complete_step(first)
        second = control.begin_step(
            forward_mode_name="extend",
            batch_size=1,
            input_num_tokens=4,
            num_extends=1,
            batch_fingerprint=19,
            stage_cache_fingerprint=103,
        )
        control.complete_step(second)
    finally:
        control.close(cancel_fatal_deadline=True)

    assert first.descriptor.step_id == 1
    assert second.descriptor.step_id == 2
    assert exits == []


def test_descriptor_divergence_poisons_the_instance_before_data_work(monkeypatch):
    _patch_groups_and_epoch(monkeypatch)

    def all_gather(gathered, local, group):
        del group
        for candidate in gathered:
            candidate.copy_(local)
        gathered[1][4] += 1

    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)
    exits = []
    control = TorchPipelineControlPlane(
        _mapping(),
        plan_digest=PipelinePlan.single(1).digest,
        step_timeout_seconds=30,
        abort_grace_seconds=30,
        terminate=exits.append,
        enable_fault_listener=False,
    )
    try:
        with pytest.raises(PipelineStepAborted, match="descriptor differs"):
            control.begin_step(
                forward_mode_name="decode",
                batch_size=2,
                input_num_tokens=2,
                num_extends=0,
                batch_fingerprint=17,
                stage_cache_fingerprint=101,
            )
        with pytest.raises(PipelineStepAborted):
            control.begin_step(
                forward_mode_name="decode",
                batch_size=2,
                input_num_tokens=2,
                num_extends=0,
                batch_fingerprint=17,
                stage_cache_fingerprint=101,
            )
    finally:
        control.close(cancel_fatal_deadline=True)

    assert exits == []


@pytest.mark.parametrize("divergent_rank", (1, 3))
def test_stage_cache_divergence_stops_every_rank(monkeypatch, divergent_rank):
    _patch_groups_and_epoch(monkeypatch)

    def all_gather(gathered, local, group):
        del group
        for candidate in gathered:
            candidate.copy_(local)
        gathered[divergent_rank][-1] += 1

    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)
    control = TorchPipelineControlPlane(
        Mapping(
            rank=0,
            world_size=4,
            pipeline_parallel_size=2,
            attn_tp_size=2,
            dense_tp_size=2,
            moe_tp_size=2,
        ),
        plan_digest=PipelinePlan.single(1).digest,
        step_timeout_seconds=30,
        abort_grace_seconds=30,
        terminate=lambda _code: None,
        enable_fault_listener=False,
    )
    try:
        with pytest.raises(PipelineStepAborted, match="cache state differs"):
            control.begin_step(
                forward_mode_name="decode",
                batch_size=1,
                input_num_tokens=1,
                num_extends=0,
                batch_fingerprint=17,
                stage_cache_fingerprint=23,
            )
    finally:
        control.close(cancel_fatal_deadline=True)


def test_peer_fault_is_validated_relayed_and_poisons_next_step(monkeypatch):
    _patch_groups_and_epoch(monkeypatch)

    def all_gather(gathered, local, group):
        del group
        for candidate in gathered:
            candidate.copy_(local)

    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)
    control = TorchPipelineControlPlane(
        Mapping(
            rank=0,
            world_size=4,
            pipeline_parallel_size=2,
            attn_tp_size=2,
            dense_tp_size=2,
            moe_tp_size=2,
        ),
        plan_digest=PipelinePlan.single(1).digest,
        step_timeout_seconds=30,
        abort_grace_seconds=30,
        terminate=lambda _code: None,
        enable_fault_listener=False,
    )
    relayed = []

    def receive(tensor, src, group):
        assert src is None
        assert group == "fault-upstream"
        tensor.copy_(
            torch.tensor(
                (_FAULT_MAGIC, 1, control._epoch, 1, 2, 1, 0xA, 0xB),
                dtype=torch.int64,
            )
        )
        return 2

    def send(tensor, dst, group):
        assert group == "fault-downstream"
        relayed.append((dst, tensor.clone()))
        return _ImmediateWork()

    monkeypatch.setattr(torch.distributed, "recv", receive)
    monkeypatch.setattr(torch.distributed, "isend", send)
    control._fault_in_group = "fault-upstream"
    control._fault_out_group = "fault-downstream"
    try:
        control._listen_for_fault()
        with pytest.raises(PipelineStepAborted, match="peer rank 2 stage 1"):
            control.begin_step(
                forward_mode_name="decode",
                batch_size=1,
                input_num_tokens=1,
                num_extends=0,
                batch_fingerprint=23,
                stage_cache_fingerprint=107,
            )
    finally:
        control.close(cancel_fatal_deadline=True)

    assert [destination for destination, _packet in relayed] == [1, 3]
    assert all(packet[4].item() == 2 for _destination, packet in relayed)


def test_fault_relay_submits_every_peer_before_waiting(monkeypatch):
    _patch_groups_and_epoch(monkeypatch)
    control = TorchPipelineControlPlane(
        Mapping(
            rank=0,
            world_size=4,
            pipeline_parallel_size=2,
            attn_tp_size=2,
            dense_tp_size=2,
            moe_tp_size=2,
        ),
        plan_digest=PipelinePlan.single(1).digest,
        step_timeout_seconds=30,
        abort_grace_seconds=30,
        terminate=lambda _code: None,
        enable_fault_listener=False,
    )
    submitted = []

    def send(tensor, dst, group):
        del tensor
        assert group == "fault-group"
        submitted.append(dst)
        return _DeferredWork(
            submitted,
            expected_submissions=3,
            completed=dst != 1,
        )

    monkeypatch.setattr(torch.distributed, "isend", send)
    packet = torch.zeros(8, dtype=torch.int64)
    try:
        assert not control._send_packets(
            packet,
            (1, 2, 3),
            group="fault-group",
            wait=True,
        )
    finally:
        control.close(cancel_fatal_deadline=True)

    assert submitted == [1, 2, 3]


def test_fatal_close_preserves_the_hard_exit_deadline(monkeypatch):
    _patch_groups_and_epoch(monkeypatch)

    def all_gather(gathered, local, group):
        del group
        for candidate in gathered:
            candidate.copy_(local)

    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)
    terminated = threading.Event()
    control = TorchPipelineControlPlane(
        _mapping(),
        plan_digest=PipelinePlan.single(1).digest,
        step_timeout_seconds=30,
        abort_grace_seconds=0.02,
        terminate=lambda _code: terminated.set(),
        enable_fault_listener=False,
    )
    step = control.begin_step(
        forward_mode_name="decode",
        batch_size=1,
        input_num_tokens=1,
        num_extends=0,
        batch_fingerprint=29,
        stage_cache_fingerprint=109,
    )

    control.abort(step, "test", RuntimeError("boom"))
    control.close()

    assert terminated.wait(timeout=1)
    control.close(cancel_fatal_deadline=True)


def test_close_during_active_step_poison_preserves_deadline(monkeypatch):
    _patch_groups_and_epoch(monkeypatch)

    def all_gather(gathered, local, group):
        del group
        for candidate in gathered:
            candidate.copy_(local)

    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)
    terminated = threading.Event()
    control = TorchPipelineControlPlane(
        _mapping(),
        plan_digest=PipelinePlan.single(1).digest,
        step_timeout_seconds=30,
        abort_grace_seconds=0.02,
        terminate=lambda _code: terminated.set(),
        enable_fault_listener=False,
    )
    control.begin_step(
        forward_mode_name="decode",
        batch_size=1,
        input_num_tokens=1,
        num_extends=0,
        batch_fingerprint=31,
        stage_cache_fingerprint=113,
    )

    control.close()

    assert terminated.wait(timeout=1)
    control.close(cancel_fatal_deadline=True)


def test_close_fails_closed_when_fault_listener_does_not_stop(monkeypatch):
    _patch_groups_and_epoch(monkeypatch)
    monkeypatch.setattr(
        torch.distributed,
        "isend",
        lambda *_args, **_kwargs: _DeferredWork([], 0, completed=False),
    )
    control = TorchPipelineControlPlane(
        _mapping(),
        plan_digest=PipelinePlan.single(1).digest,
        step_timeout_seconds=30,
        abort_grace_seconds=0.02,
        terminate=lambda _code: None,
        enable_fault_listener=False,
    )
    control._fault_out_group = "fault-group"
    control._fault_thread = _StuckThread()
    try:
        with pytest.raises(PipelineStepAborted, match="did not stop"):
            control.close()
    finally:
        control._fault_out_group = None
        control._fault_thread = None
        control.close(cancel_fatal_deadline=True)


def test_fault_listener_renews_receive_after_heartbeat(monkeypatch):
    _patch_groups_and_epoch(monkeypatch)
    control = TorchPipelineControlPlane(
        Mapping(
            rank=0,
            world_size=4,
            pipeline_parallel_size=2,
            attn_tp_size=2,
            dense_tp_size=2,
            moe_tp_size=2,
        ),
        plan_digest=PipelinePlan.single(1).digest,
        terminate=lambda _code: None,
        enable_fault_listener=False,
    )
    packets = [
        torch.tensor(
            (
                _FAULT_MAGIC,
                _HEARTBEAT_PACKET_KIND,
                control._epoch,
                0,
                1,
                0,
                0,
                0,
            ),
            dtype=torch.int64,
        ),
        torch.tensor(
            (
                _FAULT_MAGIC,
                _STOP_PACKET_KIND,
                control._epoch,
                0,
                1,
                0,
                0,
                0,
            ),
            dtype=torch.int64,
        ),
    ]
    receives = []

    def receive(tensor, src, group):
        assert src is None
        assert group == "fault-upstream"
        receives.append(True)
        tensor.copy_(packets.pop(0))
        return 1

    monkeypatch.setattr(torch.distributed, "recv", receive)
    control._fault_in_group = "fault-upstream"
    try:
        control._listen_for_fault()
        assert control._poison is None
    finally:
        control.close(cancel_fatal_deadline=True)

    assert len(receives) == 2


def test_watchdog_emits_fatal_context_before_termination(monkeypatch):
    writes = []
    terminated = threading.Event()
    exit_codes = []

    monkeypatch.setattr(
        "tokenspeed.runtime.pipeline.torch_control.os.write",
        lambda fd, payload: writes.append((fd, payload)),
    )

    def terminate(code):
        exit_codes.append(code)
        terminated.set()

    watchdog = _DeadlineWatchdog(
        terminate,
        clock=time.monotonic,
        context="global_rank=7 pipeline_stage=0",
    )
    watchdog.arm(19, time.monotonic() + 0.01, "pipeline-step-decode")

    assert terminated.wait(timeout=1)
    watchdog.close()

    assert exit_codes == [_PIPELINE_EXIT_CODE]
    assert writes and writes[0][0] == 2
    diagnostic = writes[0][1].decode()
    assert "global_rank=7 pipeline_stage=0" in diagnostic
    assert "step_id=19" in diagnostic
    assert "reason=pipeline-step-decode" in diagnostic


def test_idle_fault_channel_survives_multiple_gloo_timeouts(tmp_path):
    init_file = tmp_path / "gloo-init"
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_run_idle_fault_channel,
            args=(rank, str(init_file)),
        )
        for rank in range(2)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)

    assert [process.exitcode for process in processes] == [0, 0]
