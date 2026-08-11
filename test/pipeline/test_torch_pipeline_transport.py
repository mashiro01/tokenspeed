import pytest

torch = pytest.importorskip("torch")

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.pipeline.contracts import (
    ActivationFieldSpec,
    ActivationSchema,
    PipelineForwardMode,
    PipelinePlan,
    PipelineProtocolError,
    PipelineStepDescriptor,
)
from tokenspeed.runtime.pipeline.torch_control import PipelineStepLease
from tokenspeed.runtime.pipeline.torch_transport import (
    TorchPipelineDSparkSynchronizer,
    TorchPipelineResultSynchronizer,
    TorchPipelineTransport,
    pg_manager,
)


class _ImmediateWork:
    def is_completed(self):
        return True

    def wait(self):
        return None


class _Control:
    def __init__(self):
        self.phases = []

    def wait_work(self, work, step, phase):
        del step
        self.phases.append(phase)
        work.wait()


class _SubmissionControl:
    def __init__(self, payload_submissions):
        self.payload_submissions = payload_submissions

    def wait_work(self, work, step, phase):
        del step
        if phase.startswith("activation-payload-send"):
            assert self.payload_submissions == [2]
        work.wait()


def _step(batch_size: int = 2) -> PipelineStepLease:
    return PipelineStepLease(
        PipelineStepDescriptor(
            epoch=11,
            step_id=3,
            forward_mode=PipelineForwardMode.DECODE,
            batch_size=batch_size,
            input_num_tokens=batch_size,
            num_extends=0,
            plan_digest=PipelinePlan.single(1).digest,
            batch_fingerprint=17,
        ),
        deadline=float("inf"),
    )


def test_torch_transport_sends_fixed_header_then_tensor_payloads(monkeypatch):
    wire = []

    monkeypatch.setattr(
        pg_manager,
        "get_process_group",
        lambda backend, group, **kwargs: (backend, group, kwargs.get("role")),
    )

    def send(tensor, dst, group):
        wire.append((dst, group, tensor.clone()))
        return _ImmediateWork()

    def receive(tensor, src, group):
        dst, sent_group, value = wire.pop(0)
        assert dst == 1
        assert src == 0
        assert sent_group == group
        tensor.copy_(value)
        return _ImmediateWork()

    def p2p(op, tensor, peer, group=None, tag=0):
        del tag
        return op, tensor, peer, group

    def batch(ops):
        for op, tensor, peer, group in ops:
            if op is send:
                wire.append((peer, group, tensor.clone()))
            else:
                dst, sent_group, value = wire.pop(0)
                assert dst == 1
                assert peer == 0
                assert sent_group == group
                tensor.copy_(value)
        # NCCL coalescing reports a single aggregate Work for a P2P batch.
        return [_ImmediateWork()]

    monkeypatch.setattr(torch.distributed, "isend", send)
    monkeypatch.setattr(torch.distributed, "irecv", receive)
    monkeypatch.setattr(torch.distributed, "P2POp", p2p)
    monkeypatch.setattr(torch.distributed, "batch_isend_irecv", batch)

    first_mapping = Mapping(
        rank=0,
        world_size=2,
        pipeline_parallel_size=2,
        attn_tp_size=1,
        dense_tp_size=1,
        moe_tp_size=1,
    )
    second_mapping = Mapping(
        rank=1,
        world_size=2,
        pipeline_parallel_size=2,
        attn_tp_size=1,
        dense_tp_size=1,
        moe_tp_size=1,
    )
    schema = ActivationSchema(
        "stage-0-to-1",
        (
            ActivationFieldSpec("hidden", "bfloat16", (4,)),
            ActivationFieldSpec("residual", "float32", (4,)),
        ),
    )
    activation = schema.bind(
        (
            torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
            torch.arange(8, dtype=torch.float32).reshape(2, 4),
        )
    )

    control = _Control()
    step = _step()
    TorchPipelineTransport(first_mapping, device="cpu", control=control).send(
        step, schema, activation
    )
    received = TorchPipelineTransport(
        second_mapping, device="cpu", control=control
    ).receive(step, schema)

    assert wire == []
    assert torch.equal(received.values[0], activation.values[0])
    assert torch.equal(received.values[1], activation.values[1])
    assert control.phases == [
        "activation-header-send",
        "activation-payload-send-batch",
        "activation-header-receive",
        "activation-payload-receive-batch",
    ]


def test_torch_transport_reuses_exact_decode_receive_buffers(monkeypatch):
    wire = []

    monkeypatch.setattr(
        pg_manager,
        "get_process_group",
        lambda backend, group, **kwargs: (backend, group, kwargs.get("role")),
    )

    def send(tensor, dst, group):
        wire.append((dst, group, tensor.clone()))
        return _ImmediateWork()

    def receive(tensor, src, group):
        dst, sent_group, value = wire.pop(0)
        assert dst == 1
        assert src == 0
        assert sent_group == group
        tensor.copy_(value)
        return _ImmediateWork()

    def p2p(op, tensor, peer, group=None, tag=0):
        del tag
        return op, tensor, peer, group

    def batch(ops):
        for op, tensor, peer, group in ops:
            if op is send:
                wire.append((peer, group, tensor.clone()))
            else:
                dst, sent_group, value = wire.pop(0)
                assert dst == 1
                assert peer == 0
                assert sent_group == group
                tensor.copy_(value)
        return [_ImmediateWork()]

    monkeypatch.setattr(torch.distributed, "isend", send)
    monkeypatch.setattr(torch.distributed, "irecv", receive)
    monkeypatch.setattr(torch.distributed, "P2POp", p2p)
    monkeypatch.setattr(torch.distributed, "batch_isend_irecv", batch)

    first_mapping = Mapping(
        rank=0,
        world_size=2,
        pipeline_parallel_size=2,
        attn_tp_size=1,
        dense_tp_size=1,
        moe_tp_size=1,
    )
    second_mapping = Mapping(
        rank=1,
        world_size=2,
        pipeline_parallel_size=2,
        attn_tp_size=1,
        dense_tp_size=1,
        moe_tp_size=1,
    )
    schema = ActivationSchema(
        "stage-0-to-1",
        (ActivationFieldSpec("hidden", "bfloat16", (4,)),),
    )
    source = TorchPipelineTransport(first_mapping, device="cpu", control=_Control())
    destination = TorchPipelineTransport(
        second_mapping,
        device="cpu",
        control=_Control(),
    )
    step = _step()

    first = schema.bind((torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),))
    source.send(step, schema, first)
    received = destination.receive(step, schema)
    receive_ptr = received.values[0].data_ptr()

    second = schema.bind((torch.arange(8, 16, dtype=torch.bfloat16).reshape(2, 4),))
    source.send(step, schema, second)
    received_again = destination.receive(step, schema)

    assert received_again.values[0].data_ptr() == receive_ptr
    assert torch.equal(received_again.values[0], second.values[0])


def test_torch_transport_submits_all_payloads_before_waiting(monkeypatch):
    monkeypatch.setattr(
        pg_manager,
        "get_process_group",
        lambda backend, group, **kwargs: (backend, group, kwargs.get("role")),
    )
    payload_submissions = []

    def send(tensor, dst, group):
        del tensor, dst
        if group[0] == "nccl":
            pytest.fail("payload used an unbatched NCCL isend")
        return _ImmediateWork()

    def p2p(op, tensor, peer, group=None, tag=0):
        del tag
        return op, tensor, peer, group

    def batch(ops):
        assert len(ops) == 2
        assert all(op is send for op, _tensor, _peer, _group in ops)
        assert all(group[0] == "nccl" for _op, _tensor, _peer, group in ops)
        payload_submissions.append(len(ops))
        return [_ImmediateWork() for _ in ops]

    monkeypatch.setattr(torch.distributed, "isend", send)
    monkeypatch.setattr(torch.distributed, "P2POp", p2p)
    monkeypatch.setattr(torch.distributed, "batch_isend_irecv", batch)
    mapping = Mapping(
        rank=0,
        world_size=2,
        pipeline_parallel_size=2,
        attn_tp_size=1,
        dense_tp_size=1,
        moe_tp_size=1,
    )
    schema = ActivationSchema(
        "stage-0-to-1",
        (
            ActivationFieldSpec("hidden", "bfloat16", (4,)),
            ActivationFieldSpec("residual", "bfloat16", (4,)),
        ),
    )
    activation = schema.bind(
        (
            torch.zeros((2, 4), dtype=torch.bfloat16),
            torch.ones((2, 4), dtype=torch.bfloat16),
        )
    )

    TorchPipelineTransport(
        mapping,
        device="cpu",
        control=_SubmissionControl(payload_submissions),
    ).send(_step(), schema, activation)


def test_torch_transport_rejects_token_dimension_mismatch_before_send(monkeypatch):
    monkeypatch.setattr(
        pg_manager,
        "get_process_group",
        lambda backend, group, **kwargs: (backend, group, kwargs.get("role")),
    )
    monkeypatch.setattr(
        torch.distributed,
        "isend",
        lambda *_args, **_kwargs: pytest.fail("invalid payload reached the wire"),
    )
    mapping = Mapping(
        rank=0,
        world_size=2,
        pipeline_parallel_size=2,
        attn_tp_size=1,
        dense_tp_size=1,
        moe_tp_size=1,
    )
    schema = ActivationSchema(
        "stage-0-to-1",
        (ActivationFieldSpec("hidden", "bfloat16", (4,)),),
    )
    activation = schema.bind((torch.zeros((3, 4), dtype=torch.bfloat16),))

    with pytest.raises(PipelineProtocolError, match="token dimension disagrees"):
        TorchPipelineTransport(
            mapping,
            device="cpu",
            control=_Control(),
        ).send(_step(batch_size=2), schema, activation)


def test_pipeline_result_synchronizer_broadcasts_compact_int_packet(monkeypatch):
    published = []
    monkeypatch.setattr(
        pg_manager,
        "get_process_group",
        lambda backend, group, **kwargs: (backend, group, kwargs.get("role")),
    )

    def broadcast(tensor, src, group, async_op=False):
        assert src == 1
        assert group[0] == "nccl"
        assert async_op
        if not published:
            published.append(tensor.clone())
        else:
            tensor.copy_(published[0])
        return _ImmediateWork()

    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    final_mapping = Mapping(
        rank=1,
        world_size=2,
        pipeline_parallel_size=2,
        attn_tp_size=1,
        dense_tp_size=1,
        moe_tp_size=1,
    )
    first_mapping = Mapping(
        rank=0,
        world_size=2,
        pipeline_parallel_size=2,
        attn_tp_size=1,
        dense_tp_size=1,
        moe_tp_size=1,
    )
    control = _Control()
    final = TorchPipelineResultSynchronizer(
        final_mapping, device="cpu", control=control
    )
    first = TorchPipelineResultSynchronizer(
        first_mapping, device="cpu", control=control
    )
    step = _step()

    final.synchronize(
        step=step,
        batch_size=2,
        output_tokens=torch.tensor([7, 8]),
        accept_lengths=torch.tensor([1, 1]),
        nan_flags=torch.tensor([0, 1], dtype=torch.int32),
    )
    tokens, lengths, flags = first.synchronize(step=step, batch_size=2)

    assert tokens.tolist() == [7, 8]
    assert lengths.tolist() == [1, 1]
    assert flags.tolist() == [0, 1]


def test_pipeline_result_synchronizer_preserves_a_speculative_verify_window(
    monkeypatch,
):
    published = []
    monkeypatch.setattr(
        pg_manager,
        "get_process_group",
        lambda backend, group, **kwargs: (backend, group, kwargs.get("role")),
    )

    def broadcast(tensor, src, group, async_op=False):
        assert src == 1
        assert async_op
        if not published:
            published.append(tensor.clone())
        else:
            tensor.copy_(published[0])
        return _ImmediateWork()

    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    final_mapping = Mapping(
        rank=1,
        world_size=2,
        pipeline_parallel_size=2,
        attn_tp_size=1,
        dense_tp_size=1,
        moe_tp_size=1,
    )
    first_mapping = Mapping(
        rank=0,
        world_size=2,
        pipeline_parallel_size=2,
        attn_tp_size=1,
        dense_tp_size=1,
        moe_tp_size=1,
    )
    control = _Control()
    final = TorchPipelineResultSynchronizer(
        final_mapping, device="cpu", control=control
    )
    first = TorchPipelineResultSynchronizer(
        first_mapping, device="cpu", control=control
    )
    step = _step()

    final.synchronize(
        step=step,
        batch_size=2,
        output_tokens=torch.tensor([10, 11, 12, 13, 20, 21, 22, 23]),
        accept_lengths=torch.tensor([4, 2]),
        nan_flags=torch.tensor([0, 0], dtype=torch.int32),
        output_token_count=8,
    )
    tokens, lengths, flags = first.synchronize(
        step=step,
        batch_size=2,
        output_token_count=8,
    )

    assert tokens.tolist() == [10, 11, 12, 13, 20, 21, 22, 23]
    assert lengths.tolist() == [4, 2]
    assert flags.tolist() == [0, 0]


def test_k3_dspark_synchronizer_relays_context_to_pp0_and_candidates_back(
    monkeypatch,
):
    mailbox = {"gloo": [], "nccl": []}
    candidate_packets = []
    monkeypatch.setattr(
        pg_manager,
        "get_process_group",
        lambda backend, group, **kwargs: (backend, group, kwargs.get("role")),
    )

    def isend(tensor, dst, group):
        del dst
        mailbox[group[0]].append(tensor.clone())
        return _ImmediateWork()

    def irecv(tensor, src, group):
        del src
        tensor.copy_(mailbox[group[0]].pop(0))
        return _ImmediateWork()

    def broadcast(tensor, src, group, async_op=False):
        assert src == 0
        assert group[0] == "nccl"
        assert async_op
        if not candidate_packets:
            candidate_packets.append(tensor.clone())
        else:
            tensor.copy_(candidate_packets[0])
        return _ImmediateWork()

    monkeypatch.setattr(torch.distributed, "isend", isend)
    monkeypatch.setattr(torch.distributed, "irecv", irecv)
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    first_mapping = Mapping(
        rank=0,
        world_size=2,
        pipeline_parallel_size=2,
        attn_tp_size=1,
        dense_tp_size=1,
        moe_tp_size=1,
    )
    final_mapping = Mapping(
        rank=1,
        world_size=2,
        pipeline_parallel_size=2,
        attn_tp_size=1,
        dense_tp_size=1,
        moe_tp_size=1,
    )
    control = _Control()
    first = TorchPipelineDSparkSynchronizer(
        first_mapping,
        device="cpu",
        control=control,
        context_hidden_size=3,
        candidate_width=4,
    )
    final = TorchPipelineDSparkSynchronizer(
        final_mapping,
        device="cpu",
        control=control,
        context_hidden_size=3,
        candidate_width=4,
    )
    step = _step()
    context = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    assert final.relay_context(step=step, context=context) is None
    received = first.relay_context(step=step)
    assert torch.equal(received, context)

    local_candidates = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    assert torch.equal(
        first.broadcast_candidates(step=step, candidates=local_candidates),
        local_candidates,
    )
    assert torch.equal(
        final.broadcast_candidates(step=step),
        local_candidates,
    )

    candidate_packets.clear()
    local_widths = torch.tensor([4, 2], dtype=torch.int32)
    first_candidates, first_widths = first.broadcast_candidates_and_widths(
        step=step,
        candidates=local_candidates,
        verify_widths=local_widths,
    )
    final_candidates, final_widths = final.broadcast_candidates_and_widths(step=step)

    assert torch.equal(first_candidates, local_candidates)
    assert torch.equal(first_widths, local_widths)
    assert torch.equal(final_candidates, local_candidates)
    assert torch.equal(final_widths, local_widths)
