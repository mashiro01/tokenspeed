"""Unit coverage for eager PP warmup NaN flag sizing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tokenspeed.runtime.execution.model_executor import ModelExecutor
from tokenspeed.runtime.execution.nan_guard import NanGuard
from tokenspeed.runtime.pipeline.contracts import (
    ActivationFieldSpec,
    ActivationSchema,
    StagePlan,
)


class _PacketBus:
    def __init__(self) -> None:
        self.packet: torch.Tensor | None = None
        self.packet_lengths: list[int] = []
        self.flag_lengths: list[int] = []

    def publish(self, tokens: torch.Tensor, lengths: torch.Tensor, flags: torch.Tensor):
        self.flag_lengths.append(flags.numel())
        self.packet = torch.stack((tokens, lengths, flags), dim=1).reshape(-1).clone()
        self.packet_lengths.append(self.packet.numel())

    def consume(self, batch_size: int) -> torch.Tensor:
        assert self.packet is not None
        assert self.packet.numel() == 3 * batch_size
        return self.packet.reshape(batch_size, 3)[:, 2]


class _FakeCudaGraphWrapper:
    def __init__(self, forward_func) -> None:
        self._forward_func = forward_func

    def prewarm_comm_states(self, batch_sizes: tuple[int, ...]) -> None:
        for bs in batch_sizes:
            self._forward_func(
                bs=bs,
                ctx=SimpleNamespace(bs=bs),
                sampling_info=SimpleNamespace(),
            )


def _executor(forward_func) -> ModelExecutor:
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.nan_guard = NanGuard(max_bs=4, device="cpu")
    executor.forward_step = _FakeCudaGraphWrapper(forward_func)
    return executor


def test_pp2_eager_warmup_resets_nan_flags_to_dummy_batch_and_cleans_up():
    bus = _PacketBus()

    def final_stage(*, bs, ctx, sampling_info):
        assert final_executor.nan_guard.flags_device.numel() == bs
        final_executor.nan_guard.flags[:bs].fill_(1)
        bus.publish(
            torch.arange(bs, dtype=torch.int32),
            torch.ones(bs, dtype=torch.int32),
            final_executor.nan_guard.flags_device,
        )

    def first_stage(*, bs, ctx, sampling_info):
        assert first_executor.nan_guard.flags_device.numel() == bs
        first_executor.nan_guard.load_pipeline_flags(bus.consume(bs))
        assert first_executor.nan_guard.flags_device.tolist() == [1] * bs

    final_executor = _executor(final_stage)
    first_executor = _executor(first_stage)

    final_executor._prewarm_eager_comm_states(batch_sizes=(2,))
    first_executor._prewarm_eager_comm_states(batch_sizes=(2,))

    assert bus.packet_lengths == [6]
    assert bus.flag_lengths == [2]
    assert final_executor.nan_guard.flags_device.numel() == 0
    assert first_executor.nan_guard.flags_device.numel() == 0
    assert final_executor.nan_guard.flags.tolist() == [0, 0, 0, 0]
    assert first_executor.nan_guard.flags.tolist() == [0, 0, 0, 0]


def test_pipeline_autotune_uses_stage_executor_and_control_plane():
    class Control:
        def __init__(self) -> None:
            self.begin_calls = []
            self.completed = []

        def begin_step(self, **kwargs):
            self.begin_calls.append(kwargs)
            return "autotune-step"

        def complete_step(self, step) -> None:
            self.completed.append(step)

        def abort(self, step, phase, error):
            raise AssertionError(f"unexpected abort: {step=} {phase=} {error=}")

    class StageExecutor:
        def __init__(self) -> None:
            self.calls = []

        def forward(self, context, batch, step):
            self.calls.append((context, batch, step))
            return SimpleNamespace(final_output="tuned")

    executor = ModelExecutor.__new__(ModelExecutor)
    control = Control()
    stage_executor = StageExecutor()
    executor.config = SimpleNamespace(model_is_mrope=False)
    executor.input_buffers = SimpleNamespace(
        input_ids_buf=torch.tensor([11, 12, 13, 14, 99]),
        positions_buf=torch.tensor([0, 1, 2, 3, 99]),
        out_cache_loc_buf=torch.tensor([21, 22, 23, 24, 99]),
        req_pool_indices_buf=torch.tensor([7, 99]),
        seq_lens_buf=torch.tensor([4, 99]),
        extend_prefix_lens_buf=torch.tensor([0, 99]),
    )
    executor.pipeline_control = control
    executor.pipeline_executor = stage_executor
    ctx = SimpleNamespace(
        forward_mode=SimpleNamespace(name="EXTEND"),
        bs=1,
        input_num_tokens=4,
        num_extends=1,
        pipeline_batch_fingerprint=123,
    )

    result = executor._run_autotune_forward(ctx)

    assert result == "tuned"
    assert control.begin_calls == [
        {
            "forward_mode_name": "EXTEND",
            "batch_size": 1,
            "input_num_tokens": 4,
            "num_extends": 1,
            "batch_fingerprint": 123,
            "stage_cache_fingerprint": 0,
        }
    ]
    assert control.completed == ["autotune-step"]
    _context, batch, step = stage_executor.calls[0]
    assert step == "autotune-step"
    assert batch.input_ids.tolist() == [11, 12, 13, 14]
    assert batch.positions.tolist() == [0, 1, 2, 3]
    assert batch.out_cache_loc.tolist() == [21, 22, 23, 24]
    assert batch.req_pool_indices.tolist() == [7]
    assert batch.seq_lens.tolist() == [4]
    assert batch.extend_prefix_lens.tolist() == [0]


def test_pipeline_local_warmup_bypasses_the_pipeline_control_plane():
    schema = ActivationSchema(
        boundary_id="test/stage-0-to-1",
        fields=(
            ActivationFieldSpec(
                field_id="prefix_sum", dtype="bfloat16", trailing_shape=(4,)
            ),
        ),
    )
    stage_plan = StagePlan(
        stage_id=1,
        stage_count=2,
        first_layer=1,
        end_layer=2,
        owns_embedding=False,
        owns_head=True,
        input_schema=schema,
    )

    class Runner:
        def __init__(self) -> None:
            self.model = SimpleNamespace(pipeline_stage_plan=stage_plan)
            self.calls = []

        def forward_pipeline_stage(self, **kwargs):
            schema.validate(kwargs["incoming"])
            self.calls.append(kwargs)
            return SimpleNamespace(final_output=None)

    class PrefillGraph:
        def make_dummy_batch(self, num_tokens, _decode_wrapper):
            return SimpleNamespace(
                bs=1,
                num_extends=1,
                input_num_tokens=num_tokens,
            )

    executor = ModelExecutor.__new__(ModelExecutor)
    executor.config = SimpleNamespace(
        pipeline_stage_count=2,
        pipeline_stage_index=1,
        chunked_prefill_size=8192,
        pipeline_local_warmup_max_tokens=2048,
        model_is_mrope=False,
    )
    executor.device = "cpu"
    executor.model_runner = Runner()
    executor.prefill_graph = PrefillGraph()
    executor.forward_step = object()
    executor.input_buffers = SimpleNamespace(
        input_ids_buf=torch.ones(8192, dtype=torch.int32),
        positions_buf=torch.arange(8192, dtype=torch.int64),
        out_cache_loc_buf=torch.zeros(8192, dtype=torch.int32),
        req_pool_indices_buf=torch.zeros(1, dtype=torch.int64),
        seq_lens_buf=torch.ones(1, dtype=torch.int32),
        extend_prefix_lens_buf=torch.zeros(1, dtype=torch.int32),
    )
    executor.pipeline_executor = object()
    executor.pipeline_control = object()

    executor._warm_pipeline_stages_locally()

    assert [call["ctx"].input_num_tokens for call in executor.model_runner.calls] == [
        1,
        128,
        2048,
    ]
    assert all(call["incoming"] is not None for call in executor.model_runner.calls)


def test_pipeline_local_warmup_first_stage_uses_mrope_token_inputs():
    schema = ActivationSchema(
        boundary_id="test/stage-0-to-1",
        fields=(
            ActivationFieldSpec(
                field_id="prefix_sum", dtype="bfloat16", trailing_shape=(4,)
            ),
        ),
    )
    stage_plan = StagePlan(
        stage_id=0,
        stage_count=2,
        first_layer=0,
        end_layer=1,
        owns_embedding=True,
        owns_head=False,
        output_schema=schema,
    )

    class Runner:
        def __init__(self) -> None:
            self.model = SimpleNamespace(pipeline_stage_plan=stage_plan)
            self.calls = []

        def forward_pipeline_stage(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(activation=None)

    input_buffers = SimpleNamespace(
        input_ids_buf=torch.ones(128, dtype=torch.int32),
        positions_buf=torch.zeros(128, dtype=torch.int64),
        mrope_positions_buf=torch.zeros(3, 128, dtype=torch.int64),
        out_cache_loc_buf=torch.zeros(128, dtype=torch.int32),
        req_pool_indices_buf=torch.zeros(1, dtype=torch.int64),
        seq_lens_buf=torch.ones(1, dtype=torch.int32),
        extend_prefix_lens_buf=torch.zeros(1, dtype=torch.int32),
    )

    class PrefillGraph:
        def make_dummy_batch(self, num_tokens, _decode_wrapper):
            input_buffers.positions_buf[:num_tokens].copy_(
                torch.arange(num_tokens, dtype=torch.int64)
            )
            return SimpleNamespace(
                bs=1,
                num_extends=1,
                input_num_tokens=num_tokens,
            )

    executor = ModelExecutor.__new__(ModelExecutor)
    executor.config = SimpleNamespace(
        pipeline_stage_count=2,
        pipeline_stage_index=0,
        chunked_prefill_size=128,
        pipeline_local_warmup_max_tokens=128,
        model_is_mrope=True,
    )
    executor.device = "cpu"
    executor.model_runner = Runner()
    executor.prefill_graph = PrefillGraph()
    executor.forward_step = object()
    executor.input_buffers = input_buffers

    executor._warm_pipeline_stages_locally()

    assert [call["ctx"].input_num_tokens for call in executor.model_runner.calls] == [
        1,
        128,
    ]
    assert all(call["incoming"] is None for call in executor.model_runner.calls)
    for call in executor.model_runner.calls:
        num_tokens = call["ctx"].input_num_tokens
        expected = torch.arange(num_tokens, dtype=torch.int64).expand(3, -1)
        assert torch.equal(call["positions"], expected)
