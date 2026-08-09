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

from dataclasses import dataclass

import pytest

from tokenspeed.runtime.pipeline import (
    ActivationFieldSpec,
    ActivationSchema,
    DistributedStageExecutor,
    LocalPipelineExecutor,
    LoopbackPipelineExecutor,
    ModelRunnerPipelineStage,
    PipelineForwardBatch,
    PipelinePlan,
    PipelineProtocolError,
    StageOutput,
    StagePlan,
    WholeModelStage,
)
from tokenspeed.runtime.pipeline.contracts import StageActivation, canonical_digest


@dataclass(frozen=True)
class FakeTensor:
    dtype: str
    shape: tuple[int, ...]
    value: int = 0


class FakeStage:
    def __init__(self, plan, forward):
        self.plan = plan
        self._forward = forward
        self.idle_count = 0
        self.close_count = 0

    def forward_stage(self, context, batch, incoming):
        return self._forward(context, batch, incoming)

    def idle(self):
        self.idle_count += 1

    def close(self):
        self.close_count += 1


class QueueTransport:
    def __init__(self):
        self.payloads = []
        self.close_count = 0

    def send(self, step, schema, activation):
        del step
        schema.validate(activation)
        self.payloads.append(activation)

    def receive(self, step, schema):
        del step
        activation = self.payloads.pop(0)
        schema.validate(activation)
        return activation

    def close(self):
        self.close_count += 1


def two_stage_plan():
    schema = ActivationSchema(
        "stage-0-to-1",
        (
            ActivationFieldSpec("hidden", "bfloat16", (8,)),
            ActivationFieldSpec("residual", "float32", (8,)),
        ),
    )
    return PipelinePlan(
        (
            StagePlan(0, 2, 0, 3, True, False, output_schema=schema),
            StagePlan(1, 2, 3, 6, False, True, input_schema=schema),
        )
    )


def test_single_stage_adapter_preserves_whole_model_result_and_lifecycle():
    plan = PipelinePlan.single(6)
    lifecycle = []
    stage = WholeModelStage(
        plan.stages[0],
        lambda context, batch: (context, batch),
        idle=lambda: lifecycle.append("idle"),
        close=lambda: lifecycle.append("close"),
    )
    executor = LocalPipelineExecutor(stage)

    assert executor.forward("context", "batch") == ("context", "batch")
    executor.idle()
    executor.close()
    assert lifecycle == ["idle", "close"]


def test_loopback_pp2_validates_two_tensor_activation_and_final_authority():
    plan = two_stage_plan()
    schema = plan.stages[0].output_schema
    observed = []

    first = FakeStage(
        plan.stages[0],
        lambda _context, _batch, incoming: StageOutput(
            activation=schema.bind(
                (FakeTensor("bfloat16", (4, 8), 3), FakeTensor("float32", (4, 8), 5))
            )
        ),
    )

    def finish(_context, _batch, incoming):
        observed.extend(value.value for value in incoming.values)
        return StageOutput(final_output=sum(observed))

    second = FakeStage(plan.stages[1], finish)
    executor = LoopbackPipelineExecutor(plan, (first, second))

    assert executor.forward(object(), object()) == 8
    assert observed == [3, 5]
    executor.idle()
    executor.close()
    assert (first.idle_count, second.idle_count) == (1, 1)
    assert (first.close_count, second.close_count) == (1, 1)


def test_distributed_stage_executor_moves_only_adjacent_activation():
    plan = two_stage_plan()
    schema = plan.stages[0].output_schema
    transport = QueueTransport()
    activation = schema.bind(
        (FakeTensor("bfloat16", (2, 8), 4), FakeTensor("float32", (2, 8), 6))
    )
    first = FakeStage(
        plan.stages[0],
        lambda _context, _batch, incoming: StageOutput(activation=activation),
    )
    second = FakeStage(
        plan.stages[1],
        lambda _context, _batch, incoming: StageOutput(
            final_output=sum(value.value for value in incoming.values)
        ),
    )

    step = object()
    first_output = DistributedStageExecutor(first, transport).forward(None, None, step)
    final_output = DistributedStageExecutor(second, transport).forward(None, None, step)

    assert first_output.activation is activation
    assert final_output.final_output == 10
    assert transport.payloads == []


def test_model_runner_stage_uses_typed_forward_batch():
    plan = two_stage_plan()
    schema = plan.stages[0].output_schema
    activation = schema.bind(
        (FakeTensor("bfloat16", (1, 8)), FakeTensor("float32", (1, 8)))
    )
    observed = {}

    class FakeRunner:
        def forward_pipeline_stage(self, ctx, input_ids, positions, out_cache_loc, **kw):
            observed.update(
                ctx=ctx,
                input_ids=input_ids,
                positions=positions,
                out_cache_loc=out_cache_loc,
                incoming=kw["incoming"],
            )
            return StageOutput(activation=activation)

    stage = ModelRunnerPipelineStage(plan.stages[0], FakeRunner())
    batch = PipelineForwardBatch("ids", "positions", "cache")

    assert stage.forward_stage("ctx", batch, None).activation is activation
    assert observed == {
        "ctx": "ctx",
        "input_ids": "ids",
        "positions": "positions",
        "out_cache_loc": "cache",
        "incoming": None,
    }
    with pytest.raises(TypeError, match="PipelineForwardBatch"):
        stage.forward_stage("ctx", object(), None)


def test_loopback_rejects_schema_and_stage_output_violations():
    plan = two_stage_plan()
    schema = plan.stages[0].output_schema
    bad_dtype = FakeStage(
        plan.stages[0],
        lambda *_args: StageOutput(
            activation=schema.bind(
                (FakeTensor("float16", (1, 8)), FakeTensor("float32", (1, 8)))
            )
        ),
    )
    final = FakeStage(plan.stages[1], lambda *_args: StageOutput(final_output=1))
    with pytest.raises(PipelineProtocolError, match="expected dtype"):
        LoopbackPipelineExecutor(plan, (bad_dtype, final)).forward(None, None)

    early_final = FakeStage(
        plan.stages[0], lambda *_args: StageOutput(final_output="not-authoritative")
    )
    with pytest.raises(PipelineProtocolError, match="non-final stage"):
        LoopbackPipelineExecutor(plan, (early_final, final)).forward(None, None)

    activation = schema.bind(
        (FakeTensor("bfloat16", (1, 8)), FakeTensor("float32", (1, 8)))
    )
    first = FakeStage(plan.stages[0], lambda *_args: StageOutput(activation=activation))
    bad_final = FakeStage(
        plan.stages[1], lambda *_args: StageOutput(activation=activation)
    )
    with pytest.raises(PipelineProtocolError, match="final stage"):
        LoopbackPipelineExecutor(plan, (first, bad_final)).forward(None, None)


def test_pipeline_plan_rejects_gaps_and_boundary_divergence():
    left_schema = ActivationSchema(
        "left", (ActivationFieldSpec("hidden", "bfloat16", (8,)),)
    )
    right_schema = ActivationSchema(
        "right", (ActivationFieldSpec("hidden", "bfloat16", (8,)),)
    )
    with pytest.raises(ValueError, match="contiguous"):
        PipelinePlan(
            (
                StagePlan(0, 2, 0, 3, True, False, output_schema=left_schema),
                StagePlan(1, 2, 4, 6, False, True, input_schema=left_schema),
            )
        )
    with pytest.raises(ValueError, match="boundary schema"):
        PipelinePlan(
            (
                StagePlan(0, 2, 0, 3, True, False, output_schema=left_schema),
                StagePlan(1, 2, 3, 6, False, True, input_schema=right_schema),
            )
        )


def test_pipeline_plan_digest_is_deterministic_and_rejects_float_projection():
    assert PipelinePlan.single(6).digest == PipelinePlan.single(6).digest
    with pytest.raises(ValueError, match="floating-point"):
        canonical_digest({"duration": 1.5})


def test_pipeline_plan_and_single_stage_adapter_reject_incomplete_ownership():
    nonzero_start = StagePlan(0, 1, 1, 2, True, True)
    with pytest.raises(ValueError, match="logical layer zero"):
        PipelinePlan((nonzero_start,))

    missing_embedding = StagePlan(0, 1, 0, 2, False, True)
    with pytest.raises(ValueError, match="first stage.*embedding"):
        WholeModelStage(missing_embedding, lambda *_args: None)
    with pytest.raises(ValueError, match="first stage.*embedding"):
        LocalPipelineExecutor(FakeStage(missing_embedding, lambda *_args: None))


def test_activation_schema_rejects_digest_shape_and_non_tensor_values():
    schema = ActivationSchema(
        "boundary",
        (ActivationFieldSpec("hidden", "bfloat16", (8,)),),
    )

    with pytest.raises(PipelineProtocolError, match="digest mismatch"):
        schema.validate(
            StageActivation((FakeTensor("bfloat16", (2, 8)),), "wrong-digest")
        )
    with pytest.raises(PipelineProtocolError, match="trailing shape"):
        schema.bind((FakeTensor("bfloat16", (2, 7)),))
    with pytest.raises(PipelineProtocolError, match="expected rank"):
        schema.bind((FakeTensor("bfloat16", (1, 2, 8)),))
    dtype_only = type("DtypeOnly", (), {"dtype": "bfloat16"})()
    with pytest.raises(PipelineProtocolError, match="has no shape"):
        schema.bind((dtype_only,))


def test_activation_schema_requires_shared_leading_shape():
    schema = ActivationSchema(
        "boundary",
        (
            ActivationFieldSpec("hidden", "bfloat16", (8,)),
            ActivationFieldSpec("residual", "bfloat16", (8,)),
        ),
    )

    with pytest.raises(PipelineProtocolError, match="leading shape"):
        schema.bind(
            (
                FakeTensor("bfloat16", (2, 8)),
                FakeTensor("bfloat16", (3, 8)),
            )
        )


def test_frozen_contracts_copy_mutable_source_sequences():
    trailing_shape = [8]
    fields = [ActivationFieldSpec("hidden", "bfloat16", trailing_shape)]
    schema = ActivationSchema("boundary", fields)
    stages = [StagePlan(0, 1, 0, 2, True, True)]
    plan = PipelinePlan(stages)
    activation_values = [FakeTensor("bfloat16", (2, 8))]
    activation = StageActivation(activation_values, schema.digest)

    trailing_shape[0] = 7
    fields.clear()
    stages.append(StagePlan(0, 1, 0, 2, True, True))
    activation_values.clear()

    assert schema.fields[0].trailing_shape == (8,)
    assert len(schema.fields) == 1
    assert len(plan.stages) == 1
    assert len(activation.values) == 1


def test_loopback_rejects_non_stage_output_and_closes_in_reverse_order():
    plan = two_stage_plan()
    close_order = []

    class OrderedStage(FakeStage):
        def close(self):
            close_order.append(self.plan.stage_id)

    bad_first = OrderedStage(plan.stages[0], lambda *_args: "raw-output")
    final = OrderedStage(plan.stages[1], lambda *_args: StageOutput(final_output=1))
    executor = LoopbackPipelineExecutor(plan, (bad_first, final))

    with pytest.raises(PipelineProtocolError, match="not StageOutput"):
        executor.forward(None, None)
    executor.close()
    assert close_order == [1, 0]
