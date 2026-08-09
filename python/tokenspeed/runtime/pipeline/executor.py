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

"""Local and loopback executors for the pipeline stage contract."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from tokenspeed.runtime.pipeline.contracts import (
    PipelinePlan,
    PipelineProtocolError,
    StageActivation,
    StageModel,
    StageOutput,
    StagePlan,
)
from tokenspeed.runtime.pipeline.transport import PipelineTransport

if TYPE_CHECKING:
    from tokenspeed.runtime.pipeline.torch_control import PipelineStepLease


class WholeModelStage:
    """Adapt an existing whole-model callable to the single-stage contract."""

    def __init__(
        self,
        plan: StagePlan,
        forward: Callable[[object, object], object],
        *,
        idle: Callable[[], None] | None = None,
        close: Callable[[], None] | None = None,
    ) -> None:
        if plan.stage_count != 1:
            raise ValueError("WholeModelStage requires a single-stage plan")
        PipelinePlan((plan,))
        self._plan = plan
        self._forward = forward
        self._idle = idle
        self._close = close

    @property
    def plan(self) -> StagePlan:
        return self._plan

    def forward_stage(
        self,
        context: object,
        batch: object,
        incoming: StageActivation | None,
    ) -> StageOutput:
        if incoming is not None:
            raise PipelineProtocolError("a whole-model stage cannot consume activation input")
        return StageOutput(final_output=self._forward(context, batch))

    def idle(self) -> None:
        if self._idle is not None:
            self._idle()

    def close(self) -> None:
        if self._close is not None:
            self._close()


class LocalPipelineExecutor:
    """Execute one validated stage without transport."""

    def __init__(self, stage: StageModel) -> None:
        if stage.plan.stage_count != 1:
            raise ValueError("LocalPipelineExecutor requires a single-stage model")
        PipelinePlan((stage.plan,))
        self._stage = stage

    def forward(self, context: object, batch: object) -> object:
        output = self._stage.forward_stage(context, batch, None)
        _validate_stage_output(self._stage.plan, output)
        return output.final_output

    def idle(self) -> None:
        self._stage.idle()

    def close(self) -> None:
        self._stage.close()


class LoopbackPipelineExecutor:
    """Run several stage models synchronously while enforcing wire contracts."""

    def __init__(self, plan: PipelinePlan, stages: Sequence[StageModel]) -> None:
        if len(stages) != len(plan.stages):
            raise ValueError("stage model count does not match the pipeline plan")
        for expected, stage in zip(plan.stages, stages):
            if stage.plan != expected:
                raise ValueError(f"stage model {stage.plan.stage_id} has a different plan")
        self._plan = plan
        self._stages = tuple(stages)

    @property
    def plan(self) -> PipelinePlan:
        return self._plan

    def forward(self, context: object, batch: object) -> object:
        incoming = None
        for stage in self._stages:
            plan = stage.plan
            if plan.input_schema is not None:
                if incoming is None:
                    raise PipelineProtocolError(
                        f"stage {plan.stage_id} is missing its activation input"
                    )
                plan.input_schema.validate(incoming)
            output = stage.forward_stage(context, batch, incoming)
            _validate_stage_output(plan, output)
            if plan.owns_head:
                return output.final_output
            incoming = output.activation
        raise PipelineProtocolError("pipeline completed without final-stage output")

    def idle(self) -> None:
        for stage in self._stages:
            stage.idle()

    def close(self) -> None:
        for stage in reversed(self._stages):
            stage.close()


class DistributedStageExecutor:
    """Execute one stage and move only its adjacent activation payload.

    Final-output synchronization is deliberately outside this class: model
    runtimes must define a compact typed result contract for sampling rather
    than serializing an arbitrary Python object through the activation lane.
    """

    def __init__(self, stage: StageModel, transport: PipelineTransport) -> None:
        if stage.plan.stage_count < 2:
            raise ValueError("DistributedStageExecutor requires multiple stages")
        self._stage = stage
        self._transport = transport

    def forward(
        self,
        context: object,
        batch: object,
        step: PipelineStepLease,
    ) -> StageOutput:
        plan = self._stage.plan
        incoming = (
            None
            if plan.input_schema is None
            else self._transport.receive(step, plan.input_schema)
        )
        output = self._stage.forward_stage(context, batch, incoming)
        _validate_stage_output(plan, output)
        if output.activation is not None:
            assert plan.output_schema is not None
            self._transport.send(step, plan.output_schema, output.activation)
        return output

    def idle(self) -> None:
        self._stage.idle()

    def close(self) -> None:
        try:
            self._stage.close()
        finally:
            self._transport.close()


def _validate_stage_output(plan: StagePlan, output: StageOutput) -> None:
    if not isinstance(output, StageOutput):
        raise PipelineProtocolError(
            f"stage {plan.stage_id} returned {type(output).__name__}, not StageOutput"
        )
    if plan.owns_head:
        if output.activation is not None or output.final_output is None:
            raise PipelineProtocolError(
                f"final stage {plan.stage_id} must return only final_output"
            )
        return
    if output.final_output is not None or output.activation is None:
        raise PipelineProtocolError(
            f"non-final stage {plan.stage_id} must return only activation"
        )
    if plan.output_schema is None:
        raise PipelineProtocolError(f"stage {plan.stage_id} has no output schema")
    plan.output_schema.validate(output.activation)
