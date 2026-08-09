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

"""Typed bridge from ModelRunner inputs to the generic stage contract."""

from __future__ import annotations

from dataclasses import dataclass

from tokenspeed.runtime.pipeline.contracts import (
    StageActivation,
    StageOutput,
    StagePlan,
)


@dataclass(frozen=True)
class PipelineForwardBatch:
    input_ids: object
    positions: object
    out_cache_loc: object
    req_pool_indices: object | None = None
    seq_lens: object | None = None
    extend_prefix_lens: object | None = None
    input_embeds: object | None = None
    multimodal_context: object | None = None


class ModelRunnerPipelineStage:
    """Adapt a pipeline-aware ModelRunner without model-specific branching."""

    def __init__(self, plan: StagePlan, model_runner) -> None:
        self._plan = plan
        self._model_runner = model_runner

    @property
    def plan(self) -> StagePlan:
        return self._plan

    def forward_stage(
        self,
        context: object,
        batch: object,
        incoming: StageActivation | None,
    ) -> StageOutput:
        if not isinstance(batch, PipelineForwardBatch):
            raise TypeError("pipeline stage requires PipelineForwardBatch")
        return self._model_runner.forward_pipeline_stage(
            context,
            batch.input_ids,
            batch.positions,
            batch.out_cache_loc,
            incoming=incoming,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            extend_prefix_lens=batch.extend_prefix_lens,
            input_embeds=batch.input_embeds,
            multimodal_context=batch.multimodal_context,
        )

    def idle(self) -> None:
        return None

    def close(self) -> None:
        return None
