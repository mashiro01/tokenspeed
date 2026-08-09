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

"""Model-independent pipeline runtime contracts."""

from tokenspeed.runtime.pipeline.contracts import (
    ActivationFieldSpec,
    ActivationSchema,
    PipelineForwardMode,
    PipelinePlan,
    PipelineProtocolError,
    PipelineStepAborted,
    PipelineStepDescriptor,
    StageActivation,
    StageModel,
    StageOutput,
    StagePlan,
)
from tokenspeed.runtime.pipeline.executor import (
    DistributedStageExecutor,
    LocalPipelineExecutor,
    LoopbackPipelineExecutor,
    WholeModelStage,
)
from tokenspeed.runtime.pipeline.model_runner_stage import (
    ModelRunnerPipelineStage,
    PipelineForwardBatch,
)
from tokenspeed.runtime.pipeline.transport import PipelineTransport

__all__ = [
    "ActivationFieldSpec",
    "ActivationSchema",
    "DistributedStageExecutor",
    "LocalPipelineExecutor",
    "LoopbackPipelineExecutor",
    "ModelRunnerPipelineStage",
    "PipelineForwardMode",
    "PipelinePlan",
    "PipelineForwardBatch",
    "PipelineProtocolError",
    "PipelineStepAborted",
    "PipelineStepDescriptor",
    "PipelineTransport",
    "StageActivation",
    "StageModel",
    "StageOutput",
    "StagePlan",
    "WholeModelStage",
]
