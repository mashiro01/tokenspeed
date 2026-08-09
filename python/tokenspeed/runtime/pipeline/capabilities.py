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

"""Validated model configurations for the native pipeline runtime."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineCapability:
    architecture: str
    activation_dtypes: tuple[str, ...]
    stage_counts: tuple[int, ...]


_CAPABILITIES = {
    "KimiK3ForConditionalGeneration": PipelineCapability(
        architecture="KimiK3ForConditionalGeneration",
        activation_dtypes=("bfloat16",),
        stage_counts=(8,),
    ),
}


def require_single_stage_control(*, stage_count: int, operation: str) -> None:
    """Reject control operations that do not yet have PP transaction semantics."""

    if isinstance(stage_count, bool) or not isinstance(stage_count, int):
        raise TypeError("stage_count must be an integer")
    if stage_count < 1:
        raise ValueError("stage_count must be positive")
    if not isinstance(operation, str) or not operation:
        raise ValueError("operation must be a non-empty string")
    if stage_count > 1:
        raise RuntimeError(
            f"{operation} is unavailable with native pipeline parallelism until "
            "stage-global prepare/commit/rollback is implemented"
        )


def validate_pipeline_capability(
    *,
    architecture: str,
    activation_dtype: str,
    stage_count: int,
) -> PipelineCapability | None:
    """Fail before distributed startup for an unqualified PP configuration."""

    if stage_count == 1:
        return None
    capability = _CAPABILITIES.get(architecture)
    if capability is None:
        raise ValueError(
            "native pipeline parallelism has no qualified implementation for "
            f"architecture {architecture!r}"
        )
    if activation_dtype not in capability.activation_dtypes:
        raise ValueError(
            f"native pipeline parallelism for {architecture} requires activation "
            f"dtype in {capability.activation_dtypes}, got {activation_dtype!r}"
        )
    if stage_count not in capability.stage_counts:
        raise ValueError(
            f"native pipeline parallelism for {architecture} requires stage count "
            f"in {capability.stage_counts}, got {stage_count}"
        )
    return capability
